"""Verify temporal training controls and optionally replay selected cached tails."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np

from analyze_candidate_quality import prior_tuples
from analyze_cross_stream_context import box_iou, clustered, identity_key, sha256


def replay(report):
    import torch
    from lib.config.cadtrack.config import cfg, update_config_from_file
    from miss_rgbt.model import load_pretrained
    from probe_tail_lowrank import attach, evaluate

    torch.set_num_threads(1)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    update_config_from_file('experiments/cadtrack/cadtrack.yaml')
    model = load_pretrained(cfg, report['checkpoint'], repair=False).cuda().eval().requires_grad_(False)
    cache = torch.load(report['cache_manifests']['audit']['path'], map_location='cpu')
    baseline = evaluate(model, cache)
    assert all(row['box'] == row['native_box'] and row['loss'] == row['native_loss']
               and row['iou'] == row['native_iou'] for row in baseline['rows'])
    for mode in ['native', 'chronological']:
        adapters, handles = attach(model, 'attention')
        state = torch.load(report['arms'][mode]['selected_path'], map_location='cpu')
        adapters.load_state_dict(state['adapter'], strict=True)
        assert evaluate(model, cache) == report['audit'][mode], mode
        for handle in handles:
            handle.remove()
    return dict(native_rows=1152, selected_rows=2304, cached_tail_replay_exact=True,
                independent_image_replay=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('directory')
    parser.add_argument('--server-files', action='store_true')
    parser.add_argument('--replay', action='store_true')
    parser.add_argument('--output')
    args = parser.parse_args()
    root = Path(args.directory)
    report = json.loads((root / 'report.json').read_text())
    done = json.loads((root / 'completed.json').read_text())
    launch = json.loads((root.parent / (root.name + '.launch.json')).read_text())
    assert report['status'] == done['status'] == 'complete'
    assert report['pid'] == launch['pid']
    assert sha256(root / 'report.json') == done['report_sha256']
    assert sha256(root.parent / (root.name + '.preregistered_protocol.md')) == launch['protocol_sha256']
    assert report['source_sha256'][launch['command'][2]] == launch['script_sha256']
    assert report['baseline_unchanged'] and report['baseline_gradients_absent']
    partitions = report['partitions']
    assert [len(partitions[key]) for key in ['train', 'development', 'audit']] == [705, 88, 89]
    assert len(set(sum(partitions.values(), []))) == 882
    excluded, hashes = prior_tuples(root.parent)
    for name in ['server28_candidate_quality_20261102_v2', 'server28_quality_ranker_20261102',
                 'server28_template_chronology_20261102']:
        path = root.parent / name / 'report.json'
        previous = json.loads(path.read_text())
        hashes[name] = sha256(path)
        manifests = (previous['cache_manifests'].values() if 'cache_manifests' in previous
                     else [dict(input_manifest=previous['input_manifest'])])
        for manifest in manifests:
            excluded.update(identity_key(item['identity']) for item in manifest['input_manifest'])
    for split, partition, samples, sample_seed, reversed_images in [
            ('native_train', 'train', 256, 53101, False),
            ('chronological_train', 'train', 256, 53101, True),
            ('development', 'development', 128, 53102, True),
            ('audit', 'audit', 128, 53103, True)]:
        manifest = report['cache_manifests'][split]
        assert manifest['samples'] == samples and manifest['seed'] == sample_seed
        assert manifest['rows'] == samples * 9 and manifest['native_tail_replay_exact']
        assert manifest['sequence_names'] == partitions[partition]
        assert manifest['template_images_reversed'] == reversed_images
        assert len(manifest['input_manifest']) == len(manifest['original_order_input_sha256']) == samples
        assert all(len(value) == 64 for value in manifest['original_order_input_sha256'])
        for item in manifest['input_manifest']:
            identity = item['identity']
            assert identity_key(identity) not in excluded
            assert identity['sequence'] in partitions[partition]
            newer, older = identity['template_frames']
            assert older < newer < min(identity['search_frames'])
            assert len(item['tensors_sha256']) == 64
        if args.server_files:
            disk = json.loads((root / split / 'manifest.json').read_text())
            assert all(manifest[key] == value for key, value in disk.items())
            binding = json.loads((root / split / 'temporal_binding.json').read_text())
            assert binding == dict(original_identity_order='newer,older',
                actual_template_order='older,newer' if reversed_images else 'newer,older',
                complete_template='older' if reversed_images else 'newer')
    first = report['cache_manifests']['native_train']
    second = report['cache_manifests']['chronological_train']
    assert first['original_order_input_sha256'] == second['original_order_input_sha256']
    assert [row['identity'] for row in first['input_manifest']] == [
        row['identity'] for row in second['input_manifest']]
    sizing = report['batch_sizing']
    assert sizing == json.loads((root / 'batch_sizing.json').read_text())
    static = max(item['static_bytes'] for item in sizing['probe'].values())
    per_item = max((item['batch16_peak_bytes'] - item['static_bytes']) / 16
                   for item in sizing['probe'].values())
    proposed = min(256, 16 * int((.75 * sizing['total_bytes'] - static) / (16 * per_item)))
    assert sizing['proposed'] == proposed
    batch = max(value for value in range(16, proposed + 1, 16) if 2304 % value == 0)
    assert sizing['batch'] == batch and sizing['updates_per_epoch'] == 2304 // batch
    assert sizing['learning_rate'] == 1e-4 * batch / 32
    cells = {a + '_' + b for a in ['11', '01', '10'] for b in ['11', '01', '10']}
    audit_ids = {identity_key(item['identity']) for item in
                 report['cache_manifests']['audit']['input_manifest']}
    for mode in ['native', 'chronological']:
        arm = report['arms'][mode]
        assert arm['mode'] == 'attention' and arm['training_binding'] == mode
        assert arm['parameter_count'] == 30720
        history = arm['history']
        assert 10 <= len(history) <= 20
        assert [item['epoch'] for item in history] == list(range(1, len(history) + 1))
        reference, stale = None, 0
        for item in history:
            epoch = item['epoch']
            assert item['updates'] == epoch * (2304 // batch)
            assert item['lr'] == sizing['learning_rate'] * (1 if epoch <= 6 else .1)
            assert item['score'] == float(np.mean([cell['loss'] for cell in item['cells'].values()]))
            assert set(item['cells']) == cells and np.isfinite(item['train_mean_loss'])
            if epoch >= 7:
                if reference is None or item['score'] < reference * .998:
                    reference, stale = item['score'], 0
                else:
                    stale += 1
            assert item['stale'] == stale
            if epoch >= 10 and stale >= 3:
                assert epoch == len(history)
        assert len(history) == 20 or stale >= 3
        assert arm['selected'] == min(history[-5:], key=lambda item: (item['score'], item['epoch']))
        assert [item['epoch'] for item in arm['final_five']] == [
            item['epoch'] for item in history[-5:]]
        rows = report['audit'][mode]['rows']
        assert len(rows) == 1152
        by_tuple = defaultdict(set)
        for row in rows:
            key = identity_key(row['identity'])
            assert key in audit_ids and row['cell'] not in by_tuple[key]
            by_tuple[key].add(row['cell'])
            assert all(np.isfinite(row[field]) for field in ['loss', 'iou', 'native_loss', 'native_iou'])
            for box, score in [('box', 'iou'), ('native_box', 'native_iou')]:
                assert np.isclose(box_iou(row[box], row['target']), row[score], atol=2e-6, rtol=1e-6)
        assert len(by_tuple) == 128 and all(value == cells for value in by_tuple.values())
    effects = {}
    for name, suffix in [('complete', '11_11'), ('rgb_missing', '_01'), ('tir_missing', '_10')]:
        rows = [dict(identity=row['identity'], gain=row['iou'] - row['native_iou'])
                for row in report['audit']['chronological']['rows'] if row['cell'].endswith(suffix)]
        effects[name] = clustered(rows, 'gain', 53301)
    paired = []
    for first, second in zip(report['audit']['chronological']['rows'], report['audit']['native']['rows']):
        assert all(first[key] == second[key] for key in
                   ['identity', 'cell', 'target', 'native_box', 'native_iou', 'native_loss'])
        paired.append(dict(identity=first['identity'], gain=first['iou'] - second['iou']))
    effects['chronological_minus_native_training'] = clustered(paired, 'gain', 53302)
    advance = all(effects[name]['mean'] >= .01 and effects[name]['ci95_low'] > 0
                  for name in ['complete', 'rgb_missing', 'tir_missing'])
    advance = advance and effects['chronological_minus_native_training']['ci95_low'] > 0
    assert effects == report['effects'] and advance == report['advance'] == done['advance']
    if args.server_files:
        import torch
        names = Path('lib/train/data_specs/lasher_train.txt').read_text().splitlines()
        assert partitions == dict(train=[name for index, name in enumerate(names) if index % 5],
            development=names[::5][:88], audit=names[::5][88:])
        assert sha256(report['checkpoint']) == report['checkpoint_sha256']
        for filename, expected in report['source_sha256'].items():
            assert sha256(filename) == expected, filename
        for manifest in report['cache_manifests'].values():
            assert sha256(manifest['path']) == manifest['cache_sha256']
        for arm in report['arms'].values():
            assert sha256(arm['selected_path']) == arm['selected_sha256']
            for item in arm['final_five']:
                assert sha256(item['path']) == item['sha256']
                state = torch.load(item['path'], map_location='cpu')
                assert state['epoch'] == item['epoch'] and state['mode'] == 'attention'
                assert sum(value.numel() for value in state['adapter'].values()) == 30720
                assert all(torch.isfinite(value).all() for value in state['adapter'].values())
    result = dict(status='verified', report_sha256=done['report_sha256'], effects=effects,
        advance=advance, prior_reports_sha256=hashes, server_files_checked=args.server_files,
        replay=replay(report) if args.replay else None)
    if args.output:
        with Path(args.output).open('x', newline='\n') as stream:
            json.dump(result, stream, indent=2)
            stream.write('\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
