"""Recompute low-rank pilot evidence; optional file checks and selected-tail replay."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np

from analyze_cross_stream_context import box_iou, clustered, identity_key, previous_ids, sha256


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
    native = evaluate(model, cache)
    assert all(row['box'] == row['native_box'] and row['loss'] == row['native_loss']
               and row['iou'] == row['native_iou'] for row in native['rows'])
    for mode in ['attention', 'mlp']:
        adapters, handles = attach(model, mode)
        state = torch.load(report['arms'][mode]['selected_path'], map_location='cpu')
        adapters.load_state_dict(state['adapter'], strict=True)
        assert evaluate(model, cache) == report['audit'][mode], mode
        for handle in handles:
            handle.remove()
    return dict(native_tail_replay_exact=True, native_rows=1152,
                selected_tail_replay_exact=True, audit_rows=2304)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('directory')
    parser.add_argument('--server-files', action='store_true')
    parser.add_argument('--replay', action='store_true')
    parser.add_argument('--output')
    args = parser.parse_args()
    root = Path(args.directory)
    report = json.loads((root / 'report.json').read_text())
    completed = json.loads((root / 'completed.json').read_text())
    launch = json.loads((root.parent / (root.name + '.launch.json')).read_text())
    assert report['status'] == completed['status'] == 'complete'
    assert report['pid'] == launch['pid']
    assert sha256(root / 'report.json') == completed['report_sha256']
    snapshot = root.parent / (root.name + '.preregistered_protocol.md')
    assert sha256(snapshot) == launch['protocol_sha256']
    assert report['source_sha256'][launch['command'][2]] == launch['script_sha256']
    assert report['baseline_unchanged'] and report['baseline_gradients_absent']
    partitions = report['partitions']
    assert [len(partitions[key]) for key in ['train', 'development', 'audit']] == [705, 88, 89]
    assert len(set(sum(partitions.values(), []))) == 882
    old_ids, previous_hashes = previous_ids(root.parent)
    cross_path = root.parent / 'server28_cross_stream_context_20261102_v2/report.json'
    cross = json.loads(cross_path.read_text())
    previous_hashes[cross_path.parent.name] = sha256(cross_path)
    for manifest in cross['cache_manifests'].values():
        old_ids.update(identity_key(item['identity']) for item in manifest['input_manifest'])
    for split, samples, sample_seed in [('train', 256, 49101),
                                       ('development', 128, 49102), ('audit', 128, 49103)]:
        manifest = report['cache_manifests'][split]
        assert manifest['samples'] == samples and manifest['seed'] == sample_seed
        assert manifest['rows'] == samples * 9
        assert manifest['sequence_names'] == partitions[split] and manifest['native_tail_replay_exact']
        assert len(manifest['input_manifest']) == samples
        for item in manifest['input_manifest']:
            assert item['identity']['sequence'] in partitions[split]
            assert identity_key(item['identity']) not in old_ids
            assert len(item['tensors_sha256']) == 64
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
    effects, advances = {}, {}
    expected_cells = {a + '_' + b for a in ['11', '01', '10'] for b in ['11', '01', '10']}
    audit_ids = {identity_key(item['identity']) for item in
                 report['cache_manifests']['audit']['input_manifest']}
    for mode in ['attention', 'mlp']:
        arm = report['arms'][mode]
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
            assert set(item['cells']) == expected_cells and np.isfinite(item['train_mean_loss'])
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
            assert np.isclose(box_iou(row['box'], row['target']), row['iou'], atol=2e-6, rtol=1e-6)
            assert np.isclose(box_iou(row['native_box'], row['target']),
                              row['native_iou'], atol=2e-6, rtol=1e-6)
        assert len(by_tuple) == 128 and all(cells == expected_cells for cells in by_tuple.values())
        effects[mode] = {}
        for state in ['11', '01', '10']:
            chosen = [dict(identity=row['identity'], gain=row['iou'] - row['native_iou'])
                      for row in rows if row['cell'].endswith('_' + state)]
            effects[mode][state] = clustered(chosen, 'gain', 49301)
        advances[mode] = all(value['mean'] >= .01 and value['ci95_low'] > 0
                             for value in effects[mode].values())
    paired = []
    for first, second in zip(report['audit']['attention']['rows'], report['audit']['mlp']['rows']):
        assert all(first[key] == second[key] for key in
                   ['identity', 'cell', 'target', 'native_loss', 'native_iou', 'native_box'])
        paired.append(dict(identity=first['identity'], gain=first['iou'] - second['iou']))
    effects['attention_minus_mlp'] = clustered(paired, 'gain', 49302)
    assert effects == report['effects']
    assert advances == report['advance'] == completed['advance']
    if args.server_files:
        import torch
        names = Path('lib/train/data_specs/lasher_train.txt').read_text().splitlines()
        assert partitions == dict(
            train=[name for index, name in enumerate(names) if index % 5],
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
                assert state['epoch'] == item['epoch']
                assert sum(value.numel() for value in state['adapter'].values()) == 30720
                assert all(torch.isfinite(value).all() for value in state['adapter'].values())
    result = dict(status='verified', report_sha256=completed['report_sha256'],
        effects=effects, advance=advances, prior_reports_sha256=previous_hashes,
        server_file_hashes_checked=args.server_files, replay=replay(report) if args.replay else None)
    if args.output:
        with Path(args.output).open('x', newline='\n') as stream:
            json.dump(result, stream, indent=2)
            stream.write('\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
