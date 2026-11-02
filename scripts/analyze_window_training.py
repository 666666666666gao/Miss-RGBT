"""Recalculate window-supervision evidence and optionally replay native heads."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np

from analyze_candidate_quality import prior_tuples
from analyze_cross_stream_context import box_iou, clustered, identity_key, sha256


def replay(report, root):
    import torch
    from lib.config.cadtrack.config import cfg, update_config_from_file
    from lib.models.layers.head import build_box_head
    from lib.test.utils.hann import hann2d
    from miss_rgbt.objectives import task_loss
    from lib.utils.box_ops import box_cxcywh_to_xyxy, box_xywh_to_xyxy, generalized_box_iou

    torch.set_num_threads(1)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    update_config_from_file('experiments/cadtrack/cadtrack.yaml')
    native = torch.load(report['checkpoint'], map_location='cpu')['net']
    native_head = {key[len('box_head.'):]: value for key, value in native.items()
                   if key.startswith('box_head.')}
    cache = torch.load(root / 'audit/features.pth', map_location='cpu')
    window = hann2d(torch.tensor([16, 16]), centered=True).cuda()
    checked = 0
    for mode in ['baseline', 'raw', 'hann']:
        head = build_box_head(cfg, 768).cuda().eval()
        state = native_head if mode == 'baseline' else torch.load(
            report['arms'][mode]['selected_path'], map_location='cpu')['head']
        head.load_state_dict(state, strict=True)
        expected = report['baseline'] if mode == 'baseline' else report['audit'][mode]
        with torch.no_grad():
            for start in range(0, len(cache['labels']), 16):
                score, _, size, offset = head(cache['features'][start:start + 16].cuda())
                # Decode independently of the producer's cal_bbox call.
                peak = (score * window).flatten(1).argmax(1, keepdim=True)
                locations = peak[:, None].expand(-1, 2, 1)
                sizes = size.flatten(2).gather(2, locations).squeeze(-1)
                offsets = offset.flatten(2).gather(2, locations).squeeze(-1)
                box = torch.cat([(peak.remainder(16).float() + offsets[:, :1]) / 16,
                    ((peak // 16).float() + offsets[:, 1:]) / 16, sizes], 1)
                target = cache['boxes'][start:start + 16].cuda()
                loss = task_loss([dict(score_map=score, pred_boxes=box[:, None])],
                                 target[None], cfg)
                _, iou = generalized_box_iou(box_cxcywh_to_xyxy(box),
                                             box_xywh_to_xyxy(target).clamp(0, 1))
                for index in range(len(target)):
                    row = expected['rows'][start + index]
                    assert row['identity'] == cache['identities'][start + index]
                    assert row['cell'] == cache['labels'][start + index]
                    assert row['target'] == target[index].cpu().tolist()
                    assert row['box'] == box[index].cpu().tolist()
                    assert row['loss'] == float(loss[index]) and row['iou'] == float(iou[index])
                    checked += 1
        del head
    return dict(rows=checked, cached_head_replay_exact=True,
                independent_hann_decode=True, independent_image_replay=False)


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
    excluded, prior_hashes = prior_tuples(root.parent)
    for name in ['server28_candidate_quality_20261102_v2', 'server28_quality_ranker_20261102',
                 'server28_template_chronology_20261102', 'server28_chronology_training_20261102']:
        path = root.parent / name / 'report.json'
        prior = json.loads(path.read_text())
        prior_hashes[name] = sha256(path)
        manifests = (prior['cache_manifests'].values() if 'cache_manifests' in prior
                     else [dict(input_manifest=prior['input_manifest'])])
        for manifest in manifests:
            excluded.update(identity_key(item['identity']) for item in manifest['input_manifest'])
    for split, samples, cache_seed in [('train', 256, 54101), ('development', 128, 54102), ('audit', 128, 54103)]:
        manifest = report['cache_manifests'][split]
        assert manifest['samples'] == samples and manifest['rows'] == samples * 9
        assert manifest['seed'] == cache_seed and manifest['native_head_replay_bitwise']
        assert manifest['sequence_names'] == partitions[split]
        assert len(manifest['input_manifest']) == samples
        for item in manifest['input_manifest']:
            assert identity_key(item['identity']) not in excluded
            assert item['identity']['sequence'] in partitions[split]
            assert all(len(item[key]) == 64 for key in ['templates_sha256', 'search_sha256', 'boxes_sha256'])
    sizing = report['batch_sizing']
    assert sizing == json.loads((root / 'batch_sizing.json').read_text())
    proposed = min(256, 16 * int((.75 * sizing['total_bytes'] - sizing['static_bytes']) /
        ((sizing['peak_bytes'] - sizing['static_bytes']) / 64 * 16)))
    assert sizing['proposed'] == proposed
    batch = max(value for value in range(16, proposed + 1, 16) if 2304 % value == 0)
    assert sizing['batch'] == batch and sizing['updates_per_epoch'] == 2304 // batch
    assert sizing['learning_rate'] == 1e-5 * batch / 8
    assert report['arms']['raw']['initial_sha256'] == report['arms']['hann']['initial_sha256']
    assert report['arms']['raw']['initial_metrics'] == report['arms']['hann']['initial_metrics']
    for mode in ['raw', 'hann']:
        arm = report['arms'][mode]
        assert arm['parameter_count'] == 6474117 and arm['bn_buffers_unchanged']
        history = arm['history']
        assert 10 <= len(history) <= 20
        assert [row['epoch'] for row in history] == list(range(1, len(history) + 1))
        reference, stale = None, 0
        for row in history:
            epoch = row['epoch']
            assert row['updates'] == epoch * (2304 // batch)
            assert row['lr'] == sizing['learning_rate'] * (1 if epoch <= 6 else .1)
            assert np.isfinite(row['train_mean_loss'])
            assert row['score'] == float(np.mean([value['loss'] for value in row['cells'].values()]))
            if epoch >= 7:
                if reference is None or row['score'] < reference * .998:
                    reference, stale = row['score'], 0
                else:
                    stale += 1
            assert row['stale'] == stale
            if epoch >= 10 and stale >= 3:
                assert epoch == len(history)
        assert len(history) == 20 or stale >= 3
        assert arm['selected'] == min(history[-5:], key=lambda row: (row['score'], row['epoch']))
        assert [row['epoch'] for row in arm['final_five']] == [row['epoch'] for row in history[-5:]]
    cells = {a + '_' + b for a in ['11', '01', '10'] for b in ['11', '01', '10']}
    ids = {identity_key(item['identity']) for item in report['cache_manifests']['audit']['input_manifest']}
    for result in [report['baseline']] + list(report['audit'].values()):
        assert len(result['rows']) == 1152 and set(result['cells']) == cells
        by_tuple = defaultdict(set)
        for row in result['rows']:
            key = identity_key(row['identity'])
            assert key in ids and row['cell'] not in by_tuple[key]
            by_tuple[key].add(row['cell'])
            assert np.isfinite(row['loss']) and np.isfinite(row['iou'])
            assert np.isclose(box_iou(row['box'], row['target']), row['iou'], atol=2e-6, rtol=1e-6)
        assert len(by_tuple) == 128 and all(value == cells for value in by_tuple.values())
    paired = []
    for base, raw, hann in zip(report['baseline']['rows'], report['audit']['raw']['rows'], report['audit']['hann']['rows']):
        assert all(base[key] == raw[key] == hann[key] for key in ['identity', 'cell', 'target'])
        paired.append(dict(identity=base['identity'], cell=base['cell'],
            gain=hann['iou'] - base['iou'], control=hann['iou'] - raw['iou']))
    effects = {state: clustered([row for row in paired if row['cell'].endswith('_' + state)],
                               'gain', 54301) for state in ['01', '10']}
    effects['hann_minus_raw_training'] = clustered(
        [row for row in paired if not row['cell'].endswith('_11')], 'control', 54302)
    for cell in ['11_11', '01_11', '10_11']:
        effects[cell] = clustered([row for row in paired if row['cell'] == cell], 'gain', 54303)
    advance = all(effects[state]['mean'] >= .01 and effects[state]['ci95_low'] > 0 for state in ['01', '10'])
    advance = advance and effects['hann_minus_raw_training']['ci95_low'] > 0
    advance = advance and all(effects[cell]['mean'] >= 0 and effects[cell]['ci95_low'] > -.005
                             for cell in ['11_11', '01_11', '10_11'])
    assert effects == report['effects'] and advance == report['advance'] == done['advance']
    if args.server_files:
        import torch
        names = Path('lib/train/data_specs/lasher_train.txt').read_text().splitlines()
        assert partitions == dict(train=[name for index, name in enumerate(names) if index % 5],
            development=names[::5][:88], audit=names[::5][88:])
        assert sha256(report['checkpoint']) == report['checkpoint_sha256']
        for filename, expected in report['source_sha256'].items():
            assert sha256(filename) == expected, filename
        for split, manifest in report['cache_manifests'].items():
            assert sha256(root / split / 'features.pth') == manifest['cache_sha256']
        native = torch.load(report['checkpoint'], map_location='cpu')['net']
        for arm in report['arms'].values():
            assert sha256(arm['selected_path']) == arm['selected_sha256']
            for item in arm['final_five']:
                assert sha256(item['path']) == item['sha256']
                state = torch.load(item['path'], map_location='cpu')
                assert state['mode'] == arm['mode'] and state['epoch'] == item['epoch']
                # Buffers are part of state_dict, but not trainable parameters.
                assert all(torch.isfinite(value).all() for value in state['head'].values())
                for key, value in state['head'].items():
                    if key.endswith(('running_mean', 'running_var', 'num_batches_tracked')):
                        assert torch.equal(value, native['box_head.' + key])
                parameters = sum(value.numel() for key, value in state['head'].items()
                    if not key.endswith(('running_mean', 'running_var', 'num_batches_tracked')))
                assert parameters == 6474117
    result = dict(status='verified', report_sha256=done['report_sha256'], effects=effects,
        advance=advance, prior_reports_sha256=prior_hashes, server_files_checked=args.server_files,
        replay=replay(report, root) if args.replay else None)
    if args.output:
        save_path = Path(args.output)
        with save_path.open('x', newline='\n') as stream:
            json.dump(result, stream, indent=2)
            stream.write('\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
