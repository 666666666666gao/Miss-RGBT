"""Audit saved paired head-training histories, hashes and promotion decision."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def interval(rows, field, seed):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row['identity']['sequence']].append(row[field])
    values = np.array([np.mean(grouped[name]) for name in sorted(grouped)])
    rng = np.random.default_rng(seed)
    samples = rng.integers(0, len(values), size=(10000, len(values)))
    low, high = np.quantile(values[samples].mean(1), [.025, .975])
    return dict(mean=float(values.mean()), ci95_low=float(low), ci95_high=float(high),
                sequences=len(values), rows=len(rows))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('directory')
    parser.add_argument('--verify-server-files', action='store_true')
    parser.add_argument('--replay-native-heads', action='store_true')
    args = parser.parse_args()
    root = Path(args.directory)
    report = json.loads((root / 'report.json').read_text())
    done = json.loads((root / 'completed.json').read_text())
    assert report['status'] == done['status'] == 'complete'
    assert sha(root / 'report.json') == done['report_sha256']
    assert report['baseline_state_unchanged'] and report['baseline_gradients_absent']
    assert report['no_exact_prior_tuple_overlap']
    manifests = report['cache_manifests']
    partitions = [manifests[key]['sequence_names'] for key in ['train', 'development', 'audit']]
    assert [len(names) for names in partitions] == [705, 88, 89]
    assert len(set(sum(partitions, []))) == 882
    if args.verify_server_files:
        import torch
        training_names = Path('lib/train/data_specs/lasher_train.txt').read_text().splitlines()
        assert partitions == [[name for index, name in enumerate(training_names) if index % 5],
                              training_names[::5][:88], training_names[::5][88:]]
        old_pilot = json.loads((root.parent / 'server28_fusion_adapter_pilot_20261102/report.json').read_text())
        old_diagnostic = json.loads((root.parent / 'server28_response_geometry_20261102/report.json').read_text())
        def identity_key(identity):
            return (identity['sequence'], tuple(identity['template_frames']), tuple(identity['search_frames']))
        old_ids = {identity_key(item['identity'])
                   for manifest in old_pilot['cache_manifests'].values()
                   for item in manifest['input_manifest']}
        old_ids.update(identity_key(item['identity']) for item in old_diagnostic['manifest']['input_manifest'])
        for manifest in manifests.values():
            assert not old_ids.intersection(identity_key(item['identity'])
                                           for item in manifest['input_manifest'])
        original = torch.load(report['checkpoint'], map_location='cpu')['net']
        bn_names = {name[len('box_head.'):] for name in original
                    if name.startswith('box_head.') and
                    name.endswith(('running_mean', 'running_var', 'num_batches_tracked'))}
        assert bn_names
    for split, count in [('train', 256), ('development', 128), ('audit', 128)]:
        manifest = manifests[split]
        assert manifest['samples'] == count and manifest['rows'] == count * 9
        assert len(manifest['input_manifest']) == count
        assert manifest['native_head_replay_bitwise'] is True
        assert all(item['identity']['sequence'] in manifest['sequence_names']
                   for item in manifest['input_manifest'])
        if args.verify_server_files:
            assert sha(root / split / 'features.pth') == manifest['cache_sha256']
    arms = report['arms']
    assert arms['native']['initial_sha256'] == arms['target']['initial_sha256']
    assert arms['native']['parameters'] == arms['target']['parameters']
    expected_cells = {a + '_' + b for a in ['11', '01', '10'] for b in ['11', '01', '10']}
    for mode in ['native', 'target']:
        arm = arms[mode]
        assert arm['initial_metrics'] == report['baseline']['development']
        assert arm['bn_buffers_unchanged'] is True
        history = arm['history']
        assert 10 <= len(history) <= 20
        assert [item['epoch'] for item in history] == list(range(1, len(history) + 1))
        reference, stale = None, 0
        for row in history:
            assert row['updates'] == row['epoch'] * 36
            assert set(row['cells']) == expected_cells
            assert np.mean([v['loss'] for v in row['cells'].values()]) == row['selection_score']
            assert np.isclose(row['lr'], 8e-5 if row['epoch'] <= 6 else 8e-6)
            if row['epoch'] >= 7:
                if reference is None or row['selection_score'] < reference * .998:
                    reference, stale = row['selection_score'], 0
                else:
                    stale += 1
                assert row['stale_epochs'] == stale
        assert len(history) == 20 or stale >= 3
        assert min(history[-5:], key=lambda item: (item['selection_score'], item['epoch'])) == arm['selected']
        assert [item['epoch'] for item in arm['final_five']] == [row['epoch'] for row in history[-5:]]
        assert len(arm['steps']) == 36 * len(history)
        assert all(np.isfinite(step['loss']) and np.isfinite(step['gradient_norm'])
                   for step in arm['steps'])
        assert all(step['epoch'] == index // 36 + 1 for index, step in enumerate(arm['steps']))
        if args.verify_server_files:
            assert sha(arm['selected_path']) == arm['selected_sha256']
            for item in arm['final_five']:
                path = root / mode / ('epoch%02d.pth' % item['epoch'])
                assert sha(path) == item['sha256']
                saved = torch.load(str(path), map_location='cpu')
                assert all(torch.equal(saved[name], original['box_head.' + name]) for name in bn_names)
                assert all(torch.isfinite(tensor).all() for tensor in saved.values())
    paired, groupings = [], defaultdict(set)
    baseline_rows = report['baseline']['audit']['rows']
    native_rows = report['audit']['native']['rows']
    target_rows = report['audit']['target']['rows']
    assert len(baseline_rows) == len(native_rows) == len(target_rows) == 1152
    for baseline, native, target in zip(baseline_rows, native_rows, target_rows):
        assert baseline['identity'] == native['identity'] == target['identity']
        assert baseline['cell'] == native['cell'] == target['cell']
        identity = baseline['identity']
        assert identity['sequence'] in manifests['audit']['sequence_names']
        key = (identity['sequence'], tuple(identity['template_frames']), tuple(identity['search_frames']))
        assert baseline['cell'] not in groupings[key]
        groupings[key].add(baseline['cell'])
        assert all(np.isfinite(row[field]) for row in [baseline, native, target] for field in ['loss', 'iou'])
        paired.append(dict(identity=identity, cell=baseline['cell'],
            native_gain=target['iou'] - baseline['iou'],
            control_gain=target['iou'] - native['iou']))
    assert len(groupings) == 128 and all(cells == expected_cells for cells in groupings.values())
    for evaluation in [report['baseline']['audit'], report['audit']['native'], report['audit']['target']]:
        for cell, values in evaluation['cells'].items():
            selected = [row for row in evaluation['rows'] if row['cell'] == cell]
            assert len(selected) == 128
            assert all(np.mean([row[key] for row in selected]) == values[key] for key in ['loss', 'iou'])
    effects, gates = {}, []
    for state in ['01', '10']:
        selected = [row for row in paired if row['cell'].endswith('_' + state)]
        assert len(selected) == 384
        effects[state] = interval(selected, 'native_gain', 40104)
        gates.append(effects[state]['mean'] >= .01 and effects[state]['ci95_low'] > 0)
    effects['versus_native_training'] = interval(
        [row for row in paired if not row['cell'].endswith('_11')], 'control_gain', 40105)
    gates.append(effects['versus_native_training']['ci95_low'] > 0)
    for cell in ['11_11', '01_11', '10_11']:
        effects[cell] = interval([row for row in paired if row['cell'] == cell], 'native_gain', 40106)
        gates.append(effects[cell]['mean'] >= 0 and effects[cell]['ci95_low'] > -.005)
    assert effects == report['effects']
    assert all(gates) == report['advance'] == done['advance']
    if args.verify_server_files:
        assert sha(report['checkpoint']) == report['checkpoint_sha256']
        for path, digest in report['source_sha256'].items():
            assert sha(path) == digest, path
    replay = {}
    if args.replay_native_heads:
        assert args.verify_server_files
        from lib.config.cadtrack.config import cfg, update_config_from_file
        from lib.models.layers.head import build_box_head
        from lib.utils.box_ops import box_cxcywh_to_xyxy, box_xywh_to_xyxy, generalized_box_iou
        from miss_rgbt.objectives import task_loss
        torch.set_num_threads(1)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        update_config_from_file('experiments/cadtrack/cadtrack.yaml')
        cache = torch.load(str(root / 'audit/features.pth'), map_location='cpu')
        assert len(cache['labels']) == 1152
        for mode in ['baseline', 'native', 'target']:
            state = ({name[len('box_head.'):]: tensor for name, tensor in original.items()
                      if name.startswith('box_head.')} if mode == 'baseline' else
                     torch.load(arms[mode]['selected_path'], map_location='cpu'))
            head = build_box_head(cfg, cache['features'].shape[1]).cuda().eval()
            head.load_state_dict(state, strict=True)
            expected_rows = baseline_rows if mode == 'baseline' else report['audit'][mode]['rows']
            with torch.no_grad():
                for start in range(0, 1152, 16):
                    features = cache['features'][start:start + 16].cuda()
                    target = cache['boxes'][start:start + 16].cuda()
                    # Never pass a target score map to the native inference head.
                    score, boxes, sizes, offsets = head(features)
                    out = dict(score_map=score, pred_boxes=boxes.unsqueeze(1),
                               size_map=sizes, offset_map=offsets)
                    loss = task_loss([out], target.unsqueeze(0), cfg)
                    _, iou = generalized_box_iou(box_cxcywh_to_xyxy(boxes),
                                                 box_xywh_to_xyxy(target).clamp(0, 1))
                    for index in range(len(features)):
                        row = expected_rows[start + index]
                        assert row['identity'] == cache['identities'][start + index]
                        assert row['cell'] == cache['labels'][start + index]
                        assert float(loss[index]) == row['loss']
                        assert float(iou[index]) == row['iou']
            replay[mode] = dict(rows=1152, native_predicted_peak=True,
                                losses_and_ious_exact=True)
    result = dict(status='saved_evidence_verified', report_sha256=done['report_sha256'],
        server_file_hashes_checked=args.verify_server_files, effects=effects,
        advance=all(gates), native_head_replay=replay,
        scope='Saved arithmetic/hashes and optional direct cached-head replay; '
              'no backbone replay or external scientific review.')
    with (root / 'analysis.json').open('x', newline='\n') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
