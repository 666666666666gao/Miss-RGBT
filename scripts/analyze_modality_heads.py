"""Verify capacity, routing, training receipts and directly replay cached heads."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from scripts.analyze_head_supervision import sha, interval


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('directory')
    parser.add_argument('--replay-native-heads', action='store_true')
    args = parser.parse_args()
    root = Path(args.directory)
    report = json.loads((root / 'report.json').read_text())
    done = json.loads((root / 'completed.json').read_text())
    assert report['status'] == done['status'] == 'complete'
    assert sha(root / 'report.json') == done['report_sha256']
    assert report['baseline_state_unchanged'] and report['baseline_gradients_absent']
    assert report['no_exact_prior_tuple_overlap']
    for path, expected in report['source_sha256'].items():
        assert sha(path) == expected, path
    assert sha(report['checkpoint']) == report['checkpoint_sha256']
    original = torch.load(report['checkpoint'], map_location='cpu')['net']
    initial_digest = hashlib.sha256()
    for name, value in sorted((key[len('box_head.'):], tensor) for key, tensor in original.items()
                              if key.startswith('box_head.')):
        initial_digest.update(name.encode())
        initial_digest.update(value.contiguous().numpy().tobytes())
    bn_names = {key[len('box_head.'):] for key in original if key.startswith('box_head.')
                and key.endswith(('running_mean', 'running_var', 'num_batches_tracked'))}
    assert bn_names
    names = Path('lib/train/data_specs/lasher_train.txt').read_text().splitlines()
    expected_names = dict(train=[name for i, name in enumerate(names) if i % 5],
                          development=names[::5][:88], audit=names[::5][88:])
    assert len(set(names)) == 882
    prior = [json.loads((root.parent / name / 'report.json').read_text())
             for name in ['server28_fusion_adapter_pilot_20261102', 'server28_head_supervision_20261102']]
    diagnostic = json.loads((root.parent / 'server28_response_geometry_20261102/report.json').read_text())
    old_manifests = [manifest for record in prior for manifest in record['cache_manifests'].values()]
    old_manifests.append(diagnostic['manifest'])
    def identity_key(identity):
        return (identity['sequence'], tuple(identity['template_frames']), tuple(identity['search_frames']))
    old_ids = {identity_key(item['identity']) for manifest in old_manifests
               for item in manifest['input_manifest']}
    cache = {}
    expected_cells = {a + '_' + b for a in ['11', '01', '10'] for b in ['11', '01', '10']}
    for split, count, seed in [('train', 256, 42104), ('development', 128, 42105),
                                ('audit', 128, 42106)]:
        manifest = report['cache_manifests'][split]
        assert manifest['sequence_names'] == expected_names[split]
        assert manifest['samples'] == count and manifest['rows'] == count * 9
        assert manifest['native_head_replay_bitwise'] is True
        assert len(manifest['input_manifest']) == count
        assert not old_ids.intersection(identity_key(item['identity']) for item in manifest['input_manifest'])
        assert sha(root / split / 'features.pth') == manifest['cache_sha256']
        cache[split] = torch.load(str(root / split / 'features.pth'), map_location='cpu')
        labels = cache[split]['labels']
        assert len(labels) == count * 9 and set(labels) == expected_cells
        assert report['route_seeds'][split] == seed
        expected_random = torch.full((len(labels),), -1, dtype=torch.long)
        generator = torch.Generator().manual_seed(seed)
        for cell in sorted(expected_cells):
            indices = torch.tensor([i for i, label in enumerate(labels) if label == cell])
            assert len(indices) == count
            if cell.endswith('_11'):
                continue
            shuffled = indices[torch.randperm(count, generator=generator)]
            expected_random[shuffled[:count // 2]] = 0
            expected_random[shuffled[count // 2:]] = 1
        assert expected_random.tolist() == report['routes']['random'][split]
        expected_modality = [-1 if label.endswith('_11') else
                             (0 if label.endswith('_10') else 1) for label in labels]
        assert expected_modality == report['routes']['modality'][split]
    arms = report['arms']
    assert arms['modality']['parameters'] == arms['random']['parameters'] == 12948234
    assert arms['modality']['initial_sha256'] == arms['random']['initial_sha256']
    for mode, arm in arms.items():
        assert len(set(arm['initial_sha256'])) == 1 and arm['bn_buffers_unchanged']
        assert arm['initial_sha256'] == [initial_digest.hexdigest()] * 2
        assert report['routes'][mode]['train'].count(0) == 768
        assert report['routes'][mode]['train'].count(1) == 768
        for row, baseline in zip(arm['initial']['rows'], report['baseline']['development']['rows']):
            assert row['identity'] == baseline['identity'] and row['cell'] == baseline['cell']
            assert row['loss'] == baseline['loss'] and row['iou'] == baseline['iou']
        history = arm['history']
        assert 10 <= len(history) <= 20
        reference, stale = None, 0
        for index, row in enumerate(history):
            assert row['epoch'] == index + 1 and row['updates'] == 24 * (index + 1)
            score = np.mean([values['loss'] for cell, values in row['cells'].items()
                             if not cell.endswith('_11')])
            assert score == row['selection_score']
            for cell in ['11_11', '01_11', '10_11']:
                assert row['cells'][cell] == report['baseline']['development']['cells'][cell]
            assert np.isclose(row['lr'], 8e-5 if index < 6 else 8e-6)
            if row['epoch'] >= 7:
                if reference is None or score < reference * .998:
                    reference, stale = score, 0
                else:
                    stale += 1
                assert stale == row['stale_epochs']
        assert len(history) == 20 or stale >= 3
        assert min(history[-5:], key=lambda row: (row['selection_score'], row['epoch'])) == arm['selected']
        assert [item['epoch'] for item in arm['final_five']] == [row['epoch'] for row in history[-5:]]
        assert sha(arm['selected_path']) == arm['selected_sha256']
        assert len(arm['steps']) == len(history) * 24
        for index, step in enumerate(arm['steps']):
            assert step['epoch'] == index // 24 + 1 and step['branch'] == index % 24 // 12
            assert np.isfinite(step['loss']) and np.isfinite(step['norm'])
        for item in arm['final_five']:
            path = root / mode / ('epoch%02d.pth' % item['epoch'])
            assert sha(path) == item['sha256']
            weights = torch.load(str(path), map_location='cpu')
            assert len(weights) == 2
            for state in weights:
                assert all(torch.isfinite(tensor).all() for tensor in state.values())
                assert all(torch.equal(state[name], original['box_head.' + name]) for name in bn_names)
    paired = []
    grouped = defaultdict(set)
    base_rows = report['baseline']['audit']['rows']
    modality_rows, random_rows = [report['audit'][mode]['rows'] for mode in ['modality', 'random']]
    assert len(base_rows) == len(modality_rows) == len(random_rows) == 1152
    for index, (base, specialist, control) in enumerate(zip(base_rows, modality_rows, random_rows)):
        assert base['identity'] == specialist['identity'] == control['identity'] == cache['audit']['identities'][index]
        assert base['cell'] == specialist['cell'] == control['cell'] == cache['audit']['labels'][index]
        assert specialist['route'] == report['routes']['modality']['audit'][index]
        assert control['route'] == report['routes']['random']['audit'][index]
        key = identity_key(base['identity'])
        assert base['cell'] not in grouped[key]
        grouped[key].add(base['cell'])
        if base['cell'].endswith('_11'):
            assert base['loss'] == specialist['loss'] == control['loss']
            assert base['iou'] == specialist['iou'] == control['iou']
        assert all(np.isfinite(row[field]) for row in [base, specialist, control] for field in ['loss', 'iou'])
        paired.append(dict(identity=base['identity'], cell=base['cell'],
            baseline_gain=specialist['iou'] - base['iou'],
            control_gain=specialist['iou'] - control['iou']))
    assert len(grouped) == 128 and all(cells == expected_cells for cells in grouped.values())
    for evaluation in [report['baseline']['audit'], report['audit']['modality'], report['audit']['random']]:
        assert evaluation['complete_search_maps_equal'] is True
        for cell, values in evaluation['cells'].items():
            selected = [row for row in evaluation['rows'] if row['cell'] == cell]
            assert len(selected) == 128
            assert all(np.mean([row[field] for row in selected]) == values[field] for field in ['loss', 'iou'])
    effects, gates = {}, []
    for state in ['01', '10']:
        effects[state] = interval([row for row in paired if row['cell'].endswith('_' + state)],
                                  'baseline_gain', 42107)
        gates.append(effects[state]['mean'] >= .01 and effects[state]['ci95_low'] > 0)
    effects['versus_random'] = interval([row for row in paired if not row['cell'].endswith('_11')],
                                      'control_gain', 42108)
    gates.append(effects['versus_random']['ci95_low'] > 0)
    assert effects == report['effects'] and all(gates) == report['advance'] == done['advance']
    replay = {}
    if args.replay_native_heads:
        from lib.config.cadtrack.config import cfg, update_config_from_file
        from lib.models.layers.head import build_box_head
        from lib.utils.box_ops import box_cxcywh_to_xyxy, box_xywh_to_xyxy, generalized_box_iou
        from miss_rgbt.objectives import task_loss
        torch.set_num_threads(1)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        update_config_from_file('experiments/cadtrack/cadtrack.yaml')
        original_head = build_box_head(cfg, cache['audit']['features'].shape[1]).cuda().eval()
        original_head.load_state_dict({k[len('box_head.'):]: v for k, v in original.items()
                                      if k.startswith('box_head.')}, strict=True)
        for mode in ['baseline', 'modality', 'random']:
            heads = []
            if mode != 'baseline':
                for state in torch.load(arms[mode]['selected_path'], map_location='cpu'):
                    head = build_box_head(cfg, cache['audit']['features'].shape[1]).cuda().eval()
                    head.load_state_dict(state, strict=True)
                    heads.append(head)
            expected_rows = base_rows if mode == 'baseline' else report['audit'][mode]['rows']
            routes = [-1] * 1152 if mode == 'baseline' else report['routes'][mode]['audit']
            with torch.no_grad():
                for start in range(0, 1152, 16):
                    fused = cache['audit']['features'][start:start + 16].cuda()
                    target = cache['audit']['boxes'][start:start + 16].cuda()
                    native = original_head(fused)
                    predictions = list(native)
                    route = torch.tensor(routes[start:start + 16], device='cuda')
                    if bool((route >= 0).any()):
                        for branch, head in enumerate(heads):
                            values = head(fused)
                            for index in range(4):
                                mask = (route == branch).reshape((-1,) + (1,) * (values[index].ndim - 1))
                                predictions[index] = torch.where(mask, values[index], predictions[index])
                    complete = route == -1
                    assert all(torch.equal(predictions[i][complete], native[i][complete]) for i in range(4))
                    score, boxes, sizes, offsets = predictions
                    loss = task_loss([dict(score_map=score, pred_boxes=boxes.unsqueeze(1),
                        size_map=sizes, offset_map=offsets)], target.unsqueeze(0), cfg)
                    _, iou = generalized_box_iou(box_cxcywh_to_xyxy(boxes),
                                                 box_xywh_to_xyxy(target).clamp(0, 1))
                    for index in range(len(fused)):
                        assert float(loss[index]) == expected_rows[start + index]['loss']
                        assert float(iou[index]) == expected_rows[start + index]['iou']
            replay[mode] = dict(rows=1152, native_inference=True, exact_losses_and_ious=True,
                                complete_search_maps_equal=True)
    result = dict(status='saved_evidence_verified', report_sha256=done['report_sha256'],
        effects=effects, advance=all(gates), native_head_replay=replay,
        scope='Saved routing/arithmetic/hash checks and optional cached-head replay; '
              'no image/backbone replay or external scientific review.')
    with (root / 'analysis.json').open('x', newline='\n') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
