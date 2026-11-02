"""Verify dense-regression evidence without importing its producer arithmetic."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np
from analyze_box_consensus import sha256
from analyze_template_attention import exclusions as earlier_exclusions, iou
from analyze_head_supervision import interval


def exclusions(folder):
    used, hashes = earlier_exclusions(folder)
    for name in ['server28_template_attention_20261102', 'server28_box_consensus_20261102']:
        path = folder/name/'report.json'
        hashes[name+'/report.json'] = sha256(path)
        for item in json.loads(path.read_text())['input_manifest']:
            identity = item['identity']
            used[identity['sequence']].update(identity['template_frames']+identity['search_frames'])
    return used, hashes


def server_checks(report, root, replay):
    import torch
    from lib.config.cadtrack.config import cfg, update_config_from_file
    from lib.models.layers.head import build_box_head
    from lib.train.dataset import LasHeR
    from lib.test.utils.hann import hann2d
    from lib.utils.heapmap_utils import generate_heatmap
    from lib.utils.box_ops import box_cxcywh_to_xyxy, box_xywh_to_xyxy, generalized_box_iou
    from miss_rgbt.objectives import task_loss

    torch.set_num_threads(1)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    update_config_from_file('experiments/cadtrack/cadtrack.yaml')
    for path, expected in report['source_sha256'].items():
        assert sha256(path) == expected, path
    assert sha256(report['checkpoint']) == report['checkpoint_sha256']
    names = Path('lib/train/data_specs/lasher_train.txt').read_text().splitlines()
    assert report['partitions'] == dict(train=[name for index, name in enumerate(names) if index % 5],
        development=names[::5][:88], audit=names[::5][88:])
    dataset = LasHeR(root='/data/wangwj/dataset/LasHeR/trainingset', split='train')
    for manifest in report['cache_manifests'].values():
        assert sha256(manifest['cache_path']) == manifest['cache_sha256']
        for item in manifest['input_manifest']:
            identity = item['identity']
            info = dataset.get_sequence_info(dataset.sequence_list.index(identity['sequence']))
            assert info['visible'][identity['template_frames']+identity['search_frames']].all()
    native = torch.load(report['checkpoint'], map_location='cpu')['net']
    native_head = {key[len('box_head.'):]: value for key, value in native.items()
                   if key.startswith('box_head.')}
    verified_weights = 0
    for arm in report['arms'].values():
        assert sha256(arm['selected_path']) == arm['selected_sha256']
        for item in arm['final_five']:
            assert sha256(item['path']) == item['sha256']
            checkpoint = torch.load(item['path'], map_location='cpu')
            assert checkpoint['epoch'] == item['epoch'] and checkpoint['mode'] == arm['mode']
            state = checkpoint['head']
            assert all(torch.isfinite(value).all() for value in state.values())
            for key, value in state.items():
                if key.endswith(('running_mean', 'running_var', 'num_batches_tracked')):
                    assert torch.equal(value, native_head[key])
            assert sum(value.numel() for key, value in state.items()
                if not key.endswith(('running_mean', 'running_var', 'num_batches_tracked'))) == 6474117
            verified_weights += 1
    witness = report['objective_witness']
    assert sha256(witness['path']) == witness['sha256']
    saved = torch.load(witness['path'], map_location='cpu')
    target = saved['target'].double().numpy()
    gt = np.clip(np.concatenate([target[:, :2], target[:, :2]+target[:, 2:]], -1), 0, 1)
    offset = saved['offset_map'].double().numpy().reshape(16, 2, 256)
    size = saved['size_map'].double().numpy().reshape(16, 2, 256)
    index = np.arange(256)
    center = np.stack([(index[None] % 16+offset[:, 0])/16,
                       (index[None]//16+offset[:, 1])/16], -1)
    wh = size.transpose(0, 2, 1)
    corners = np.concatenate([center-wh/2, center+wh/2], -1)
    truth = gt[:, None]
    intersection = np.maximum(0, np.minimum(corners[..., 2:], truth[..., 2:])-
        np.maximum(corners[..., :2], truth[..., :2])).prod(-1)
    union = wh.prod(-1)+(truth[..., 2:]-truth[..., :2]).prod(-1)-intersection
    enclosure = (np.maximum(corners[..., 2:], truth[..., 2:])-
                 np.minimum(corners[..., :2], truth[..., :2])).prod(-1)
    giou = intersection/union-(enclosure-union)/enclosure
    regression = 2*(1-giou)+5*np.abs(corners-truth).mean(-1)
    np.testing.assert_allclose(regression, saved['regression'].numpy(), atol=3e-6, rtol=1e-6)
    heatmap = generate_heatmap(saved['target'][None], 256, 16)[0].flatten(1).numpy()
    peak = heatmap.argmax(-1)
    support = (np.abs(index[None] % 16-peak[:, None] % 16) <= 1) & (
               np.abs(index[None]//16-peak[:, None]//16) <= 1)
    assert np.array_equal(support, saved['support'].numpy())
    window = hann2d(torch.tensor([16, 16]), centered=True)
    location = (saved['score_map']*window).flatten(1).numpy().argmax(-1)
    expected = (saved['native_loss'].double().numpy()-regression[np.arange(16), location]
                +(regression*support).sum(-1)/support.sum(-1))
    np.testing.assert_allclose(expected, saved['dense_loss'].numpy(), atol=4e-6, rtol=1e-6)
    gradient = saved['offset_gradient'].numpy()
    assert np.isfinite(gradient).all()
    assert (gradient*np.logical_not(support[:, None].reshape(16, 1, 16, 16)) == 0).all()
    # Reconstruct map-level regression and differentiate a separate expression.
    offset_tensor = saved['offset_map'].clone().requires_grad_(True)
    size_tensor = saved['size_map']
    locations = torch.arange(256)
    dx = (locations.remainder(16)[None]+offset_tensor.flatten(2)[:, 0])/16
    dy = (locations.div(16, rounding_mode='floor')[None]+offset_tensor.flatten(2)[:, 1])/16
    boxes = torch.stack([dx, dy, size_tensor.flatten(2)[:, 0], size_tensor.flatten(2)[:, 1]], -1)
    predicted = box_cxcywh_to_xyxy(boxes).reshape(-1, 4)
    target_tensor = box_xywh_to_xyxy(saved['target']).clamp(0, 1)[:, None].expand(-1, 256, -1).reshape(-1, 4)
    g, _ = generalized_box_iou(predicted, target_tensor)
    reg = (2*(1-g)+5*(predicted-target_tensor).abs().mean(-1)).reshape(16, 256)
    support_tensor = torch.from_numpy(support)
    expression = ((reg*support_tensor).sum(-1)/support_tensor.sum(-1)).sum()
    independently_differentiated = torch.autograd.grad(expression, offset_tensor)[0]
    gradient_error = float((independently_differentiated-saved['offset_gradient']).abs().max())
    torch.testing.assert_close(independently_differentiated, saved['offset_gradient'],
                               atol=2e-6, rtol=2e-6)
    checked = 0
    if replay:
        cache = torch.load(root/'audit/features.pth', map_location='cpu')
        for mode in ['baseline', 'native', 'dense']:
            head = build_box_head(cfg, 768).cuda().eval()
            state = native_head if mode == 'baseline' else torch.load(
                report['arms'][mode]['selected_path'], map_location='cpu')['head']
            head.load_state_dict(state, strict=True)
            expected_rows = (report['baseline'] if mode == 'baseline' else report['audit'][mode])['rows']
            with torch.no_grad():
                for start in range(0, len(cache['labels']), 16):
                    score, _, sizes, offsets = head(cache['features'][start:start+16].cuda())
                    location = (score*window.cuda()).flatten(1).argmax(1, keepdim=True)
                    gather = location[:, None].expand(-1, 2, 1)
                    sizes = sizes.flatten(2).gather(2, gather).squeeze(-1)
                    offsets = offsets.flatten(2).gather(2, gather).squeeze(-1)
                    box = torch.cat([(location.remainder(16).float()+offsets[:, :1])/16,
                        ((location//16).float()+offsets[:, 1:])/16, sizes], 1)
                    target_batch = cache['boxes'][start:start+16].cuda()
                    loss = task_loss([dict(score_map=score, pred_boxes=box[:, None])], target_batch[None], cfg)
                    _, overlaps = generalized_box_iou(box_cxcywh_to_xyxy(box),
                        box_xywh_to_xyxy(target_batch).clamp(0, 1))
                    for local in range(len(box)):
                        row = expected_rows[start+local]
                        assert row['box'] == box[local].cpu().tolist()
                        assert row['target'] == target_batch[local].cpu().tolist()
                        assert row['iou'] == float(overlaps[local]) and row['loss'] == float(loss[local])
                        assert row['identity'] == cache['identities'][start+local]
                        assert row['cell'] == cache['labels'][start+local]
                        checked += 1
            del head
    return dict(final_five_weights=verified_weights, objective_rows=16,
        objective_offset_gradient_max_error=gradient_error, replay_rows=checked,
        limitation='Independent formula/map gradient and native head decode; shared native head/taskloss; no image replay.')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--server-files', action='store_true')
    parser.add_argument('--replay', action='store_true')
    args = parser.parse_args()
    root = Path(args.root)
    report = json.loads((root/'report.json').read_text())
    completed = json.loads((root/'completed.json').read_text())
    launch = json.loads(Path(str(root)+'.launch.json').read_text())
    assert report['status'] == completed['status'] == 'complete'
    assert completed['report_sha256'] == sha256(root/'report.json')
    assert launch['pid'] == report['pid']
    assert launch['protocol_sha256'] == sha256(Path(str(root)+'.preregistered_protocol.md'))
    assert launch['script_sha256'] == sha256(root.parent.parent/'scripts/train_dense_geometry.py')
    assert report['baseline_unchanged'] and report['baseline_gradients_absent']
    used, hashes = exclusions(root.parent)
    for key, value in report['prior_reports_sha256'].items():
        assert hashes[key] == value
    assert [len(report['partitions'][key]) for key in ['train', 'development', 'audit']] == [705, 88, 89]
    assert len(set(sum(report['partitions'].values(), []))) == 882
    for split, count, seed in [('train', 256, 62101), ('development', 128, 62102), ('audit', 128, 62103)]:
        manifest = report['cache_manifests'][split]
        assert manifest['samples'] == count and manifest['rows'] == count*9
        assert manifest['seed'] == seed and manifest['native_head_replay_bitwise']
        assert manifest['sequence_names'] == report['partitions'][split]
        assert len(manifest['input_manifest']) == count
        for item in manifest['input_manifest']:
            identity = item['identity']
            assert identity['sequence'] in manifest['sequence_names']
            assert not used[identity['sequence']].intersection(
                identity['template_frames']+identity['search_frames'])
    sizing = report['batch_sizing']
    proposed = int(64*(.8*sizing['total_bytes']-sizing['static_bytes'])/
                   (sizing['peak_bytes']-sizing['static_bytes']))
    batch = max(value for value in range(32, min(proposed, 2304)+1, 32) if 2304 % value == 0)
    assert sizing['batch'] == batch and sizing['proposed'] == proposed
    assert sizing['learning_rate'] == 1e-5*batch/8 and sizing['updates_per_epoch'] == 2304//batch
    assert report['arms']['native']['initial_sha256'] == report['arms']['dense']['initial_sha256']
    assert report['arms']['native']['initial_metrics'] == report['arms']['dense']['initial_metrics']
    for arm in report['arms'].values():
        assert arm['parameter_count'] == 6474117 and arm['bn_buffers_unchanged']
        history = arm['history']
        assert 10 <= len(history) <= 20
        reference, stale = None, 0
        for epoch, row in enumerate(history, 1):
            assert row['epoch'] == epoch and row['updates'] == epoch*2304//batch
            assert row['lr'] == sizing['learning_rate']*(1 if epoch <= 6 else .1)
            assert row['score'] == float(np.mean([value['loss'] for value in row['cells'].values()]))
            assert np.isfinite(row['train_mean_loss'])
            if epoch >= 7:
                if reference is None or row['score'] < reference*.998:
                    reference, stale = row['score'], 0
                else:
                    stale += 1
            assert row['stale'] == stale
            if epoch >= 10 and stale >= 3:
                assert epoch == len(history)
        assert len(history) == 20 or stale >= 3
        assert arm['selected'] == min(history[-5:], key=lambda row: (row['score'], row['epoch']))
        assert [r['epoch'] for r in arm['final_five']] == [r['epoch'] for r in history[-5:]]
    cells = {a+'_'+b for a in ['11', '01', '10'] for b in ['11', '01', '10']}
    audit_ids = {(item['identity']['sequence'], tuple(item['identity']['template_frames']),
                  tuple(item['identity']['search_frames']))
                 for item in report['cache_manifests']['audit']['input_manifest']}
    for result in [report['baseline']]+list(report['audit'].values()):
        assert len(result['rows']) == 1152 and set(result['cells']) == cells
        seen = defaultdict(set)
        for row in result['rows']:
            identity = row['identity']
            key = (identity['sequence'], tuple(identity['template_frames']), tuple(identity['search_frames']))
            assert key in audit_ids
            assert row['cell'] not in seen[key]
            seen[key].add(row['cell'])
            assert abs(iou(row['box'], row['target'])-row['iou']) < 2e-6
            assert np.isfinite(row['loss'])
        assert len(seen) == 128 and all(value == cells for value in seen.values())
    paired = []
    for base, native, dense in zip(report['baseline']['rows'],
                                 report['audit']['native']['rows'], report['audit']['dense']['rows']):
        assert all(base[key] == native[key] == dense[key] for key in ['identity', 'cell', 'target'])
        paired.append(dict(identity=base['identity'], cell=base['cell'],
            gain=dense['iou']-base['iou'], control=dense['iou']-native['iou']))
    effects = {state: interval([row for row in paired if row['cell'].endswith('_'+state)], 'gain', 62301)
               for state in ['01', '10']}
    effects['versus_native_training'] = interval(
        [row for row in paired if not row['cell'].endswith('_11')], 'control', 62302)
    for cell in ['11_11', '01_11', '10_11']:
        effects[cell] = interval([row for row in paired if row['cell'] == cell], 'gain', 62303)
    advance = (all(effects[state]['mean'] >= .01 and effects[state]['ci95_low'] > 0
        for state in ['01', '10']) and effects['versus_native_training']['ci95_low'] > 0
        and all(effects[cell]['mean'] >= 0 and effects[cell]['ci95_low'] > -.005
                for cell in ['11_11', '01_11', '10_11'])
        and all(value['sequences'] >= 32 for value in effects.values()))
    assert effects == report['effects'] and bool(advance) == report['advance'] == completed['advance']
    result = dict(status='verified', report_sha256=completed['report_sha256'],
        analyzer_sha256=sha256(__file__), effects=effects, advance=bool(advance),
        independent_prior_reports_sha256=hashes,
        server_checks=server_checks(report, root, args.replay) if args.server_files else None,
        limitation='Cached-head training diagnostic; no official metrics, full-image replay or module validation.')
    with Path(args.output).open('x', newline='\n') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
