"""Paired train-only native-head objective control with native inference."""
import argparse
import copy
import json
import os
from pathlib import Path
import time

import cv2
import numpy as np
import torch

from lib.config.cadtrack.config import cfg, update_config_from_file
from lib.utils.box_ops import box_cxcywh_to_xyxy, box_xywh_to_xyxy, generalized_box_iou
from lib.utils.heapmap_utils import generate_heatmap
from miss_rgbt.model import load_pretrained
from miss_rgbt.objectives import task_loss
from scripts.probe_fusion_adapter import CHECKPOINT, CHECKPOINT_SHA, build_cache, digest, head_output, seed
from scripts.diagnose_joint_utility_sets import sha256
from scripts.analyze_response_geometry import clustered


def save(path, value):
    with path.open('w', newline='\n') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')


def measure(head, cache):
    rows = []
    with torch.no_grad():
        for start in range(0, len(cache['labels']), 16):
            fused = cache['features'][start:start + 16].cuda()
            target = cache['boxes'][start:start + 16].cuda()
            out = head_output(head, fused)
            loss = task_loss([out], target.unsqueeze(0), cfg)
            _, iou = generalized_box_iou(box_cxcywh_to_xyxy(out['pred_boxes']).flatten(0, 1),
                                         box_xywh_to_xyxy(target).clamp(0, 1))
            assert torch.isfinite(loss).all() and torch.isfinite(iou).all()
            for index in range(len(fused)):
                position = start + index
                rows.append(dict(identity=cache['identities'][position],
                    cell=cache['labels'][position], loss=float(loss[index]), iou=float(iou[index])))
    cells = {cell: {key: float(np.mean([row[key] for row in rows if row['cell'] == cell]))
                    for key in ['loss', 'iou']} for cell in sorted(set(cache['labels']))}
    return dict(rows=rows, cells=cells,
                selection_score=float(np.mean([cell['loss'] for cell in cells.values()])))


def train(mode, original, cache, development, folder):
    folder.mkdir()
    head = copy.deepcopy(original).cuda().eval().requires_grad_(True)
    initial = digest(head.state_dict())
    buffers = digest(dict(head.named_buffers()))
    initial_metrics = measure(head, development)
    optimizer = torch.optim.AdamW(head.parameters(), lr=8e-5, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[6], gamma=.1)
    history, steps, reference, stale = [], [], None, 0
    torch.cuda.reset_peak_memory_stats()
    for epoch in range(1, 21):
        order = torch.randperm(len(cache['labels']), generator=torch.Generator().manual_seed(41000 + epoch))
        for start in range(0, len(order), 64):
            indices = order[start:start + 64]
            fused = cache['features'][indices].cuda()
            target = cache['boxes'][indices].cuda()
            optimizer.zero_grad(set_to_none=True)
            if mode == 'target':
                heatmap = generate_heatmap(target.unsqueeze(0), cfg.DATA.SEARCH.SIZE,
                                           cfg.MODEL.BACKBONE.STRIDE)[0]
                score, boxes, size, offset = head(fused, heatmap)
                out = dict(score_map=score, pred_boxes=boxes.unsqueeze(1),
                           size_map=size, offset_map=offset)
            else:
                out = head_output(head, fused)
            loss = task_loss([out], target.unsqueeze(0), cfg).mean()
            assert torch.isfinite(loss)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(head.parameters(), .1)
            assert torch.isfinite(norm)
            optimizer.step()
            steps.append(dict(epoch=epoch, loss=float(loss.detach()), gradient_norm=float(norm)))
        measured = measure(head, development)
        score = measured['selection_score']
        if epoch >= 7:
            if reference is None or score < reference * .998:
                reference, stale = score, 0
            else:
                stale += 1
        record = dict(epoch=epoch, updates=len(steps), selection_score=score,
            cells=measured['cells'], stale_epochs=stale, lr=optimizer.param_groups[0]['lr'])
        history.append(record)
        torch.save(head.state_dict(), str(folder / ('epoch%02d.pth' % epoch)))
        save(folder / 'history.json', history)
        save(folder / 'steps.json', steps)
        print(json.dumps(dict(stage='epoch', mode=mode, **record)), flush=True)
        if epoch >= 10 and stale >= 3:
            break
        scheduler.step()
    selected = min(history[-5:], key=lambda row: (row['selection_score'], row['epoch']))
    path = folder / ('epoch%02d.pth' % selected['epoch'])
    head.load_state_dict(torch.load(str(path), map_location='cpu'), strict=True)
    assert measure(head, development)['selection_score'] == selected['selection_score']
    assert digest(dict(head.named_buffers())) == buffers
    result = dict(mode=mode, initial_sha256=initial, initial_metrics=initial_metrics,
        history=history, selected=selected, selected_path=str(path), selected_sha256=sha256(path),
        final_five=[dict(epoch=row['epoch'], sha256=sha256(folder / ('epoch%02d.pth' % row['epoch'])))
                    for row in history[-5:]], steps=steps, bn_buffers_unchanged=True,
        parameters=sum(p.numel() for p in head.parameters()),
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        peak_reserved_bytes=torch.cuda.max_memory_reserved())
    save(folder / 'selection.json', result)
    return head, result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=False)
    started = time.time()
    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    seed(40101)
    assert sha256(CHECKPOINT) == CHECKPOINT_SHA
    update_config_from_file('experiments/cadtrack/cadtrack.yaml')
    model = load_pretrained(cfg, CHECKPOINT, repair=False).cuda().eval().requires_grad_(False)
    before = digest(model.state_dict())
    previous = json.loads(Path('reports/server28_fusion_adapter_pilot_20261102/report.json').read_text())
    names = previous['launch']['split_names']
    assert [len(names[key]) for key in ['train', 'development', 'audit']] == [705, 88, 89]
    old = [item['identity'] for manifest in previous['cache_manifests'].values()
           for item in manifest['input_manifest']]
    diagnostic = json.loads(Path('reports/server28_response_geometry_20261102/report.json').read_text())
    old.extend(item['identity'] for item in diagnostic['manifest']['input_manifest'])
    def key(identity):
        return (identity['sequence'], tuple(identity['template_frames']), tuple(identity['search_frames']))
    old_ids = {key(item) for item in old}
    cache, manifests = {}, {}
    for split, count, cache_seed in [('train', 256, 40101), ('development', 128, 40102),
                                      ('audit', 128, 40103)]:
        cache[split], manifests[split] = build_cache(model, names[split], count, cache_seed,
                                                    False, root / split)
        assert not old_ids.intersection(key(item['identity'])
                                       for item in manifests[split]['input_manifest'])
    baseline = {split: measure(model.baseline.box_head, cache[split])
                for split in ['development', 'audit']}
    arms, heads = {}, {}
    for mode in ['native', 'target']:
        heads[mode], arms[mode] = train(mode, model.baseline.box_head, cache['train'],
                                        cache['development'], root / mode)
        assert arms[mode]['initial_metrics'] == baseline['development']
    assert arms['native']['initial_sha256'] == arms['target']['initial_sha256']
    audit = {mode: measure(heads[mode], cache['audit']) for mode in ['native', 'target']}
    paired = []
    for base, native, target in zip(baseline['audit']['rows'], audit['native']['rows'], audit['target']['rows']):
        assert base['identity'] == native['identity'] == target['identity']
        assert base['cell'] == native['cell'] == target['cell']
        paired.append(dict(identity=base['identity'], cell=base['cell'],
            native_gain=target['iou'] - base['iou'], control_gain=target['iou'] - native['iou']))
    effects, gates = {}, []
    for state in ['01', '10']:
        subset = [row for row in paired if row['cell'].endswith('_' + state)]
        effects[state] = clustered(subset, 'native_gain', 40104)
        gates.append(effects[state]['mean'] >= .01 and effects[state]['ci95_low'] > 0)
    missing = [row for row in paired if not row['cell'].endswith('_11')]
    effects['versus_native_training'] = clustered(missing, 'control_gain', 40105)
    gates.append(effects['versus_native_training']['ci95_low'] > 0)
    for cell in ['11_11', '01_11', '10_11']:
        effects[cell] = clustered([row for row in paired if row['cell'] == cell], 'native_gain', 40106)
        gates.append(effects[cell]['mean'] >= 0 and effects[cell]['ci95_low'] > -.005)
    assert digest(model.state_dict()) == before
    assert all(p.grad is None for p in model.parameters())
    sources = [Path(__file__), Path('scripts/probe_fusion_adapter.py'),
        Path('scripts/analyze_response_geometry.py'), Path('scripts/diagnose_joint_utility_sets.py'),
        Path('miss_rgbt/model.py'), Path('miss_rgbt/data.py'), Path('miss_rgbt/objectives.py'),
        Path('lib/models/layers/head.py'), Path('lib/utils/heapmap_utils.py')]
    report = dict(status='complete', pid=os.getpid(), checkpoint=CHECKPOINT,
        checkpoint_sha256=CHECKPOINT_SHA, baseline_state_sha256=before,
        baseline_state_unchanged=True, baseline_gradients_absent=True,
        no_exact_prior_tuple_overlap=True, cache_manifests=manifests,
        source_sha256={str(p): sha256(p) for p in sources}, baseline=baseline,
        arms=arms, audit=audit, effects=effects, advance=all(gates), seconds=time.time() - started,
        scope='Paired cached native-head training control; no official metrics or novel module.')
    save(root / 'report.json', report)
    save(root / 'completed.json', dict(status='complete',
        report_sha256=sha256(root / 'report.json'), advance=all(gates), seconds=report['seconds']))
    print(json.dumps(dict(stage='complete', effects=effects, advance=all(gates),
                          seconds=report['seconds'])), flush=True)


if __name__ == '__main__':
    main()
