"""Capacity-matched train-only modality-versus-random head routing."""
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
from miss_rgbt.model import load_pretrained
from miss_rgbt.objectives import task_loss
from scripts.probe_fusion_adapter import CHECKPOINT, CHECKPOINT_SHA, build_cache, digest, head_output, seed
from scripts.diagnose_joint_utility_sets import sha256
from scripts.analyze_response_geometry import clustered
from scripts.probe_head_supervision import save


def routing(cache, mode, route_seed):
    route = torch.full((len(cache['labels']),), -1, dtype=torch.long)
    rng = torch.Generator().manual_seed(route_seed)
    for cell in sorted(set(cache['labels'])):
        if cell.endswith('_11'):
            continue
        indices = torch.tensor([i for i, label in enumerate(cache['labels']) if label == cell])
        if mode == 'modality':
            route[indices] = 0 if cell.endswith('_10') else 1
        else:
            assert mode == 'random' and len(indices) % 2 == 0
            order = indices[torch.randperm(len(indices), generator=rng)]
            route[order[:len(order) // 2]] = 0
            route[order[len(order) // 2:]] = 1
    assert (route == 0).sum() == (route == 1).sum() == len(route) // 3
    return route


def measure(heads, original, cache, route):
    rows = []
    with torch.no_grad():
        for start in range(0, len(route), 16):
            fused = cache['features'][start:start + 16].cuda()
            target = cache['boxes'][start:start + 16].cuda()
            native = head_output(original, fused)
            out = dict(native)
            selected = route[start:start + 16].cuda()
            if bool((selected >= 0).any()):
                for index, head in enumerate(heads):
                    candidate = head_output(head, fused)
                    for key in out:
                        mask = (selected == index).reshape((-1,) + (1,) * (out[key].ndim - 1))
                        out[key] = torch.where(mask, candidate[key], out[key])
            complete = selected == -1
            if bool(complete.any()):
                assert all(torch.equal(out[key][complete], native[key][complete]) for key in out)
            loss = task_loss([out], target.unsqueeze(0), cfg)
            _, iou = generalized_box_iou(box_cxcywh_to_xyxy(out['pred_boxes']).flatten(0, 1),
                                         box_xywh_to_xyxy(target).clamp(0, 1))
            assert torch.isfinite(loss).all() and torch.isfinite(iou).all()
            for index in range(len(fused)):
                position = start + index
                rows.append(dict(identity=cache['identities'][position],
                    cell=cache['labels'][position], route=int(route[position]),
                    loss=float(loss[index]), iou=float(iou[index])))
    cells = {cell: {key: float(np.mean([row[key] for row in rows if row['cell'] == cell]))
                    for key in ['loss', 'iou']} for cell in sorted(set(cache['labels']))}
    return dict(rows=rows, cells=cells, complete_search_maps_equal=True,
        selection_score=float(np.mean([value['loss'] for cell, value in cells.items()
                                      if not cell.endswith('_11')])))


def fit(mode, original, cache, development, train_route, development_route, folder):
    folder.mkdir()
    heads = [copy.deepcopy(original).cuda().eval().requires_grad_(True) for _ in range(2)]
    initial_sha = [digest(head.state_dict()) for head in heads]
    assert initial_sha[0] == initial_sha[1]
    buffers = digest(dict(original.named_buffers()))
    optimizers = [torch.optim.AdamW(head.parameters(), lr=8e-5, weight_decay=1e-4) for head in heads]
    schedulers = [torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[6], gamma=.1)
                  for optimizer in optimizers]
    indices = [(train_route == index).nonzero().flatten() for index in range(2)]
    assert [len(items) for items in indices] == [768, 768]
    initial = measure(heads, original, development, development_route)
    history, steps, reference, stale = [], [], None, 0
    torch.cuda.reset_peak_memory_stats()
    for epoch in range(1, 21):
        for branch in range(2):
            order = indices[branch][torch.randperm(768, generator=torch.Generator().manual_seed(
                43000 + epoch * 2 + branch))]
            for start in range(0, len(order), 64):
                batch = order[start:start + 64]
                optimizers[branch].zero_grad(set_to_none=True)
                fused = cache['features'][batch].cuda()
                target = cache['boxes'][batch].cuda()
                out = head_output(heads[branch], fused)
                loss = task_loss([out], target.unsqueeze(0), cfg).mean()
                assert torch.isfinite(loss)
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(heads[branch].parameters(), .1)
                assert torch.isfinite(norm)
                optimizers[branch].step()
                steps.append(dict(epoch=epoch, branch=branch, loss=float(loss.detach()), norm=float(norm)))
        measured = measure(heads, original, development, development_route)
        score = measured['selection_score']
        if epoch >= 7:
            if reference is None or score < reference * .998:
                reference, stale = score, 0
            else:
                stale += 1
        record = dict(epoch=epoch, updates=len(steps), selection_score=score,
            cells=measured['cells'], stale_epochs=stale, lr=optimizers[0].param_groups[0]['lr'])
        history.append(record)
        torch.save([head.state_dict() for head in heads], str(folder / ('epoch%02d.pth' % epoch)))
        save(folder / 'history.json', history)
        save(folder / 'steps.json', steps)
        print(json.dumps(dict(stage='epoch', mode=mode, **record)), flush=True)
        if epoch >= 10 and stale >= 3:
            break
        for scheduler in schedulers:
            scheduler.step()
    selected = min(history[-5:], key=lambda row: (row['selection_score'], row['epoch']))
    path = folder / ('epoch%02d.pth' % selected['epoch'])
    for head, state in zip(heads, torch.load(str(path), map_location='cpu')):
        head.load_state_dict(state, strict=True)
        assert digest(dict(head.named_buffers())) == buffers
    assert measure(heads, original, development, development_route)['selection_score'] == selected['selection_score']
    result = dict(mode=mode, initial_sha256=initial_sha, initial=initial,
        history=history, steps=steps, selected=selected, selected_path=str(path),
        selected_sha256=sha256(path), final_five=[dict(epoch=row['epoch'],
            sha256=sha256(folder / ('epoch%02d.pth' % row['epoch']))) for row in history[-5:]],
        parameters=sum(p.numel() for head in heads for p in head.parameters()),
        bn_buffers_unchanged=True, peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        peak_reserved_bytes=torch.cuda.max_memory_reserved())
    save(folder / 'selection.json', result)
    return heads, result


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
    seed(42101)
    assert sha256(CHECKPOINT) == CHECKPOINT_SHA
    update_config_from_file('experiments/cadtrack/cadtrack.yaml')
    model = load_pretrained(cfg, CHECKPOINT, repair=False).cuda().eval().requires_grad_(False)
    original = model.baseline.box_head
    before = digest(model.state_dict())
    prior = json.loads(Path('reports/server28_fusion_adapter_pilot_20261102/report.json').read_text())
    names = prior['launch']['split_names']
    prior_manifests = list(prior['cache_manifests'].values())
    previous_head = json.loads(Path('reports/server28_head_supervision_20261102/report.json').read_text())
    previous_diagnostic = json.loads(Path('reports/server28_response_geometry_20261102/report.json').read_text())
    prior_manifests.extend(previous_head['cache_manifests'].values())
    prior_manifests.append(previous_diagnostic['manifest'])
    def key(identity):
        return (identity['sequence'], tuple(identity['template_frames']), tuple(identity['search_frames']))
    old_ids = {key(item['identity']) for manifest in prior_manifests for item in manifest['input_manifest']}
    caches, manifests = {}, {}
    for split, samples, cache_seed in [('train', 256, 42101), ('development', 128, 42102),
                                       ('audit', 128, 42103)]:
        caches[split], manifests[split] = build_cache(
            model, names[split], samples, cache_seed, False, root / split)
        assert not old_ids.intersection(key(item['identity']) for item in manifests[split]['input_manifest'])
    route_seeds = dict(train=42104, development=42105, audit=42106)
    routes = {mode: {split: routing(cache, mode, route_seeds[split])
                    for split, cache in caches.items()} for mode in ['modality', 'random']}
    baseline = {split: measure([], original, cache, torch.full((len(cache['labels']),), -1))
                for split, cache in caches.items() if split != 'train'}
    heads, arms = {}, {}
    for mode in ['modality', 'random']:
        heads[mode], arms[mode] = fit(mode, original, caches['train'], caches['development'],
            routes[mode]['train'], routes[mode]['development'], root / mode)
        for initial, base in zip(arms[mode]['initial']['rows'], baseline['development']['rows']):
            assert initial['loss'] == base['loss'] and initial['iou'] == base['iou']
    assert arms['modality']['parameters'] == arms['random']['parameters']
    audit = {mode: measure(heads[mode], original, caches['audit'], routes[mode]['audit'])
             for mode in heads}
    paired = []
    for base, specialist, control in zip(baseline['audit']['rows'], audit['modality']['rows'],
                                         audit['random']['rows']):
        assert base['identity'] == specialist['identity'] == control['identity']
        assert base['cell'] == specialist['cell'] == control['cell']
        if base['cell'].endswith('_11'):
            assert base['loss'] == specialist['loss'] == control['loss']
            assert base['iou'] == specialist['iou'] == control['iou']
        paired.append(dict(identity=base['identity'], cell=base['cell'],
            baseline_gain=specialist['iou'] - base['iou'],
            control_gain=specialist['iou'] - control['iou']))
    effects, gates = {}, []
    for state in ['01', '10']:
        effects[state] = clustered([row for row in paired if row['cell'].endswith('_' + state)],
                                   'baseline_gain', 42107)
        gates.append(effects[state]['mean'] >= .01 and effects[state]['ci95_low'] > 0)
    effects['versus_random'] = clustered([row for row in paired if not row['cell'].endswith('_11')],
                                       'control_gain', 42108)
    gates.append(effects['versus_random']['ci95_low'] > 0)
    assert digest(model.state_dict()) == before and all(p.grad is None for p in model.parameters())
    sources = [Path(__file__), Path('scripts/probe_head_supervision.py'),
        Path('scripts/probe_fusion_adapter.py'), Path('scripts/analyze_response_geometry.py'),
        Path('scripts/diagnose_joint_utility_sets.py'), Path('miss_rgbt/model.py'),
        Path('miss_rgbt/data.py'), Path('miss_rgbt/objectives.py'), Path('lib/models/layers/head.py')]
    report = dict(status='complete', pid=os.getpid(), checkpoint=CHECKPOINT,
        checkpoint_sha256=CHECKPOINT_SHA, baseline_state_sha256=before,
        baseline_state_unchanged=True, baseline_gradients_absent=True,
        cache_manifests=manifests, no_exact_prior_tuple_overlap=True,
        source_sha256={str(path): sha256(path) for path in sources},
        route_seeds=route_seeds, routes={mode: {split: route.tolist() for split, route in group.items()}
                                       for mode, group in routes.items()},
        arms=arms, baseline=baseline, audit=audit, effects=effects, advance=all(gates),
        seconds=time.time() - started,
        scope='Ordinary capacity-matched cached-head routing control; no official score or module.')
    save(root / 'report.json', report)
    save(root / 'completed.json', dict(status='complete', report_sha256=sha256(root / 'report.json'),
        advance=all(gates), seconds=report['seconds']))
    print(json.dumps(dict(stage='complete', effects=effects, advance=all(gates),
                          seconds=report['seconds'])), flush=True)


if __name__ == '__main__':
    main()
