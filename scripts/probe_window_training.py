"""Matched raw/Hann regression supervision with identical native CENTER heads."""
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
from lib.test.utils.hann import hann2d
from lib.utils.box_ops import box_cxcywh_to_xyxy, box_xywh_to_xyxy, generalized_box_iou
from miss_rgbt.model import load_pretrained
from miss_rgbt.objectives import task_loss
from scripts import diagnose_joint_utility_sets
from probe_fusion_adapter import build_cache, head_output
from diagnose_absent_position import CHECKPOINT, CHECKPOINT_SHA, clustered, digest, seed, sha256
from diagnose_template_chronology import old_ids, PROJECT


def save(path, value):
    with Path(path).open('x', newline='\n') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')


def decode(head, features, window, use_window):
    out = head_output(head, features)
    if use_window:
        out['pred_boxes'] = head.cal_bbox(
            out['score_map'] * window, out['size_map'], out['offset_map']).unsqueeze(1)
    return out


def measure(head, cache, window):
    rows = []
    with torch.no_grad():
        for start in range(0, len(cache['labels']), 16):
            target = cache['boxes'][start:start + 16].cuda()
            out = decode(head, cache['features'][start:start + 16].cuda(), window, True)
            loss = task_loss([out], target.unsqueeze(0), cfg)
            _, iou = generalized_box_iou(box_cxcywh_to_xyxy(out['pred_boxes']).flatten(0, 1),
                                         box_xywh_to_xyxy(target).clamp(0, 1))
            assert torch.isfinite(loss).all() and torch.isfinite(iou).all()
            for index in range(len(target)):
                position = start + index
                rows.append(dict(identity=cache['identities'][position],
                    cell=cache['labels'][position], loss=float(loss[index]), iou=float(iou[index]),
                    box=out['pred_boxes'][index, 0].cpu().tolist(),
                    target=target[index].cpu().tolist()))
    cells = {cell: {key: float(np.mean([row[key] for row in rows if row['cell'] == cell]))
                   for key in ['loss', 'iou']} for cell in sorted(set(cache['labels']))}
    return dict(rows=rows, cells=cells,
                score=float(np.mean([value['loss'] for value in cells.values()])))


def profile(original, cache, window, root):
    probe = copy.deepcopy(original).cuda().eval().requires_grad_(True)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    static = torch.cuda.memory_allocated()
    optimizer = torch.optim.AdamW(probe.parameters(), lr=1e-5, weight_decay=1e-4)
    out = decode(probe, cache['features'][:64].cuda(), window, True)
    loss = task_loss([out], cache['boxes'][:64].cuda().unsqueeze(0), cfg).mean()
    loss.backward()
    assert torch.isfinite(loss)
    optimizer.step()
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated()
    total = torch.cuda.get_device_properties(0).total_memory
    proposed = min(256, 16 * int((.75 * total - static) / ((peak - static) / 64 * 16)))
    assert proposed >= 16
    batch = max(value for value in range(16, proposed + 1, 16)
                if len(cache['labels']) % value == 0)
    result = dict(probe_batch=64, static_bytes=static, peak_bytes=peak, total_bytes=total,
        proposed=proposed, batch=batch, learning_rate=1e-5 * batch / 8,
        reference_batch=8, reference_lr=1e-5, updates_per_epoch=len(cache['labels']) // batch,
        policy='75% linear memory estimate; cap256 keeps at least9 finite-cache updates/epoch.')
    save(root / 'batch_sizing.json', result)
    del probe, optimizer, out, loss
    torch.cuda.empty_cache()
    return result


def fit(mode, original, cache, development, window, sizing, root):
    root.mkdir()
    head = copy.deepcopy(original).cuda().eval().requires_grad_(True)
    initial = digest(head.state_dict())
    buffers = digest(dict(head.named_buffers()))
    baseline = measure(head, development, window)
    optimizer = torch.optim.AdamW(head.parameters(), lr=sizing['learning_rate'], weight_decay=1e-4)
    torch.cuda.reset_peak_memory_stats()
    history, reference, stale = [], None, 0
    started = time.time()
    for epoch in range(1, 21):
        lr = sizing['learning_rate'] * (1 if epoch <= 6 else .1)
        for group in optimizer.param_groups:
            group['lr'] = lr
        order = torch.randperm(len(cache['labels']),
            generator=torch.Generator().manual_seed(54200 + epoch))
        losses = []
        for start in range(0, len(order), sizing['batch']):
            indices = order[start:start + sizing['batch']]
            optimizer.zero_grad(set_to_none=True)
            out = decode(head, cache['features'][indices].cuda(), window, mode == 'hann')
            loss = task_loss([out], cache['boxes'][indices].cuda().unsqueeze(0), cfg).mean()
            assert torch.isfinite(loss)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(head.parameters(), .1)
            assert torch.isfinite(norm)
            optimizer.step()
            losses.append(float(loss.detach()))
        result = measure(head, development, window)
        if epoch >= 7:
            if reference is None or result['score'] < reference * .998:
                reference, stale = result['score'], 0
            else:
                stale += 1
        record = dict(epoch=epoch, updates=epoch * sizing['updates_per_epoch'], lr=lr,
            score=result['score'], cells=result['cells'], train_mean_loss=float(np.mean(losses)),
            stale=stale, seconds=time.time() - started)
        history.append(record)
        torch.save(dict(head=head.state_dict(), mode=mode, epoch=epoch),
                   str(root / ('epoch%02d.pth' % epoch)))
        save(root / ('epoch%02d.json' % epoch), record)
        print(json.dumps(dict(stage='epoch', mode=mode, **record)), flush=True)
        if epoch >= 10 and stale >= 3:
            break
    selected = min(history[-5:], key=lambda row: (row['score'], row['epoch']))
    path = root / ('epoch%02d.pth' % selected['epoch'])
    head.load_state_dict(torch.load(str(path), map_location='cpu')['head'], strict=True)
    assert measure(head, development, window)['score'] == selected['score']
    assert digest(dict(head.named_buffers())) == buffers
    result = dict(mode=mode, history=history, selected=selected, selected_path=str(path),
        selected_sha256=sha256(path), initial_sha256=initial, initial_metrics=baseline,
        initial_buffers_sha256=buffers, bn_buffers_unchanged=True,
        parameter_count=sum(value.numel() for value in head.parameters()),
        peak_allocated=torch.cuda.max_memory_allocated(), peak_reserved=torch.cuda.max_memory_reserved(),
        final_five=[dict(epoch=row['epoch'], path=str(root / ('epoch%02d.pth' % row['epoch'])),
            sha256=sha256(root / ('epoch%02d.pth' % row['epoch']))) for row in history[-5:]])
    save(root / 'selection.json', result)
    return result


def statistics(baseline, audits):
    paired = []
    for base, raw, hann in zip(baseline['rows'], audits['raw']['rows'], audits['hann']['rows']):
        assert all(base[key] == raw[key] == hann[key] for key in ['identity', 'cell', 'target'])
        paired.append(dict(identity=base['identity'], cell=base['cell'],
            gain=hann['iou'] - base['iou'], control=hann['iou'] - raw['iou']))
    effects = {}
    for state in ['01', '10']:
        effects[state] = clustered([row for row in paired if row['cell'].endswith('_' + state)],
                                   'gain', 54301)
    effects['hann_minus_raw_training'] = clustered(
        [row for row in paired if not row['cell'].endswith('_11')], 'control', 54302)
    for cell in ['11_11', '01_11', '10_11']:
        effects[cell] = clustered([row for row in paired if row['cell'] == cell], 'gain', 54303)
    advance = all(effects[state]['mean'] >= .01 and effects[state]['ci95_low'] > 0
                  for state in ['01', '10'])
    advance = advance and effects['hann_minus_raw_training']['ci95_low'] > 0
    advance = advance and all(effects[cell]['mean'] >= 0 and effects[cell]['ci95_low'] > -.005
                             for cell in ['11_11', '01_11', '10_11'])
    return effects, advance


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    root = Path(args.output)
    root.mkdir(exist_ok=False)
    started = time.time()
    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    update_config_from_file('experiments/cadtrack/cadtrack.yaml')
    assert sha256(CHECKPOINT) == CHECKPOINT_SHA
    model = load_pretrained(cfg, CHECKPOINT, repair=False).cuda().eval().requires_grad_(False)
    before = digest(model.state_dict())
    names = Path('lib/train/data_specs/lasher_train.txt').read_text().splitlines()
    assert len(names) == len(set(names)) == 882
    partitions = dict(train=[name for index, name in enumerate(names) if index % 5],
                      development=names[::5][:88], audit=names[::5][88:])
    excluded = old_ids()
    for name in ['server28_template_chronology_20261102', 'server28_chronology_training_20261102']:
        prior = json.loads((PROJECT / 'reports' / name / 'report.json').read_text())
        manifests = (prior['cache_manifests'].values() if 'cache_manifests' in prior
                     else [dict(input_manifest=prior['input_manifest'])])
        for manifest in manifests:
            for item in manifest['input_manifest']:
                value = item['identity']
                excluded.add((value['sequence'], tuple(value['template_frames']), tuple(value['search_frames'])))
    window = hann2d(torch.tensor([16, 16]), centered=True).cuda()
    caches, manifests = {}, {}
    for split, samples, cache_seed in [('train', 256, 54101), ('development', 128, 54102)]:
        caches[split], manifests[split] = build_cache(
            model, partitions[split], samples, cache_seed, False, root / split)
        for item in manifests[split]['input_manifest']:
            value = item['identity']
            assert (value['sequence'], tuple(value['template_frames']), tuple(value['search_frames'])) not in excluded
    sizing = profile(model.baseline.box_head, caches['train'], window, root)
    arms = {mode: fit(mode, model.baseline.box_head, caches['train'], caches['development'],
                     window, sizing, root / mode) for mode in ['raw', 'hann']}
    assert arms['raw']['initial_sha256'] == arms['hann']['initial_sha256']
    assert arms['raw']['initial_metrics'] == arms['hann']['initial_metrics']
    caches['audit'], manifests['audit'] = build_cache(
        model, partitions['audit'], 128, 54103, False, root / 'audit')
    for item in manifests['audit']['input_manifest']:
        value = item['identity']
        assert (value['sequence'], tuple(value['template_frames']), tuple(value['search_frames'])) not in excluded
    baseline = measure(model.baseline.box_head, caches['audit'], window)
    audits = {}
    for mode in ['raw', 'hann']:
        head = copy.deepcopy(model.baseline.box_head).cuda().eval()
        head.load_state_dict(torch.load(arms[mode]['selected_path'], map_location='cpu')['head'], strict=True)
        audits[mode] = measure(head, caches['audit'], window)
        del head
    effects, advance = statistics(baseline, audits)
    assert digest(model.state_dict()) == before
    assert all(parameter.grad is None for parameter in model.parameters())
    sources = [Path(__file__), Path(__file__).with_name('probe_fusion_adapter.py'),
        Path(diagnose_joint_utility_sets.__file__),
        Path(__file__).with_name('diagnose_absent_position.py'),
        Path(__file__).with_name('diagnose_template_chronology.py'),
        Path('lib/train/data_specs/lasher_train.txt'), Path('experiments/cadtrack/cadtrack.yaml'),
        Path('lib/test/utils/hann.py'), Path('lib/utils/heapmap_utils.py'), Path('lib/utils/box_ops.py')]
    sources += sorted(Path('miss_rgbt').rglob('*.py')) + sorted(Path('lib/models').rglob('*.py'))
    report = dict(status='complete', pid=os.getpid(), checkpoint=CHECKPOINT,
        checkpoint_sha256=CHECKPOINT_SHA, partitions=partitions, cache_manifests=manifests,
        batch_sizing=sizing, arms=arms, baseline=baseline, audit=audits, effects=effects, advance=advance,
        baseline_state_sha256=before, baseline_unchanged=True, baseline_gradients_absent=True,
        source_sha256={str(path): sha256(path) for path in sources}, seconds=time.time() - started,
        scope='Train882 matched cached native-head Hann/raw supervision; not official tracking scores.')
    save(root / 'report.json', report)
    save(root / 'completed.json', dict(status='complete', report_sha256=sha256(root / 'report.json'),
        advance=advance, seconds=report['seconds']))
    print(json.dumps(dict(stage='complete', effects=effects, advance=advance,
                          seconds=report['seconds'])), flush=True)


if __name__ == '__main__':
    main()
