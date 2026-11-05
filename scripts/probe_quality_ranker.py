"""Matched quality/response-supervised ranking of fixed native top-eight boxes."""
import argparse
import copy
import json
import os
from pathlib import Path
import time

import cv2
import numpy as np
import torch
from torch import nn

from lib.config.cadtrack.config import cfg, update_config_from_file
from lib.test.utils.hann import hann2d
from miss_rgbt.data import make_loader
from miss_rgbt.model import load_pretrained
from diagnose_absent_position import (
    CHECKPOINT, CHECKPOINT_SHA, capture_identities, clustered, digest, seed, sha256)
from diagnose_candidate_quality import dense_boxes, dense_iou, excluded_tuples, stable_descending

PROJECT = Path('/home/gaob/Miss-RGBT')
STATES = [('11', [1., 1.]), ('01', [0., 1.]), ('10', [1., 0.])]


def save(path, value):
    with Path(path).open('x', newline='\n') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')


class Ranker(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = nn.LayerNorm(768)
        self.hidden = nn.Linear(780, 64)
        self.output = nn.Linear(64, 1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, inputs, prior):
        value = torch.cat([self.norm(inputs[..., :768]), inputs[..., 768:]], -1)
        residual = self.output(torch.nn.functional.gelu(self.hidden(value))).squeeze(-1)
        return prior.log() + residual


def cache_split(model, names, samples, sample_seed, root, excluded):
    root.mkdir()
    seed(sample_seed)
    loader = make_loader(cfg, '/data/wangwj/dataset/LasHeR/trainingset',
                         'train', 16, samples, 0)
    loader.dataset.datasets[0].sequence_list = names
    identities = capture_identities(loader)
    window = hann2d(torch.tensor([16, 16]), centered=True).cuda()
    captured = {}

    def capture(module, inputs, output):
        captured['features'] = output.detach()

    handle = model.baseline.decode_fuse_search.register_forward_hook(capture)
    keys = ['inputs', 'prior', 'ious', 'boxes', 'target', 'indices']
    values = {key: [] for key in keys}
    labels, row_ids, manifest, count = [], [], [], 0
    with torch.no_grad():
        for data in loader:
            templates = [value.cuda() for value in data['template_images']]
            search = data['search_images'][0].cuda()
            target = data['search_anno'][0].cuda()
            batch_ids = identities[count:count + len(search)]
            assert len(batch_ids) == len(search) and len(templates) == 2
            for index, identity in enumerate(batch_ids):
                key = (identity['sequence'], tuple(identity['template_frames']),
                       tuple(identity['search_frames']))
                assert key not in excluded and identity['sequence'] in names
                manifest.append(dict(identity=identity, tensors_sha256=digest(dict(
                    templates=data['template_images'][:, index],
                    search=data['search_images'][0, index], boxes=target[index]))))
            count += len(search)
            for dynamic_name, dynamic in STATES:
                for search_name, present in STATES:
                    dynamic_flags = search.new_tensor(dynamic).expand(len(search), -1)
                    search_flags = search.new_tensor(present).expand(len(search), -1)
                    out = model(templates, [search], [
                        search.new_ones(len(search), 2), dynamic_flags, search_flags])[0]
                    raw = out['score_map'].flatten(1)
                    weighted = (out['score_map'] * window).flatten(1)
                    indices = stable_descending(weighted)[:, :8]
                    batch_index = torch.arange(len(search), device=search.device)[:, None]
                    geometry = dense_boxes(out)[batch_index, indices]
                    ious = dense_iou(dense_boxes(out), target)[batch_index, indices]
                    prior = weighted.gather(1, indices)
                    assert (prior > 0).all()
                    locations = torch.stack([(indices % 16).float() / 16,
                                              (indices // 16).float() / 16], -1)
                    flags = torch.cat([dynamic_flags, search_flags], -1)[:, None].expand(-1, 8, -1)
                    fused = captured['features'].flatten(2).transpose(1, 2)[batch_index, indices]
                    scores = torch.stack([raw.gather(1, indices), prior], -1)
                    inputs = torch.cat([fused, geometry, locations, scores, flags], -1)
                    assert inputs.shape == (len(search), 8, 780)
                    selected = model.baseline.box_head.cal_bbox(
                        out['score_map'] * window, out['size_map'], out['offset_map'])
                    assert torch.equal(selected, geometry[:, 0])
                    for key, value in dict(inputs=inputs, prior=prior, ious=ious,
                        boxes=geometry, target=target, indices=indices).items():
                        values[key].append(value.cpu())
                    labels.extend([dynamic_name + '_' + search_name] * len(search))
                    row_ids.extend(batch_ids)
            print(json.dumps(dict(stage='cache', split=root.name, samples=count,
                                  expected=samples)), flush=True)
    handle.remove()
    assert count == samples
    cache = {key: torch.cat(items) for key, items in values.items()}
    cache.update(labels=labels, identities=row_ids)
    path = root / 'candidates.pth'
    torch.save(cache, str(path))
    record = dict(samples=samples, rows=len(labels), seed=sample_seed,
        sequence_names=names, input_manifest=manifest, path=str(path),
        cache_sha256=sha256(path), native_hann_box_exact=True)
    save(root / 'manifest.json', record)
    return cache, record


def targets(cache, index, mode):
    if mode == 'quality':
        return (cache['ious'][index].cuda() / .1).softmax(-1)
    prior = cache['prior'][index].cuda()
    return prior / prior.sum(-1, keepdim=True)


def evaluate(ranker, cache, mode):
    rows, losses = [], []
    with torch.no_grad():
        for start in range(0, len(cache['labels']), 128):
            index = torch.arange(start, min(start + 128, len(cache['labels'])))
            logits = ranker(cache['inputs'][index].cuda(), cache['prior'][index].cuda())
            prediction = logits.argmax(-1).cpu()
            loss = -(targets(cache, index, mode) * logits.log_softmax(-1)).sum(-1)
            assert torch.isfinite(loss).all()
            losses.extend(loss.cpu().tolist())
            for local, position in enumerate(index.tolist()):
                choice = int(prediction[local])
                rows.append(dict(identity=cache['identities'][position],
                    cell=cache['labels'][position], choice=choice,
                    native_iou=float(cache['ious'][position, 0]),
                    iou=float(cache['ious'][position, choice]),
                    oracle_iou=float(cache['ious'][position].max()),
                    box=cache['boxes'][position, choice].tolist(),
                    native_box=cache['boxes'][position, 0].tolist(),
                    target=cache['target'][position].tolist(),
                    index=int(cache['indices'][position, choice])))
    cells = {cell: float(np.mean([losses[index] for index, row in enumerate(rows)
                                  if row['cell'] == cell]))
             for cell in sorted(set(cache['labels']))}
    return dict(score=float(np.mean(list(cells.values()))), cells=cells, rows=rows)


def fit(initial, mode, train, development, root):
    root.mkdir()
    ranker = Ranker().cuda()
    ranker.load_state_dict(initial, strict=True)
    initial_result = evaluate(ranker, development, mode)
    assert all(row['choice'] == 0 for row in initial_result['rows'])
    save(root / 'initial.json', dict(score=initial_result['score'], native_selection_exact=True))
    optimizer = torch.optim.AdamW(ranker.parameters(), lr=8e-4, weight_decay=1e-4)
    history, reference, stale = [], None, 0
    torch.cuda.reset_peak_memory_stats()
    started = time.time()
    for epoch in range(1, 21):
        lr = 8e-4 if epoch <= 6 else 8e-5
        for group in optimizer.param_groups:
            group['lr'] = lr
        order = torch.randperm(len(train['labels']),
            generator=torch.Generator().manual_seed(51200 + epoch))
        step_losses = []
        for start in range(0, len(order), 256):
            index = order[start:start + 256]
            assert len(index) == 256
            optimizer.zero_grad(set_to_none=True)
            logits = ranker(train['inputs'][index].cuda(), train['prior'][index].cuda())
            loss = -(targets(train, index, mode) * logits.log_softmax(-1)).sum(-1).mean()
            assert torch.isfinite(loss)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(ranker.parameters(), .1)
            assert torch.isfinite(norm)
            optimizer.step()
            step_losses.append(float(loss.detach()))
        result = evaluate(ranker, development, mode)
        if epoch >= 7:
            if reference is None or result['score'] < reference * .998:
                reference, stale = result['score'], 0
            else:
                stale += 1
        row = dict(epoch=epoch, updates=epoch * 18, lr=lr, stale=stale,
            train_mean_loss=float(np.mean(step_losses)), score=result['score'],
            cells=result['cells'], seconds=time.time() - started)
        history.append(row)
        torch.save(dict(ranker=ranker.state_dict(), epoch=epoch, mode=mode),
                   str(root / ('epoch%02d.pth' % epoch)))
        save(root / ('epoch%02d.json' % epoch), row)
        print(json.dumps(dict(stage='epoch', mode=mode, **row)), flush=True)
        if epoch >= 10 and stale >= 3:
            break
    chosen = min(history[-5:], key=lambda row: (row['score'], row['epoch']))
    path = root / ('epoch%02d.pth' % chosen['epoch'])
    ranker.load_state_dict(torch.load(str(path), map_location='cpu')['ranker'], strict=True)
    assert evaluate(ranker, development, mode)['score'] == chosen['score']
    result = dict(mode=mode, history=history, selected=chosen, selected_path=str(path),
        selected_sha256=sha256(path), initial_state_sha256=digest(initial),
        parameter_count=sum(parameter.numel() for parameter in ranker.parameters()),
        peak_allocated=torch.cuda.max_memory_allocated(), peak_reserved=torch.cuda.max_memory_reserved(),
        final_five=[dict(epoch=row['epoch'], path=str(root / ('epoch%02d.pth' % row['epoch'])),
                        sha256=sha256(root / ('epoch%02d.pth' % row['epoch']))) for row in history[-5:]])
    save(root / 'selection.json', result)
    return ranker, result


def statistics(audit):
    effects = {}
    for category, suffix in [('complete', '11_11'), ('rgb_missing', '_01'), ('tir_missing', '_10')]:
        rows = [dict(identity=row['identity'], gain=row['iou'] - row['native_iou'])
                for row in audit['quality']['rows'] if row['cell'].endswith(suffix)]
        effects[category] = clustered(rows, 'gain', 51301)
    comparisons = []
    for quality, response in zip(audit['quality']['rows'], audit['response']['rows']):
        assert quality['identity'] == response['identity'] and quality['cell'] == response['cell']
        comparisons.append(dict(identity=quality['identity'], gain=quality['iou'] - response['iou']))
    effects['quality_minus_response'] = clustered(comparisons, 'gain', 51302)
    advance = all(effects[name]['mean'] >= .01 and effects[name]['ci95_low'] > 0
        for name in ['complete', 'rgb_missing', 'tir_missing'])
    advance = advance and effects['quality_minus_response']['ci95_low'] > 0
    return effects, advance


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    root = Path(args.output)
    root.mkdir(exist_ok=False)
    started = time.time()
    seed(2027)
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
    held = names[::5]
    partitions = dict(train=[name for index, name in enumerate(names) if index % 5],
                      development=held[:88], audit=held[88:])
    excluded = excluded_tuples()
    diagnostic = json.loads((PROJECT / 'reports/server28_candidate_quality_20261102_v2'
                              / 'report.json').read_text())
    assert diagnostic['advance']
    for item in diagnostic['input_manifest']:
        value = item['identity']
        excluded.add((value['sequence'], tuple(value['template_frames']), tuple(value['search_frames'])))
    caches, manifests = {}, {}
    for split, samples, sample_seed in [('train', 512, 51101), ('development', 128, 51102)]:
        caches[split], manifests[split] = cache_split(
            model, partitions[split], samples, sample_seed, root / split, excluded)
    seed(2027)
    initial = copy.deepcopy(Ranker().state_dict())
    rankers, arms = {}, {}
    for mode in ['quality', 'response']:
        rankers[mode], arms[mode] = fit(initial, mode, caches['train'], caches['development'], root / mode)
    caches['audit'], manifests['audit'] = cache_split(
        model, partitions['audit'], 128, 51103, root / 'audit', excluded)
    audit = {mode: evaluate(rankers[mode], caches['audit'], mode) for mode in rankers}
    effects, advance = statistics(audit)
    assert before == digest(model.state_dict())
    assert all(parameter.grad is None for parameter in model.parameters())
    sources = [Path(__file__), Path(__file__).with_name('diagnose_candidate_quality.py'),
        Path(__file__).with_name('diagnose_absent_position.py'),
        Path('lib/train/data_specs/lasher_train.txt'), Path('experiments/cadtrack/cadtrack.yaml'),
        Path('lib/test/utils/hann.py')]
    sources += sorted(Path('miss_rgbt').rglob('*.py'))
    sources += sorted(Path('lib/models').rglob('*.py'))
    report = dict(status='complete', pid=os.getpid(), checkpoint=CHECKPOINT,
        checkpoint_sha256=CHECKPOINT_SHA, partitions=partitions, cache_manifests=manifests,
        arms=arms, audit=audit, effects=effects, advance=advance,
        baseline_unchanged=True, baseline_gradients_absent=True, baseline_state_sha256=before,
        source_sha256={str(path): sha256(path) for path in sources}, seconds=time.time() - started,
        scope='Train882 cached fixed-candidate ranking pilot; not official or closed-loop tracking.')
    save(root / 'report.json', report)
    save(root / 'completed.json', dict(status='complete', report_sha256=sha256(root / 'report.json'),
        advance=advance, seconds=report['seconds']))
    print(json.dumps(dict(stage='complete', effects=effects, advance=advance,
                          seconds=report['seconds'])), flush=True)


if __name__ == '__main__':
    main()
