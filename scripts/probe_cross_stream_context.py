"""Matched train882 pilot of pre-fusion cross-stream context residuals."""
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
from lib.utils.box_ops import box_cxcywh_to_xyxy, box_xywh_to_xyxy, generalized_box_iou
from miss_rgbt.data import make_loader
from miss_rgbt.model import load_pretrained
from miss_rgbt.objectives import task_loss
from diagnose_absent_position import (
    CHECKPOINT, CHECKPOINT_SHA, MAPS, capture_identities, clustered, digest, seed, sha256)
from diagnose_late_fusion_streams import read_prior_tuple_ids

PROJECT = Path('/home/gaob/Miss-RGBT')


def save(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')


class ContextResidual(nn.Module):
    def __init__(self, dim, tokens, mode):
        super().__init__()
        assert mode in ('aligned', 'shuffled')
        self.mode = mode
        self.norm = nn.LayerNorm(dim)
        self.query = nn.Linear(dim, 64)
        self.key = nn.Linear(dim, 64)
        self.value = nn.Linear(dim, 64)
        self.output = nn.Linear(64, dim)
        permutation = torch.randperm(tokens, generator=torch.Generator().manual_seed(48100))
        self.register_buffer('permutation', permutation)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, features, flags):
        tokens = features.flatten(2).transpose(1, 2)
        rgb, tir = tokens.chunk(2, -1)
        rgb_present = flags[:, -2:-1, None].bool()
        source = torch.where(rgb_present, rgb, tir)
        missing = torch.where(rgb_present, tir, rgb)
        batch, length, _ = source.shape
        q = self.query(self.norm(missing)).view(batch, length, 4, 16).transpose(1, 2)
        k = self.key(self.norm(source)).view(batch, length, 4, 16).transpose(1, 2)
        v = self.value(self.norm(source)).view(batch, length, 4, 16).transpose(1, 2)
        if self.mode == 'shuffled':
            v = v[:, :, self.permutation]
        else:
            # Both arms perform the same gather; only K/V correspondence differs.
            v = v[:, :, torch.arange(length, device=v.device)]
        attention = ((q @ k.transpose(-2, -1)) * .25).softmax(-1)
        residual = self.output((attention @ v).transpose(1, 2).reshape(batch, length, 64))
        single = flags[:, -2:].sum(-1).eq(1)[:, None, None]
        rgb = torch.where(single & ~rgb_present, rgb + residual, rgb)
        tir = torch.where(single & rgb_present, tir + residual, tir)
        return torch.cat([rgb, tir], -1).transpose(1, 2).reshape_as(features).contiguous()


def output_from_map(model, features):
    fused = model.baseline.decode_fuse_search(features)
    score, boxes, size, offset = model.baseline.box_head(fused)
    return dict(score_map=score, pred_boxes=boxes.unsqueeze(1),
                size_map=size, offset_map=offset)


def item_metrics(output, boxes):
    losses = task_loss([output], boxes.unsqueeze(0), cfg)
    _, iou = generalized_box_iou(box_cxcywh_to_xyxy(output['pred_boxes']).flatten(0, 1),
                                 box_xywh_to_xyxy(boxes).clamp(0, 1))
    assert torch.isfinite(losses).all() and torch.isfinite(iou).all()
    return losses, iou


def prior_ids():
    result = read_prior_tuple_ids()
    for name in ['server28_late_fusion_streams_20261102',
                 'server28_absent_position_20261102_v4']:
        report = json.loads((PROJECT / 'reports' / name / 'report.json').read_text())
        for row in report['rows']:
            item = row['identity']
            result.add((item['sequence'], tuple(item['template_frames']),
                        tuple(item['search_frames'])))
    return result


def validate_control(dim, tokens):
    seed(2027)
    aligned = ContextResidual(dim, tokens, 'aligned').cuda()
    shuffled = ContextResidual(dim, tokens, 'shuffled').cuda()
    shuffled.load_state_dict(aligned.state_dict(), strict=True)
    raw = torch.randn(4, 2 * dim, int(tokens ** .5), int(tokens ** .5), device='cuda')
    flags = raw.new_tensor([[1, 1, 1, 1], [0, 1, 1, 1],
                            [1, 1, 0, 1], [1, 1, 1, 0]])
    with torch.no_grad():
        assert torch.equal(aligned(raw, flags), raw)
        assert torch.equal(shuffled(raw, flags), raw)
        nn.init.normal_(aligned.output.weight, std=.01)
        shuffled.load_state_dict(aligned.state_dict(), strict=True)
        first, second = aligned(raw, flags), shuffled(raw, flags)
        assert torch.equal(first[:2], raw[:2]) and torch.equal(second[:2], raw[:2])
        assert not torch.equal(first[2:], second[2:])
        # The available stream remains native in either missing-state direction.
        assert torch.equal(first[2, dim:], raw[2, dim:])
        assert torch.equal(first[3, :dim], raw[3, :dim])
        assert torch.equal(second[2, dim:], raw[2, dim:])
        assert torch.equal(second[3, :dim], raw[3, :dim])
    return dict(zero_init_exact=True, complete_search_exact=True,
                available_stream_exact=True, shuffled_context_differs=True)


def cache_split(model, names, samples, sample_seed, fitting, root, old_ids):
    root.mkdir()
    seed(sample_seed)
    loader = make_loader(cfg, '/data/wangwj/dataset/LasHeR/trainingset',
                         'train', 16, samples, 0)
    loader.dataset.datasets[0].sequence_list = list(names)
    identities = capture_identities(loader)
    captured = {}

    def capture(module, inputs):
        captured['features'] = inputs[0].detach()

    hook = model.baseline.decode_fuse_search.register_forward_pre_hook(capture)
    features, boxes, flags, labels, row_ids, manifest = [], [], [], [], [], []
    count = 0
    states = [('11', [1., 1.]), ('01', [0., 1.]), ('10', [1., 0.])]
    with torch.no_grad():
        for data in loader:
            templates = [image.cuda() for image in data['template_images']]
            search = data['search_images'][0].cuda()
            target = data['search_anno'][0].cuda()
            batch = search.shape[0]
            batch_ids = identities[count:count + batch]
            assert len(batch_ids) == batch and len(templates) == 2
            for index, identity in enumerate(batch_ids):
                key = (identity['sequence'], tuple(identity['template_frames']),
                       tuple(identity['search_frames']))
                assert key not in old_ids
                assert identity['sequence'] in names
                manifest.append(dict(identity=identity, tensors_sha256=digest(dict(
                    templates=data['template_images'][:, index],
                    search=data['search_images'][0, index], boxes=target[index]))))
            count += batch
            for dynamic_name, dynamic in states:
                for search_name, available in states:
                    if fitting and search_name == '11':
                        continue
                    dynamic_flags = search.new_tensor(dynamic).expand(batch, -1)
                    search_flags = search.new_tensor(available).expand(batch, -1)
                    native = model(templates, [search], [
                        search.new_ones(batch, 2), dynamic_flags, search_flags])[0]
                    raw = captured['features'].clone()
                    replay = output_from_map(model, raw)
                    assert all(torch.equal(native[key], replay[key]) for key in MAPS)
                    features.append(raw.cpu())
                    boxes.append(target.cpu())
                    flags.append(torch.cat([dynamic_flags, search_flags], -1).cpu())
                    labels.extend([dynamic_name + '_' + search_name] * batch)
                    row_ids.extend(batch_ids)
            print(json.dumps(dict(stage='cache', split=root.name, samples=count,
                                  expected=samples)), flush=True)
    hook.remove()
    assert count == samples
    cached = dict(features=torch.cat(features), boxes=torch.cat(boxes),
                  flags=torch.cat(flags), labels=labels, identities=row_ids)
    torch.save(cached, str(root / 'features.pth'))
    record = dict(samples=samples, seed=sample_seed, sequence_names=names,
        input_manifest=manifest, rows=len(labels),
        path=str(root / 'features.pth'), cache_sha256=sha256(root / 'features.pth'),
        native_map_replay_exact=True)
    save(root / 'manifest.json', record)
    return cached, record


def measure(adapter, model, cache):
    rows = []
    with torch.no_grad():
        for start in range(0, len(cache['labels']), 16):
            raw = cache['features'][start:start + 16].cuda()
            flags = cache['flags'][start:start + 16].cuda()
            boxes = cache['boxes'][start:start + 16].cuda()
            native = output_from_map(model, raw)
            changed = adapter(raw, flags)
            predicted = output_from_map(model, changed)
            native_loss, native_iou = item_metrics(native, boxes)
            loss, iou = item_metrics(predicted, boxes)
            complete = flags[:, -2:].eq(1).all(-1)
            if bool(complete.any()):
                assert torch.equal(raw[complete], changed[complete])
                assert all(torch.equal(native[key][complete], predicted[key][complete]) for key in MAPS)
            for index in range(len(raw)):
                position = start + index
                rows.append(dict(identity=cache['identities'][position],
                    cell=cache['labels'][position], loss=float(loss[index]),
                    iou=float(iou[index]), native_loss=float(native_loss[index]),
                    native_iou=float(native_iou[index]),
                    box=predicted['pred_boxes'][index, 0].cpu().tolist(),
                    target=boxes[index].cpu().tolist(),
                    native_box=native['pred_boxes'][index, 0].cpu().tolist()))
    cells = {cell: {key: float(np.mean([row[key] for row in rows if row['cell'] == cell]))
                   for key in ['loss', 'iou', 'native_loss', 'native_iou']}
             for cell in sorted(set(cache['labels']))}
    score = float(np.mean([value['loss'] for cell, value in cells.items()
                           if not cell.endswith('_11')]))
    return dict(rows=rows, cells=cells, score=score, complete_maps_equal=True)


def fit(mode, initial, model, train, development, folder):
    folder.mkdir()
    adapter = ContextResidual(train['features'].shape[1] // 2,
        train['features'].shape[-1] ** 2, mode).cuda()
    adapter.load_state_dict(initial, strict=True)
    initial_record = measure(adapter, model, development)
    assert all(row['native_loss'] == row['loss'] and row['native_iou'] == row['iou']
               for row in initial_record['rows'])
    save(folder / 'initial.json', initial_record)
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=4e-4, weight_decay=1e-4)
    history, reference, stale, updates = [], None, 0, 0
    torch.cuda.reset_peak_memory_stats()
    started = time.time()
    for epoch in range(1, 21):
        lr = 4e-4 if epoch <= 6 else 4e-5
        for group in optimizer.param_groups:
            group['lr'] = lr
        order = torch.randperm(len(train['labels']),
                               generator=torch.Generator().manual_seed(48200 + epoch))
        step_losses = []
        for start in range(0, len(order), 32):
            index = order[start:start + 32]
            raw = train['features'][index].cuda()
            flags = train['flags'][index].cuda()
            boxes = train['boxes'][index].cuda()
            optimizer.zero_grad(set_to_none=True)
            predicted = output_from_map(model, adapter(raw, flags))
            loss = task_loss([predicted], boxes.unsqueeze(0), cfg).mean()
            assert torch.isfinite(loss)
            loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(adapter.parameters(), .1)
            assert torch.isfinite(grad_norm)
            optimizer.step()
            updates += 1
            step_losses.append(float(loss.detach()))
        evaluation = measure(adapter, model, development)
        score = evaluation['score']
        if epoch >= 7:
            if reference is None or score < reference * .998:
                reference, stale = score, 0
            else:
                stale += 1
        record = dict(epoch=epoch, updates=updates, lr=lr, score=score,
            stale=stale, train_mean_loss=float(np.mean(step_losses)),
            development_cells=evaluation['cells'], seconds=time.time() - started)
        history.append(record)
        path = folder / ('epoch%02d.pth' % epoch)
        torch.save(dict(adapter=adapter.state_dict(), mode=mode, epoch=epoch), str(path))
        save(folder / ('epoch%02d.json' % epoch), record)
        print(json.dumps(dict(stage='epoch', mode=mode, **record)), flush=True)
        if epoch >= 10 and stale >= 3:
            break
    chosen = min(history[-5:], key=lambda row: (row['score'], row['epoch']))
    path = folder / ('epoch%02d.pth' % chosen['epoch'])
    adapter.load_state_dict(torch.load(str(path), map_location='cpu')['adapter'], strict=True)
    assert measure(adapter, model, development)['score'] == chosen['score']
    result = dict(mode=mode, history=history, selected=chosen, selected_path=str(path),
        selected_sha256=sha256(path), parameter_count=sum(p.numel() for p in adapter.parameters()),
        initial_state_sha256=digest(initial), peak_allocated=torch.cuda.max_memory_allocated(),
        peak_reserved=torch.cuda.max_memory_reserved(), final_five=[
            dict(epoch=row['epoch'], path=str(folder / ('epoch%02d.pth' % row['epoch'])),
                 sha256=sha256(folder / ('epoch%02d.pth' % row['epoch'])))
            for row in history[-5:]])
    save(folder / 'selection.json', result)
    return adapter, result


def effects_from(audit):
    pairs = []
    for aligned, control in zip(audit['aligned']['rows'], audit['shuffled']['rows']):
        assert aligned['identity'] == control['identity'] and aligned['cell'] == control['cell']
        assert aligned['native_iou'] == control['native_iou']
        pairs.append(dict(identity=aligned['identity'], cell=aligned['cell'],
            iou_gain=aligned['iou'] - aligned['native_iou'],
            control_gain=aligned['iou'] - control['iou']))
    effects = {state: clustered(
        [row for row in pairs if row['cell'].endswith('_' + state)], 'iou_gain', 48301)
        for state in ['01', '10']}
    effects['vs_shuffled'] = clustered(
        [row for row in pairs if not row['cell'].endswith('_11')], 'control_gain', 48302)
    advance = all(effects[state]['mean'] >= .01 and effects[state]['ci95_low'] > 0
                  for state in ['01', '10']) and effects['vs_shuffled']['ci95_low'] > 0
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
    assert sha256(CHECKPOINT) == CHECKPOINT_SHA
    update_config_from_file('experiments/cadtrack/cadtrack.yaml')
    model = load_pretrained(cfg, CHECKPOINT, repair=False).cuda().eval().requires_grad_(False)
    baseline_digest = digest(model.state_dict())
    control_checks = validate_control(model.baseline.backbone.embed_dim,
                                     model.baseline.feat_len_s)
    print(json.dumps(dict(stage='control_checks', **control_checks)), flush=True)
    train_file = Path('lib/train/data_specs/lasher_train.txt')
    names = train_file.read_text().splitlines()
    assert len(names) == len(set(names)) == 882
    holdout = names[::5]
    partitions = dict(train=[name for index, name in enumerate(names) if index % 5],
                      development=holdout[:88], audit=holdout[88:])
    assert [len(partitions[key]) for key in ['train', 'development', 'audit']] == [705, 88, 89]
    old_ids = prior_ids()
    caches, manifests = {}, {}
    # Audit features are not sampled/read until both models have stopped and selected.
    for split, samples, cache_seed in [('train', 256, 48101), ('development', 128, 48102)]:
        caches[split], manifests[split] = cache_split(
            model, partitions[split], samples, cache_seed, split == 'train', root / split, old_ids)
    seed(2027)
    prototype = ContextResidual(model.baseline.backbone.embed_dim,
                               model.baseline.feat_len_s, 'aligned')
    initial = copy.deepcopy(prototype.state_dict())
    adapters, arms = {}, {}
    for mode in ['aligned', 'shuffled']:
        adapters[mode], arms[mode] = fit(mode, initial, model, caches['train'],
                                       caches['development'], root / mode)
    assert arms['aligned']['parameter_count'] == arms['shuffled']['parameter_count']
    caches['audit'], manifests['audit'] = cache_split(
        model, partitions['audit'], 128, 48103, False, root / 'audit', old_ids)
    audit = {mode: measure(adapters[mode], model, caches['audit'])
             for mode in ['aligned', 'shuffled']}
    effects, advance = effects_from(audit)
    assert digest(model.state_dict()) == baseline_digest
    assert all(parameter.grad is None for parameter in model.parameters())
    sources = [Path(__file__), Path(__file__).with_name('diagnose_absent_position.py'),
        Path(__file__).with_name('diagnose_late_fusion_streams.py'), train_file,
        Path('miss_rgbt/model.py'), Path('miss_rgbt/data.py'), Path('miss_rgbt/objectives.py'),
        Path('lib/models/cadtrack/cadtrack.py'), Path('lib/models/cadtrack/vit_care.py'),
        Path('lib/models/layers/head.py'), Path('experiments/cadtrack/cadtrack.yaml')]
    report = dict(status='complete', pid=os.getpid(), checkpoint=CHECKPOINT,
        checkpoint_sha256=CHECKPOINT_SHA, baseline_state_sha256=baseline_digest,
        baseline_unchanged=True, baseline_gradients_absent=True,
        partitions=partitions, cache_manifests=manifests, arms=arms, audit=audit,
        control_checks=control_checks,
        effects=effects, advance=advance, source_sha256={str(p): sha256(p) for p in sources},
        seconds=time.time() - started,
        scope='Train882 cached pre-fusion pilot, not official metrics or full-tracker convergence.')
    save(root / 'report.json', report)
    save(root / 'completed.json', dict(status='complete', advance=advance,
        report_sha256=sha256(root / 'report.json'), seconds=report['seconds']))
    print(json.dumps(dict(stage='complete', effects=effects, advance=advance)), flush=True)


if __name__ == '__main__':
    main()
