"""Matched train882 attention/MLP low-rank adaptation in the final two blocks."""
import argparse
import json
import os
from pathlib import Path
import time

import cv2
import numpy as np
import torch
from torch import nn

from lib.config.cadtrack.config import cfg, update_config_from_file
from miss_rgbt.data import make_loader
from miss_rgbt.model import load_pretrained
from miss_rgbt.objectives import task_loss
from diagnose_absent_position import (
    CHECKPOINT, CHECKPOINT_SHA, MAPS, capture_identities, clustered,
    digest, seed, sha256)
from probe_cross_stream_context import item_metrics, prior_ids

PROJECT = Path('/home/gaob/Miss-RGBT')
STATES = [('11', [1., 1.]), ('01', [0., 1.]), ('10', [1., 0.])]


def save(path, value):
    with Path(path).open('x', newline='\n') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')


class LowRank(nn.Module):
    def __init__(self, layer, rank):
        super().__init__()
        self.down = nn.Linear(layer.in_features, rank, bias=False)
        self.up = nn.Linear(rank, layer.out_features, bias=False)
        nn.init.zeros_(self.up.weight)

    def forward(self, value):
        return self.up(self.down(value))


def attach(model, mode):
    seed(2027)
    modules, handles = nn.ModuleList(), []
    for block in model.baseline.backbone.blocks[10:]:
        layer = block.attn.qkv if mode == 'attention' else block.mlp.fc2
        adapter = LowRank(layer, 5 if mode == 'attention' else 4).cuda()
        modules.append(adapter)

        def residual(module, inputs, output, adapter=adapter):
            return output + adapter(inputs[0])

        handles.append(layer.register_forward_hook(residual))
    assert sum(parameter.numel() for parameter in modules.parameters()) == 30720
    return modules, handles


def tail(model, rgb, tir):
    for block in model.baseline.backbone.blocks[10:]:
        rgb, tir = block(rgb, tir)
    backbone = model.baseline.backbone
    features = model.finish_features(rgb, tir, 2 * backbone.pos_embed_z.shape[1],
                                     backbone.pos_embed_x.shape[1])
    return model.baseline.forward_head(features)


def old_tuples():
    identities = prior_ids()
    report = json.loads((PROJECT / 'reports/server28_cross_stream_context_20261102_v2'
                         / 'report.json').read_text())
    for manifest in report['cache_manifests'].values():
        for item in manifest['input_manifest']:
            value = item['identity']
            identities.add((value['sequence'], tuple(value['template_frames']),
                            tuple(value['search_frames'])))
    return identities


def cache_split(model, names, samples, sample_seed, folder, excluded):
    folder.mkdir()
    seed(sample_seed)
    loader = make_loader(cfg, '/data/wangwj/dataset/LasHeR/trainingset',
                         'train', 16, samples, 0)
    loader.dataset.datasets[0].sequence_list = names
    identities = capture_identities(loader)
    captured = {}

    def capture(module, inputs):
        captured['rgb'], captured['tir'] = [value.detach() for value in inputs]

    handle = model.baseline.backbone.blocks[10].register_forward_pre_hook(capture)
    rgb_rows, tir_rows, targets, native_boxes, native_losses, native_ious = [], [], [], [], [], []
    labels, row_ids, input_manifest = [], [], []
    count = 0
    with torch.no_grad():
        for data in loader:
            templates = [value.cuda() for value in data['template_images']]
            search = data['search_images'][0].cuda()
            boxes = data['search_anno'][0].cuda()
            batch_ids = identities[count:count + len(search)]
            assert len(batch_ids) == len(search) and len(templates) == 2
            for index, identity in enumerate(batch_ids):
                key = (identity['sequence'], tuple(identity['template_frames']),
                       tuple(identity['search_frames']))
                assert key not in excluded and identity['sequence'] in names
                input_manifest.append(dict(identity=identity, tensors_sha256=digest(dict(
                    templates=data['template_images'][:, index],
                    search=data['search_images'][0, index], boxes=boxes[index]))))
            count += len(search)
            for dynamic_name, dynamic in STATES:
                for search_name, present in STATES:
                    availability = [search.new_ones(len(search), 2),
                        search.new_tensor(dynamic).expand(len(search), -1),
                        search.new_tensor(present).expand(len(search), -1)]
                    native = model(templates, [search], availability)[0]
                    rgb, tir = captured['rgb'].clone(), captured['tir'].clone()
                    replay = tail(model, rgb, tir)
                    assert all(torch.equal(native[key], replay[key]) for key in MAPS)
                    loss, iou = item_metrics(native, boxes)
                    rgb_rows.append(rgb.cpu())
                    tir_rows.append(tir.cpu())
                    targets.append(boxes.cpu())
                    native_boxes.append(native['pred_boxes'][:, 0].cpu())
                    native_losses.append(loss.cpu())
                    native_ious.append(iou.cpu())
                    labels.extend([dynamic_name + '_' + search_name] * len(search))
                    row_ids.extend(batch_ids)
            print(json.dumps(dict(stage='cache', split=folder.name, samples=count,
                                  expected=samples)), flush=True)
    handle.remove()
    assert count == samples
    cache = dict(rgb=torch.cat(rgb_rows), tir=torch.cat(tir_rows), boxes=torch.cat(targets),
        native_boxes=torch.cat(native_boxes), native_losses=torch.cat(native_losses),
        native_ious=torch.cat(native_ious), labels=labels, identities=row_ids)
    path = folder / 'tokens.pth'
    torch.save(cache, str(path))
    manifest = dict(samples=samples, rows=len(labels), seed=sample_seed,
        sequence_names=names, input_manifest=input_manifest, path=str(path),
        cache_sha256=sha256(path), native_tail_replay_exact=True)
    save(folder / 'manifest.json', manifest)
    return cache, manifest


def evaluate(model, cache):
    rows = []
    # Use the same batch16 layout as the native cache producer.
    with torch.no_grad():
        for start in range(0, len(cache['labels']), 16):
            out = tail(model, cache['rgb'][start:start + 16].cuda(),
                       cache['tir'][start:start + 16].cuda())
            target = cache['boxes'][start:start + 16].cuda()
            losses, ious = item_metrics(out, target)
            for index in range(len(target)):
                position = start + index
                rows.append(dict(identity=cache['identities'][position],
                    cell=cache['labels'][position], loss=float(losses[index]),
                    iou=float(ious[index]), native_loss=float(cache['native_losses'][position]),
                    native_iou=float(cache['native_ious'][position]),
                    box=out['pred_boxes'][index, 0].cpu().tolist(),
                    native_box=cache['native_boxes'][position].tolist(),
                    target=target[index].cpu().tolist()))
    cells = {cell: {key: float(np.mean([row[key] for row in rows if row['cell'] == cell]))
                   for key in ['loss', 'iou', 'native_loss', 'native_iou']}
             for cell in sorted(set(cache['labels']))}
    return dict(rows=rows, cells=cells,
                score=float(np.mean([value['loss'] for value in cells.values()])))


def size_batch(model, train, folder):
    measurements = {}
    for mode in ['attention', 'mlp']:
        adapters, handles = attach(model, mode)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        static = torch.cuda.memory_allocated()
        out = tail(model, train['rgb'][:16].cuda(), train['tir'][:16].cuda())
        loss = task_loss([out], train['boxes'][:16].cuda().unsqueeze(0), cfg).mean()
        loss.backward()
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated()
        assert torch.isfinite(loss)
        for handle in handles:
            handle.remove()
        measurements[mode] = dict(static_bytes=static, batch16_peak_bytes=peak)
        del adapters, out, loss
    torch.cuda.empty_cache()
    total = torch.cuda.get_device_properties(0).total_memory
    static = max(item['static_bytes'] for item in measurements.values())
    per_item = max((item['batch16_peak_bytes'] - item['static_bytes']) / 16
                   for item in measurements.values())
    # Leave room for the next larger matmul workspace and allocator reservations.
    proposed = min(256, 16 * int((.75 * total - static) / (16 * per_item)))
    assert proposed >= 16
    # Equal full minibatches, without row duplication or a variable final update.
    choices = [value for value in range(16, proposed + 1, 16)
               if len(train['labels']) % value == 0]
    batch = max(choices)
    record = dict(probe=measurements, total_bytes=total, proposed=proposed,
        batch=batch, reference_batch=32, reference_lr=1e-4,
        learning_rate=1e-4 * batch / 32, updates_per_epoch=len(train['labels']) // batch,
        policy='Largest divisor multiple16 below75% linear allocated-memory estimate; max256.')
    save(folder / 'batch_sizing.json', record)
    print(json.dumps(dict(stage='batch_sizing', **record)), flush=True)
    return record


def fit(model, mode, train, development, sizing, folder):
    folder.mkdir()
    adapters, handles = attach(model, mode)
    before = evaluate(model, development)
    assert all(row['box'] == row['native_box'] and row['loss'] == row['native_loss']
               and row['iou'] == row['native_iou'] for row in before['rows'])
    save(folder / 'initial.json', dict(score=before['score'], cells=before['cells'],
                                     native_exact=True))
    batch, initial_lr = sizing['batch'], sizing['learning_rate']
    optimizer = torch.optim.AdamW(adapters.parameters(), lr=initial_lr, weight_decay=1e-4)
    history, reference, stale = [], None, 0
    started = time.time()
    torch.cuda.reset_peak_memory_stats()
    for epoch in range(1, 21):
        lr = initial_lr if epoch <= 6 else initial_lr * .1
        for group in optimizer.param_groups:
            group['lr'] = lr
        order = torch.randperm(len(train['labels']),
            generator=torch.Generator().manual_seed(49200 + epoch))
        step_losses = []
        for start in range(0, len(order), batch):
            index = order[start:start + batch]
            optimizer.zero_grad(set_to_none=True)
            out = tail(model, train['rgb'][index].cuda(), train['tir'][index].cuda())
            loss = task_loss([out], train['boxes'][index].cuda().unsqueeze(0), cfg).mean()
            assert torch.isfinite(loss)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(adapters.parameters(), .1)
            assert torch.isfinite(norm)
            optimizer.step()
            step_losses.append(float(loss.detach()))
        result = evaluate(model, development)
        if epoch >= 7:
            if reference is None or result['score'] < reference * .998:
                reference, stale = result['score'], 0
            else:
                stale += 1
        record = dict(epoch=epoch, updates=epoch * sizing['updates_per_epoch'], lr=lr,
            train_mean_loss=float(np.mean(step_losses)), score=result['score'],
            cells=result['cells'], stale=stale, seconds=time.time() - started)
        history.append(record)
        path = folder / ('epoch%02d.pth' % epoch)
        torch.save(dict(adapter=adapters.state_dict(), mode=mode, epoch=epoch), str(path))
        save(folder / ('epoch%02d.json' % epoch), record)
        print(json.dumps(dict(stage='epoch', mode=mode, **record)), flush=True)
        if epoch >= 10 and stale >= 3:
            break
    selected = min(history[-5:], key=lambda item: (item['score'], item['epoch']))
    selected_path = folder / ('epoch%02d.pth' % selected['epoch'])
    adapters.load_state_dict(torch.load(str(selected_path), map_location='cpu')['adapter'],
                             strict=True)
    assert evaluate(model, development)['score'] == selected['score']
    final_five = [dict(epoch=item['epoch'], path=str(folder / ('epoch%02d.pth' % item['epoch'])),
        sha256=sha256(folder / ('epoch%02d.pth' % item['epoch']))) for item in history[-5:]]
    result = dict(mode=mode, parameter_count=30720, history=history, selected=selected,
        selected_path=str(selected_path), selected_sha256=sha256(selected_path),
        final_five=final_five, peak_allocated=torch.cuda.max_memory_allocated(),
        peak_reserved=torch.cuda.max_memory_reserved())
    save(folder / 'selection.json', result)
    for handle in handles:
        handle.remove()
    return result


def statistics(audits):
    effects, advances = {}, {}
    for mode in ['attention', 'mlp']:
        effects[mode] = {}
        for state in ['11', '01', '10']:
            rows = [dict(identity=row['identity'], gain=row['iou'] - row['native_iou'])
                    for row in audits[mode]['rows'] if row['cell'].endswith('_' + state)]
            effects[mode][state] = clustered(rows, 'gain', 49301)
        advances[mode] = all(value['mean'] >= .01 and value['ci95_low'] > 0
                             for value in effects[mode].values())
    paired = []
    for first, second in zip(audits['attention']['rows'], audits['mlp']['rows']):
        assert first['identity'] == second['identity'] and first['cell'] == second['cell']
        paired.append(dict(identity=first['identity'], gain=first['iou'] - second['iou']))
    effects['attention_minus_mlp'] = clustered(paired, 'gain', 49302)
    return effects, advances


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
    original = digest(model.state_dict())
    names = Path('lib/train/data_specs/lasher_train.txt').read_text().splitlines()
    assert len(names) == len(set(names)) == 882
    held = names[::5]
    partitions = dict(train=[name for index, name in enumerate(names) if index % 5],
                      development=held[:88], audit=held[88:])
    excluded = old_tuples()
    caches, manifests = {}, {}
    for split, samples, sample_seed in [('train', 256, 49101), ('development', 128, 49102)]:
        caches[split], manifests[split] = cache_split(
            model, partitions[split], samples, sample_seed, root / split, excluded)
    sizing = size_batch(model, caches['train'], root)
    arms = {mode: fit(model, mode, caches['train'], caches['development'], sizing, root / mode)
            for mode in ['attention', 'mlp']}
    caches['audit'], manifests['audit'] = cache_split(
        model, partitions['audit'], 128, 49103, root / 'audit', excluded)
    audits = {}
    for mode in ['attention', 'mlp']:
        adapters, handles = attach(model, mode)
        adapters.load_state_dict(torch.load(arms[mode]['selected_path'],
                                           map_location='cpu')['adapter'], strict=True)
        audits[mode] = evaluate(model, caches['audit'])
        for handle in handles:
            handle.remove()
    assert original == digest(model.state_dict())
    assert all(parameter.grad is None for parameter in model.parameters())
    effects, advances = statistics(audits)
    sources = [Path(__file__), Path(__file__).with_name('probe_cross_stream_context.py'),
        Path(__file__).with_name('diagnose_absent_position.py'),
        Path(__file__).with_name('diagnose_late_fusion_streams.py'),
        Path('miss_rgbt/model.py'), Path('miss_rgbt/objectives.py'), Path('miss_rgbt/data.py'),
        Path('lib/models/cadtrack/vit_care.py'), Path('lib/models/cadtrack/cadtrack.py'),
        Path('lib/models/layers/head.py'), Path('experiments/cadtrack/cadtrack.yaml'),
        Path('lib/train/data_specs/lasher_train.txt')]
    report = dict(status='complete', pid=os.getpid(), checkpoint=CHECKPOINT,
        checkpoint_sha256=CHECKPOINT_SHA, partitions=partitions, cache_manifests=manifests,
        batch_sizing=sizing, arms=arms, audit=audits, effects=effects, advance=advances,
        baseline_state_sha256=original, baseline_unchanged=True, baseline_gradients_absent=True,
        source_sha256={str(path): sha256(path) for path in sources},
        seconds=time.time() - started,
        scope='Train882 cached final-two-block low-rank pilot; not official tracking metrics.')
    save(root / 'report.json', report)
    save(root / 'completed.json', dict(status='complete', report_sha256=sha256(root / 'report.json'),
        advance=advances, seconds=report['seconds']))
    print(json.dumps(dict(stage='complete', effects=effects, advance=advances,
                          seconds=report['seconds'])), flush=True)


if __name__ == '__main__':
    main()
