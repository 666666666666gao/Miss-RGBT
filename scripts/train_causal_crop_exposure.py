"""Matched full-image training from native versus frozen-prediction crops."""
import argparse
import copy
import json
import math
import os
from pathlib import Path
import time

import cv2
import numpy as np
import torch

from lib.config.cadtrack.config import cfg, update_config_from_file
from lib.train.dataset import LasHeR
from lib.train.data.processing_utils import sample_target, transform_image_to_crop
from miss_rgbt.model import load_pretrained
from miss_rgbt.objectives import task_loss
from miss_rgbt.train import fix_bn
from miss_rgbt.tracker import MissingTracker
import diagnose_absent_position as common
import diagnose_burst_mask_rollout as burst
import diagnose_causal_crop_exposure as causal

PROJECT = Path('/home/gaob/Miss-RGBT')
STATES = causal.STATES


def save(path, value):
    with Path(path).open('x', newline='\n') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')


def fresh_clips(dataset, names, count, length, sample_seed, used):
    rng = np.random.default_rng(sample_seed)
    selected = []
    for offset in rng.permutation(len(names)):
        name = names[offset]
        index = dataset.sequence_list.index(name)
        info = dataset.get_sequence_info(index)
        boxes = info['bbox'].numpy()
        valid = (np.isfinite(boxes).all(1) & (boxes[:, 2:] > 0).all(1)
                 & info['visible'].numpy().astype(bool))
        starts = [start for start in range(len(boxes) - length + 1)
                  if valid[start:start + length].all()
                  and not used[name].intersection(range(start, start + length))]
        if starts:
            start = int(rng.choice(starts))
            schedule = ['11'] + rng.choice(list(STATES), length - 1, p=[.4, .3, .3]).tolist()
            selected.append(dict(sequence=name, dataset_index=index,
                frame_ids=list(range(start, start + length)), mixed_schedule=schedule))
        if len(selected) == count:
            break
    assert len(selected) == count
    return selected


def load_clip(dataset, clip):
    info = dataset.get_sequence_info(clip['dataset_index'])
    frames, annotations, _ = dataset.get_frames(clip['dataset_index'], clip['frame_ids'], info)
    boxes = [value.tolist() for value in annotations['bbox']]
    return frames, boxes


def crop_label(image, extract, target):
    crop, factor, _ = sample_target(image, extract, 4., output_sz=256)
    label = transform_image_to_crop(torch.tensor(target), torch.tensor(extract),
                                    factor, torch.tensor([256., 256.]), normalize=True)
    valid = bool((label[:2] >= 0).all() and (label[:2] + label[2:] <= 1).all()
                 and (label[2:] > 0).all() and torch.isfinite(label).all())
    return crop, label, valid


def collect(tracker, dataset, clips, root):
    common.seed(58111)
    records, manifests, rejected = [], [], []
    for clip_index, clip in enumerate(clips):
        frames, boxes = load_clip(dataset, clip)
        tracker.initialize(frames[0], dict(init_bbox=boxes[0]))
        for index in range(1, len(frames)):
            flags = clip['mixed_schedule'][index]
            if index % 8 == 0:
                gt = torch.tensor(boxes[index])
                size = gt[2:] * torch.exp(torch.randn(2) * .25)
                center = gt[:2] + .5 * gt[2:] + size.prod().sqrt() * 3. * (torch.rand(2) - .5)
                jittered = torch.cat([center - .5 * size, size]).tolist()
                predicted = list(tracker.state)
                native_crop, native_target, native_valid = crop_label(frames[index], jittered, boxes[index])
                causal_crop, causal_target, causal_valid = crop_label(frames[index], predicted, boxes[index])
                identity = dict(sequence=clip['sequence'], frame=clip['frame_ids'][index])
                if native_valid and causal_valid:
                    tensors = dict(templates=torch.cat(tracker.templates).cpu(),
                        native_search=tracker.preprocessor.process(native_crop)[0].cpu(),
                        causal_search=tracker.preprocessor.process(causal_crop)[0].cpu(),
                        native_target=native_target, causal_target=causal_target,
                        availability=torch.cat(tracker.availability + [
                            torch.tensor([STATES[flags]], device='cuda')]).cpu())
                    records.append(tensors)
                    manifests.append(dict(identity=identity, state=flags, gt=boxes[index],
                        jittered_crop_box=jittered, predicted_crop_box=predicted,
                        tensors_sha256=common.digest(tensors)))
                else:
                    rejected.append(dict(identity=identity, native_valid=native_valid,
                                         causal_valid=causal_valid))
            tracker.track(frames[index], STATES[flags])
        print(json.dumps(dict(stage='collect', clips=clip_index + 1, accepted=len(records))), flush=True)
    assert len(records) >= 128
    cache = {key: torch.stack([record[key] for record in records]) for key in records[0]}
    path = root / 'image_cache.pth'
    torch.save(cache, str(path))
    manifest = dict(samples=len(records), candidates=len(records) + len(rejected),
        accepted=manifests, rejected=rejected, clips=clips,
        cache_path=str(path), cache_sha256=common.sha256(path),
        image_tensors_only=True, backbone_features_cached=False)
    save(root / 'image_manifest.json', manifest)
    return cache, manifest


def training_step(model, cache, indices, mode):
    templates = [cache['templates'][indices, index].cuda() for index in range(2)]
    common_search = cache['native_search'][indices].cuda()
    common_target = cache['native_target'][indices].cuda()
    available = [cache['availability'][indices, index].cuda() for index in range(3)]
    complete = model(templates, [common_search])
    loss_c = task_loss(complete, common_target[None], cfg).mean()
    assert torch.isfinite(loss_c)
    loss_c.backward()
    value_c = float(loss_c.detach())
    del complete, loss_c
    search = cache[mode + '_search'][indices].cuda()
    target = cache[mode + '_target'][indices].cuda()
    missing = model(templates, [search], available)
    loss_m = task_loss(missing, target[None], cfg).mean()
    assert torch.isfinite(loss_m)
    loss_m.backward()
    return value_c, float(loss_m.detach())


def optimizer_for(model, lr):
    return torch.optim.AdamW([
        dict(params=[p for n, p in model.named_parameters() if 'backbone' in n], lr=lr * .1),
        dict(params=[p for n, p in model.named_parameters() if 'backbone' not in n], lr=lr)],
        weight_decay=1e-4)


def profile(cache, root):
    model = load_pretrained(cfg, common.CHECKPOINT, repair=False).cuda().train()
    fix_bn(model, True)
    optimizer = optimizer_for(model, 1e-5)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    static = torch.cuda.memory_allocated()
    training_step(model, cache, torch.arange(4), 'causal')
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), .1)
    assert torch.isfinite(norm)
    optimizer.step()
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated()
    total = torch.cuda.get_device_properties(0).total_memory
    # Account for persistent parameter, gradient and Adam moment storage separately.
    parameters = sum(value.numel() * value.element_size() for value in model.parameters())
    fixed = static + 3 * parameters
    activation = max(peak - fixed, 1)
    proposed = min(32, 4 * int((.8 * total - fixed) / (activation / 4 * 4)))
    assert proposed >= 4
    result = dict(probe_batch=4, static_bytes=static, peak_bytes=peak, total_bytes=total,
        parameters_bytes=parameters, fixed_estimate_bytes=fixed,
        activation_estimate_bytes=activation, batch=proposed, lr=1e-5 * proposed / 8,
        dropped_remainder=len(cache['native_search']) % proposed,
        updates_per_epoch=len(cache['native_search']) // proposed,
        policy='80percent estimated full-image memory, multiple4 cap32; no OOM retry.')
    save(root / 'batch_sizing.json', result)
    del optimizer, model
    torch.cuda.empty_cache()
    return result


def rollout(tracker, dataset, clips, label, keep_rows):
    tracker.network.eval()
    rows, manifests = [], []
    with torch.no_grad():
        for clip in clips:
            frames, boxes = load_clip(dataset, clip)
            for trajectory in ['complete', 'mixed']:
                schedule = ['11'] * len(frames) if trajectory == 'complete' else clip['mixed_schedule']
                tracker.initialize(frames[0], dict(init_bbox=boxes[0]))
                for index in range(1, len(frames)):
                    result = tracker.track(frames[index], STATES[schedule[index]])
                    row = dict(sequence=clip['sequence'], frame=clip['frame_ids'][index],
                        trajectory=trajectory, state=schedule[index],
                        iou=burst.box_iou(result['target_bbox'], boxes[index]))
                    assert np.isfinite(row['iou'])
                    if keep_rows:
                        row.update(box=list(result['target_bbox']), gt=boxes[index],
                                   confidence=result['best_score'])
                    rows.append(row)
            if keep_rows:
                manifests.append(dict(sequence=clip['sequence'], frame_ids=clip['frame_ids'],
                    image_sha256=burst.sha_image_sequence(frames), gt=boxes))
    groups = dict(complete=[row for row in rows if row['trajectory'] == 'complete'],
        missing=[row for row in rows if row['trajectory'] == 'mixed' and row['state'] != '11'])
    means = {}
    for key, values in groups.items():
        grouped = {}
        for name in sorted({row['sequence'] for row in values}):
            grouped[name] = float(np.mean([row['iou'] for row in values if row['sequence'] == name]))
        assert len(grouped) == len(clips)
        means[key] = dict(mean=float(np.mean(list(grouped.values()))), sequences=grouped)
    return dict(label=label, means=means, score=float(np.mean([value['mean'] for value in means.values()])),
                rows=rows if keep_rows else None, image_manifest=manifests)


def fit(mode, cache, sizing, dataset, dev_clips, root):
    folder = root / mode
    folder.mkdir()
    model = load_pretrained(cfg, common.CHECKPOINT, repair=False).cuda()
    tracker = MissingTracker(cfg, str(causal.WRAPPER))
    tracker.network = model
    initial = common.digest(model.state_dict())
    buffers = common.digest(dict(model.named_buffers()))
    optimizer = optimizer_for(model, sizing['lr'])
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    history, reference, stale = [], None, 0
    started = time.time()
    for epoch in range(1, 13):
        model.train()
        fix_bn(model, True)
        lr = sizing['lr'] * (1 if epoch < 5 else .1)
        optimizer.param_groups[0]['lr'] = lr * .1
        optimizer.param_groups[1]['lr'] = lr
        order = torch.randperm(len(cache['native_search']),
                               generator=torch.Generator().manual_seed(58200 + epoch))
        losses = []
        for step in range(sizing['updates_per_epoch']):
            common.seed(58400 + epoch * 1000 + step)
            indices = order[step * sizing['batch']:(step + 1) * sizing['batch']]
            optimizer.zero_grad(set_to_none=True)
            first, second = training_step(model, cache, indices, mode)
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), .1)
            assert torch.isfinite(norm)
            optimizer.step()
            losses.append(first + second)
        development = rollout(tracker, dataset, dev_clips, mode, False)
        score = development['score']
        if epoch >= 5:
            if reference is None or score > reference * 1.002:
                reference, stale = score, 0
            else:
                stale += 1
        record = dict(epoch=epoch, updates=epoch * sizing['updates_per_epoch'],
            lr=lr, train_mean_loss=float(np.mean(losses)), score=score, stale=stale,
            development=development, seconds=time.time() - started)
        history.append(record)
        assert common.digest(dict(model.named_buffers())) == buffers
        path = folder / ('epoch%02d.pth' % epoch)
        torch.save(dict(model=model.state_dict(), mode=mode, epoch=epoch,
                        config=dict(repair=False, repair_width=128, repair_layer=6)), str(path))
        save(folder / ('epoch%02d.json' % epoch), record)
        print(json.dumps(dict(stage='epoch', mode=mode, epoch=epoch, score=score,
                              stale=stale, seconds=record['seconds'])), flush=True)
        if epoch >= 8 and stale >= 3:
            break
    selected = max(history[-5:], key=lambda item: (item['score'], -item['epoch']))
    path = folder / ('epoch%02d.pth' % selected['epoch'])
    result = dict(mode=mode, initial_sha256=initial, initial_buffers_sha256=buffers,
        parameter_count=sum(value.numel() for value in model.parameters()),
        history=history, selected=selected, selected_path=str(path), selected_sha256=common.sha256(path),
        bn_buffers_unchanged=True, peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        final_five=[dict(epoch=item['epoch'], path=str(folder / ('epoch%02d.pth' % item['epoch'])),
            sha256=common.sha256(folder / ('epoch%02d.pth' % item['epoch']))) for item in history[-5:]])
    save(folder / 'selection.json', result)
    del model, tracker, optimizer
    torch.cuda.empty_cache()
    return result


def statistics(audits):
    paired = []
    for base, native, changed in zip(audits['e0']['rows'], audits['native']['rows'], audits['causal']['rows']):
        assert all(base[key] == native[key] == changed[key]
                   for key in ['sequence', 'frame', 'trajectory', 'state', 'gt'])
        paired.append(dict(identity=dict(sequence=base['sequence']), trajectory=base['trajectory'],
            state=base['state'], gain=changed['iou'] - base['iou'],
            control=changed['iou'] - native['iou']))
    effects = {}
    for state in ['01', '10']:
        values = [row for row in paired if row['trajectory'] == 'mixed' and row['state'] == state]
        effects[state] = common.clustered(values, 'gain', 58301)
    missing = [row for row in paired if row['trajectory'] == 'mixed' and row['state'] != '11']
    effects['causal_minus_native'] = common.clustered(missing, 'control', 58301)
    effects['complete'] = common.clustered(
        [row for row in paired if row['trajectory'] == 'complete'], 'gain', 58301)
    advance = all(effects[state]['mean'] >= .01 and effects[state]['ci95_low'] > 0
                  and effects[state]['sequences'] == 16 for state in ['01', '10'])
    advance = (advance and effects['causal_minus_native']['ci95_low'] > 0
        and effects['causal_minus_native']['sequences'] == 16
        and effects['complete']['mean'] >= 0 and effects['complete']['ci95_low'] > -.01
        and effects['complete']['sequences'] == 16)
    return effects, bool(advance)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    root = Path(args.output)
    root.mkdir(exist_ok=False)
    started = time.time()
    common.seed(58101)
    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    update_config_from_file('experiments/cadtrack/cadtrack.yaml')
    assert common.sha256(common.CHECKPOINT) == common.CHECKPOINT_SHA
    names = Path('lib/train/data_specs/lasher_train.txt').read_text().splitlines()
    assert len(names) == len(set(names)) == 882
    partitions = dict(train=[name for index, name in enumerate(names) if index % 5],
                      development=names[::5][:88], audit=names[::5][88:])
    used, prior_hashes = causal.excluded_frames()
    previous = PROJECT / 'reports/server28_causal_crop_exposure_20261102/report.json'
    prior = json.loads(previous.read_text())
    assert prior['advance']
    prior_hashes[previous.parent.name] = common.sha256(previous)
    for clip in prior['clips']:
        used[clip['sequence']].update(clip['frame_ids'])
    dataset = LasHeR(root=burst.DATASET, split='train')
    train_clips = fresh_clips(dataset, partitions['train'], 32, 96, 58101, used)
    dev_clips = fresh_clips(dataset, partitions['development'], 8, 48, 58102, used)
    save(root / 'initial_clips.json', dict(train=train_clips, development=dev_clips))
    sources = [Path(__file__), Path(causal.__file__), Path(common.__file__), Path(burst.__file__),
        Path('lib/train/data_specs/lasher_train.txt'), Path('experiments/cadtrack/cadtrack.yaml'),
        Path('lib/train/data/processing_utils.py'), Path('lib/utils/heapmap_utils.py'),
        Path('lib/utils/box_ops.py'), Path('lib/test/tracker/data_utils.py'),
        Path('lib/test/utils/hann.py')]
    sources += sorted(Path('miss_rgbt').rglob('*.py')) + sorted(Path('lib/models').rglob('*.py'))
    source_hashes = {str(path.resolve()): common.sha256(path) for path in sources}
    save(root / 'source_sha256.json', source_hashes)
    collector = MissingTracker(cfg, str(causal.WRAPPER))
    collector.network.eval().requires_grad_(False)
    frozen = common.digest(collector.network.state_dict())
    cache, manifest = collect(collector, dataset, train_clips, root)
    assert frozen == common.digest(collector.network.state_dict())
    assert all(value.grad is None for value in collector.network.parameters())
    del collector
    torch.cuda.empty_cache()
    sizing = profile(cache, root)
    arms = {mode: fit(mode, cache, sizing, dataset, dev_clips, root) for mode in ['native', 'causal']}
    assert arms['native']['initial_sha256'] == arms['causal']['initial_sha256'] == frozen
    assert arms['native']['parameter_count'] == arms['causal']['parameter_count']
    audit_clips = fresh_clips(dataset, partitions['audit'], 16, 96, 58103, used)
    save(root / 'audit_clips.json', audit_clips)
    audits = {}
    for mode in ['e0', 'native', 'causal']:
        path = str(causal.WRAPPER) if mode == 'e0' else arms[mode]['selected_path']
        tracker = MissingTracker(cfg, path)
        tracker.network.eval().requires_grad_(False)
        before = common.digest(tracker.network.state_dict())
        audits[mode] = rollout(tracker, dataset, audit_clips, mode, True)
        assert common.digest(tracker.network.state_dict()) == before
        assert all(value.grad is None for value in tracker.network.parameters())
        save(root / ('audit_' + mode + '.json'), audits[mode])
        print(json.dumps(dict(stage='audit', mode=mode, score=audits[mode]['score'])), flush=True)
        del tracker
        torch.cuda.empty_cache()
    effects, advance = statistics(audits)
    report = dict(status='complete', pid=os.getpid(), checkpoint=common.CHECKPOINT,
        checkpoint_sha256=common.CHECKPOINT_SHA, original_state_sha256=frozen,
        partitions=partitions, training_manifest=manifest, development_clips=dev_clips,
        audit_clips=audit_clips, batch_sizing=sizing, arms=arms, audits=audits,
        effects=effects, advance=advance, prior_reports_sha256=prior_hashes,
        source_sha256=source_hashes, seconds=time.time() - started,
        scope='Train882 matched full-image offline crop exposure, closed-loop internal IoU; no official scores.')
    save(root / 'report.json', report)
    save(root / 'completed.json', dict(status='complete', advance=advance,
        report_sha256=common.sha256(root / 'report.json'), seconds=report['seconds']))
    print(json.dumps(dict(stage='complete', effects=effects, advance=advance,
                          seconds=report['seconds'])), flush=True)


if __name__ == '__main__':
    main()
