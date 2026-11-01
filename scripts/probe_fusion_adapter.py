"""Train-only paired optimization probe; not a tracking benchmark."""
import argparse
import datetime
import hashlib
import json
import os
import random
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn

from lib.config.cadtrack.config import cfg, update_config_from_file
from miss_rgbt.data import make_loader
from miss_rgbt.model import load_pretrained
from miss_rgbt.objectives import task_loss
from scripts.diagnose_joint_utility_sets import capture_identities, sha256, tensor_sha256


CHECKPOINT = '/home/gaob/Miss-RGBT/checkpoints/pretrained/CADTrack_ep0020.pth.tar'
CHECKPOINT_SHA = 'd20c8641e4263df3ceadf9e430d0ac96c181c13ad482bca28c2dddef592a8bb8'
STATES = [('11', [1., 1.]), ('01', [0., 1.]), ('10', [1., 0.])]
MAPS = ['score_map', 'size_map', 'offset_map', 'pred_boxes']


def seed(value):
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    torch.cuda.manual_seed_all(value)


def digest(state):
    value = hashlib.sha256()
    for key, tensor in sorted(state.items()):
        value.update(key.encode())
        value.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return value.hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def head_output(head, features):
    score, boxes, size, offset = head(features)
    return dict(score_map=score, pred_boxes=boxes.unsqueeze(1),
                size_map=size, offset_map=offset)


class FusionAdapter(nn.Module):
    def __init__(self, channels, mode):
        super().__init__()
        assert mode in ['local', 'global']
        self.mode = mode
        self.layers = nn.Sequential(nn.Conv2d(channels + 4, 32, 1),
                                    nn.GELU(), nn.Conv2d(32, channels, 1))
        nn.init.zeros_(self.layers[-1].weight)
        nn.init.zeros_(self.layers[-1].bias)

    def forward(self, features, flags):
        if bool(flags.eq(1).all()):
            return features
        inputs = features
        if self.mode == 'global':
            inputs = inputs.mean((2, 3), keepdim=True).expand_as(features)
        availability = flags[:, :, None, None].expand(-1, -1, *features.shape[2:])
        correction = self.layers(torch.cat([inputs, availability], dim=1))
        gate = (1 - flags.prod(1))[:, None, None, None]
        return features + gate * correction


def build_cache(model, names, samples, sample_seed, fitting, folder):
    seed(sample_seed)
    loader = make_loader(cfg, '/data/wangwj/dataset/LasHeR/trainingset',
                         'train', 16, samples, 0)
    loader.dataset.datasets[0].sequence_list = list(names)
    identities = capture_identities(loader)
    generator = torch.Generator().manual_seed(sample_seed + 100)
    choices = torch.tensor([item[1] for item in STATES])
    captured = {}

    def capture(module, inputs, output):
        captured['features'] = output.detach()

    hook = model.baseline.decode_fuse_search.register_forward_hook(capture)
    features, boxes, flags, labels, sources = [], [], [], [], []
    input_manifest = []
    offset = 0
    parity = True
    with torch.no_grad():
        for batch_index, data in enumerate(loader):
            templates = [image.cuda() for image in data['template_images']]
            search = data['search_images'][0].cuda()
            target = data['search_anno'][0].cpu()
            batch = search.shape[0]
            assert len(templates) == 2
            batch_identities = identities[offset:offset + batch]
            assert len(batch_identities) == batch
            for index, identity in enumerate(batch_identities):
                input_manifest.append(dict(
                    identity=identity,
                    templates_sha256=tensor_sha256(data['template_images'][:, index]),
                    search_sha256=tensor_sha256(data['search_images'][0, index]),
                    boxes_sha256=tensor_sha256(target[index])))
            offset += batch
            if fitting:
                dynamic = choices[torch.multinomial(torch.tensor([.4, .3, .3]),
                    batch, replacement=True, generator=generator)].cuda()
                conditions = [('mixed_' + key, dynamic, state)
                              for key, state in STATES[1:]]
            else:
                conditions = [(dynamic_key + '_' + search_key,
                    search.new_tensor(dynamic_state).expand(batch, -1), search_state)
                    for dynamic_key, dynamic_state in STATES
                    for search_key, search_state in STATES]
            for label, dynamic_flags, search_state in conditions:
                search_flags = search.new_tensor(search_state).expand(batch, -1)
                availability = [search.new_ones(batch, 2), dynamic_flags, search_flags]
                native = model(templates, [search], availability)[0]
                fused = captured['features']
                replay = head_output(model.baseline.box_head, fused)
                equal = all(torch.equal(native[key], replay[key]) for key in MAPS)
                assert equal, (folder.name, label, 'head replay')
                parity = parity and equal
                features.append(fused.cpu())
                boxes.append(target)
                flags.append(torch.cat([dynamic_flags, search_flags], dim=1).cpu())
                labels.extend([label] * batch)
                sources.extend(batch_identities)
            print(json.dumps(dict(stage='cache', split=folder.name,
                                  samples=offset, expected=samples)), flush=True)
    hook.remove()
    assert offset == samples
    assert all(identity['sequence'] in names for identity in sources)
    cached = dict(features=torch.cat(features), boxes=torch.cat(boxes),
                  flags=torch.cat(flags), labels=labels, identities=sources)
    folder.mkdir()
    path = folder / 'features.pth'
    torch.save(cached, str(path))
    manifest = dict(samples=samples, rows=len(labels), seed=sample_seed,
                    sequence_names=list(names), input_manifest=input_manifest,
                    cache_sha256=sha256(path), native_head_replay_bitwise=parity,
                    dtype=str(cached['features'].dtype))
    save(folder / 'manifest.json', manifest)
    return cached, manifest


def measure(adapter, cache, head, baseline=None):
    rows = []
    with torch.no_grad():
        for start in range(0, len(cache['labels']), 16):
            end = start + 16
            features = cache['features'][start:end].cuda()
            flags = cache['flags'][start:end].cuda()
            target = cache['boxes'][start:end].cuda()
            native = head_output(head, features)
            adapted = head_output(head, adapter(features, flags))
            native_loss = task_loss([native], target.unsqueeze(0), cfg)
            adapted_loss = task_loss([adapted], target.unsqueeze(0), cfg)
            assert torch.isfinite(native_loss).all() and torch.isfinite(adapted_loss).all()
            complete = flags.eq(1).all(1)
            if bool(complete.any()):
                for key in MAPS:
                    assert torch.equal(native[key][complete], adapted[key][complete]), key
            for index in range(features.shape[0]):
                position = start + index
                row = dict(identity=cache['identities'][position],
                           cell=cache['labels'][position],
                           native_loss=float(native_loss[index]),
                           adapted_loss=float(adapted_loss[index]))
                if baseline is not None:
                    assert row['native_loss'] == baseline[position]['native_loss']
                rows.append(row)
    cells = {}
    for cell in sorted(set(row['cell'] for row in rows)):
        selected = [row for row in rows if row['cell'] == cell]
        cells[cell] = dict(native=float(np.mean([row['native_loss'] for row in selected])),
                           adapted=float(np.mean([row['adapted_loss'] for row in selected])))
    score = np.mean([value['adapted'] for cell, value in cells.items()
                     if cell.split('_')[-1] != '11'])
    return dict(selection_score=float(score), cells=cells, rows=rows)


def fit_arm(mode, initial, train, development, head, folder):
    folder.mkdir()
    adapter = FusionAdapter(train['features'].shape[1], mode).cuda()
    adapter.load_state_dict(initial, strict=True)
    initial_result = measure(adapter, development, head)
    assert all(row['native_loss'] == row['adapted_loss']
               for row in initial_result['rows'])
    save(folder / 'initial.json', initial_result)
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=8e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[6], gamma=.1)
    history, steps, reference, stale = [], [], None, 0
    started = time.time()
    torch.cuda.reset_peak_memory_stats()
    for epoch in range(1, 21):
        generator = torch.Generator().manual_seed(40000 + epoch)
        order = torch.randperm(len(train['labels']), generator=generator)
        for start in range(0, len(order), 64):
            index = order[start:start + 64]
            optimizer.zero_grad(set_to_none=True)
            fused = train['features'][index].cuda()
            available = train['flags'][index].cuda()
            target = train['boxes'][index].cuda()
            prediction = head_output(head, adapter(fused, available))
            loss = task_loss([prediction], target.unsqueeze(0), cfg).mean()
            assert torch.isfinite(loss)
            loss.backward()
            norm = nn.utils.clip_grad_norm_(adapter.parameters(), .1)
            assert torch.isfinite(norm)
            optimizer.step()
            steps.append(dict(epoch=epoch, update=len(steps) + 1, loss=float(loss.detach()),
                              gradient_norm=float(norm)))
        evaluation = measure(adapter, development, head, initial_result['rows'])
        score = evaluation['selection_score']
        if epoch >= 7:
            if reference is None or score < reference * .998:
                reference, stale = score, 0
            else:
                stale += 1
        record = dict(epoch=epoch, updates=len(steps), selection_score=score,
                      cells=evaluation['cells'], stale_epochs=stale,
                      lr=optimizer.param_groups[0]['lr'], seconds=time.time() - started)
        history.append(record)
        torch.save(dict(adapter=adapter.state_dict(), epoch=epoch, mode=mode),
                   str(folder / ('epoch%02d.pth' % epoch)))
        save(folder / 'history.json', history)
        save(folder / 'steps.json', steps)
        print(json.dumps(dict(stage='epoch', arm=mode, **record)), flush=True)
        if epoch >= 10 and stale >= 3:
            break
        scheduler.step()
    selected = min(history[-5:], key=lambda row: (row['selection_score'], row['epoch']))
    path = folder / ('epoch%02d.pth' % selected['epoch'])
    adapter.load_state_dict(torch.load(str(path), map_location='cpu')['adapter'], strict=True)
    replay = measure(adapter, development, head, initial_result['rows'])
    assert replay['selection_score'] == selected['selection_score']
    final_five = [dict(epoch=row['epoch'], sha256=sha256(
        folder / ('epoch%02d.pth' % row['epoch']))) for row in history[-5:]]
    result = dict(arm=mode, selected=selected, final_five=final_five,
                  selected_path=str(path), selected_sha256=sha256(path), history=history,
                  updates=len(steps), stopped_epoch=history[-1]['epoch'],
                  parameter_count=sum(parameter.numel() for parameter in adapter.parameters()),
                  initial_adapter_sha256=digest(initial),
                  peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                  peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                  seconds=time.time() - started,
                  complete_history_maps_bitwise_equal=True)
    save(folder / 'selection.json', result)
    return adapter, result


def paired_interval(rows, field, sample_seed):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row['sequence']].append(row[field])
    means = np.array([np.mean(grouped[name]) for name in sorted(grouped)])
    rng = np.random.default_rng(sample_seed)
    draws = rng.integers(0, len(means), size=(10000, len(means)))
    low, high = np.quantile(means[draws].mean(1), [.025, .975])
    return dict(mean=float(means.mean()), ci95_low=float(low), ci95_high=float(high),
                sequences=len(means), rows=len(rows))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    started = time.time()
    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    seed(2027)
    assert sha256(CHECKPOINT) == CHECKPOINT_SHA
    update_config_from_file('experiments/cadtrack/cadtrack.yaml')
    model = load_pretrained(cfg, CHECKPOINT, repair=False).cuda().eval()
    model.requires_grad_(False)
    before = digest(model.state_dict())
    sequence_path = Path('lib/train/data_specs/lasher_train.txt')
    names = sequence_path.read_text().splitlines()
    assert len(names) == len(set(names)) == 882
    fitting = [name for index, name in enumerate(names) if index % 5]
    development = names[::5]
    split_names = dict(train=fitting, development=development[:88], audit=development[88:])
    assert not (set(fitting) & set(development))
    assert not (set(split_names['development']) & set(split_names['audit']))
    sources = {str(path): sha256(path) for path in [Path(__file__),
        Path('scripts/diagnose_joint_utility_sets.py'), Path('miss_rgbt/data.py'),
        Path('miss_rgbt/model.py'), Path('miss_rgbt/objectives.py'),
        Path('lib/models/cadtrack/cadtrack.py'), Path('lib/models/layers/head.py'),
        Path('experiments/cadtrack/cadtrack.yaml')]}
    launch = dict(pid=os.getpid(), started_utc=datetime.datetime.utcnow().isoformat() + 'Z',
                  checkpoint=CHECKPOINT, checkpoint_sha256=CHECKPOINT_SHA,
                  baseline_initial_state_sha256=before, source_sha256=sources,
                  split_names=split_names, train_list_sha256=sha256(sequence_path),
                  physical_gpu=os.environ['CUDA_VISIBLE_DEVICES'],
                  optimizer='AdamW', batch_size=64, learning_rate=8e-4,
                  protocol='train882 adaptation fit/development/audit only; no val97 or official tests')
    save(output / 'launch.json', launch)
    cached, manifests = {}, {}
    for split, samples, cache_seed in [('train', 256, 38101),
                                       ('development', 128, 38102), ('audit', 128, 38103)]:
        cached[split], manifests[split] = build_cache(
            model, split_names[split], samples, cache_seed, split == 'train', output / split)
    assert digest(model.state_dict()) == before
    cache_peak_allocated_bytes = torch.cuda.max_memory_allocated()
    seed(2027)
    initial = FusionAdapter(cached['train']['features'].shape[1], 'local').state_dict()
    initial = {key: value.clone() for key, value in initial.items()}
    arms, results = {}, {}
    for mode in ['local', 'global']:
        arms[mode], results[mode] = fit_arm(
            mode, initial, cached['train'], cached['development'],
            model.baseline.box_head, output / mode)
    assert results['local']['parameter_count'] == results['global']['parameter_count']
    assert results['local']['initial_adapter_sha256'] == results['global']['initial_adapter_sha256']
    audit = {mode: measure(arms[mode], cached['audit'], model.baseline.box_head)
             for mode in ['local', 'global']}
    paired = []
    for local, control in zip(audit['local']['rows'], audit['global']['rows']):
        assert local['identity'] == control['identity'] and local['cell'] == control['cell']
        assert local['native_loss'] == control['native_loss']
        paired.append(dict(sequence=local['identity']['sequence'], cell=local['cell'],
                           native_loss=local['native_loss'],
                           native_gain=local['native_loss'] - local['adapted_loss'],
                           relative_gain=(local['native_loss'] - local['adapted_loss']) /
                           max(local['native_loss'], 1e-6),
                           control_gain=control['adapted_loss'] - local['adapted_loss']))
    effects, gates = {}, []
    for state in ['01', '10']:
        subset = [row for row in paired if row['cell'].endswith('_' + state)]
        effects[state] = dict(native_gain=paired_interval(subset, 'native_gain', 50101),
                              relative_gain=paired_interval(subset, 'relative_gain', 50102))
        native = defaultdict(list)
        for row in subset:
            native[row['sequence']].append(row['native_loss'])
        macro_native = float(np.mean([np.mean(values) for values in native.values()]))
        effects[state]['relative_macro_loss_improvement'] = (
            effects[state]['native_gain']['mean'] / macro_native)
        gates.append(effects[state]['native_gain']['ci95_low'] > 0
                     and effects[state]['relative_macro_loss_improvement'] >= .01)
    missing = [row for row in paired if not row['cell'].endswith('_11')]
    effects['local_vs_global'] = paired_interval(missing, 'control_gain', 50103)
    gates.append(effects['local_vs_global']['ci95_low'] > 0)
    after = digest(model.state_dict())
    assert before == after
    assert all(parameter.grad is None for parameter in model.parameters())
    report = dict(launch=launch, cache_manifests=manifests, arms=results,
                  cache_peak_allocated_bytes=cache_peak_allocated_bytes,
                  audit=audit, effects=effects, baseline_final_state_sha256=after,
                  baseline_state_unchanged=True, baseline_gradients_absent=True,
                  complete_history_all_map_parity=True,
                  advance_to_fresh_train882_closed_loop=all(gates),
                  seconds=time.time() - started,
                  scope='Cached-feature training diagnostic; not benchmark metrics, full-tracker convergence, or a validated module.')
    save(output / 'report.json', report)
    save(output / 'completed.json', dict(
        status='complete', report_sha256=sha256(output / 'report.json'),
        baseline_state_unchanged=True,
        advance_to_fresh_train882_closed_loop=all(gates),
        seconds=report['seconds']))
    print(json.dumps(dict(stage='complete', effects=effects,
                         advance=all(gates), seconds=report['seconds'])), flush=True)


if __name__ == '__main__':
    main()
