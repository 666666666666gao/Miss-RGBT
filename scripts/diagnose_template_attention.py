"""Measure template foreground/background attention on fixed native candidates."""
import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import time

import cv2
import numpy as np
import torch
from lib.config.cadtrack.config import cfg, update_config_from_file
from lib.test.utils.hann import hann2d
from miss_rgbt.data import make_loader
from miss_rgbt.model import load_pretrained
import diagnose_absent_position as common
import diagnose_candidate_quality as candidates
import diagnose_last_observed_template as history

PROJECT = history.PROJECT
STATES = [('11', [1., 1.]), ('01', [0., 1.]), ('10', [1., 0.])]


def exclusions():
    used, hashes = history.exclusions()
    path = PROJECT / 'reports/server28_last_observed_template_20261102/selected_clips.json'
    hashes[path.parent.name + '/selected_clips.json'] = common.sha256(path)
    for clip in json.loads(path.read_text()):
        used[clip['sequence']].update(clip['frame_ids'])
    return used, hashes


def query_weights(box):
    index = torch.arange(8, device=box.device).float() / 8
    yy, xx = torch.meshgrid(index, index, indexing='ij')
    low = torch.stack([xx, yy], -1).reshape(1, 64, 2)
    high = low + 1. / 8
    intersection = (torch.minimum(high, (box[:, :2] + box[:, 2:])[:, None])
                    - torch.maximum(low, box[:, None, :2])).clamp_min(0).prod(-1) * 64
    foreground, background = intersection, 1 - intersection
    assert (foreground.sum(-1) > 0).all() and (background.sum(-1) > 0).all()
    return foreground, background


def read_attention(qkv, weights, heads):
    batch, tokens, width = qkv.shape
    assert tokens == 385 and width == 2304
    values = qkv.reshape(batch, tokens, 3, heads, 768 // heads).permute(2, 0, 3, 1, 4)
    query, key = values[0], values[1]
    attention = ((query[:, :, 1:65] @ key.transpose(-2, -1))
                 * ((768 // heads) ** -.5)).softmax(-1)[..., 129:]
    outputs = []
    for weight in weights:
        value = (attention * weight[:, None, :, None]).sum(2) / weight.sum(-1)[:, None, None]
        outputs.append(value.mean(1))
    return torch.stack(outputs, 1)


def pair_counts(ious, scores):
    count, correct = 0, 0.
    for first in range(8):
        for second in range(first + 1, 8):
            delta = ious[first] - ious[second]
            if abs(delta) >= .05:
                count += 1
                sign = (scores[first] - scores[second]) * delta
                correct += 1. if sign > 0 else .5 if sign == 0 else 0.
    return count, correct


def summarize(rows):
    groups = dict(complete=[row for row in rows if row['cell'] == '11_11'],
        rgb_missing=[row for row in rows if row['cell'].endswith('_01')],
        tir_missing=[row for row in rows if row['cell'].endswith('_10')])
    results = {}
    for category, values in groups.items():
        totals = defaultdict(lambda: np.zeros(3, dtype=np.float64))
        for row in values:
            totals[row['identity']['sequence']] += [
                row['pair_count'], row['foreground_correct'], row['background_correct']]
        qualified = [totals[name] for name in sorted(totals) if totals[name][0] > 0]
        assert qualified
        matrix = np.asarray(qualified)
        fg, bg = matrix[:, 1] / matrix[:, 0], matrix[:, 2] / matrix[:, 0]
        rng = np.random.default_rng(60301)
        draws = rng.integers(0, len(fg), (10000, len(fg)))
        estimates = {}
        for label, data in [('foreground', fg), ('background', bg), ('difference', fg - bg)]:
            low, high = np.quantile(data[draws].mean(1), [.025, .975])
            estimates[label] = dict(mean=float(data.mean()), ci95_low=float(low), ci95_high=float(high))
        estimates.update(sequences=len(fg), pairs=int(matrix[:, 0].sum()), rows=len(values))
        for key in ['foreground_gain', 'background_gain']:
            estimates[key] = common.clustered(values, key, 60301)
        results[category] = estimates
    advance = all(value['sequences'] >= 64 and value['pairs'] >= 100
        and value['foreground']['ci95_low'] > .5 and value['difference']['mean'] >= .05
        and value['difference']['ci95_low'] > 0 for value in results.values())
    return results, bool(advance)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    root = Path(args.output)
    root.mkdir(exist_ok=False)
    started = time.time()
    common.seed(60101)
    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    update_config_from_file('experiments/cadtrack/cadtrack.yaml')
    assert common.sha256(common.CHECKPOINT) == common.CHECKPOINT_SHA
    used, previous_hashes = exclusions()
    model = load_pretrained(cfg, common.CHECKPOINT, repair=False).cuda().eval().requires_grad_(False)
    before = common.digest(model.state_dict())
    names_path = Path('lib/train/data_specs/lasher_train.txt')
    names = names_path.read_text().splitlines()
    assert len(names) == len(set(names)) == 882
    fitting = [name for index, name in enumerate(names) if index % 5]
    loader = make_loader(cfg, history.training.burst.DATASET, 'train', 16, 128, 0)
    dataset = loader.dataset.datasets[0]
    dataset.sequence_list = fitting
    original_info = dataset.get_sequence_info

    def excluded_info(index):
        info = dict(original_info(index))
        info['visible'] = info['visible'].clone()
        info['visible'][sorted(used[dataset.sequence_list[index]])] = False
        return info

    dataset.get_sequence_info = excluded_info
    identities = common.capture_identities(loader)
    window = hann2d(torch.tensor([16, 16]), centered=True).cuda()
    rows, manifests, raw_qkv = [], [], {}
    image_values = {key: [] for key in ['templates', 'search', 'target', 'template_box']}
    map_values = {key: [] for key in ['score_map', 'size_map', 'offset_map', 'attention',
                                     'weights', 'target', 'indices']}
    count, parity_calls = 0, 0
    with torch.no_grad():
        for data in loader:
            templates = [value.cuda() for value in data['template_images']]
            search = data['search_images'][0].cuda()
            target = data['search_anno'][0].cuda()
            template_box = data['template_anno'][0].cuda()
            weights = query_weights(template_box)
            batch_ids = identities[count:count + len(search)]
            for item, identity in enumerate(batch_ids):
                assert identity['sequence'] in fitting
                assert not used[identity['sequence']].intersection(
                    identity['template_frames'] + identity['search_frames'])
                tensors = dict(templates=data['template_images'][:, item],
                    search=data['search_images'][0, item], target=data['search_anno'][0, item],
                    template_box=data['template_anno'][0, item])
                manifests.append(dict(identity=identity, tensors_sha256=common.digest(tensors)))
            image_values['templates'].append(data['template_images'].transpose(0, 1))
            image_values['search'].append(data['search_images'][0])
            image_values['target'].append(data['search_anno'][0])
            image_values['template_box'].append(data['template_anno'][0])
            for dynamic_name, dynamic in STATES:
                for search_name, flags in STATES:
                    cell = dynamic_name + '_' + search_name
                    availability = [search.new_ones(len(search), 2),
                        search.new_tensor(dynamic).expand(len(search), -1),
                        search.new_tensor(flags).expand(len(search), -1)]
                    plain = model(templates, [search], availability)[0] if count == 0 else None
                    captured = []
                    projection = model.baseline.backbone.blocks[5].attn.qkv
                    heads = model.baseline.backbone.blocks[5].attn.num_heads

                    def capture(module, inputs, output):
                        captured.append(read_attention(output, weights, heads))
                        if count == 0:
                            raw_qkv.setdefault(cell, []).append(output[:1].cpu())

                    handle = projection.register_forward_hook(capture)
                    output = model(templates, [search], availability)[0]
                    handle.remove()
                    assert len(captured) == 2
                    if plain is not None:
                        assert all(torch.equal(output[key], plain[key]) for key in common.MAPS)
                        parity_calls += 1
                    attention = sum(value * flags[index] for index, value in enumerate(captured)) / sum(flags)
                    assert torch.isfinite(attention).all() and (attention > 0).all()
                    inspections = candidates.inspect_output(model, output, target, window)
                    indices = candidates.stable_descending((output['score_map'] * window).flatten(1))[:, :8]
                    for item, (identity, inspected) in enumerate(zip(batch_ids, inspections)):
                        fg = attention[item, 0, indices[item]].cpu().tolist()
                        bg = attention[item, 1, indices[item]].cpu().tolist()
                        choices = inspected['candidates']
                        ious = [candidate['iou'] for candidate in choices]
                        prior = np.asarray([candidate['hann_score'] for candidate in choices])
                        pair_count, foreground_correct = pair_counts(ious, fg)
                        count_bg, background_correct = pair_counts(ious, bg)
                        assert pair_count == count_bg
                        fg_choice = int((prior * fg).argmax())
                        bg_choice = int((prior * bg).argmax())
                        rows.append(dict(identity=identity, cell=cell, target=inspected['target'],
                            candidates=choices, foreground=fg, background=bg,
                            pair_count=pair_count, foreground_correct=foreground_correct,
                            background_correct=background_correct, native_iou=ious[0],
                            foreground_choice=fg_choice, background_choice=bg_choice,
                            foreground_gain=ious[fg_choice] - ious[0],
                            background_gain=ious[bg_choice] - ious[0]))
                    for key in ['score_map', 'size_map', 'offset_map']:
                        map_values[key].append(output[key].cpu())
                    for key, value in dict(attention=attention, weights=torch.stack(weights, 1),
                                           target=target, indices=indices).items():
                        map_values[key].append(value.cpu())
            count += len(search)
            print(json.dumps(dict(stage='batch', tuples=count, rows=len(rows),
                                  seconds=time.time() - started)), flush=True)
    assert count == 128 and len(rows) == 1152 and parity_calls == 9
    images = {key: torch.cat(values) for key, values in image_values.items()}
    maps = {key: torch.cat(values) for key, values in map_values.items()}
    maps.update(window=window.cpu(), raw_qkv=raw_qkv)
    torch.save(images, str(root / 'images.pth'))
    torch.save(maps, str(root / 'maps.pth'))
    effects, advance = summarize(rows)
    assert common.digest(model.state_dict()) == before
    assert all(value.grad is None for value in model.parameters())
    sources = [Path(__file__), Path(history.__file__), Path(history.training.__file__),
        Path(common.__file__), Path(candidates.__file__), names_path,
        Path('experiments/cadtrack/cadtrack.yaml'), Path('lib/train/data/processing_utils.py'),
        Path('lib/train/data/sampler.py'), Path('lib/train/dataset/lasher.py'),
        Path('lib/test/utils/hann.py'), Path('lib/utils/box_ops.py')]
    sources += sorted(Path('miss_rgbt').rglob('*.py')) + sorted(Path('lib/models').rglob('*.py'))
    report = dict(status='complete', pid=os.getpid(), seed=60101, fitting_names=fitting,
        input_manifest=manifests, rows=rows, effects=effects, advance=advance,
        prior_reports_sha256=previous_hashes, checkpoint=common.CHECKPOINT,
        checkpoint_sha256=common.CHECKPOINT_SHA, baseline_state_sha256=before,
        controls=dict(hook_parity_calls=parity_calls, frozen_state=True, gradients_absent=True),
        caches={key: dict(path=str(root / (key + '.pth')), sha256=common.sha256(root / (key + '.pth')))
                for key in ['images', 'maps']},
        source_sha256={str(path.resolve()): common.sha256(path) for path in sources},
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        peak_reserved_bytes=torch.cuda.max_memory_reserved(), seconds=time.time() - started,
        scope='Frozen contextual attention information on augmented train882 crops, not official tracking metrics.')
    history.training.save(root / 'report.json', report)
    history.training.save(root / 'completed.json', dict(status='complete', advance=advance,
        report_sha256=common.sha256(root / 'report.json'), seconds=report['seconds']))
    print(json.dumps(dict(stage='complete', effects=effects, advance=advance,
                          seconds=report['seconds'])), flush=True)


if __name__ == '__main__':
    main()
