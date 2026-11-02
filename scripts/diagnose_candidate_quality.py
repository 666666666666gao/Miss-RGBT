"""Fresh train882 diagnostic of ranking headroom with fixed native geometry."""
import argparse
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
from miss_rgbt.data import make_loader
from miss_rgbt.model import load_pretrained
from diagnose_absent_position import (
    CHECKPOINT, CHECKPOINT_SHA, capture_identities, clustered, digest, seed, sha256)
from probe_tail_lowrank import old_tuples

PROJECT = Path('/home/gaob/Miss-RGBT')
STATES = [('11', [1., 1.]), ('01', [0., 1.]), ('10', [1., 0.])]


def save(path, value):
    with Path(path).open('x', newline='\n') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')


def excluded_tuples():
    excluded = old_tuples()
    path = PROJECT / 'reports/server28_tail_lowrank_20261102/report.json'
    record = json.loads(path.read_text())
    for manifest in record['cache_manifests'].values():
        for item in manifest['input_manifest']:
            value = item['identity']
            excluded.add((value['sequence'], tuple(value['template_frames']),
                          tuple(value['search_frames'])))
    return excluded


def dense_boxes(output):
    batch, _, side, width = output['score_map'].shape
    assert side == width == 16
    index = torch.arange(side * side, device=output['score_map'].device)
    offset = output['offset_map'].flatten(2)
    size = output['size_map'].flatten(2)
    cx = (index.remainder(side)[None].float() + offset[:, 0]) / side
    cy = (index.div(side, rounding_mode='floor')[None].float() + offset[:, 1]) / side
    return torch.stack([cx, cy, size[:, 0], size[:, 1]], -1)


def dense_iou(boxes, target):
    first = torch.cat([boxes[..., :2] - boxes[..., 2:] / 2,
                       boxes[..., :2] + boxes[..., 2:] / 2], -1)
    second = box_xywh_to_xyxy(target).clamp(0, 1)[:, None]
    overlap = (torch.minimum(first[..., 2:], second[..., 2:]) -
               torch.maximum(first[..., :2], second[..., :2])).clamp_min(0).prod(-1)
    union = ((first[..., 2:] - first[..., :2]).prod(-1) +
             (second[..., 2:] - second[..., :2]).prod(-1) - overlap)
    return overlap / union


def stable_descending(values):
    order = np.argsort(-values.detach().cpu().numpy(), axis=-1, kind='mergesort')
    return torch.from_numpy(order.copy()).to(values.device)


def inspect_output(model, output, target, window):
    boxes = dense_boxes(output)
    ious = dense_iou(boxes, target)
    raw = output['score_map'].flatten(1)
    weighted = (output['score_map'] * window).flatten(1)
    order = stable_descending(weighted)[:, :8]
    raw_index, hann_index = raw.argmax(1), weighted.argmax(1)
    batch_index = torch.arange(len(target), device=target.device)
    assert torch.equal(order[:, 0], hann_index)
    assert torch.equal(boxes[batch_index, raw_index], output['pred_boxes'][:, 0])
    hann_box = model.baseline.box_head.cal_bbox(output['score_map'] * window,
                                               output['size_map'], output['offset_map'])
    assert torch.equal(boxes[batch_index, hann_index], hann_box)
    _, native_iou = generalized_box_iou(
        box_cxcywh_to_xyxy(output['pred_boxes']).flatten(0, 1),
        box_xywh_to_xyxy(target).clamp(0, 1))
    assert torch.allclose(ious[batch_index, raw_index], native_iou, atol=2e-6, rtol=1e-6)
    assert torch.isfinite(boxes).all() and torch.isfinite(ious).all()
    rows = []
    for item in range(len(target)):
        candidates = []
        for location in order[item].tolist():
            candidates.append(dict(index=location, box=boxes[item, location].cpu().tolist(),
                raw_score=float(raw[item, location]), hann_score=float(weighted[item, location]),
                iou=float(ious[item, location])))
        raw_location = int(raw_index[item])
        best_dense = int(ious[item].argmax())
        raw_top8 = stable_descending(raw[item])[:8]
        selected = candidates[0]['iou']
        oracle = max(candidate['iou'] for candidate in candidates)
        rows.append(dict(target=target[item].cpu().tolist(),
            raw_index=raw_location, raw_box=boxes[item, raw_location].cpu().tolist(),
            raw_iou=float(ious[item, raw_location]), raw_score=float(raw[item, raw_location]),
            hann_index=int(hann_index[item]), hann_iou=selected,
            candidates=candidates, top8_oracle_iou=oracle,
            top8_oracle_gain=oracle - selected,
            top8_uniform_iou=float(np.mean([candidate['iou'] for candidate in candidates])),
            raw_top8_oracle_iou=float(ious[item, raw_top8].max()),
            dense_oracle_index=best_dense,
            dense_oracle_box=boxes[item, best_dense].cpu().tolist(),
            dense_oracle_iou=float(ious[item, best_dense]),
            hann_minus_raw=selected - float(ious[item, raw_location])))
    return rows


def statistics(rows):
    categories = dict(
        complete=[row for row in rows if row['cell'] == '11_11'],
        rgb_missing=[row for row in rows if row['cell'].endswith('_01')],
        tir_missing=[row for row in rows if row['cell'].endswith('_10')])
    effects = {name: clustered(values, 'top8_oracle_gain', 50102)
               for name, values in categories.items()}
    advance = all(value['mean'] >= .02 and value['ci95_low'] > 0
                  for value in effects.values())
    return effects, advance


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    root = Path(args.output)
    root.mkdir(exist_ok=False)
    started = time.time()
    seed(50101)
    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    update_config_from_file('experiments/cadtrack/cadtrack.yaml')
    assert sha256(CHECKPOINT) == CHECKPOINT_SHA
    model = load_pretrained(cfg, CHECKPOINT, repair=False).cuda().eval().requires_grad_(False)
    before = digest(model.state_dict())
    window = hann2d(torch.tensor([16, 16]), centered=True).cuda()
    train_file = Path('lib/train/data_specs/lasher_train.txt')
    names = train_file.read_text().splitlines()
    assert len(names) == len(set(names)) == 882
    fitting = [name for index, name in enumerate(names) if index % 5]
    loader = make_loader(cfg, '/data/wangwj/dataset/LasHeR/trainingset',
                         'train', 16, 128, 0)
    loader.dataset.datasets[0].sequence_list = fitting
    identities = capture_identities(loader)
    excluded = excluded_tuples()
    manifest, rows, count = [], [], 0
    maps = {key: [] for key in ['score_map', 'size_map', 'offset_map', 'pred_boxes', 'boxes']}
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
                assert key not in excluded and identity['sequence'] in fitting
                manifest.append(dict(identity=identity, tensors_sha256=digest(dict(
                    templates=data['template_images'][:, index],
                    search=data['search_images'][0, index], boxes=target[index]))))
            count += len(search)
            for dynamic_name, dynamic in STATES:
                for search_name, available in STATES:
                    output = model(templates, [search], [
                        search.new_ones(len(search), 2),
                        search.new_tensor(dynamic).expand(len(search), -1),
                        search.new_tensor(available).expand(len(search), -1)])[0]
                    inspected = inspect_output(model, output, target, window)
                    for identity, row in zip(batch_ids, inspected):
                        rows.append(dict(identity=identity,
                                         cell=dynamic_name + '_' + search_name, **row))
                    for key in ['score_map', 'size_map', 'offset_map', 'pred_boxes']:
                        maps[key].append(output[key].cpu())
                    maps['boxes'].append(target.cpu())
            print(json.dumps(dict(stage='batch', samples=count, expected=128)), flush=True)
    assert count == 128 and len(rows) == 1152
    cache = {key: torch.cat(values) for key, values in maps.items()}
    cache['window'] = window.cpu()
    cache_path = root / 'maps.pth'
    torch.save(cache, str(cache_path))
    assert before == digest(model.state_dict())
    assert all(parameter.grad is None for parameter in model.parameters())
    effects, advance = statistics(rows)
    sources = [Path(__file__), Path(__file__).with_name('probe_tail_lowrank.py'),
        Path(__file__).with_name('probe_cross_stream_context.py'),
        Path(__file__).with_name('diagnose_absent_position.py'),
        Path(__file__).with_name('diagnose_late_fusion_streams.py'), train_file,
        Path('experiments/cadtrack/cadtrack.yaml'), Path('lib/test/utils/hann.py'),
        Path('lib/utils/box_ops.py')]
    sources += sorted(Path('miss_rgbt').rglob('*.py'))
    sources += sorted(Path('lib/models').rglob('*.py'))
    report = dict(status='complete', pid=os.getpid(), checkpoint=CHECKPOINT,
        checkpoint_sha256=CHECKPOINT_SHA, seed=50101, fitting_names=fitting,
        input_manifest=manifest, rows=rows, effects=effects, advance=advance,
        map_cache=dict(path=str(cache_path), sha256=sha256(cache_path)),
        controls=dict(raw_decode_exact=True, hann_decode_exact=True,
                      native_iou_matches=True, baseline_unchanged=True, gradients_absent=True),
        baseline_state_sha256=before,
        source_sha256={str(path): sha256(path) for path in sources},
        peak_allocated=torch.cuda.max_memory_allocated(),
        peak_reserved=torch.cuda.max_memory_reserved(), seconds=time.time() - started,
        scope='Train882 fixed native geometry and GT ranking oracle; not deployable or official scores.')
    save(root / 'report.json', report)
    save(root / 'completed.json', dict(status='complete', report_sha256=sha256(root / 'report.json'),
        advance=advance, seconds=report['seconds']))
    print(json.dumps(dict(stage='complete', effects=effects, advance=advance,
                          seconds=report['seconds'])), flush=True)


if __name__ == '__main__':
    main()
