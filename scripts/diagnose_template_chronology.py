"""Train882 paired diagnostic of which temporal template remains complete."""
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
    CHECKPOINT, CHECKPOINT_SHA, MAPS, capture_identities, clustered, digest, seed, sha256)
from diagnose_candidate_quality import excluded_tuples

PROJECT = Path('/home/gaob/Miss-RGBT')
STATES = [('11', [1., 1.]), ('01', [0., 1.]), ('10', [1., 0.])]


def save(path, value):
    with Path(path).open('x', newline='\n') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')


def old_ids():
    excluded = excluded_tuples()
    for name in ['server28_candidate_quality_20261102_v2', 'server28_quality_ranker_20261102']:
        record = json.loads((PROJECT / 'reports' / name / 'report.json').read_text())
        manifests = (record['cache_manifests'].values() if 'cache_manifests' in record
                     else [dict(input_manifest=record['input_manifest'])])
        for manifest in manifests:
            for item in manifest['input_manifest']:
                value = item['identity']
                excluded.add((value['sequence'], tuple(value['template_frames']),
                              tuple(value['search_frames'])))
    return excluded


def measurement(model, output, target, window):
    box = model.baseline.box_head.cal_bbox(
        output['score_map'] * window, output['size_map'], output['offset_map'])
    _, iou = generalized_box_iou(box_cxcywh_to_xyxy(box),
                                 box_xywh_to_xyxy(target).clamp(0, 1))
    assert torch.isfinite(box).all() and torch.isfinite(iou).all()
    return box, iou


def summarize(rows):
    effects = {}
    for state in ['11', '01', '10']:
        selected = [dict(identity=row['identity'], gain=row['chronological_iou'] - row['native_iou'])
                    for row in rows if row['dynamic'] != '11' and row['search'] == state]
        effects[state] = clustered(selected, 'gain', 52102)
    return effects


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    root = Path(args.output)
    root.mkdir(exist_ok=False)
    started = time.time()
    seed(52101)
    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    update_config_from_file('experiments/cadtrack/cadtrack.yaml')
    assert sha256(CHECKPOINT) == CHECKPOINT_SHA
    model = load_pretrained(cfg, CHECKPOINT, repair=False).cuda().eval().requires_grad_(False)
    before = digest(model.state_dict())
    names_path = Path('lib/train/data_specs/lasher_train.txt')
    names = names_path.read_text().splitlines()
    assert len(names) == len(set(names)) == 882
    fitting = [name for index, name in enumerate(names) if index % 5]
    loader = make_loader(cfg, '/data/wangwj/dataset/LasHeR/trainingset', 'train', 16, 128, 0)
    loader.dataset.datasets[0].sequence_list = fitting
    identities = capture_identities(loader)
    excluded = old_ids()
    window = hann2d(torch.tensor([16, 16]), centered=True).cuda()
    rows, manifest, count = [], [], 0
    max_sham_box_error, max_sham_map_error = 0., 0.
    for data in loader:
        templates = [value.cuda() for value in data['template_images']]
        search = data['search_images'][0].cuda()
        target = data['search_anno'][0].cuda()
        batch_ids = identities[count:count + len(search)]
        assert len(batch_ids) == len(search) and len(templates) == 2
        for index, identity in enumerate(batch_ids):
            key = (identity['sequence'], tuple(identity['template_frames']), tuple(identity['search_frames']))
            assert key not in excluded and identity['sequence'] in fitting
            newer, older = identity['template_frames']
            assert older < newer < min(identity['search_frames'])
            manifest.append(dict(identity=identity, age_gap=newer - older,
                tensors_sha256=digest(dict(templates=data['template_images'][:, index],
                    search=data['search_images'][0, index], target=target[index]))))
        count += len(search)
        with torch.no_grad():
            for dynamic_name, dynamic in STATES:
                for search_name, present in STATES:
                    full = search.new_ones(len(search), 2)
                    flags = search.new_tensor(dynamic).expand(len(search), -1)
                    observed = search.new_tensor(present).expand(len(search), -1)
                    native = model(templates, [search], [full, flags, observed])[0]
                    # Same images and masks, reversed together: permutation control.
                    sham = model(list(reversed(templates)), [search], [flags, full, observed])[0]
                    # Old initial template complete; recent dynamic template masked.
                    changed = model(list(reversed(templates)), [search], [full, flags, observed])[0]
                    native_box, native_iou = measurement(model, native, target, window)
                    sham_box, sham_iou = measurement(model, sham, target, window)
                    changed_box, changed_iou = measurement(model, changed, target, window)
                    box_error = float((native_box - sham_box).abs().max())
                    map_error = max(float((native[key] - sham[key]).abs().max()) for key in MAPS)
                    max_sham_box_error = max(max_sham_box_error, box_error)
                    max_sham_map_error = max(max_sham_map_error, map_error)
                    for index, identity in enumerate(batch_ids):
                        rows.append(dict(identity=identity, dynamic=dynamic_name, search=search_name,
                            target=target[index].cpu().tolist(),
                            native_box=native_box[index].cpu().tolist(),
                            native_iou=float(native_iou[index]),
                            sham_box=sham_box[index].cpu().tolist(), sham_iou=float(sham_iou[index]),
                            chronological_box=changed_box[index].cpu().tolist(),
                            chronological_iou=float(changed_iou[index])))
        print(json.dumps(dict(stage='batch', samples=count, expected=128)), flush=True)
    assert count == 128 and len(rows) == 1152
    assert before == digest(model.state_dict())
    assert all(parameter.grad is None for parameter in model.parameters())
    effects = summarize(rows)
    shams = [dict(identity=row['identity'], gain=row['sham_iou'] - row['native_iou']) for row in rows]
    sham_effect = clustered(shams, 'gain', 52102)
    # Numerical permutation sensitivity is measured, not assumed exactly zero.
    sham_small = max(abs(row['sham_iou'] - row['native_iou']) for row in rows) <= 1e-4
    material = any(abs(value['mean']) >= .01 and
        (value['ci95_low'] > 0 or value['ci95_high'] < 0) for value in effects.values())
    sources = [Path(__file__), Path(__file__).with_name('diagnose_candidate_quality.py'),
        Path(__file__).with_name('diagnose_absent_position.py'), names_path,
        Path('experiments/cadtrack/cadtrack.yaml'), Path('lib/test/utils/hann.py')]
    sources += sorted(Path('miss_rgbt').rglob('*.py'))
    sources += sorted(Path('lib/models').rglob('*.py'))
    report = dict(status='complete', pid=os.getpid(), seed=52101, checkpoint=CHECKPOINT,
        checkpoint_sha256=CHECKPOINT_SHA, fitting_names=fitting, input_manifest=manifest,
        rows=rows, effects=effects, sham_effect=sham_effect,
        sham_max_box_error=max_sham_box_error, sham_max_map_error=max_sham_map_error,
        sham_small=sham_small, material_temporal_mask_effect=material,
        diagnostic_supported=material and sham_small, baseline_unchanged=True,
        baseline_gradients_absent=True, baseline_state_sha256=before,
        source_sha256={str(path): sha256(path) for path in sources},
        seconds=time.time() - started,
        scope='Train882 template-age/missingness binding diagnostic, not a better policy or official score.')
    save(root / 'report.json', report)
    save(root / 'completed.json', dict(status='complete', report_sha256=sha256(root / 'report.json'),
        diagnostic_supported=report['diagnostic_supported'], seconds=report['seconds']))
    print(json.dumps(dict(stage='complete', effects=effects, sham_effect=sham_effect,
        sham_small=sham_small, diagnostic_supported=report['diagnostic_supported'],
        seconds=report['seconds'])), flush=True)


if __name__ == '__main__':
    main()
