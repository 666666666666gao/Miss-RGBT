"""Fresh train-only response/geometry diagnostic; no optimizer or adapter."""
import argparse
import json
import os
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from lib.config.cadtrack.config import cfg, update_config_from_file
from lib.utils.box_ops import box_cxcywh_to_xyxy, box_xywh_to_xyxy, generalized_box_iou
from lib.utils.heapmap_utils import generate_heatmap
from miss_rgbt.model import load_pretrained
from miss_rgbt.objectives import task_loss
from scripts.probe_fusion_adapter import (
    CHECKPOINT, CHECKPOINT_SHA, build_cache, digest, head_output, seed)
from scripts.diagnose_joint_utility_sets import sha256
from scripts.analyze_fusion_adapter_pilot import effect


def save(path, value):
    with path.open('x', newline='\n') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')


def components(out, target):
    predicted = box_cxcywh_to_xyxy(out['pred_boxes']).flatten(0, 1)
    gt_box = box_xywh_to_xyxy(target).clamp(0, 1)
    giou, iou = generalized_box_iou(predicted, gt_box)
    regression = 2 * (1 - giou) + 5 * (predicted - gt_box).abs().mean(-1)
    heatmap = generate_heatmap(target.unsqueeze(0), cfg.DATA.SEARCH.SIZE,
                               cfg.MODEL.BACKBONE.STRIDE)[0]
    score = out['score_map'].flatten(1)
    gt = heatmap.flatten(1)
    positive = gt.eq(1).to(score.dtype)
    negative = gt.lt(1).to(score.dtype)
    focal = -((score.log() * (1 - score).pow(2) * positive).sum(-1) +
              ((1 - score).log() * score.pow(2) * (1 - gt).pow(4) * negative).sum(-1))
    focal = focal * score.shape[0] / positive.sum().clamp_min(1)
    return regression, focal, iou, heatmap


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=False)
    start = time.time()
    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    seed(39101)
    assert sha256(CHECKPOINT) == CHECKPOINT_SHA
    update_config_from_file('experiments/cadtrack/cadtrack.yaml')
    model = load_pretrained(cfg, CHECKPOINT, repair=False).cuda().eval()
    model.requires_grad_(False)
    before = digest(model.state_dict())
    train_list = Path('lib/train/data_specs/lasher_train.txt')
    names = train_list.read_text().splitlines()
    assert len(names) == len(set(names)) == 882
    fitting = [name for index, name in enumerate(names) if index % 5]
    prior_root = Path('reports/server28_fusion_adapter_pilot_20261102')
    prior_report = json.loads((prior_root / 'report.json').read_text())
    assert fitting == prior_report['launch']['split_names']['train']
    cache, manifest = build_cache(model, fitting, 128, 39101, False, root / 'fresh')
    # New tuple RNG; verify exact frame tuples do not repeat any previous pilot split.
    def identity_key(identity):
        return (identity['sequence'], tuple(identity['template_frames']),
                tuple(identity['search_frames']))
    old_ids = {identity_key(item['identity'])
               for split in prior_report['cache_manifests'].values()
               for item in split['input_manifest']}
    assert not old_ids.intersection(identity_key(item['identity'])
                                   for item in manifest['input_manifest'])
    rows = []
    for start_index in range(0, len(cache['labels']), 16):
        features = cache['features'][start_index:start_index + 16].cuda().requires_grad_(True)
        target = cache['boxes'][start_index:start_index + 16].cuda()
        out = head_output(model.baseline.box_head, features)
        regression, focal, iou, heatmap = components(out, target)
        native_loss = task_loss([out], target.unsqueeze(0), cfg)
        assert torch.allclose(regression + focal, native_loss, atol=1e-6, rtol=1e-6)
        g_reg = torch.autograd.grad(regression.sum(), features, retain_graph=True)[0].flatten(1)
        g_focal = torch.autograd.grad(focal.sum(), features)[0].flatten(1)
        reg_norm = g_reg.norm(dim=1)
        focal_norm = g_focal.norm(dim=1)
        assert (reg_norm > 0).all() and (focal_norm > 0).all()
        cosine = (g_reg * g_focal).sum(1) / (reg_norm * focal_norm)
        with torch.no_grad():
            oracle_box = model.baseline.box_head.cal_bbox(
                heatmap.unsqueeze(1), out['size_map'], out['offset_map'])
            _, oracle_iou = generalized_box_iou(
                box_cxcywh_to_xyxy(oracle_box), box_xywh_to_xyxy(target).clamp(0, 1))
            predicted_index = out['score_map'].flatten(1).argmax(1)
            target_index = heatmap.flatten(1).argmax(1)
            side = heatmap.shape[-1]
            dx = (predicted_index % side - target_index % side).float()
            dy = (predicted_index // side - target_index // side).float()
            peak_error = (dx.square() + dy.square()).sqrt()
            positives = heatmap.eq(1).flatten(1).sum(1)
            for index in range(features.shape[0]):
                position = start_index + index
                values = dict(regression=float(regression[index]),
                    focal=float(focal[index]), total=float(native_loss[index]),
                    iou=float(iou[index]), oracle_iou=float(oracle_iou[index]),
                    oracle_gain=float(oracle_iou[index] - iou[index]),
                    peak_error_cells=float(peak_error[index]),
                    peak_matches=float(predicted_index[index] == target_index[index]),
                    gradient_cosine=float(cosine[index]),
                    gradient_conflict=float(cosine[index] < 0),
                    regression_gradient_norm=float(reg_norm[index]),
                    focal_gradient_norm=float(focal_norm[index]),
                    target_peak_count=int(positives[index]))
                assert all(np.isfinite(value) for value in values.values())
                rows.append(dict(identity=cache['identities'][position],
                                 cell=cache['labels'][position], **values))
        if len(rows) % 144 == 0:
            print(json.dumps(dict(stage='head_diagnostic', rows=len(rows))), flush=True)
    assert len(rows) == 1152
    cells = {}
    for cell in sorted(set(row['cell'] for row in rows)):
        selected = [dict(sequence=row['identity']['sequence'], **row)
                    for row in rows if row['cell'] == cell]
        assert len(selected) == 128
        cells[cell] = {key: effect(selected, key, 39102) for key in
            ['regression', 'focal', 'total', 'iou', 'oracle_iou', 'oracle_gain',
             'peak_error_cells', 'peak_matches', 'gradient_cosine', 'gradient_conflict']}
    assert before == digest(model.state_dict())
    assert all(parameter.grad is None for parameter in model.parameters())
    sources = [Path(__file__), Path('scripts/probe_fusion_adapter.py'),
        Path('scripts/diagnose_joint_utility_sets.py'),
        Path('scripts/analyze_fusion_adapter_pilot.py'), Path('miss_rgbt/objectives.py'),
        Path('miss_rgbt/model.py'), Path('miss_rgbt/data.py'),
        Path('lib/models/layers/head.py'), Path('lib/utils/box_ops.py'),
        Path('lib/utils/heapmap_utils.py'), Path('experiments/cadtrack/cadtrack.yaml')]
    report = dict(status='complete', pid=os.getpid(), seed=39101,
        checkpoint=CHECKPOINT, checkpoint_sha256=CHECKPOINT_SHA,
        baseline_state_sha256=before, baseline_state_unchanged=True,
        baseline_gradients_absent=True, source_sha256={str(p): sha256(p) for p in sources},
        train_list_sha256=sha256(train_list), manifest=manifest,
        no_exact_prior_pilot_tuple_overlap=True, rows=rows, cells=cells,
        seconds=time.time() - start,
        scope='Exploratory fresh train882 crops; GT peak oracle is diagnosis only. '
              'Not a deployable policy, tracking benchmark, module or training promotion.')
    save(root / 'report.json', report)
    save(root / 'completed.json', dict(status='complete',
        report_sha256=sha256(root / 'report.json'), seconds=report['seconds']))
    print(json.dumps(dict(stage='complete', seconds=report['seconds'],
                          cells=cells)), flush=True)


if __name__ == '__main__':
    main()
