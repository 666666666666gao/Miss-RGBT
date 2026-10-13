"""CPU arithmetic checks against the native focal implementation; no benchmark."""
import json
from pathlib import Path

import torch

from lib.config.cadtrack.config import cfg, update_config_from_file
from lib.train.actors.cadtrack import CADTRACKActor
from lib.utils.focal_loss import FocalLoss
from lib.utils.box_ops import giou_loss
from miss_rgbt.objectives import task_loss


torch.manual_seed(2027)
update_config_from_file('experiments/cadtrack/cadtrack.yaml')
# Native actor uses .cuda() internally; use native primitives on CPU instead.
from lib.utils.box_ops import box_cxcywh_to_xyxy, box_xywh_to_xyxy
from lib.utils.heapmap_utils import generate_heatmap

report = dict(scope='CPU synthetic loss arithmetic only; no model, dataset or metric result', cases={})
for name, boxes in [('one_positive_each', [[.3, .3, .2, .2], [.4, .4, .2, .2]]),
                    ('mixed_empty_target', [[.3, .3, .2, .2], [2., 2., .2, .2]]),
                    ('all_empty_targets', [[2., 2., .2, .2], [2., 2., .2, .2]])]:
    target = torch.tensor([boxes])
    predicted = torch.tensor([[[.5, .5, .2, .2]], [[.5, .5, .2, .2]]])
    score = torch.rand(2, 1, 16, 16).clamp(.001, .999)
    pred_xyxy = box_cxcywh_to_xyxy(predicted).flatten(0, 1)
    gt_xyxy = box_xywh_to_xyxy(target[0]).clamp(0, 1)
    giou, _ = giou_loss(pred_xyxy, gt_xyxy)
    gt = generate_heatmap(target, 256, 16)[0].unsqueeze(1)
    native = 2 * giou + 5 * torch.nn.functional.l1_loss(pred_xyxy, gt_xyxy) + FocalLoss()(score, gt)
    actual = task_loss([dict(pred_boxes=predicted, score_map=score)], target, cfg).mean()
    difference = float((native - actual).abs())
    assert torch.allclose(native, actual, atol=2e-5, rtol=1e-6), (name, native, actual)
    report['cases'][name] = dict(native=float(native), per_sample_mean=float(actual),
        absolute_difference=difference, positive_peaks=gt.eq(1).flatten(1).sum(-1).tolist())
Path('reports/focal_normalization_validation.json').write_text(json.dumps(report, indent=2))
print('FOCAL_NATIVE_ALIGNMENT_PASS ' + json.dumps(report), flush=True)
