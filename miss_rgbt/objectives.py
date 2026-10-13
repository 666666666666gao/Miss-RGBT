import torch
from torch.nn import functional as F

from lib.utils.box_ops import box_cxcywh_to_xyxy, box_xywh_to_xyxy, generalized_box_iou
from lib.utils.heapmap_utils import generate_heatmap


def task_loss(outputs, boxes, cfg):
    """Native 2*GIoU + 5*L1 + focal, retaining a loss per sample."""
    heatmaps = generate_heatmap(boxes, cfg.DATA.SEARCH.SIZE, cfg.MODEL.BACKBONE.STRIDE)
    losses = []
    for index, out in enumerate(outputs):
        predicted = box_cxcywh_to_xyxy(out['pred_boxes']).flatten(0, 1)
        target = box_xywh_to_xyxy(boxes[index]).clamp(0, 1)
        giou, _ = generalized_box_iou(predicted, target)
        l1 = (predicted - target).abs().mean(-1)
        score = out['score_map'].flatten(1)
        gt = heatmaps[index].flatten(1)
        positive = gt.eq(1).to(score.dtype)
        negative = gt.lt(1).to(score.dtype)
        positive_term = score.log() * (1 - score).pow(2) * positive
        negative_term = (1 - score).log() * score.pow(2) * (1 - gt).pow(4) * negative
        positives = positive.sum(-1)
        focal = -(positive_term.sum(-1) + negative_term.sum(-1)) / positives.clamp_min(1)
        losses.append(2 * (1 - giou) + 5 * l1 + focal)
    return torch.stack(losses).sum(0)


def response_kd(student, teacher):
    losses = []
    for source, target in zip(student, teacher):
        # Native CENTER head supplies clipped sigmoid probabilities.
        s = source['score_map'].flatten(1)
        t = target['score_map'].detach().flatten(1)
        s = torch.log(s) - torch.log1p(-s)
        t = torch.log(t) - torch.log1p(-t)
        losses.append(F.kl_div(F.log_softmax(s, -1), F.softmax(t, -1), reduction='none').sum(-1))
    return torch.stack(losses).sum(0)


def training_loss(complete, missing, boxes, cfg, forward_weight=0., reverse_weight=0., margin=0.):
    loss_c = task_loss(complete, boxes, cfg)
    loss_m = task_loss(missing, boxes, cfg)
    reliable = (loss_m.detach() + margin < loss_c.detach()).to(loss_c.dtype)
    kd_forward = response_kd(missing, complete)
    kd_reverse = response_kd(complete, missing)
    total = loss_c.mean() + loss_m.mean() + forward_weight * kd_forward.mean()
    # Mean over all samples keeps feedback weak when reliable samples are rare.
    total = total + reverse_weight * (reliable * kd_reverse).mean()
    return total, dict(task_c=float(loss_c.mean().detach()), task_m=float(loss_m.mean().detach()),
                      kd_forward=float(kd_forward.mean().detach()),
                      kd_reverse=float((reliable * kd_reverse).mean().detach()),
                      reverse_fraction=float(reliable.mean()))
