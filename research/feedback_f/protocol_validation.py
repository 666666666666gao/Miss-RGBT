"""Fixed val97 diagnostics for the OSTrack F protocol, never official scores."""
import torch

from lib.test.utils.hann import hann2d
from lib.utils.box_ops import box_cxcywh_to_xyxy, box_iou, box_xywh_to_xyxy
from miss_rgbt.objectives import task_loss

STATES = [('11', [1., 1.]), ('01', [0., 1.]), ('10', [1., 0.])]
IOU_THRESHOLD = .5


def crop_iou(outputs, boxes, head):
    """Decode with the tracker's Hann window; retain one IoU per search/sample."""
    assert len(outputs) == boxes.shape[0] == 3
    values = []
    for index, out in enumerate(outputs):
        score = out['score_map']
        size = score.shape[-1]
        assert score.shape[-2] == size == head.feat_sz
        window = hann2d(torch.tensor([size, size]), centered=True).to(score)
        predicted = head.cal_bbox(window * score, out['size_map'], out['offset_map'])
        assert predicted.shape == boxes[index].shape
        iou, union = box_iou(box_cxcywh_to_xyxy(predicted),
                             box_xywh_to_xyxy(boxes[index]))
        assert torch.isfinite(iou).all() and (union > 0).all()
        values.append(iou)
    return torch.stack(values)


def compare_rounds(first, final):
    """Success crossings use IoU >= .5; expose both all-frame and eligible rates."""
    first = torch.tensor(first, dtype=torch.float64)
    final = torch.tensor(final, dtype=torch.float64)
    assert first.shape == final.shape and first.ndim == 1 and first.numel() > 0
    assert torch.isfinite(first).all() and torch.isfinite(final).all()
    assert ((first >= 0) & (first <= 1)).all()
    assert ((final >= 0) & (final <= 1)).all()
    success1, success2 = first >= IOU_THRESHOLD, final >= IOU_THRESHOLD
    corrected = int((~success1 & success2).sum())
    destroyed = int((success1 & ~success2).sum())
    failed1, passed1 = int((~success1).sum()), int(success1.sum())
    count = first.numel()
    return dict(frames=count, iou_threshold=IOU_THRESHOLD,
        round1_mean_iou=float(first.mean()), round2_mean_iou=float(final.mean()),
        mean_iou_delta=float((final - first).mean()),
        higher_iou=int((final > first).sum()), lower_iou=int((final < first).sum()),
        equal_iou=int((final == first).sum()),
        round1_failed=failed1, round1_succeeded=passed1,
        corrected=corrected, destroyed=destroyed,
        correction_fraction=corrected / count, destruction_fraction=destroyed / count,
        correction_rate_given_failure=corrected / failed1 if failed1 else None,
        destruction_rate_given_success=destroyed / passed1 if passed1 else None)


def validate(model, batches, cfg, feedback_mode):
    """Use all 128 cached tuples and final-round task loss for model selection."""
    assert feedback_mode in ('single', 'none', 'shallow', 'deep-no-response', 'deep')
    device = next(model.parameters()).device
    devices = [] if device.type == 'cpu' else [device.index]
    selector = model.utility_repair.selection_rng_state.clone()
    training = [(module, module.training) for module in model.modules()]
    model.utility_repair.reset_selection()
    model.eval()
    totals = {label: 0. for label, _ in STATES}
    records = {label: [] for label, _ in STATES}
    paired = feedback_mode != 'single'
    count = 0
    with torch.no_grad(), torch.random.fork_rng(devices=devices):
        for batch_index, data in enumerate(batches):
            templates = [value.to(device) for value in data['template_images']]
            searches = [value.to(device) for value in data['search_images']]
            boxes = data['search_anno'].to(device)
            assert len(templates) == 2 and len(searches) == boxes.shape[0] == 3
            size = searches[0].shape[0]
            for label, flags in STATES:
                available = [searches[0].new_ones(size, 2) for _ in templates]
                available += [searches[0].new_tensor(flags).expand(size, -1)
                              for _ in searches]
                outputs = model(templates, searches, available)
                final_loss = task_loss(outputs, boxes, cfg)
                assert final_loss.shape == (size,) and torch.isfinite(final_loss).all()
                totals[label] += float(final_loss.sum())
                if paired:
                    first = [out['round1'] for out in outputs]
                    first_loss = task_loss(first, boxes, cfg)
                    assert torch.isfinite(first_loss).all()
                    records[label].append(dict(batch_index=batch_index,
                        round1_task_loss=first_loss.cpu().tolist(),
                        round2_task_loss=final_loss.cpu().tolist(),
                        round1_iou=crop_iou(first, boxes, model.baseline.box_head)
                            .cpu().tolist(),
                        round2_iou=crop_iou(outputs, boxes, model.baseline.box_head)
                            .cpu().tolist()))
            count += size
    assert count == 128
    model.utility_repair.selection_rng_state.copy_(selector)
    for module, was_training in training:
        module.training = was_training
    result = dict(samples=count, task_loss={key: value / count for key, value in totals.items()},
        mean_task_loss=sum(totals.values()) / (3 * count),
        scope='Internal fixed val97 crop diagnostics only; not official tracking metrics')
    if paired:
        comparison = {}
        for label, _ in STATES:
            first = [value for row in records[label] for search in row['round1_iou']
                     for value in search]
            final = [value for row in records[label] for search in row['round2_iou']
                     for value in search]
            assert len(first) == len(final) == 3 * count
            comparison[label] = compare_rounds(first, final)
        result.update(round_comparison=comparison, round_samples=records,
            round_comparison_scope='384 independent cached search crops per state; '
                'Hann-windowed native CENTER decode; normalized crop coordinates, '
                'unclipped GT and prediction; success is IoU >= 0.5; '
                'no closed-loop state, template updates or full-image clip_box')
    return result
