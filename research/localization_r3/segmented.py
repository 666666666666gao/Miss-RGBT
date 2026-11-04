"""Isolated reverse-KD-off backward candidate; not wired into frozen training."""
import torch

from miss_rgbt.objectives import response_kd, task_loss
from miss_rgbt.train import sample_masks
from research.localization_r3.objective import seeded_call


def backward_segmented(model, templates, searches, boxes, cfg, microstep,
                       accumulation=1, stochastic_seed=74101):
    """Accumulate C and M gradients; caller owns clipping and optimizer.step."""
    assert microstep >= 0 and accumulation >= 1
    device = searches[0].device
    base = stochastic_seed + 3 * microstep
    mask_config = dict(complete_rgb_missing_tir_missing_probability=[.4, .3, .3],
                       patch_mask_probability=.1)
    available, patch = seeded_call(base + 2, device,
        lambda: sample_masks(templates, searches, mask_config))
    complete = seeded_call(base, device, lambda: model(templates, searches))
    teacher = [dict(score_map=out['score_map'].detach()) for out in complete]
    loss_c = task_loss(complete, boxes, cfg)
    task_c = loss_c.detach()
    (loss_c.mean() / accumulation).backward()
    del complete, loss_c
    missing = seeded_call(base + 1, device,
        lambda: model(templates, searches, available, patch))
    loss_m = task_loss(missing, boxes, cfg)
    kd = response_kd(missing, teacher)
    ((loss_m.mean() + .1 * kd.mean()) / accumulation).backward()
    # Reverse KD is telemetry only, matching the zero-weight native objective.
    with torch.no_grad():
        reliable = (loss_m.detach() + .1 < task_c).to(task_c.dtype)
        reverse = reliable * response_kd(teacher, missing)
        total = task_c.mean() + loss_m.detach().mean() + .1 * kd.detach().mean()
        status = dict(task_c=float(task_c.mean()), task_m=float(loss_m.mean()),
            kd_forward=float(kd.mean()), kd_reverse=float(reverse.mean()),
            reverse_fraction=float(reliable.mean()))
    return float(total), status, available, patch
