"""Matched R3 C-only/C+M objectives; no campaign or training authorization."""
import torch

from miss_rgbt.objectives import task_loss, training_loss
from miss_rgbt.train import sample_masks

ARMS = ('complete_only', 'complete_missing')


def seeded_call(seed, device, function):
    devices = [] if device.type == 'cpu' else [device.index]
    with torch.random.fork_rng(devices=devices):
        torch.random.default_generator.manual_seed(seed)
        if device.type == 'cuda':
            with torch.cuda.device(device):
                torch.cuda.manual_seed(seed)
        return function()


def forward_objective(model, templates, searches, boxes, cfg, arm, microstep,
                      stochastic_seed=74101):
    """Pair C stochastic draws and candidate masks without an unused M forward."""
    assert arm in ARMS and microstep >= 0
    device = searches[0].device
    seeds = dict(complete=stochastic_seed + 3 * microstep,
                 missing=stochastic_seed + 3 * microstep + 1,
                 masks=stochastic_seed + 3 * microstep + 2)
    mask_config = dict(complete_rgb_missing_tir_missing_probability=[.4, .3, .3],
                       patch_mask_probability=.1)
    availability, patch = seeded_call(seeds['masks'], device,
        lambda: sample_masks(templates, searches, mask_config))
    complete = seeded_call(seeds['complete'], device,
                           lambda: model(templates, searches))
    if arm == 'complete_only':
        loss = task_loss(complete, boxes, cfg).mean()
        status = dict(task_c=float(loss.detach()), task_m=None,
                      kd_forward=None, kd_reverse=None, reverse_fraction=None)
        missing = None
    else:
        missing = seeded_call(seeds['missing'], device,
            lambda: model(templates, searches, availability, patch))
        loss, status = training_loss(complete, missing, boxes, cfg, .1, 0., .1)
    assert torch.isfinite(loss)
    status.update(arm=arm, microstep=microstep, model_forwards=1 if missing is None else 2,
        mask_applied_to_model=missing is not None, seeds=seeds,
        scope='Training objective only; not evaluation or efficacy')
    return loss, status, complete, missing, availability, patch
