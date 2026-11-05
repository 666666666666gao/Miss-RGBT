"""RPPM primitives; the old four-arm campaign remains unchanged."""
import torch

from research.curriculum_p.schedule import ARMS as CONTROL_ARMS
from research.curriculum_p.schedule import sample_masks as control_sample_masks
from research.curriculum_p.schedule import schedule as control_schedule

ARMS = CONTROL_ARMS + ('paced',)


def schedule(arm):
    return control_schedule('progressive' if arm == 'paced' else arm)


class CompleteLossEMA:
    def __init__(self, momentum=.99):
        self.momentum = momentum
        self.value = None

    def update(self, losses):
        detached = losses.detach()
        mean = detached.mean()
        if self.value is None:
            self.value = mean.clone()
        else:
            self.value.mul_(self.momentum).add_(mean, alpha=1. - self.momentum)
        hard = detached > self.value
        return hard, dict(ema_c=float(self.value),
                          hard_fraction=float(hard.float().mean()))


def sample_masks(templates, searches, config, epoch, hard, generator):
    """Sample categorical [1-p_i,p_i/2,p_i/2] with paired native RNG.

    Native draws supply the p_epoch states and all patch masks. Independent
    uniforms retain each missing state with probability p_i/p_epoch. Thus
    hard=False is exactly the native progressive path, including RNG state.
    The caller owns and checkpoints the independent generator.
    """
    assert config['curriculum_arm'] == 'paced'
    batch = searches[0].shape[0]
    assert hard.shape == (batch,) and hard.dtype == torch.bool
    assert hard.device == searches[0].device
    base_config = dict(config, curriculum_arm='progressive')
    available, patch = control_sample_masks(templates, searches, base_config, epoch)
    probability = schedule('paced')[epoch - 1]
    low = max(.2, probability - .3)
    per_sample = torch.where(hard, torch.full_like(hard, low, dtype=torch.float),
                             torch.full_like(hard, probability, dtype=torch.float))
    for state in available[1:]:
        uniform = torch.rand(batch, device=hard.device, generator=generator)
        restore = hard & (state.sum(-1) == 1) & (uniform >= per_sample / probability)
        state[restore] = 1.
    return available, patch, dict(
        p_epoch=probability, p_hard=low,
        mean_missing_probability=float(per_sample.mean()),
        hard_fraction=float(hard.float().mean()))
