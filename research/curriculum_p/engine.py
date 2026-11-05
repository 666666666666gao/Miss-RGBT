"""Six-arm microbatch execution; not a campaign launcher."""
import torch

from miss_rgbt.objectives import task_loss
from research.curriculum_p.objectives import StateLossEMA, training_loss
from research.curriculum_p.pacing import CompleteLossEMA
from research.curriculum_p.pacing import sample_masks as paced_masks
from research.curriculum_p.pacing import schedule
from research.curriculum_p.schedule import sample_masks as control_masks

ARMS = ('fixed60', 'fixed50', 'progressive', 'shuffled', 'paced', 'paced+norm')


class CurriculumState:
    def __init__(self, arm, device, seed):
        assert arm in ARMS
        self.arm = arm
        self.complete = CompleteLossEMA()
        self.missing = StateLossEMA(device)
        self.generator = torch.Generator(device=device).manual_seed(seed)
        self.microsteps = 0

    def state_dict(self):
        return dict(
            arm=self.arm, microsteps=self.microsteps,
            complete_momentum=self.complete.momentum,
            missing_momentum=self.missing.momentum,
            ema_c=None if self.complete.value is None else
            self.complete.value.detach().cpu().clone(),
            ema_m=self.missing.values.detach().cpu().clone(),
            generator=self.generator.get_state().clone())

    def load_state_dict(self, value):
        assert value['arm'] == self.arm
        assert value['complete_momentum'] == self.complete.momentum
        assert value['missing_momentum'] == self.missing.momentum
        device = self.missing.values.device
        self.complete.value = (None if value['ema_c'] is None else
                               value['ema_c'].to(device).clone())
        self.missing.values.copy_(value['ema_m'].to(device))
        self.generator.set_state(value['generator'])
        self.microsteps = value['microsteps']


def forward_microbatch(model, templates, searches, boxes, cfg, config,
                       epoch, state, utility_probe=None, probe_view=None):
    """Run C before sampling M in every arm; the caller owns backpropagation.

    Utility probes enter the same forward as in the original A trainer.
    Utility supervision, gradient accumulation and optimization stay with
    the caller. Return both branches so that supervision needs no new forward.
    """
    assert state.arm == config['curriculum_arm'] and 1 <= epoch <= 20
    assert config['patch_mask_probability'] == .1
    assert probe_view in (None, 'C', 'M')
    assert (utility_probe is None) == (probe_view is None)
    complete = model(templates, searches,
                     utility_probe=utility_probe if probe_view == 'C' else None)
    hard, statistics = state.complete.update(task_loss(complete, boxes, cfg))
    arm = 'paced' if state.arm == 'paced+norm' else state.arm
    local = dict(config, curriculum_arm=arm)
    if arm == 'paced':
        available, patch, pacing = paced_masks(
            templates, searches, local, epoch, hard, state.generator)
        statistics.update(pacing)
    else:
        available, patch = control_masks(templates, searches, local, epoch)
        probability = schedule(arm)[epoch - 1]
        statistics.update(p_epoch=probability, p_hard=probability,
                          mean_missing_probability=probability)
    missing = model(templates, searches, available, patch,
                    utility_probe=utility_probe if probe_view == 'M' else None)
    loss, status = training_loss(
        complete, missing, boxes, cfg,
        config['forward_weight'], config['reverse_weight'],
        config['reverse_margin'], state_norm=state.arm == 'paced+norm',
        normalizer=state.missing, search_availability=available[len(templates):])
    assert torch.isfinite(loss), status
    state.microsteps += 1
    statistics.update(curriculum_arm=state.arm, curriculum_microstep=state.microsteps,
                      state_norm=state.arm == 'paced+norm',
                      actual_search_missing_fraction=float(
                          torch.stack(available[len(templates):]).sum(-1).eq(1)
                          .float().mean()))
    status.update(statistics)
    return loss, status, complete, missing, available, patch
