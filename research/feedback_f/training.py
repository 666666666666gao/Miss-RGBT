"""Shared F training batch interface; no dataset access or job launch."""
import torch

from miss_rgbt.model import load_pretrained
from miss_rgbt.objectives import training_loss, utility_supervision
from miss_rgbt.train import sample_masks
from model import FeedbackTrackerModel, two_round_training_loss

ARMS = ('single', 'none', 'shallow', 'deep')


def step_masks(templates, searches, config, microstep):
    """Native masks keyed by step, independent of model forward RNG draws."""
    device = searches[0].device
    with torch.random.fork_rng(devices=[device.index]):
        with torch.cuda.device(device):
            torch.cuda.manual_seed(config['mask_seed'] + microstep)
        return sample_masks(templates, searches, config)


def build_model(cfg, checkpoint, task_utility, arm, width=128):
    assert arm in ARMS and task_utility['selection'] == 'utility'
    native = load_pretrained(cfg, checkpoint, repair=False, task_utility=task_utility)
    if arm == 'single':
        return native
    # Extra module initialization must not advance the caller's sampler RNG.
    with torch.random.fork_rng(devices=[]):
        model = FeedbackTrackerModel(native.baseline, task_utility=task_utility,
                                     feedback_mode=arm, feedback_width=width)
    model.utility_repair.load_state_dict(native.utility_repair.state_dict(), strict=True)
    return model


def batch_loss(model, templates, searches, boxes, availability, patch,
               cfg, config, probe=None, probe_view=None):
    assert (probe is None and probe_view is None) or (
        probe is not None and probe_view in ['C', 'M'])
    complete = model(templates, searches, utility_probe=probe if probe_view == 'C' else None)
    missing = model(templates, searches, availability, patch,
                    utility_probe=probe if probe_view == 'M' else None)
    objective = (two_round_training_loss if isinstance(model, FeedbackTrackerModel)
                 else training_loss)
    loss, status = objective(complete, missing, boxes, cfg,
                            forward_weight=config['forward_weight'],
                            reverse_weight=0., margin=config['reverse_margin'])
    if probe is not None:
        auxiliary, utility = utility_supervision(
            complete if probe_view == 'C' else missing, boxes, cfg)
        loss = loss + config['utility_weight'] * auxiliary
        status.update(utility=utility, probe_view=probe_view)
    return loss, status
