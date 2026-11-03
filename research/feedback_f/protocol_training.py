"""OSTrack/RPPM batch bridge for F; not a campaign launcher."""
import torch

from miss_rgbt.objectives import response_kd, task_loss, training_loss
from research.curriculum_p.engine import ARMS as CURRICULA
from research.curriculum_p.pacing import sample_masks as paced_masks, schedule
from research.curriculum_p.protocol_train import build as build_native
from research.curriculum_p.schedule import sample_masks as control_masks
from research.feedback_f.model import FeedbackTrackerModel
from research.protocol_r.common import INITIALIZER_SHA, sha

ARMS = ('single', 'none', 'shallow', 'deep')


def build(config):
    assert config['init'] == 'ostrack'
    assert config['initializer_sha256'] == INITIALIZER_SHA
    assert sha(config['checkpoint']) == INITIALIZER_SHA
    arm = config['feedback']['mode']
    assert arm in ARMS and config['task_utility']['selection'] == 'utility'
    native, evidence = build_native(config)
    if arm == 'single':
        return native, evidence
    # Additional modules cannot shift the paired data/initialization RNG.
    with torch.random.fork_rng(devices=[]):
        model = FeedbackTrackerModel(
            native.baseline, task_utility=config['task_utility'],
            feedback_mode=arm, feedback_width=config['feedback']['width'])
    model.utility_repair.load_state_dict(native.utility_repair.state_dict(), strict=True)
    return model, evidence


def step_masks(templates, searches, config, epoch, hard, state):
    """Pair mask draws, while allowing genuine model-dependent pacing decisions."""
    device = searches[0].device
    devices = [] if device.type == 'cpu' else [device.index]
    arm = 'paced' if state.arm == 'paced+norm' else state.arm
    local = dict(config, curriculum_arm=arm)
    with torch.random.fork_rng(devices=devices):
        value = config['mask_seed'] + state.microsteps
        torch.random.default_generator.manual_seed(value)
        if device.type == 'cuda':
            with torch.cuda.device(device):
                torch.cuda.manual_seed(value)
        if arm == 'paced':
            return paced_masks(templates, searches, local, epoch, hard, state.generator)
        available, patch = control_masks(templates, searches, local, epoch)
    probability = schedule(arm)[epoch - 1]
    return available, patch, dict(p_epoch=probability, p_hard=probability,
                                 mean_missing_probability=probability)


def round_loss(rounds, boxes, cfg, config, normalizer, availability):
    """Average round objectives; update missing-state EMA once across all rounds."""
    count = len(rounds)
    assert count in (1, 2)
    if config['curriculum_arm'] != 'paced+norm':
        results = [training_loss(c, m, boxes, cfg, config['forward_weight'],
                                config['reverse_weight'], config['reverse_margin'])
                   for c, m in rounds]
        return sum(result[0] for result in results) / count, dict(
            rounds=[result[1] for result in results], state_norm=False)
    complete = torch.stack([task_loss(c, boxes, cfg) for c, _ in rounds])
    per_search = torch.stack([
        torch.stack([task_loss([out], boxes[i:i + 1], cfg)
                     for i, out in enumerate(m)]) for _, m in rounds])
    raw_missing = per_search.sum(1)
    states = torch.stack(availability).repeat(count, 1, 1)
    normalized, statistics = normalizer.normalize(per_search.flatten(0, 1), states)
    normalized = normalized / count
    forward = torch.stack([response_kd(m, c) for c, m in rounds])
    reverse = torch.stack([response_kd(c, m) for c, m in rounds])
    reliable = (raw_missing.detach() + config['reverse_margin'] <
                complete.detach()).to(complete.dtype)
    loss = complete.mean() + normalized.mean() + config['forward_weight'] * forward.mean()
    loss = loss + config['reverse_weight'] * (reliable * reverse).mean()
    statistics.update(state_norm=True, normalization_observations=per_search.numel(),
                      rounds=[dict(task_c=float(complete[i].mean().detach()),
                                   task_m=float(raw_missing[i].mean().detach()),
                                   kd_forward=float(forward[i].mean().detach()),
                                   kd_reverse=float((reliable[i] * reverse[i]).mean().detach()))
                              for i in range(count)],
                      task_m_normalized=float(normalized.mean().detach()))
    return loss, statistics


def forward_microbatch(model, templates, searches, boxes, cfg, config,
                       epoch, state, utility_probe=None, probe_view=None):
    """Return existing C/M outputs for caller-owned Utility supervision/backprop."""
    assert config['feedback']['mode'] in ARMS
    assert state.arm == config['curriculum_arm'] and state.arm in CURRICULA
    assert 1 <= epoch <= 20 and config['patch_mask_probability'] == .1
    assert probe_view in (None, 'C', 'M')
    assert (utility_probe is None) == (probe_view is None)
    complete = model(templates, searches,
                     utility_probe=utility_probe if probe_view == 'C' else None)
    hard, statistics = state.complete.update(task_loss(complete, boxes, cfg))
    available, patch, pacing = step_masks(
        templates, searches, config, epoch, hard, state)
    missing = model(templates, searches, available, patch,
                    utility_probe=utility_probe if probe_view == 'M' else None)
    rounds = [(complete, missing)]
    if config['feedback']['mode'] != 'single':
        rounds.insert(0, ([out['round1'] for out in complete],
                          [out['round1'] for out in missing]))
    loss, status = round_loss(rounds, boxes, cfg, config, state.missing,
                              available[len(templates):])
    assert torch.isfinite(loss), status
    state.microsteps += 1
    statistics.update(pacing)
    statistics.update(curriculum_arm=state.arm, curriculum_microstep=state.microsteps,
                      feedback_mode=config['feedback']['mode'],
                      actual_search_missing_fraction=float(
                          torch.stack(available[len(templates):]).sum(-1).eq(1)
                          .float().mean()))
    status.update(statistics)
    return loss, status, complete, missing, available, patch
