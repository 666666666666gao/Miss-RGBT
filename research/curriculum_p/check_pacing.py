"""CPU-only synthetic RPPM evidence, never tracking benchmark results."""
import argparse
import hashlib
import json
from pathlib import Path

import torch

from lib.config.cadtrack.config import cfg, update_config_from_file
from miss_rgbt.objectives import task_loss, training_loss as native_training_loss
from research.curriculum_p.objectives import StateLossEMA, training_loss
from research.curriculum_p.pacing import CompleteLossEMA, sample_masks, schedule
from research.curriculum_p.schedule import sample_masks as control_sample_masks


def check_masks():
    batch = 8192
    templates = [torch.empty(batch, 1) for _ in range(2)]
    searches = [torch.empty(batch, 1) for _ in range(3)]
    observations = []
    assert schedule('paced') == [.2] * 5 + [.5] * 10 + [.8] * 5
    for epoch in (1, 5, 6, 15, 16, 20):
        generator = torch.Generator().manual_seed(651104)
        torch.manual_seed(651103)
        native, native_patch = control_sample_masks(
            templates, searches, dict(curriculum_arm='progressive'), epoch)
        native_rng = torch.get_rng_state().clone()
        torch.manual_seed(651103)
        actual, patch, stats = sample_masks(
            templates, searches, dict(curriculum_arm='paced'), epoch,
            torch.zeros(batch, dtype=torch.bool), generator)
        assert torch.equal(torch.get_rng_state(), native_rng)
        assert all(torch.equal(a, b) for a, b in zip(native, actual))
        assert all(torch.equal(a, b) for a, b in zip(native_patch, patch))
        for mode in ('all', 'mixed'):
            hard = torch.ones(batch, dtype=torch.bool)
            if mode == 'mixed':
                hard[1::2] = False
            generator.manual_seed(651104)
            torch.manual_seed(651103)
            actual, patch, stats = sample_masks(
                templates, searches, dict(curriculum_arm='paced'), epoch,
                hard, generator)
            assert torch.equal(torch.get_rng_state(), native_rng)
            assert all(torch.equal(a, b) for a, b in zip(native_patch, patch))
            assert torch.equal(actual[0], torch.ones(batch, 2))
            assert all(value.sum(-1).ge(1).all() for value in actual)
            probability = max(.2, schedule('paced')[epoch - 1] - .3)
            assert stats['p_hard'] == probability
            if epoch <= 5:
                assert all(torch.equal(a, b) for a, b in zip(native, actual))
            if mode == 'mixed':
                assert all(torch.equal(a[~hard], b[~hard])
                           for a, b in zip(native, actual))
            missing = torch.stack(actual[1:])[:, hard]
            frequencies = [(missing == missing.new_tensor(state)).all(-1)
                           .float().mean().item()
                           for state in ((1, 1), (0, 1), (1, 0))]
            expected = (1 - probability, probability / 2, probability / 2)
            assert all(abs(a - b) < .02 for a, b in zip(frequencies, expected))
            observations.append(dict(epoch=epoch, mode=mode, p_hard=probability,
                                     hard_state_frequencies=frequencies))
    ema = CompleteLossEMA()
    hard, stats = ema.update(torch.tensor([2., 4.], requires_grad=True))
    assert stats['ema_c'] == 3. and hard.tolist() == [False, True]
    hard, stats = ema.update(torch.tensor([10., 12.]))
    assert abs(stats['ema_c'] - 3.08) < 1e-6
    assert hard.all() and not ema.value.requires_grad
    return observations


def make_outputs():
    return [dict(pred_boxes=torch.tensor(
        [[[.4, .4, .2, .2]], [[.6, .6, .3, .3]], [[.5, .5, .2, .2]]],
        requires_grad=True),
        score_map=torch.rand(3, 1, 16, 16).mul(.8).add(.1).requires_grad_())
        for _ in range(3)]


def check_loss():
    torch.manual_seed(651105)
    boxes = torch.tensor([[[.3, .3, .2, .2], [.4, .4, .3, .3], [.4, .4, .2, .2]]]
                         * 3)
    complete, missing = make_outputs(), make_outputs()
    native, native_status = native_training_loss(
        complete, missing, boxes, cfg, .1, .02, .1)
    normalizer = StateLossEMA('cpu')
    actual, status = training_loss(
        complete, missing, boxes, cfg, .1, .02, .1,
        normalizer=normalizer, state_norm=False)
    assert torch.equal(actual, native) and status == native_status
    assert torch.equal(normalizer.values, torch.ones(3))
    leaves = [out[key] for outputs in (complete, missing) for out in outputs
              for key in ('pred_boxes', 'score_map')]
    native_grad = torch.autograd.grad(native, leaves, retain_graph=True)
    actual_grad = torch.autograd.grad(actual, leaves, retain_graph=True)
    assert all(torch.equal(a, b) for a, b in zip(native_grad, actual_grad))
    frames = torch.stack([task_loss([out], boxes[i:i + 1], cfg)
                          for i, out in enumerate(missing)])
    assert torch.equal(frames.sum(0), task_loss(missing, boxes, cfg))
    states = torch.tensor([[1., 1.], [0., 1.], [1., 0.]])
    availability = [states.roll(i, 0) for i in range(3)]
    values, counts = [], []
    for state in states:
        selected = torch.stack(availability).eq(state).all(-1)
        values.append(.99 + .01 * float(frames.detach()[selected].mean()))
        counts.append(int(selected.sum()))
    expected = torch.tensor(values)
    normalized, status = training_loss(
        complete, missing, boxes, cfg, .1, .02, .1, state_norm=True,
        normalizer=normalizer, search_availability=availability)
    assert torch.allclose(normalizer.values, expected, atol=1e-6, rtol=0.)
    indices = torch.stack(availability).unsqueeze(-2).eq(states).all(-1).long().argmax(-1)
    manual = (frames / normalizer.values[indices]).sum(0).mean()
    assert abs(status['task_m_normalized'] - float(manual)) < 1e-6
    assert status['reverse_fraction'] == native_status['reverse_fraction']
    assert torch.allclose(normalized, native - frames.sum(0).mean() + manual)
    gradients = torch.autograd.grad(normalized, leaves)
    assert all(torch.isfinite(grad).all() for grad in gradients)
    sparse = StateLossEMA('cpu')
    sparse.normalize(torch.ones(2, 3) * 4, states[:1].expand(2, 3, 2))
    assert torch.equal(sparse.values[1:], torch.ones(2))
    assert not normalizer.values.requires_grad
    return dict(norm_off_loss_and_gradients_bitwise=True,
                per_search_native_loss_parity=True,
                per_state_observations=counts, ema_m=status,
                normalized_gradients_finite=True,
                unobserved_state_unchanged=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    assert not torch.cuda.is_initialized()
    torch.set_num_threads(1)
    update_config_from_file('experiments/cadtrack/cadtrack.yaml')
    result = dict(status='PASS', scope='CPU synthetic helpers only; not a launch gate',
                  masks=check_masks(), loss=check_loss(),
                  all_false_native_mask_patch_rng_bitwise=True,
                  mixed_easy_samples_native_bitwise=True,
                  device='cpu', cuda_initialized=torch.cuda.is_initialized())
    assert not result['cuda_initialized']
    result['source_sha256'] = {
        str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in [Path(__file__), Path(__file__).with_name('pacing.py'),
                     Path(__file__).with_name('objectives.py'),
                     Path(__file__).with_name('schedule.py')]}
    with Path(args.output).open('x') as stream:
        stream.write(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
