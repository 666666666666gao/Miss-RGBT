"""CPU batch/gradient and real OSTrack construction checks; no image evidence."""
import argparse
import gc
import io
from pathlib import Path

import torch

from lib.config.cadtrack.config import cfg, update_config_from_file
from miss_rgbt.objectives import response_kd, task_loss, training_loss
from research.curriculum_p.check_engine import compare_trees, inputs
from research.curriculum_p.engine import ARMS as CURRICULA, CurriculumState
from research.curriculum_p.objectives import STATES
from research.feedback_f.model import SearchFeedback
from research.feedback_f.protocol_training import ARMS, build, forward_microbatch, step_masks
from research.protocol_r.common import INITIALIZER_SHA, save, sha, tensor_sha


class RoundModel(torch.nn.Module):
    def __init__(self, arm):
        super().__init__()
        self.arm = arm
        self.weight = torch.nn.Parameter(torch.tensor(.25))
        self.calls = []

    def prediction(self, image, mask, bias):
        batch = image.shape[0]
        score = torch.sigmoid(self.weight * image[:, :1, None, None] * mask +
                              torch.rand(batch, 1, 16, 16) * .1 + bias)
        box = torch.sigmoid(self.weight + bias).expand(batch, 1, 4) * .5
        return dict(score_map=score, pred_boxes=box)

    def forward(self, templates, searches, availability=None, patch=None,
                utility_probe=None):
        self.calls.append((availability is None, utility_probe))
        outputs = []
        for index, image in enumerate(searches):
            mask = 1. if availability is None else (
                availability[len(templates) + index][:, :1, None, None])
            first = self.prediction(image, mask, 0.)
            if self.arm == 'single':
                outputs.append(first)
            else:
                second = self.prediction(image, mask, .15)
                second['round1'] = first
                outputs.append(second)
        return outputs


def config(curriculum, arm):
    return dict(curriculum_arm=curriculum, feedback=dict(mode=arm, width=128),
                patch_mask_probability=.1, mask_seed=65503,
                forward_weight=.1, reverse_weight=0., reverse_margin=.1)


def independent_objective(rounds, boxes, config, availability, previous):
    if config['curriculum_arm'] != 'paced+norm':
        losses = [training_loss(c, m, boxes, cfg, .1, 0., .1)[0] for c, m in rounds]
        return sum(losses) / len(losses), previous
    losses = [torch.stack([task_loss([out], boxes[i:i + 1], cfg)
                           for i, out in enumerate(m)]) for _, m in rounds]
    expected = previous.clone()
    states = torch.stack(availability)
    # Independent per-state update from the mean of the round observations.
    averaged = torch.stack(losses).mean(0)
    for index, flags in enumerate(STATES):
        selected = states.eq(states.new_tensor(flags)).all(-1)
        if selected.any():
            expected[index] = .99 * previous[index] + .01 * averaged.detach()[selected].mean()
    denominators = torch.zeros_like(averaged)
    for index, flags in enumerate(STATES):
        denominators[states.eq(states.new_tensor(flags)).all(-1)] = expected[index]
    values = [task_loss(c, boxes, cfg).mean() + (m / denominators).sum(0).mean() +
              .1 * response_kd(missing, c).mean()
              for (c, missing), m in zip(rounds, losses)]
    return sum(values) / len(values), expected


def check_batches():
    templates, searches, boxes = inputs()
    rows = []
    for curriculum in CURRICULA:
        for arm in ARMS:
            for epoch in (1, 6, 16):
                torch.manual_seed(65501)
                model = RoundModel(arm)
                state = CurriculumState(curriculum, 'cpu', 65502)
                # Nontrivial old EMA makes repeated updates detectably different.
                state.missing.values.copy_(torch.tensor([2., 3., 4.]))
                previous = state.missing.values.clone()
                local = config(curriculum, arm)
                result = forward_microbatch(
                    model, templates, searches, boxes, cfg, local, epoch, state)
                loss, status, complete, missing, available, patch = result
                rounds = [(complete, missing)]
                if arm != 'single':
                    rounds.insert(0, ([out['round1'] for out in complete],
                                      [out['round1'] for out in missing]))
                expected, ema = independent_objective(
                    rounds, boxes, local, available[len(templates):], previous)
                assert torch.allclose(loss, expected, atol=1e-5, rtol=1e-6)
                gradient = torch.autograd.grad(loss, model.weight, retain_graph=True)[0]
                reference = torch.autograd.grad(expected, model.weight)[0]
                assert torch.isfinite(gradient) and gradient.abs() > 0
                assert torch.allclose(gradient, reference, atol=1e-5, rtol=1e-6)
                assert torch.allclose(state.missing.values, ema, atol=1e-6, rtol=1e-6)
                assert model.calls == [(True, None), (False, None)]
                assert state.microsteps == 1
                assert status['state_norm'] == (curriculum == 'paced+norm')
                rows.append(dict(curriculum=curriculum, arm=arm, epoch=epoch,
                                 objective_gradient_ema_match=True))
    return rows


def check_masks_and_reload():
    templates, searches, boxes = inputs()
    rows = []
    for curriculum in CURRICULA:
        local = config(curriculum, 'deep')
        state = CurriculumState(curriculum, 'cpu', 65502)
        hard = torch.tensor([True, False, True, False])
        rng = torch.get_rng_state().clone()
        expected = step_masks(templates, searches, local, 16, hard, state)
        assert torch.equal(rng, torch.get_rng_state())
        # The same hard flags + keyed random draws must survive extra forwards.
        torch.rand(137)
        other = CurriculumState(curriculum, 'cpu', 65502)
        compare_trees(expected, step_masks(templates, searches, local, 16, hard, other))
        model = RoundModel('deep')
        forward_microbatch(model, templates, searches, boxes, cfg, local, 6, state)
        buffer = io.BytesIO()
        torch.save(state.state_dict(), buffer)
        restored = CurriculumState(curriculum, 'cpu', 1)
        restored.load_state_dict(torch.load(io.BytesIO(buffer.getvalue()), map_location='cpu'))
        compare_trees(state.state_dict(), restored.state_dict())
        rng = torch.get_rng_state().clone()
        first = forward_microbatch(model, templates, searches, boxes, cfg, local, 16, state)
        torch.set_rng_state(rng)
        second = forward_microbatch(model, templates, searches, boxes, cfg, local, 16, restored)
        compare_trees(first, second)
        compare_trees(state.state_dict(), restored.state_dict())
        gradients = [torch.autograd.grad(result[0], model.weight)[0]
                     for result in (first, second)]
        assert torch.equal(*gradients)
        rows.append(dict(curriculum=curriculum, mask_rng_isolated=True,
                         state_and_next_gradient_exact=True))
    for view in ('C', 'M'):
        model = RoundModel('deep')
        probe = dict(search_index=1, regions=torch.tensor([[0]]))
        forward_microbatch(model, templates, searches, boxes, cfg, config('paced', 'deep'),
                          6, CurriculumState('paced', 'cpu', 1), probe, view)
        assert [call[1] for call in model.calls] == (
            [probe, None] if view == 'C' else [None, probe])
    return rows


def check_initialization(initializer):
    common = dict(init='ostrack', seed=65501, checkpoint=initializer,
                  initializer_sha256=INITIALIZER_SHA,
                  task_utility=dict(width=128, depth=1, region_grid=4,
                                    selected_regions=4, refinement_limit=.1,
                                    selection='utility', selection_seed=65501))
    reference, reference_rng, rows = None, None, []
    for arm in ARMS:
        model, evidence = build(dict(common, feedback=dict(mode=arm, width=128)))
        shared = {name: tensor_sha(value) for name, value in model.state_dict().items()
                  if not name.startswith('feedback.')}
        rng = torch.get_rng_state().clone()
        if arm == 'single':
            reference, reference_rng = shared, rng
        assert shared == reference and torch.equal(rng, reference_rng)
        actual_parameters = dict(model.named_parameters())
        assert all(tensor_sha(actual_parameters[name]) == digest
                   for name, digest in evidence['parameters_sha256'].items())
        if arm in ('shallow', 'deep-no-response', 'deep'):
            assert model.feedback.delta.weight.eq(0).all()
            assert model.feedback.delta.bias.eq(0).all()
        rows.append(dict(arm=arm, shared_parameters_buffers_exact=True,
                         native_initialization_evidence_exact=True,
                         caller_cpu_rng_exact=True))
        del model, actual_parameters
        gc.collect()
    return rows


def check_response_control():
    torch.manual_seed(65503)
    shallow = (torch.randn(2, 4, 4), torch.randn(2, 4, 4))
    deep = (torch.randn(2, 4, 4), torch.randn(2, 4, 4))
    prediction = dict(score_map=torch.randn(2, 1, 2, 2),
                      size_map=torch.randn(2, 2, 2, 2),
                      offset_map=torch.randn(2, 2, 2, 2))
    available = torch.ones(2, 4, 2)
    response_changed = dict((key, value + .25) for key, value in prediction.items())
    deep_changed = tuple(value + .25 for value in deep)
    rows = {}
    for mode in ('shallow', 'deep-no-response', 'deep'):
        module = SearchFeedback(4, 8, mode)
        with torch.no_grad():
            module.delta.weight.normal_(std=.1)
            module.delta.bias.normal_(std=.1)
        output = module(shallow, deep, prediction, available)
        changed_response = module(shallow, deep, response_changed, available)
        changed_deep = module(shallow, deep_changed, prediction, available)
        response_sensitive = any(not torch.equal(a, b)
                                 for a, b in zip(output, changed_response))
        deep_sensitive = any(not torch.equal(a, b)
                             for a, b in zip(output, changed_deep))
        assert response_sensitive == (mode == 'deep')
        assert deep_sensitive == (mode != 'shallow')
        rows[mode] = dict(response_sensitive=response_sensitive,
                          deep_sensitive=deep_sensitive)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--initializer', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    assert not Path(args.output).exists() and not torch.cuda.is_initialized()
    torch.set_num_threads(1)
    update_config_from_file('experiments/cadtrack/cadtrack.yaml')
    batches, masks = check_batches(), check_masks_and_reload()
    response_control = check_response_control()
    initialization = check_initialization(args.initializer)
    assert not torch.cuda.is_initialized()
    sources = [Path(__file__), Path(__file__).with_name('protocol_training.py'),
               Path(__file__).with_name('model.py')]
    save(args.output, dict(
        status='PASS', scope='CPU synthetic batch gradients and actual OSTrack construction; '
            'no actual-network forward, images, GPU smoke, training or official metrics',
        batches=batches, mask_state=masks, initialization=initialization,
        response_control=response_control,
        utility_probe_routing=True, missing_ema_updates_per_microbatch=1,
        cuda_initialized=False, initializer_sha256=sha(args.initializer),
        source_sha256={str(path.resolve()): sha(path) for path in sources}))
    print('PASS: 90 batch cases, 6 mask/state roundtrips, 5 real CPU initializations')


if __name__ == '__main__':
    main()
