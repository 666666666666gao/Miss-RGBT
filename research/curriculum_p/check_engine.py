"""CPU stochastic microbatch and checkpoint-state checks; no image evidence."""
import argparse
import hashlib
import io
import json
from pathlib import Path

import torch

from lib.config.cadtrack.config import cfg, update_config_from_file
from miss_rgbt.objectives import task_loss, training_loss
from research.curriculum_p.engine import ARMS, CurriculumState, forward_microbatch
from research.curriculum_p.pacing import CompleteLossEMA, sample_masks as paced_masks
from research.curriculum_p.schedule import sample_masks as control_masks


class StochasticModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(.25))
        self.calls = []

    def forward(self, templates, searches, availability=None, patch=None,
                utility_probe=None):
        self.calls.append(dict(complete=availability is None, probe=utility_probe))
        outputs = []
        for index, image in enumerate(searches):
            batch = image.shape[0]
            content = image[:, 0].reshape(batch, 1, 1, 1)
            mask = 1. if availability is None else \
                availability[len(templates) + index][:, 0].reshape(batch, 1, 1, 1)
            logits = self.weight * content * mask
            score = torch.sigmoid(logits + torch.rand(batch, 1, 16, 16) * .1)
            box = torch.sigmoid(self.weight).expand(batch, 1, 4) * .5
            outputs.append(dict(score_map=score, pred_boxes=box))
        return outputs


def inputs():
    templates = [torch.ones(4, 2) for _ in range(2)]
    searches = [torch.tensor([[1., 2.], [3., 4.], [5., 6.], [7., 8.]])
                for _ in range(3)]
    boxes = torch.tensor([[[.3, .3, .2, .2]] * 4] * 3)
    return templates, searches, boxes


def config(arm):
    return dict(curriculum_arm=arm, patch_mask_probability=.1,
                forward_weight=.1, reverse_weight=0., reverse_margin=.1)


def compare_trees(left, right):
    assert type(left) is type(right)
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            compare_trees(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            compare_trees(a, b)
    else:
        assert left == right


def check_order_and_native_parity():
    templates, searches, boxes = inputs()
    results = []
    for arm in ARMS:
        for epoch in (1, 6, 16):
            torch.manual_seed(651201)
            model = StochasticModel()
            state = CurriculumState(arm, 'cpu', 651202)
            result = forward_microbatch(model, templates, searches, boxes, cfg,
                                        config(arm), epoch, state)
            loss, status, complete, missing, available, patch = result
            final_rng = torch.get_rng_state().clone()
            gradient = torch.autograd.grad(loss, model.weight)[0]
            assert [x['complete'] for x in model.calls] == [True, False]
            assert state.microsteps == 1
            assert status['state_norm'] == (arm == 'paced+norm')
            assert torch.isfinite(gradient)

            # An independent explicit C-first execution is the reference.
            torch.manual_seed(651201)
            reference = StochasticModel()
            reference_c = reference(templates, searches)
            hard, _ = CompleteLossEMA().update(task_loss(reference_c, boxes, cfg))
            local = config('paced' if arm == 'paced+norm' else arm)
            if arm in ('paced', 'paced+norm'):
                generator = torch.Generator().manual_seed(651202)
                expected, expected_patch, _ = paced_masks(
                    templates, searches, local, epoch, hard, generator)
                assert torch.equal(generator.get_state(), state.generator.get_state())
            else:
                expected, expected_patch = control_masks(
                    templates, searches, local, epoch)
            reference_m = reference(templates, searches, expected, expected_patch)
            compare_trees(complete, reference_c)
            compare_trees(missing, reference_m)
            compare_trees(available, expected)
            compare_trees(patch, expected_patch)
            assert torch.equal(final_rng, torch.get_rng_state())
            if arm != 'paced+norm':
                native, native_status = training_loss(
                    reference_c, reference_m, boxes, cfg, .1, 0., .1)
                assert torch.equal(loss, native)
                assert all(status[key] == value for key, value in native_status.items())
                assert torch.equal(gradient, torch.autograd.grad(native, reference.weight)[0])
                assert torch.equal(state.missing.values, torch.ones(3))
            results.append(dict(arm=arm, epoch=epoch,
                                native_mask_patch_rng_bitwise=True,
                                norm_off_native_loss_gradient_bitwise=arm != 'paced+norm',
                                finite_gradient=True))
    return results


def check_state_reload():
    templates, searches, boxes = inputs()
    checks = []
    for arm in ARMS:
        torch.manual_seed(651203)
        model = StochasticModel()
        state = CurriculumState(arm, 'cpu', 651204)
        forward_microbatch(model, templates, searches, boxes, cfg, config(arm), 6, state)
        saved = state.state_dict()
        buffer = io.BytesIO()
        torch.save(saved, buffer)
        buffer.seek(0)
        restored = CurriculumState(arm, 'cpu', 1)
        restored.load_state_dict(torch.load(buffer, map_location='cpu'))
        compare_trees(saved, restored.state_dict())
        rng = torch.get_rng_state().clone()
        first = forward_microbatch(model, templates, searches, boxes, cfg,
                                   config(arm), 16, state)
        first_gradient = torch.autograd.grad(first[0], model.weight)[0]
        torch.set_rng_state(rng)
        second = forward_microbatch(model, templates, searches, boxes, cfg,
                                    config(arm), 16, restored)
        compare_trees(first, second)
        assert torch.equal(first_gradient,
                           torch.autograd.grad(second[0], model.weight)[0])
        compare_trees(state.state_dict(), restored.state_dict())
        assert saved['microsteps'] == 1 and restored.microsteps == 2
        assert torch.equal(saved['ema_m'], torch.load(io.BytesIO(buffer.getvalue()))['ema_m'])
        checks.append(dict(arm=arm, state_roundtrip=True,
                           next_microbatch_and_gradient_bitwise=True))
    # Probe routing must reuse the two existing forwards.
    for view in ('C', 'M'):
        model = StochasticModel()
        probe = dict(search_index=1, regions=torch.tensor([[0]]))
        forward_microbatch(model, templates, searches, boxes, cfg, config('paced'),
                          6, CurriculumState('paced', 'cpu', 1), probe, view)
        expected = [probe, None] if view == 'C' else [None, probe]
        assert [x['probe'] for x in model.calls] == expected
    return checks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    assert not torch.cuda.is_initialized()
    torch.set_num_threads(1)
    update_config_from_file('experiments/cadtrack/cadtrack.yaml')
    result = dict(
        status='PASS', scope='CPU synthetic microbatch execution only; not a launch gate',
        execution=check_order_and_native_parity(), checkpoint_state=check_state_reload(),
        c_before_masks_all_six_arms=True, utility_probe_routing=True,
        device='cpu', cuda_initialized=torch.cuda.is_initialized(),
        source_sha256={str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest()
                       for path in [Path(__file__), Path(__file__).with_name('engine.py'),
                                    Path(__file__).with_name('pacing.py'),
                                    Path(__file__).with_name('objectives.py'),
                                    Path(__file__).with_name('schedule.py')]})
    assert not result['cuda_initialized']
    with Path(args.output).open('x') as stream:
        stream.write(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
