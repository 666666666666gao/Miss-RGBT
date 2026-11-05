"""Native network CPU check on synthetic images; no training deployment."""
import argparse
import copy
import json
from pathlib import Path
import time

import torch

from lib.config.cadtrack.config import cfg, update_config_from_file
from miss_rgbt.train import fix_bn
from research.localization_r3.objective import forward_objective
from research.localization_r3.segmented import backward_segmented
from research.localization_r3.train import build
from research.protocol_r.common import read


def compare_named(left, right, rtol, atol):
    assert left.keys() == right.keys()
    maximum, count = 0., 0
    for name, first in left.items():
        second = right[name]
        assert (first is None) == (second is None), name
        if first is None:
            continue
        assert torch.isfinite(first).all() and torch.isfinite(second).all(), name
        error = float((first - second).abs().max())
        maximum = max(maximum, error)
        assert torch.allclose(first, second, rtol=rtol, atol=atol), (name, error)
        count += 1
    return dict(tensors=count, maximum_absolute_error=maximum, rtol=rtol, atol=atol)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    started = time.time()
    torch.set_num_threads(1)
    config = read(args.config)
    update_config_from_file(str(Path(config['snapshot']) /
                                'experiments/cadtrack/cadtrack.yaml'))
    joint, _ = build(config)
    joint.train()
    fix_bn(joint, True)
    split = copy.deepcopy(joint)
    torch.manual_seed(94103)
    templates = [torch.randn(1, 6, 128, 128) for _ in range(2)]
    searches = [torch.randn(1, 6, 256, 256) for _ in range(3)]
    boxes = torch.tensor([[[.3, .3, .2, .2]]] * 3)
    optimizer = []
    for model in (joint, split):
        optimizer.append(torch.optim.AdamW([
            dict(params=[p for name, p in model.named_parameters()
                         if 'backbone' in name], lr=1e-5),
            dict(params=[p for name, p in model.named_parameters()
                         if 'backbone' not in name], lr=1e-4)],
            weight_decay=1e-4))
    before = torch.get_rng_state().clone()
    loss, status, complete, missing, masks, patch = forward_objective(
        joint, templates, searches, boxes, cfg, 'complete_missing', 0)
    loss.backward()
    expected_loss = float(loss.detach())
    del loss, complete, missing
    actual_loss, actual, actual_masks, actual_patch = backward_segmented(
        split, templates, searches, boxes, cfg, 0)
    assert abs(expected_loss - actual_loss) < 1e-4
    for key in actual:
        assert abs(actual[key] - status[key]) < 1e-4, key
    assert torch.equal(before, torch.get_rng_state())
    assert all(torch.equal(a, b) for a, b in
               zip(masks + patch, actual_masks + actual_patch))
    gradients = compare_named(
        {name: p.grad for name, p in joint.named_parameters()},
        {name: p.grad for name, p in split.named_parameters()}, 2e-4, 2e-5)
    buffers = compare_named(dict(joint.named_buffers()), dict(split.named_buffers()), 0., 0.)
    norms = []
    for model, opt in zip((joint, split), optimizer):
        norms.append(float(torch.nn.utils.clip_grad_norm_(model.parameters(), .1)))
        opt.step()
    parameters = compare_named(dict(joint.named_parameters()),
                               dict(split.named_parameters()), 1e-6, 1e-7)
    states = []
    for opt in optimizer:
        states.append({str(index) + ':' + key: value
                       for index, item in enumerate(opt.state.values())
                       for key, value in item.items()})
    optimizer_state = compare_named(states[0], states[1], 2e-4, 2e-6)
    assert not torch.cuda.is_initialized()
    result = dict(status='passed', synthetic_images=True, batch=1, microsteps=1,
        native_model_parameters=sum(p.numel() for p in joint.parameters()),
        searches=3, templates=2, freeze_all_bn=True, cuda_initialized=False,
        joint_loss=expected_loss, segmented_loss=actual_loss, gradients=gradients,
        buffers=buffers, parameters_after_adamw=parameters,
        optimizer_state=optimizer_state, preclip_gradient_norms=norms,
        seconds=time.time() - started,
        scope='Native network on synthetic CPU images, not native batch4 real-image '
              'smoke, GPU memory/performance, campaign acceptance or efficacy')
    with Path(args.output).open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
