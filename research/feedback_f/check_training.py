"""Actual-network synthetic optimizer checks for the shared F batch interface."""
import argparse
import gc
import json
from pathlib import Path
import time

import torch

from lib.config.cadtrack.config import cfg, update_config_from_file
from miss_rgbt.objectives import training_loss, utility_supervision
from miss_rgbt.train import fix_bn
from check_structure import sha
from training import ARMS, batch_loss, build_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    output = Path(args.output)
    assert not output.exists()
    assert sha(args.checkpoint) == 'd20c8641e4263df3ceadf9e430d0ac96c181c13ad482bca28c2dddef592a8bb8'
    torch.set_num_threads(1)
    update_config_from_file('experiments/cadtrack/cadtrack.yaml')
    utility = dict(width=128, depth=1, region_grid=4, selected_regions=4,
                   refinement_limit=.1, selection='utility', selection_seed=65303)
    config = dict(forward_weight=.1, reverse_margin=0., utility_weight=.1)
    torch.manual_seed(65303)
    templates = [torch.randn(1, 6, 128, 128, device='cuda') for _ in range(2)]
    searches = [torch.randn(1, 6, 256, 256, device='cuda') for _ in range(3)]
    boxes = searches[0].new_tensor([[[.3, .3, .25, .25]]] * 3)
    availability = [searches[0].new_ones(1, 2) for _ in templates]
    availability += [searches[0].new_tensor([[0., 1.]]) for _ in searches]
    patch = [searches[0].new_ones(1, 256, 2) for _ in searches]
    for value in patch:
        value[:, ::4] = 0
    probe = dict(search_index=1, regions=torch.zeros(1, 1, device='cuda', dtype=torch.long))
    rows, initial = {}, None
    started = time.time()
    for arm in ARMS:
        torch.manual_seed(65304)
        model = build_model(cfg, args.checkpoint, utility, arm)
        rng = torch.get_rng_state().clone()
        shared = {name: sha_tensor(value) for name, value in model.state_dict().items()
                  if not name.startswith('feedback.')}
        if initial is None:
            initial, expected_rng = shared, rng
        assert shared == initial and torch.equal(rng, expected_rng)
        model.cuda().train()
        fix_bn(model, True)
        bn = {name: value.clone() for name, value in model.named_buffers()
              if any(key in name for key in
                     ['running_mean', 'running_var', 'num_batches_tracked'])}
        optimizer = torch.optim.AdamW(model.parameters(), lr=.00001, weight_decay=.0001)
        cases = []
        for view in ['C', 'M']:
            optimizer.zero_grad(set_to_none=True)
            # Identical RNG makes a fresh independent objective reconstruction exact.
            torch.manual_seed(65305)
            loss, status = batch_loss(model, templates, searches, boxes, availability,
                                     patch, cfg, config, probe, view)
            torch.manual_seed(65305)
            with torch.no_grad():
                complete = model(templates, searches, utility_probe=probe if view == 'C' else None)
                missing = model(templates, searches, availability, patch,
                                utility_probe=probe if view == 'M' else None)
                expected, _ = training_loss(complete, missing, boxes, cfg, forward_weight=.1)
                if arm != 'single':
                    first, _ = training_loss([out['round1'] for out in complete],
                        [out['round1'] for out in missing], boxes, cfg, forward_weight=.1)
                    expected = .5 * (expected + first)
                auxiliary, _ = utility_supervision(complete if view == 'C' else missing, boxes, cfg)
                expected = expected + .1 * auxiliary
                assert torch.equal(loss.detach(), expected)
                del complete, missing, expected, auxiliary
            assert torch.isfinite(loss)
            named = dict(model.named_parameters())
            key = 'baseline.backbone.blocks.11.attn.qkv.weight'
            before = named[key].detach().clone()
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), .1)
            assert torch.isfinite(norm) and norm > 0
            assert named['utility_repair.selector.0.weight'].grad.norm() > 0
            if arm in ['shallow', 'deep']:
                assert named['feedback.delta.weight'].grad.norm() > 0
            optimizer.step()
            assert not torch.equal(before, named[key])
            assert all(torch.equal(value, dict(model.named_buffers())[name])
                       for name, value in bn.items())
            cases.append(dict(probe_view=view, loss=float(loss.detach()),
                              grad_norm=float(norm), exact_objective=True,
                              backbone_updated=True, bn_unchanged=True))
            del loss, before
        rows[arm] = cases
        del model, optimizer, named, bn
        gc.collect()
        torch.cuda.empty_cache()
    paths = [Path(__file__), Path(__file__).with_name('training.py'),
             Path(__file__).with_name('model.py')]
    report = dict(status='PASS', seed=65303, scope='Synthetic tensors; actual E0 network',
        shared_initial_state_exact=True, caller_cpu_rng_after_build_exact=True,
        arms=rows, seconds=time.time()-started,
        peak_gpu_bytes=torch.cuda.max_memory_allocated(),
        source_sha256={str(path.resolve()): sha(path) for path in paths},
        limitations=['No image training, convergence or official metrics',
                     'No formal trainer loop or benchmark launcher',
                     'No measured actual-workload latency or correction/destruction rates'])
    output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report), flush=True)


def sha_tensor(value):
    import hashlib
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


if __name__ == '__main__':
    main()
