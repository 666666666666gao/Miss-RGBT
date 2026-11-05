"""Synthetic full-update memory/time measurement; no image or weight files."""
import argparse
import json
import os
from pathlib import Path
import time

import torch

from lib.config.cadtrack.config import cfg, update_config_from_file
from miss_rgbt.train import fix_bn
from check_structure import sha
from training import ARMS, batch_loss, build_model, step_masks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--arm', required=True, choices=ARMS)
    parser.add_argument('--microbatch', required=True, type=int)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    output = Path(args.output)
    assert not output.exists() and args.microbatch > 0
    expected = 'd20c8641e4263df3ceadf9e430d0ac96c181c13ad482bca28c2dddef592a8bb8'
    assert sha(args.checkpoint) == expected
    torch.set_num_threads(1)
    torch.manual_seed(65307)
    torch.cuda.manual_seed_all(65307)
    update_config_from_file('experiments/cadtrack/cadtrack.yaml')
    utility = dict(width=128, depth=1, region_grid=4, selected_regions=4,
                   refinement_limit=.1, selection='utility', selection_seed=65307)
    config = dict(forward_weight=.1, reverse_margin=.1, utility_weight=.1,
        mask_seed=65308, patch_mask_probability=.1,
        complete_rgb_missing_tir_missing_probability=[.4, .3, .3])
    model = build_model(cfg, args.checkpoint, utility, args.arm).cuda().train()
    fix_bn(model, True)
    batch, accumulation = args.microbatch, 2
    learning_rate = .000125 * (batch*accumulation)/10
    groups = [
        dict(params=[p for name, p in model.named_parameters()
                     if p.requires_grad and 'backbone' in name], lr=learning_rate*.1),
        dict(params=[p for name, p in model.named_parameters()
                     if p.requires_grad and 'backbone' not in name], lr=learning_rate)]
    optimizer = torch.optim.AdamW(groups, weight_decay=.0001)
    templates = [torch.randn(batch, 6, 128, 128, device='cuda') for _ in range(2)]
    searches = [torch.randn(batch, 6, 256, 256, device='cuda') for _ in range(3)]
    boxes = searches[0].new_tensor([[[.3, .3, .25, .25]]]*3).expand(-1, batch, -1)
    generator = torch.Generator().manual_seed(65307)
    rows = []
    for update in range(3):
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        losses, views = [], []
        for index in range(accumulation):
            microstep = update*accumulation+index
            available, patch = step_masks(templates, searches, config, microstep)
            probe, view = None, None
            if microstep % 4 == 0:
                probe_id = microstep//4
                view = 'C' if probe_id % 2 == 0 else 'M'
                regions = torch.randint(model.utility_repair.region_tokens.shape[0],
                    (batch, 1), generator=generator).cuda()
                probe = dict(search_index=(probe_id//2) % len(searches), regions=regions)
            loss, _ = batch_loss(model, templates, searches, boxes, available,
                                 patch, cfg, config, probe, view)
            assert torch.isfinite(loss)
            (loss/accumulation).backward()
            losses.append(float(loss.detach()))
            views.append(view)
            del loss
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), .1)
        assert torch.isfinite(norm)
        optimizer.step()
        torch.cuda.synchronize()
        rows.append(dict(update=update+1, seconds=time.perf_counter()-started,
            losses=losses, probe_views=views, grad_norm=float(norm),
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_reserved_bytes=torch.cuda.max_memory_reserved()))
    paths = [Path(__file__), Path(__file__).with_name('training.py'),
             Path(__file__).with_name('model.py')]
    paths += [Path('miss_rgbt')/name for name in ['model.py', 'objectives.py', 'train.py']]
    report = dict(status='PASS', scope='Synthetic resident tensors; three full AdamW updates',
        arm=args.arm, microbatch=batch, accumulation=accumulation,
        effective_batch=batch*accumulation, learning_rate=learning_rate,
        backbone_learning_rate=learning_rate*.1, seed=65307,
        checkpoint_sha256=expected, cuda_visible_devices=os.environ['CUDA_VISIBLE_DEVICES'],
        gpu=torch.cuda.get_device_name(0),
        total_gpu_bytes=torch.cuda.get_device_properties(0).total_memory,
        torch_version=torch.__version__, rows=rows,
        peak_allocated_bytes=max(row['peak_allocated_bytes'] for row in rows),
        peak_reserved_bytes=max(row['peak_reserved_bytes'] for row in rows),
        source_sha256={str(path.resolve()): sha(path) for path in paths},
        limitations=['No real images, data loading, cropping, transfers or official metrics',
            'Three updates do not measure convergence or sustained training throughput',
            'Synthetic memory fit is preliminary; actual image smoke remains required',
            'No checkpoint persisted or formal F campaign launched'])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
