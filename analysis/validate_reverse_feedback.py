"""Audit real E4 feedback gradients and an AdamW step; never a performance result."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random

import numpy as np
import torch

from lib.config.cadtrack.config import cfg, update_config_from_file
from lib.models.cadtrack import build_cadtrack
from miss_rgbt.data import make_loader
from miss_rgbt.model import MissingTrackerModel
from miss_rgbt.objectives import response_kd, task_loss, training_loss
from miss_rgbt.train import fix_bn, sample_masks


def seed(value):
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    torch.cuda.manual_seed_all(value)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def parameter_gradients(loss, parameters, retain_graph):
    values = torch.autograd.grad(loss, list(parameters.values()),
                                 allow_unused=True, retain_graph=retain_graph)
    return {name: None if value is None else value.detach().cpu()
            for name, value in zip(parameters, values)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    assert os.environ['CUDA_VISIBLE_DEVICES'] in ['2', '3']
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    checkpoint = Path(args.checkpoint)
    saved = torch.load(str(checkpoint), map_location='cpu')
    config = saved['config']
    assert config['stage'] == 'E4' and config['repair']
    assert config['reverse_weight'] == 0 and saved['updates'] > 0
    torch.set_num_threads(1)
    seed(config['seed'])
    update_config_from_file('experiments/cadtrack/cadtrack.yaml')
    model = MissingTrackerModel(build_cadtrack(cfg, training=False), repair=True,
                               width=config['repair_width'], repair_layer=config['repair_layer'])
    model.load_state_dict(saved['model'], strict=True)
    model.cuda().train()
    fix_bn(model)
    seed(config['seed'])
    # Eight predeclared real training batches; no artificial gate override.
    loader = make_loader(cfg, config['dataset'], 'train', config['microbatch'],
                         8 * config['microbatch'], 0)
    report = dict(scope='real-data gradient and optimizer audit, not E5 accuracy',
                  checkpoint=str(checkpoint), checkpoint_bytes=checkpoint.stat().st_size,
                  checkpoint_sha256=sha(checkpoint),
                  seed=config['seed'], native_margin=config['reverse_margin'],
                  probe_reverse_weight=.02, maximum_batches=8, batches=[],
                  update_scope='one microbatch gradient, without training accumulation')
    parameters = {name: p for name, p in model.named_parameters() if p.requires_grad}
    for index, data in enumerate(loader):
        templates = [image.cuda() for image in data['template_images']]
        searches = [image.cuda() for image in data['search_images']]
        boxes = data['search_anno'].cuda()
        available, patch = sample_masks(templates, searches, config)
        complete = model(templates, searches)
        missing = model(templates, searches, available, patch)
        loss_c = task_loss(complete, boxes, cfg)
        loss_m = task_loss(missing, boxes, cfg)
        reliable = (loss_m.detach() + config['reverse_margin'] < loss_c.detach()).to(loss_c.dtype)
        report['batches'].append(dict(batch=index, reliable=reliable.cpu().tolist(),
            task_c=loss_c.detach().cpu().tolist(), task_m=loss_m.detach().cpu().tolist(),
            search_sha256=hashlib.sha256(data['search_images'].numpy().tobytes()).hexdigest()))
        if reliable.sum() == 0:
            del complete, missing, loss_c, loss_m
            continue
        reverse = (reliable * response_kd(complete, missing)).mean()
        maps_c = [item['score_map'] for item in complete]
        maps_m = [item['score_map'] for item in missing]
        map_gradients = torch.autograd.grad(reverse, maps_c + maps_m,
                                            allow_unused=True, retain_graph=True)
        assert all(value is None for value in map_gradients[len(maps_c):])
        assert any(value is not None and value.abs().sum() > 0
                   for value in map_gradients[:len(maps_c)])
        baseline, _ = training_loss(complete, missing, boxes, cfg,
            config['forward_weight'], 0., config['reverse_margin'])
        active, _ = training_loss(complete, missing, boxes, cfg,
            config['forward_weight'], .02, config['reverse_margin'])
        grad_base = parameter_gradients(baseline, parameters, True)
        grad_reverse = parameter_gradients(reverse, parameters, True)
        grad_active = parameter_gradients(active, parameters, False)
        strengths = {name: float(value.abs().sum()) for name, value in grad_reverse.items()
                     if name.startswith('baseline.') and value is not None}
        recipient = max(strengths, key=strengths.get)
        assert strengths[recipient] > 0
        assert all(value is None or value.abs().sum() == 0
                   for name, value in grad_reverse.items() if name.startswith('repair.'))
        for name, value in grad_active.items():
            if value is None:
                assert grad_base[name] is None and grad_reverse[name] is None
                continue
            assert torch.isfinite(value).all()
            expected = grad_base[name]
            if grad_reverse[name] is not None:
                expected = expected + .02 * grad_reverse[name]
            assert torch.allclose(value, expected, atol=1e-6, rtol=1e-4), name
        del complete, missing, loss_c, loss_m, baseline, active, reverse, map_gradients, maps_c, maps_m
        groups = [dict(params=[p for n, p in parameters.items() if p.requires_grad and 'backbone' in n],
                       lr=config['learning_rate'] * config['backbone_multiplier']),
                  dict(params=[p for n, p in parameters.items() if p.requires_grad and 'backbone' not in n],
                       lr=config['learning_rate'])]
        optimizer = torch.optim.AdamW(groups, weight_decay=config['weight_decay'])
        optimizer.load_state_dict(saved['optimizer'])
        for name, parameter in parameters.items():
            parameter.grad = None if grad_base[name] is None else grad_base[name].to(parameter.device)
        torch.nn.utils.clip_grad_norm_(model.parameters(), config['clip_grad_norm'])
        optimizer.step()
        control = parameters[recipient].detach().cpu().clone()
        control_repair = {name: p.detach().cpu().clone() for name, p in parameters.items()
                          if name.startswith('repair.')}
        with torch.no_grad():
            for name, parameter in parameters.items():
                parameter.copy_(saved['model'][name])
        optimizer.load_state_dict(saved['optimizer'])
        for name, parameter in parameters.items():
            parameter.grad = None if grad_active[name] is None else grad_active[name].to(parameter.device)
        torch.nn.utils.clip_grad_norm_(model.parameters(), config['clip_grad_norm'])
        optimizer.step()
        difference = float((parameters[recipient].detach().cpu() - control).abs().max())
        assert difference > 0, recipient
        # Global norm clipping can also change repair updates indirectly.
        repair_difference = max(float((p.detach().cpu() - control_repair[name]).abs().max())
                                for name, p in parameters.items() if name.startswith('repair.'))
        report.update(status='DIRECTION_AND_UPDATE_VERIFIED', selected_batch=index,
            missing_target_gradient_absent=True, complete_response_gradient_nonzero=True,
            repair_direct_reverse_gradient_zero=True, gradient_decomposition_verified=True,
            recipient=recipient, recipient_reverse_gradient_l1=strengths[recipient],
            recipient_update_difference_from_reverse_zero=difference,
            repair_update_difference_from_reverse_zero=repair_difference,
            optimizer_source='existing E4 AdamW state; identical state before both steps')
        optimizer.zero_grad(set_to_none=True)
        model.eval()
        with torch.no_grad():
            expected = model(templates, searches, available, patch)
            artifact = output / 'probe_model.pth.tar'
            torch.save(dict(model=model.state_dict(), config=config), str(artifact))
            reloaded = MissingTrackerModel(build_cadtrack(cfg, training=False), repair=True,
                width=config['repair_width'], repair_layer=config['repair_layer'])
            reloaded.load_state_dict(torch.load(str(artifact), map_location='cpu')['model'], strict=True)
            restored = reloaded.cuda().eval()(templates, searches, available, patch)
            errors = {key: max(float((a[key] - b[key]).abs().max()) for a, b in zip(expected, restored))
                      for key in ['pred_boxes', 'score_map', 'size_map', 'offset_map']}
            assert max(errors.values()) == 0, errors
        report.update(strict_reload_max_difference=errors, probe_checkpoint_sha256=sha(artifact))
        break
    else:
        report['status'] = 'REAL_RELIABILITY_GATE_NOT_TRIGGERED'
    report['source_sha256'] = {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in [Path(__file__), Path('miss_rgbt/model.py'), Path('miss_rgbt/objectives.py'),
                     Path('miss_rgbt/train.py'), Path('miss_rgbt/data.py')]}
    (output / 'report.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
