"""CPU mechanics witness on real-model BN modules; no task forward or training."""
import argparse
import hashlib
import json
import os
from pathlib import Path

import torch

from lib.config.cadtrack.config import cfg, update_config_from_file
from miss_rgbt.model import load_pretrained
from miss_rgbt.train import fix_bn


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def witness(path):
    config = json.loads(path.read_text())
    assert config['freeze_all_bn'] is True and config['stage'] in ['E3', 'E4']
    torch.manual_seed(config['seed'])
    model = load_pretrained(cfg, config['checkpoint'], repair=config['repair'],
        width=config['repair_width'], repair_layer=config['repair_layer']).train()
    modules = {name: module for name, module in model.named_modules()
        if isinstance(module, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d))}
    assert len(modules) == 13 and all(isinstance(module, torch.nn.BatchNorm2d) for module in modules.values())
    fusion = 'baseline.decode_fuse_search.1'
    assert fusion in modules and sum(name.startswith('baseline.box_head.') for name in modules) == 12
    initial = {name: {key: value.detach().clone() for key, value in module.state_dict().items()}
        for name, module in modules.items()}
    parameter_names = list(model.state_dict())
    results = []
    for scope in ['head_only_default', 'all_bn']:
        model.train()
        model.zero_grad(set_to_none=True)
        for name, module in modules.items():
            module.load_state_dict(initial[name], strict=True)
        if scope == 'head_only_default':
            fix_bn(model)
        else:
            fix_bn(model, freeze_all=True)
        assert all(module.training for module in model.modules()
            if not isinstance(module, torch.nn.BatchNorm2d))
        assert all(module.weight.requires_grad and module.bias.requires_grad for module in modules.values())
        optimizer = torch.optim.AdamW([parameter for module in modules.values() for parameter in module.parameters()],
            lr=config['learning_rate'], weight_decay=config['weight_decay'])
        for index, (name, module) in enumerate(modules.items()):
            generator = torch.Generator().manual_seed(config['seed'] + index)
            value = torch.randn(2, module.num_features, 2, 2, generator=generator) * 3 + 2
            output = module(value)
            loss = output.square().mean() + output.mean()
            assert torch.isfinite(loss)
            loss.backward()
        gradients = {name: {key: float(parameter.grad.norm()) for key, parameter in module.named_parameters()}
            for name, module in modules.items()}
        assert all(torch.isfinite(parameter.grad).all() and parameter.grad.abs().sum() > 0
            for module in modules.values() for parameter in module.parameters())
        optimizer.step()
        rows = []
        for name, module in modules.items():
            current = module.state_dict()
            increment = int(current['num_batches_tracked'] - initial[name]['num_batches_tracked'])
            expected = int(scope == 'head_only_default' and name == fusion)
            assert increment == expected
            stats_unchanged = all(torch.equal(current[key], initial[name][key])
                for key in ['running_mean', 'running_var', 'num_batches_tracked'])
            assert stats_unchanged == (expected == 0)
            affine_changed = all(not torch.equal(current[key], initial[name][key]) for key in ['weight', 'bias'])
            assert affine_changed
            rows.append(dict(name=name, training=module.training, counter_increment=increment,
                running_buffers_unchanged=stats_unchanged, affine_grad_norm=gradients[name],
                affine_updated=affine_changed))
        assert list(model.state_dict()) == parameter_names
        results.append(dict(scope=scope, all_non_bn_modules_training=True, modules=rows))
    return dict(config=str(path), config_sha256=sha(path), checkpoint_sha256=sha(config['checkpoint']),
        repair=config['repair'], bn_modules=len(modules), total_parameters=sum(p.numel() for p in model.parameters()),
        state_keys=len(parameter_names), cases=results)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--configs', nargs='+', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    assert os.environ['CUDA_VISIBLE_DEVICES'] == ''
    torch.set_num_threads(1)
    update_config_from_file('experiments/cadtrack/cadtrack.yaml')
    records = [witness(Path(path)) for path in args.configs]
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    sources = ['miss_rgbt/train.py', 'scripts/validate_bn_freeze.py', 'experiments/cadtrack/cadtrack.yaml']
    report = dict(scope='CPU-only direct calls to the 13 BN modules in the actual checkpoint-initialized E3/E4 model. '
        'One tiny input and one BN-affine AdamW update per mode; no image, task-model forward, GPU, formal training or benchmark.',
        torch_version=torch.__version__, logical_torch_threads=torch.get_num_threads(),
        source_sha256={path: sha(path) for path in sources}, records=records)
    (output/'completed.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
