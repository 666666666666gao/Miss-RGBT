"""Audit frozen R RNG dependencies without touching live trainers or CUDA."""
import argparse
import ast
import hashlib
import importlib
import inspect
import json
from pathlib import Path
import textwrap
import time

import torch

from lib.config.cadtrack.config import cfg
from miss_rgbt.data import make_loader
from miss_rgbt.train import fix_bn, sample_masks
from research.protocol_r.common import (
    REPORT, RUNS, SNAPSHOT, build, check_sources, read, save, seed, sha, tensor_sha,
)


def source_record(function):
    text = inspect.getsource(function)
    # Class methods arrive indented; AST needs their original function scope.
    tree = ast.parse(textwrap.dedent(text))
    names = []
    random_names = {
        'rand', 'randn', 'rand_like', 'randn_like', 'randint', 'multinomial',
        'bernoulli', 'bernoulli_', 'normal_', 'uniform_', 'random', 'dropout',
        'drop_path', 'manual_seed', 'manual_seed_all', 'fork_rng',
    }
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = (node.func.attr if isinstance(node.func, ast.Attribute) else
                    node.func.id if isinstance(node.func, ast.Name) else '')
            if name in random_names:
                names.append(dict(name=name, relative_line=node.lineno))
    path = Path(inspect.getsourcefile(function)).resolve()
    return dict(function=function.__module__ + '.' + function.__qualname__,
                source_file=str(path), source_file_sha256=sha(path),
                function_source_sha256=hashlib.sha256(text.encode()).hexdigest(),
                random_related_calls=names, source=text)


def setup_witness(model, config):
    """Exercise the actual post-seed CPU setup, without consuming loader items."""
    seed(config['seed'])
    expected = torch.get_rng_state().clone()
    stages = []

    def unchanged(name):
        assert torch.equal(torch.get_rng_state(), expected), name
        stages.append(dict(stage=name, cpu_rng_sha256=tensor_sha(expected)))

    loader = make_loader(cfg, config['dataset'], 'train', config['microbatch'],
                         config['samples_per_epoch'], config['workers'])
    unchanged('native_loader_construction')
    groups = [
        dict(params=[p for n, p in model.named_parameters() if 'backbone' in n],
             lr=config['learning_rate'] * config['backbone_multiplier']),
        dict(params=[p for n, p in model.named_parameters() if 'backbone' not in n],
             lr=config['learning_rate']),
    ]
    optimizer = torch.optim.AdamW(groups, weight_decay=config['weight_decay'])
    unchanged('native_adamw_construction')
    assert sha(config['development_cache']) == config['development_cache_sha256']
    batches = torch.load(config['development_cache'], map_location='cpu')
    unchanged('actual_development_cache_load')
    model.train()
    fix_bn(model)
    optimizer.zero_grad(set_to_none=True)
    unchanged('train_bn_and_zero_grad')
    assert len(loader) == 5000 and len(loader) % config['accumulation'] == 0
    assert batches
    return dict(arm=config['arm'], stages=stages, loader_batches=len(loader),
                development_batches=len(batches),
                no_images_decoded=True, no_loader_iterator_created=True,
                optimizer_steps_executed=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    destination = Path(args.output)
    assert not destination.exists() and not torch.cuda.is_initialized()
    started = time.time()
    torch.set_num_threads(1)
    check_sources()
    configs = {arm: read(RUNS / arm / 'run_config.json') for arm in ('r1', 'r2')}
    model, initialization = build(configs['r1'])
    assert model.repair is None and model.utility_repair is None
    assert all(not value.is_cuda for value in model.parameters())
    setups = [setup_witness(model, config) for config in configs.values()]
    witness_root = REPORT.parent / 'server28_protocol_r_epoch_boundary_20261103'
    witness = read(witness_root / 'report.json')
    witness_done = read(witness_root / 'completed.json')
    assert witness['status'] == 'passed' and witness_done['status'] == 'complete'
    assert sha(witness_root / 'report.json') == witness_done['report_sha256']
    assert all(stage['cpu_rng_sha256'] == witness['initial_cpu_rng_sha256']
               for setup in setups for stage in setup['stages'])
    for arm in configs:
        assert sha(RUNS / arm / 'run_config.json') == witness['run_config_sha256'][arm]
        assert read(RUNS / arm / 'initial_state.json') == initialization
    before_fork = torch.get_rng_state().clone()
    with torch.random.fork_rng(devices=[]):
        torch.rand(7)
        assert not torch.equal(torch.get_rng_state(), before_fork)
    assert torch.equal(torch.get_rng_state(), before_fork)
    model_sources = {}
    stochastic_modules = []
    for name, module in model.named_modules():
        cls = type(module)
        key = cls.__module__ + '.' + cls.__qualname__
        if key not in model_sources:
            model_sources[key] = source_record(cls.forward)
        if cls.__name__ in ('Dropout', 'DropPath'):
            stochastic_modules.append(dict(
                module=name, type=key,
                probability=module.p if cls.__name__ == 'Dropout' else module.drop_prob,
                input_device_bound=True))
    mask_source = source_record(sample_masks)
    assert 'device = searches[0].device' in mask_source['source']
    assert 'torch.multinomial(probability' in mask_source['source']
    assert 'device=device' in mask_source['source']
    drop_classes = {type(module) for module in model.modules()
                    if type(module).__name__ == 'DropPath'}
    assert len(drop_classes) == 1
    drop_class = next(iter(drop_classes))
    drop_module = importlib.import_module(drop_class.__module__)
    drop_source = source_record(drop_module.drop_path)
    assert 'x.new_empty(shape).bernoulli_' in drop_source['source']
    optimizer_module = importlib.import_module(torch.optim.AdamW.__module__)
    helpers = [
        source_record(torch.optim.AdamW.__init__),
        source_record(torch.optim.AdamW.step),
        source_record(optimizer_module.adamw),
        source_record(optimizer_module._single_tensor_adamw),
        source_record(optimizer_module._multi_tensor_adamw),
        source_record(torch.random.fork_rng),
    ]
    for helper in helpers[:5]:
        assert not helper['random_related_calls'], helper['function']
    trainer = SNAPSHOT / 'research/protocol_r/train.py'
    tree = ast.parse(trainer.read_text())
    validate = next(node for node in tree.body
                    if isinstance(node, ast.FunctionDef) and node.name == 'validate')
    following = next(node for node in tree.body if node.lineno > validate.lineno)
    validation_source = '\n'.join(
        trainer.read_text().splitlines()[validate.lineno - 1:following.lineno - 1])
    assert 'torch.random.fork_rng' in validation_source
    # This inventory is source evidence, not a recording of the original CUDA run.
    sources = dict(read(REPORT / 'source_sha256.json'))
    sources[str(witness_root / 'report.json')] = witness_done['report_sha256']
    sources[str(witness_root / 'completed.json')] = sha(witness_root / 'completed.json')
    records = list(model_sources.values()) + [mask_source, drop_source] + helpers
    for entry in records:
        sources[entry['source_file']] = entry['source_file_sha256']
    sources[str(Path(__file__).resolve())] = sha(__file__)
    for path, expected in sources.items():
        assert sha(path) == expected, path
    assert not torch.cuda.is_initialized()
    result = dict(
        status='passed', seconds=time.time() - started, torch_version=torch.__version__,
        native_cpu_setup_witnesses=setups,
        setup_matches_completed_replay_initial_rng=True,
        native_initialization_matches_both_original_runs=True,
        cpu_fork_rng_restoration_executed=True,
        initialization_parameters_sha256=initialization['parameters_sha256'],
        stochastic_modules=stochastic_modules, model_forward_sources=model_sources,
        mask_source=mask_source, drop_path_source=drop_source,
        installed_torch_helpers=helpers, validation_source=validation_source,
        run_config_sha256={arm: sha(RUNS / arm / 'run_config.json')
                           for arm in configs},
        source_sha256=sources, cuda_initialized=False,
        original_cuda_rng_observed=False,
        actual_original_inputs_independently_certified=False,
        all_historical_frame_exclusions_verified=False, ohcas_data_gate_passed=False,
        scope='Actual frozen native CPU initialization/loader/AdamW/cache setup plus '
              'source inventory of all model forward types, device-bound stochastic '
              'layers, masks, installed AdamW and validation RNG restoration. '
              'No model forward/backward, optimizer step, original process RNG read, '
              'later original input recording, training or official tracking metric.')
    save(destination, result)
    print(json.dumps({key: result[key] for key in (
        'status', 'seconds', 'cuda_initialized',
        'actual_original_inputs_independently_certified', 'ohcas_data_gate_passed')}),
        flush=True)


if __name__ == '__main__':
    main()
