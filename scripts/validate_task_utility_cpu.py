"""Two real-data CPU updates; no GPU use or benchmark performance claim."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import time

import cv2
import numpy as np
import torch

from lib.config.cadtrack.config import cfg, update_config_from_file
from miss_rgbt.data import make_loader
from miss_rgbt.model import load_pretrained
from miss_rgbt.objectives import training_loss, utility_supervision
from miss_rgbt.train import fix_bn, sample_masks


def differences(first, second):
    return {key: max(float((a[key] - b[key]).abs().max()) for a, b in zip(first, second))
            for key in ['pred_boxes', 'score_map', 'size_map', 'offset_map']}


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--threads', type=int, default=2)
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    config = json.loads(Path(args.config).read_text())
    assert config['stage'] == 'A' and config['freeze_all_bn'] and not config['repair']
    random.seed(config['seed'])
    np.random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    torch.set_num_threads(args.threads)
    cv2.setNumThreads(0)
    update_config_from_file('experiments/cadtrack/cadtrack.yaml')
    sources = {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
               for folder in ['miss_rgbt', 'lib'] for path in sorted(Path(folder).rglob('*.py'))}
    sources[args.config] = hashlib.sha256(Path(args.config).read_bytes()).hexdigest()
    sources[__file__] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (output / 'source_sha256.json').write_text(json.dumps(sources, indent=2))
    loader = make_loader(cfg, config['dataset'], 'train', 1, 2, 0)
    data = next(iter(loader))
    templates, searches = list(data['template_images']), list(data['search_images'])
    boxes = data['search_anno']
    model = load_pretrained(cfg, config['checkpoint'], repair=False,
        repair_layer=config['repair_layer'], task_utility=config['task_utility']).eval()
    assert all(parameter.device.type == 'cpu' for parameter in model.parameters())
    report = dict(scope='CPU implementation validation: one actual training tuple reused for two updates; no benchmark or GPU claim',
        torch_version=torch.__version__, threads=args.threads, config=config,
        baseline_checkpoint_sha256=file_sha256(Path(config['checkpoint'])),
        templates=list(data['template_images'].shape), searches=list(data['search_images'].shape),
        input_sha256={key: hashlib.sha256(data[key].numpy().tobytes()).hexdigest()
                      for key in ['template_images', 'search_images', 'search_anno']},
        module_parameters=sum(p.numel() for p in model.utility_repair.parameters()),
        total_parameters=sum(p.numel() for p in model.parameters()))
    assert report['baseline_checkpoint_sha256'] == 'd20c8641e4263df3ceadf9e430d0ac96c181c13ad482bca28c2dddef592a8bb8'
    with torch.no_grad():
        original = model.baseline(templates, searches)
        current = model(templates, searches)
        parity = differences(original, current)
        assert max(parity.values()) == 0., parity
        report['zero_initialized_complete_parity'] = parity
    del original, current
    model.train()
    fix_bn(model, True)
    bn = {name: value.clone() for name, value in model.state_dict().items()
          if name.endswith(('running_mean', 'running_var', 'num_batches_tracked'))}
    assert len(bn) == 39
    groups = [dict(params=[p for name, p in model.named_parameters() if p.requires_grad and 'backbone' in name],
                   lr=config['learning_rate'] * config['backbone_multiplier']),
              dict(params=[p for name, p in model.named_parameters() if p.requires_grad and 'backbone' not in name],
                   lr=config['learning_rate'])]
    optimizer = torch.optim.AdamW(groups, weight_decay=config['weight_decay'])
    received = []
    hook = model.utility_repair.correction.register_forward_pre_hook(
        lambda _, inputs: received.append(list(inputs[0].shape)))
    updates = []
    started = time.time()
    for update in range(2):
        optimizer.zero_grad(set_to_none=True)
        available, patch = sample_masks(templates, searches, config)
        probe = dict(search_index=update, regions=torch.tensor([[update]], dtype=torch.long))
        cpu_rng = torch.get_rng_state().clone()
        selection_rng = model.utility_repair.selection_rng_state.clone()
        complete = model(templates, searches, utility_probe=probe if update == 0 else None)
        if update == 0:
            after_rng = torch.get_rng_state().clone()
            after_selection = model.utility_repair.selection_rng_state.clone()
            torch.set_rng_state(cpu_rng)
            model.utility_repair.selection_rng_state.copy_(selection_rng)
            with torch.no_grad():
                without_probe = model(templates, searches)
            parity = differences(complete, without_probe)
            assert max(parity.values()) == 0., parity
            assert torch.equal(after_rng, torch.get_rng_state())
            assert torch.equal(after_selection, model.utility_repair.selection_rng_state)
            report['probe_preserves_main_forward_and_rng'] = parity
            pair = complete[0]['utility_probe']
            parity = differences([pair['off']], [pair['on']])
            assert max(parity.values()) == 0., parity
            report['zero_initialized_paired_probe_parity'] = parity
            del without_probe
        missing = model(templates, searches, available, patch,
                        utility_probe=probe if update == 1 else None)
        loss, status = training_loss(complete, missing, boxes, cfg,
            config['forward_weight'], config['reverse_weight'], config['reverse_margin'])
        probed = complete if update == 0 else missing
        auxiliary, record = utility_supervision(probed, boxes, cfg)
        pair = probed[update]['utility_probe']
        assert all(not value.requires_grad for side in ['off', 'on'] for value in pair[side].values())
        loss = loss + config['utility_weight'] * auxiliary
        assert torch.isfinite(loss), status
        old_delta = model.utility_repair.delta.weight.detach().clone()
        old_selector = model.utility_repair.selector[-1].weight.detach().clone()
        loss.backward()
        delta_gradient = float(model.utility_repair.delta.weight.grad.abs().sum())
        selector_gradient = float(model.utility_repair.selector[-1].weight.grad.abs().sum())
        assert delta_gradient > 0. and selector_gradient > 0.
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config['clip_grad_norm'])
        assert torch.isfinite(norm)
        optimizer.step()
        assert not torch.equal(old_delta, model.utility_repair.delta.weight)
        assert not torch.equal(old_selector, model.utility_repair.selector[-1].weight)
        assert all(torch.equal(value, model.state_dict()[name]) for name, value in bn.items())
        record.update(update=update + 1, view='C' if update == 0 else 'M',
                      loss=float(loss.detach()), delta_gradient_l1=delta_gradient,
                      selector_gradient_l1=selector_gradient, status=status)
        updates.append(record)
        print(json.dumps(record), flush=True)
    hook.remove()
    report.update(updates=updates, update_seconds=time.time() - started,
                  all_39_bn_buffers_unchanged=True, correction_input_shapes=received)
    optimizer.zero_grad(set_to_none=True)
    model.eval()
    missing_available = [torch.ones(1, 2)] + [torch.tensor([[0., 1.]])
                                            for _ in range(len(templates) - 1 + len(searches))]
    changed = [image.clone() for image in templates + searches]
    for image in changed[1:]:
        image[:, :3] = torch.randn_like(image[:, :3]) * 100
    with torch.no_grad():
        selection_rng = model.utility_repair.selection_rng_state.clone()
        first = model(templates, searches, missing_available)
        model.utility_repair.selection_rng_state.copy_(selection_rng)
        second = model(changed[:len(templates)], changed[len(templates):], missing_available)
        parity = differences(first, second)
        assert max(parity.values()) == 0., parity
        report['trained_unobserved_rgb_perturbation_parity'] = parity
        selection_rng = model.utility_repair.selection_rng_state.clone()
        enabled = model(templates, searches)
        model.utility_repair.selection_rng_state.copy_(selection_rng)
        limit = model.utility_repair.refinement_limit
        model.utility_repair.refinement_limit = 0.
        disabled = model(templates, searches)
        model.utility_repair.refinement_limit = limit
        error = differences(enabled, disabled)
        assert max(error.values()) > 0., error
        report['trained_complete_refinement_path_difference'] = error
    checkpoint = output / 'cpu_native_two_updates.pth.tar'
    torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(), config=config,
                    epoch=0, updates=2), str(checkpoint))
    with torch.no_grad():
        expected = model(templates, searches, missing_available)
        saved = torch.load(str(checkpoint), map_location='cpu')
        restored = load_pretrained(cfg, saved['config']['checkpoint'], repair=False,
            repair_layer=saved['config']['repair_layer'], task_utility=saved['config']['task_utility']).eval()
        restored.load_state_dict(saved['model'], strict=True)
        actual = restored(templates, searches, missing_available)
        error = differences(expected, actual)
        assert max(error.values()) == 0., error
        report['strict_reload_parity'] = error
    assert all(hashlib.sha256(Path(name).read_bytes()).hexdigest() == value for name, value in sources.items())
    report.update(status='CPU_NATIVE_PASS', checkpoint_bytes=checkpoint.stat().st_size,
                  checkpoint_sha256=file_sha256(checkpoint), checkpoint_epoch=0, checkpoint_updates=2)
    (output / 'completed.json').write_text(json.dumps(report, indent=2))
    print('CPU_NATIVE_PASS ' + json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
