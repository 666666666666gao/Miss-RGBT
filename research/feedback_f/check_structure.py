"""Synthetic-tensor F checks on the actual pretrained network, not a benchmark."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import time

import torch

from lib.config.cadtrack.config import cfg, update_config_from_file
from lib.models.cadtrack import build_cadtrack
from miss_rgbt.model import MissingTrackerModel, load_pretrained
from miss_rgbt.objectives import task_loss, training_loss, utility_supervision
from miss_rgbt.train import fix_bn
from model import FeedbackTrackerModel, two_round_training_loss


KEYS = ['pred_boxes', 'score_map', 'size_map', 'offset_map']


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def equal(left, right):
    return all(torch.equal(left[key], right[key]) for key in KEYS)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    output = Path(args.output)
    assert not output.exists()
    expected = 'd20c8641e4263df3ceadf9e430d0ac96c181c13ad482bca28c2dddef592a8bb8'
    assert sha(args.checkpoint) == expected
    started = time.time()
    torch.set_num_threads(1)
    torch.manual_seed(65301)
    torch.cuda.manual_seed_all(65301)
    update_config_from_file('experiments/cadtrack/cadtrack.yaml')
    utility = dict(width=128, depth=1, region_grid=4, selected_regions=4,
                   refinement_limit=.1, selection='utility', selection_seed=65301)
    native = load_pretrained(cfg, args.checkpoint, task_utility=utility).cuda().eval()
    # Nonzero repair makes omission of block6 repair observable.
    with torch.no_grad():
        native.utility_repair.delta.weight.normal_(std=.001)
    model = FeedbackTrackerModel(native.baseline, task_utility=utility).cuda().eval()
    model.utility_repair.load_state_dict(native.utility_repair.state_dict(), strict=True)
    templates = [torch.randn(1, 6, 128, 128, device='cuda') for _ in range(2)]
    search = torch.randn(1, 6, 256, 256, device='cuda')
    block_calls, repair_calls = [0] * 12, [0]

    def block_hook(index):
        def record(module, inputs, outputs):
            block_calls[index] += 1
        return record

    def repair_hook(module, inputs, outputs):
        repair_calls[0] += 1

    checks = {}
    with torch.no_grad():
        parity = {}
        for label, flags in [('11', [1., 1.]), ('01', [0., 1.]), ('10', [1., 0.])]:
            observed = search.new_tensor([flags])
            availability = [search.new_ones(1, 2), observed, observed]
            baseline = native(templates, [search], availability)[0]
            actual = model(templates, [search], availability)[0]
            assert equal(baseline, actual) and equal(actual['round1'], actual)
            parity[label] = True
        checks['zero_residual_native_round1_round2_exact'] = parity
        handles = [block.register_forward_hook(block_hook(index))
                   for index, block in enumerate(model.baseline.backbone.blocks)]
        handles.append(model.utility_repair.register_forward_hook(repair_hook))
        model(templates, [search])
        for handle in handles:
            handle.remove()
        assert block_calls == [1] * 3 + [2] * 9 and repair_calls == [2]
        checks['block_calls'] = block_calls
        checks['repair_calls'] = repair_calls[0]
        for mode in ['shallow', 'deep']:
            model.feedback.mode = mode
            actual = model(templates, [search])[0]
            assert equal(actual, native(templates, [search])[0])
        residual = model.feedback
        model.feedback = None
        actual = model(templates, [search])[0]
        assert equal(actual, native(templates, [search])[0])
        model.feedback = residual
        checks['no_residual_and_shallow_controls_native_parity_exact'] = True
        model.feedback.mode = 'deep'
        with torch.enable_grad():
            initial = model(templates, [search])
            initial_loss = task_loss(initial, search.new_tensor([[[.3, .3, .25, .25]]]), cfg).mean()
            initial_loss.backward()
            gradient = model.feedback.delta.weight.grad
            assert torch.isfinite(gradient).all() and float(gradient.norm()) > 0
            checks['zero_initialized_delta_gradient_norm'] = float(gradient.norm())
            model.zero_grad()
            del initial, initial_loss, gradient
        with torch.no_grad():
            model.feedback.delta.weight.normal_(std=.001)
        tail_inputs, feedback_inputs = [], []

        def record_tail(module, inputs):
            tail_inputs.append(tuple(value.detach().clone() for value in inputs))

        def record_feedback(module, inputs):
            feedback_inputs.append(inputs)

        handles = [model.baseline.backbone.blocks[3].register_forward_pre_hook(record_tail),
                   model.feedback.register_forward_pre_hook(record_feedback)]
        actual = model(templates, [search])[0]
        for handle in handles:
            handle.remove()
        assert len(tail_inputs) == 2
        for first_input, second_input in zip(*tail_inputs):
            assert torch.equal(first_input[:, :-256], second_input[:, :-256])
        assert any(not torch.equal(first_input[:, -256:], second_input[:, -256:])
                   for first_input, second_input in zip(*tail_inputs))
        checks['feedback_preserves_query_template_prefix_and_changes_search'] = True
        assert not equal(actual['round1'], actual)
        checks['nonzero_feedback_changes_complete_prediction'] = True
        shallow, deep, prediction, observed = feedback_inputs[0]
        changed_deep = tuple(torch.randn_like(value) for value in deep)
        changed_prediction = {key: value + .1 for key, value in prediction.items()
                              if key in KEYS}
        for mode in ['deep', 'shallow']:
            model.feedback.mode = mode
            original = model.feedback(shallow, deep, prediction, observed)
            for source_deep, source_prediction in [
                    (changed_deep, prediction), (deep, changed_prediction)]:
                changed = model.feedback(shallow, source_deep, source_prediction, observed)
                same = all(torch.equal(left, right) for left, right in zip(original, changed))
                assert same == (mode == 'shallow')
        checks['shallow_control_ignores_deep_features_and_prediction_exact'] = True
        checks['deep_feedback_responds_to_deep_features_and_prediction'] = True
        model.feedback.mode = 'deep'
        invariance = {}
        for label, flags in [('01', [0., 1.]), ('10', [1., 0.])]:
            observed = search.new_tensor([flags])
            availability = [search.new_ones(1, 2), observed, observed]
            altered_templates = [value.clone() for value in templates]
            altered_search = search.clone()
            channels = slice(0, 3) if flags[0] == 0 else slice(3, 6)
            altered_templates[1][:, channels] = 100 * torch.randn_like(
                altered_templates[1][:, channels])
            altered_search[:, channels] = 100 * torch.randn_like(altered_search[:, channels])
            clean = model(templates, [search], availability)[0]
            altered = model(altered_templates, [altered_search], availability)[0]
            assert equal(clean, altered) and equal(clean['round1'], altered['round1'])
            invariance[label] = True
        checks['unavailable_sensor_input_invariance_exact'] = invariance
        patch = search.new_ones(1, 256, 2)
        patch[:, 0] = 0
        altered_search = search.clone()
        altered_search[:, :, :16, :16] = 100 * torch.randn_like(
            altered_search[:, :, :16, :16])
        clean = model(templates, [search], patch_keep=[patch])[0]
        altered = model(templates, [altered_search], patch_keep=[patch])[0]
        assert equal(clean, altered) and equal(clean['round1'], altered['round1'])
        checks['masked_patch_input_invariance_exact'] = True

    model.train()
    fix_bn(model, True)
    bn_before = {name: value.clone() for name, value in model.named_buffers()
                 if any(key in name for key in
                        ['running_mean', 'running_var', 'num_batches_tracked'])}
    boxes = search.new_tensor([[[.3, .3, .25, .25]]] * 3)
    searches = [search, search * .9, search * 1.1]
    availability = [search.new_ones(1, 2) for _ in templates]
    availability += [search.new_tensor([[0., 1.]]) for _ in searches]
    probe = dict(search_index=0, regions=torch.zeros(1, 1, device='cuda', dtype=torch.long))
    complete = model(templates, searches, utility_probe=probe)
    missing = model(templates, searches, availability)
    loss, _ = two_round_training_loss(complete, missing, boxes, cfg, forward_weight=.1)
    first, _ = training_loss([out['round1'] for out in complete],
                            [out['round1'] for out in missing], boxes, cfg, forward_weight=.1)
    second, _ = training_loss(complete, missing, boxes, cfg, forward_weight=.1)
    assert torch.equal(loss, .5 * (first + second))
    auxiliary, _ = utility_supervision(complete, boxes, cfg)
    loss = loss + .1 * auxiliary
    assert torch.isfinite(loss)
    before = model.feedback.delta.weight.detach().clone()
    optimizer = torch.optim.SGD(model.parameters(), lr=.00001)
    loss.backward()
    gradients = {}
    named = dict(model.named_parameters())
    for name in ['feedback.delta.weight', 'feedback.hidden.0.weight',
                 'baseline.backbone.blocks.0.attn.qkv.weight',
                 'baseline.backbone.blocks.11.attn.qkv.weight',
                 'utility_repair.selector.0.weight']:
        gradient = named[name].grad
        assert gradient is not None and torch.isfinite(gradient).all()
        norm = float(gradient.norm())
        assert norm > 0, name
        gradients[name] = norm
    assert all(torch.isfinite(value.grad).all() for value in model.parameters()
               if value.grad is not None)
    optimizer.step()
    assert not torch.equal(before, model.feedback.delta.weight)
    assert all(torch.equal(value, dict(model.named_buffers())[name])
               for name, value in bn_before.items())
    optimizer.zero_grad()
    checks['three_search_dual_round_task_and_utility_loss'] = float(loss.detach())
    checks['objective_scale_exact'] = True
    checks['gradient_norms'] = gradients
    checks['feedback_parameter_updated'] = True
    checks['training_bn_buffers_unchanged_exact'] = True
    del complete, missing, loss, first, second, auxiliary, optimizer, native

    model.eval()
    with torch.no_grad():
        expected_output = model(templates, [search])[0]
        checkpoint = io.BytesIO()
        torch.save(dict(model=model.state_dict(), feedback_mode='deep', feedback_width=128,
                        task_utility=utility), checkpoint)
        checkpoint.seek(0)
        saved = torch.load(checkpoint, map_location='cpu')
        restored = FeedbackTrackerModel(build_cadtrack(cfg, training=False),
            task_utility=saved['task_utility'], feedback_mode=saved['feedback_mode'],
            feedback_width=saved['feedback_width'])
        restored.load_state_dict(saved['model'], strict=True)
        restored.cuda().eval()
        actual = restored(templates, [search])[0]
        assert equal(expected_output, actual)
        checks['fresh_model_strict_reload_prediction_exact'] = True
    files = [Path(__file__), Path(__file__).with_name('model.py')]
    files += [Path('miss_rgbt') / name for name in ['model.py', 'objectives.py', 'task_utility.py']]
    report = dict(status='PASS', scope='Synthetic tensors, actual E0 network; no dataset or benchmark',
        seed=65301, checkpoint_sha256=expected, checks=checks,
        source_sha256={str(path.resolve()): sha(path) for path in files},
        torch_version=torch.__version__, gpu=torch.cuda.get_device_name(0),
        peak_gpu_bytes=torch.cuda.max_memory_allocated(), seconds=time.time()-started,
        limitations=['No actual image training or efficacy evidence',
                     'No full tracker/template-update integration check',
                     'No random-region selection protocol',
                     'No formal trainer or official evaluator integration',
                     'No measured correction/destruction rates or latency comparison'])
    output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
