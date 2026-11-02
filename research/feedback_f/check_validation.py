"""Actual-network synthetic checks for F internal round-loss reporting."""
import argparse
import gc
import json
from pathlib import Path

import torch

import train_partitioned_a as base
from check_structure import sha
from training import build_model
from validation import validate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    output = Path(args.output)
    assert not output.exists()
    assert sha(args.checkpoint) == base.prior.common.CHECKPOINT_SHA
    torch.set_num_threads(1)
    torch.manual_seed(65309)
    torch.cuda.manual_seed_all(65309)
    base.update_config_from_file('experiments/cadtrack/cadtrack.yaml')
    utility = dict(width=128, depth=1, region_grid=4, selected_regions=4,
                   refinement_limit=.1, selection='utility', selection_seed=65309)
    data = dict(template_images=torch.randn(2, 2, 6, 128, 128),
                search_images=torch.randn(3, 2, 6, 256, 256),
                search_anno=torch.tensor([[[.3, .3, .25, .25]]*2]*3))
    results = {}
    for arm in ['single', 'none', 'deep']:
        model = build_model(base.cfg, args.checkpoint, utility, arm).cuda().eval()
        selector = model.utility_repair.selection_rng_state.clone()
        cpu_rng = torch.get_rng_state().clone()
        cuda_rng = torch.cuda.get_rng_state().clone()
        report = validate(model, [data], base.cfg)
        assert torch.equal(selector, model.utility_repair.selection_rng_state)
        assert torch.equal(cpu_rng, torch.get_rng_state())
        assert torch.equal(cuda_rng, torch.cuda.get_rng_state())
        expected = base.validate(model, [data])
        for key in ['samples', 'task_loss', 'mean_task_loss']:
            assert report[key] == expected[key]
        if arm == 'single':
            assert 'round_samples' not in report and 'round_comparison' not in report
        else:
            for state in ['11', '01', '10']:
                comparison = report['round_comparison'][state]
                assert comparison['equal_loss'] == 2
                assert comparison['mean_delta'] == 0
        results[arm] = dict(native_final_loss_exact=True,
            zero_residual_round_ties=(arm != 'single'), rng_and_selector_preserved=True)
        if arm == 'deep':
            with torch.no_grad():
                model.feedback.delta.weight.normal_(std=.01)
            observed = []
            hook = model.register_forward_hook(lambda module, inputs, outputs:
                                               observed.append(outputs))
            report = validate(model, [data], base.cfg)
            hook.remove()
            changed = False
            for state, outputs in zip(['11', '01', '10'], observed):
                with torch.no_grad():
                    first = base.task_loss([out['round1'] for out in outputs],
                                           data['search_anno'].cuda(), base.cfg).cpu().tolist()
                    final = base.task_loss(outputs, data['search_anno'].cuda(), base.cfg).cpu().tolist()
                row = report['round_samples'][state][0]
                assert row == dict(batch_index=0, round1=first, round2=final)
                stats = report['round_comparison'][state]
                assert stats['lower_loss'] == sum(b < a for a, b in zip(first, final))
                assert stats['higher_loss'] == sum(b > a for a, b in zip(first, final))
                assert stats['equal_loss'] == sum(b == a for a, b in zip(first, final))
                assert sum(stats[key] for key in ['lower_loss', 'higher_loss', 'equal_loss']) == 2
                assert stats['mean_delta'] == (sum(final)-sum(first))/2
                changed = changed or first != final
            assert changed
            results['deep_nonzero'] = dict(raw_losses_recomputed_exact=True,
                count_arithmetic_exact=True, states=report['round_comparison'])
            del observed, outputs, report
        del model
        gc.collect()
        torch.cuda.empty_cache()
    paths = [Path(__file__), Path(__file__).with_name('validation.py'),
             Path(__file__).with_name('train.py'), Path(__file__).with_name('model.py'),
             Path(__file__).with_name('training.py'), Path(base.__file__),
             Path('miss_rgbt/objectives.py')]
    result = dict(status='PASS', seed=65309,
        scope='Synthetic batch, actual E0 network; internal task-loss reporting only',
        checks=results, peak_gpu_bytes=torch.cuda.max_memory_allocated(),
        source_sha256={str(path.resolve()): sha(path) for path in paths},
        limitations=['No real images or learned F weights',
            'No closed-loop correction/destruction rates or official accuracy',
            'No F image-training epoch or checkpoint selection executed'])
    output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
