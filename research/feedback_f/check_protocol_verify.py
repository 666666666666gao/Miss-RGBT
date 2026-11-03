"""CPU fixtures for F saved-evidence checks, not a formal-run verification."""
import argparse
import copy
from pathlib import Path
import tempfile
import unittest

import torch

from lib.config.cadtrack.config import cfg, update_config_from_file
from research.curriculum_p.check_engine import inputs
from research.curriculum_p.engine import ARMS as CURRICULA, CurriculumState
from research.feedback_f.check_protocol_runner import ProbeModel, optimizer
from research.feedback_f.check_protocol_training import config
from research.feedback_f.check_protocol_validation import DiagnosticModel
from research.feedback_f.protocol_runner import step
from research.feedback_f.protocol_training import ARMS
from research.feedback_f.protocol_validation import validate
from research.feedback_f.protocol_verify import check_diagnostic, check_feedback, check_step
from research.protocol_r.common import save, sha, tensor_sha


def check_steps():
    templates, searches, boxes = inputs()
    data = dict(template_images=torch.stack([value[:2] for value in templates]),
                search_images=torch.stack([value[:2] for value in searches]),
                search_anno=boxes[:, :2])
    tester, results, rejected = unittest.TestCase(), [], []
    for curriculum in CURRICULA:
        for mode in ARMS:
            local = dict(config(curriculum, mode), accumulation=2, microbatch=2,
                samples_per_epoch=20000, learning_rate=5e-5, backbone_multiplier=.1,
                utility_probe_interval=4, utility_weight=.1, warmup_updates=500,
                lr_drop_epoch=10, clip_grad_norm=.1)
            model = ProbeModel(mode)
            opt = optimizer(model)
            state = CurriculumState(curriculum, 'cpu', 65502)
            generator = torch.Generator().manual_seed(65503)
            updates = 0
            for index in range(24):
                row, updates = step(model, opt, local, 1, index,
                                     data, state, generator, updates)
                if 'update' in row:
                    row.update(peak_gpu_bytes=1, seconds=float(index))
                check_step(local, index, row)
                mutations = [
                    ('wrong_mode', dict(row, feedback_mode='other')),
                    ('wrong_round_count', dict(row, rounds=row['rounds'] + row['rounds'])),
                    ('wrong_total_loss', dict(row, loss=row['loss'] + 1.)),
                ]
                bad = copy.deepcopy(row)
                bad['rounds'][0]['kd_forward'] = float('nan')
                mutations.append(('nonfinite_round', bad))
                if curriculum == 'paced+norm':
                    mutations.append(('wrong_normalization_count',
                        dict(row, normalization_observations=1)))
                for name, altered in mutations:
                    with tester.assertRaises(AssertionError):
                        check_step(local, index, altered)
                    rejected.append(name)
            results.append(dict(curriculum=curriculum, feedback_mode=mode,
                                microsteps=24, updates=updates))
    return results, sorted(set(rejected))


def summary_for(raw):
    return dict({key: value for key, value in raw.items() if key != 'round_samples'},
                diagnostic_path='synthetic', diagnostic_sha256='synthetic')


def check_diagnostics():
    boxes = torch.tensor([[[.375, .375, .25, .25]] * 64] * 3)
    data = dict(template_images=torch.ones(2, 64, 1),
                search_images=torch.ones(3, 64, 1), search_anno=boxes)
    tester, results, rejected = unittest.TestCase(), [], []
    for mode in ARMS:
        local = dict(feedback=dict(mode=mode), development_samples=128)
        raw = validate(DiagnosticModel(mode), [data, data], cfg, mode)
        check_diagnostic(local, summary_for(raw), raw)
        # JSON serialization must preserve all comparisons used by the verifier.
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'diagnostic.json'
            save(path, raw)
            from research.protocol_r.common import read
            loaded = read(path)
            check_diagnostic(local, summary_for(loaded), loaded)
        mutations = []
        bad_summary = summary_for(raw)
        bad_summary['mean_task_loss'] += 1
        with tester.assertRaises(AssertionError):
            check_diagnostic(local, bad_summary, raw)
        rejected.append('summary_sidecar_mismatch')
        if mode != 'single':
            for name in ('incorrect_crossing_count', 'wrong_loss_mean', 'invalid_iou',
                         'missing_search', 'missing_tuple', 'wrong_batch_index'):
                bad = copy.deepcopy(raw)
                row = bad['round_samples']['01'][0]
                if name == 'incorrect_crossing_count':
                    bad['round_comparison']['01']['corrected'] += 1
                elif name == 'wrong_loss_mean':
                    row['round2_task_loss'][0] += 128
                elif name == 'invalid_iou':
                    row['round1_iou'][0][0] = 2.
                elif name == 'missing_search':
                    row['round2_iou'].pop()
                elif name == 'missing_tuple':
                    row['round1_task_loss'].pop()
                else:
                    row['batch_index'] = 3
                mutations.append((name, bad))
        else:
            mutations.append(('single_with_round_diagnostics',
                              dict(raw, round_samples={})))
        for name, bad in mutations:
            with tester.assertRaises(AssertionError):
                check_diagnostic(local, summary_for(bad), bad)
            rejected.append(name)
        results.append(mode)
    return results, sorted(set(rejected))


def check_weights():
    tester, results = unittest.TestCase(), []
    for mode in ARMS:
        model = {'baseline.weight': torch.ones(2)}
        initial = {}
        if mode in ('shallow', 'deep'):
            value = torch.zeros(2, 3)
            initial['feedback.delta.weight'] = dict(shape=[2, 3], sha256=tensor_sha(value))
            model['feedback.delta.weight'] = value + 1.
        local = dict(feedback=dict(mode=mode))
        changed = check_feedback(dict(model=model), local, initial)
        assert bool(changed) == (mode in ('shallow', 'deep'))
        if initial:
            with tester.assertRaises(AssertionError):
                check_feedback(dict(model={'feedback.delta.weight': torch.zeros(2, 3)}),
                               local, initial)
            with tester.assertRaises(AssertionError):
                check_feedback(dict(model={'feedback.delta.weight': torch.ones(3, 2)}),
                               local, initial)
        results.append(mode)
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    assert not Path(args.output).exists() and not torch.cuda.is_initialized()
    torch.set_num_threads(1)
    update_config_from_file('experiments/cadtrack/cadtrack.yaml')
    steps, rejected_steps = check_steps()
    diagnostics, rejected_diagnostics = check_diagnostics()
    weights = check_weights()
    assert not torch.cuda.is_initialized()
    sources = [Path(__file__), Path(__file__).with_name('protocol_verify.py'),
        Path(__file__).with_name('protocol_runner.py'),
        Path(__file__).with_name('protocol_training.py'),
        Path(__file__).with_name('protocol_validation.py'),
        Path(__file__).parents[1] / 'curriculum_p/protocol_verify.py']
    save(args.output, dict(status='PASS', cases=steps, diagnostic_modes=diagnostics,
        feedback_weight_fixture_modes=weights, rejected_steps=rejected_steps,
        rejected_diagnostics=rejected_diagnostics, cuda_initialized=False,
        source_sha256={str(path.resolve()): sha(path) for path in sources},
        scope='CPU synthetic F evidence checks only; full-run orchestration, '
              'real saved checkpoints, native optimizer-state audit, parent/smoke gates, '
              'paired full runs and official scores not exercised'))
    print('PASS: 24 mode/curriculum cases, 576 steps, four diagnostic/weight fixture modes')


if __name__ == '__main__':
    main()
