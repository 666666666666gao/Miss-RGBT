"""CPU-only synthetic contracts for P* selection, not real parent evidence."""
import argparse
from contextlib import ExitStack
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

import torch

from research.feedback_f import protocol_parent as parent
from research.protocol_r.common import read, save, sha


def fixture(root):
    campaign, queue = root / 'campaign', root / 'queue'
    campaign.mkdir()
    queue.mkdir()
    verifier = root / 'snapshot/research/curriculum_p/protocol_verify.py'
    verifier.parent.mkdir(parents=True)
    verifier.write_bytes((Path(parent.__file__).parents[1] /
                         'curriculum_p/protocol_verify.py').read_bytes())
    source = {str(verifier): sha(verifier)}
    save(queue / 'source_sha256.json', source)
    save(queue / 'completed.json', dict(status='complete', arms=list(parent.ARMS),
                                      full_evaluations=60, pairing_verified=True))
    paired = dict(status='passed', arms={}, pairing=dict(synthetic=True))
    for index, arm in enumerate(parent.ARMS):
        run = root / 'runs' / arm
        run.mkdir(parents=True)
        controller = campaign / (arm + '_formal_controller')
        controller.mkdir()
        manifest = root / 'sources.json'
        save(manifest, source)
        gate = root / 'gate.json'
        save(gate, dict(ready=True, protocol_arm='r1', parents_sha256=source))
        config = dict(curriculum_arm=arm, stage='A', init='ostrack', epochs=20,
            early_stop=False, checkpoint_epochs=[10, 16, 17, 18, 19, 20],
            freeze_all_bn=True, repair=False, task_utility=dict(selection='utility'),
            utility_probe_interval=4, utility_weight=.1, patch_mask_probability=.1,
            forward_weight=.1, reverse_weight=0., samples_per_epoch=20000, workers=2,
            microbatch=4, accumulation=2, warmup_updates=500,
            official_test_selection_used=False, resume_supported=False,
            development_selection='synthetic_internal_mean',
            output=str(run), snapshot=str(root / 'snapshot'),
            source_manifest=str(manifest), source_manifest_sha256=sha(manifest),
            gate=str(gate), gate_sha256=sha(gate),
            checkpoint=str(verifier), initializer_sha256=sha(verifier),
            development_cache=str(verifier), development_cache_sha256=sha(verifier))
        path = campaign / (arm + '.json')
        save(path, config)
        save(run / 'run_config.json', config)
        # Fixed50/progressive tie deliberately; registered arm order wins.
        loss = 1. if index in (1, 2) else 2. + index
        history = [dict(epoch=epoch, updates=epoch * 2500, samples=128,
            task_loss=dict.fromkeys(('11', '01', '10'), loss),
            mean_task_loss=loss, seconds=float(epoch)) for epoch in range(1, 21)]
        save(run / 'development_history.json', history)
        (run / 'input_manifest.jsonl').write_text('synthetic\n')
        checkpoints = []
        for epoch in range(16, 21):
            checkpoint = run / ('epoch%02d.pth.tar' % epoch)
            checkpoint.write_text('synthetic checkpoint %s %s' % (arm, epoch))
            checkpoints.append(dict(epoch=epoch, path=str(checkpoint),
                                    sha256=sha(checkpoint)))
        done = dict(status='complete', epochs=20, updates=50000, selected_epoch=16,
            early_stopped=False, official_test_selection_used=False,
            all_checkpoints_retained=True, resume_supported=False,
            selection_rule=config['development_selection'], final_five=checkpoints)
        save(run / 'completed.json', done)
        bound = {str(item): sha(item) for item in (
            path, run / 'completed.json', run / 'run_config.json',
            run / 'development_history.json', run / 'input_manifest.jsonl')}
        verified = dict(status='passed', arm=arm, completed_sha256=sha(run / 'completed.json'),
                        selected_epoch=16, evidence_sha256=bound,
                        verifier_sha256=sha(verifier))
        save(campaign / (arm + '_training_verified.json'), verified)
        paired['arms'][arm] = verified
        jobs = []
        for checkpoint in checkpoints:
            for condition in ('complete', 'miss'):
                output = root / 'results/rppm_ostrack_20261103' / arm / (
                    'epoch%02d_%s' % (checkpoint['epoch'], condition))
                output.mkdir(parents=True)
                # Distinct scores prevent accidental use of Complete/Miss maxima.
                metrics = dict(synthetic=index * 100 + checkpoint['epoch'] +
                               (1000 if condition == 'miss' else 0))
                for name in ('run.json', 'completed.json', 'metrics.json', 'all_receipts.json'):
                    save(output / name, metrics)
                jobs.append(dict(arm=arm, epoch=checkpoint['epoch'], condition=condition,
                    checkpoint=checkpoint['path'], checkpoint_sha256=checkpoint['sha256'],
                    development_selected=checkpoint['epoch'] == 16, output=str(output),
                    expected_sequences=479, expected_frames=337353,
                    independently_recomputed=True, metrics=metrics,
                    completed_sha256=sha(output / 'completed.json')))
        save(controller / 'official_sources.json', source)
        save(controller / 'official_completed.json', dict(status='complete',
            full_evaluations=10, sequences_per_evaluation=479, frames_per_evaluation=337353,
            selected_epoch=16, official_test_selection_used=False,
            all_checkpoints_retained=True, rows=jobs))
    save(campaign / 'paired_training_verified.json', paired)
    return campaign, queue, paired


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    tester = unittest.TestCase()
    rejected = []
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        waiting = parent.select_parent(root, root)
        assert not waiting['ready'] and not waiting['f_training_authorized']
        campaign, queue, paired = fixture(root)
        calls = []

        def official(destination, checkpoint, condition, expected):
            calls.append((str(destination), checkpoint, condition))
            assert read(destination / 'metrics.json') == expected
            return expected

        # Only the outer gate is exercised here; no claim of independent full
        # tracking/paired-step verification is made by these two explicit stubs.
        with ExitStack() as stack:
            stack.enter_context(patch.object(parent, 'ROOT', root))
            stack.enter_context(patch.object(parent, 'verify_pairs', return_value=paired['pairing']))
            stack.enter_context(patch.dict('sys.modules', {
                'verify_recovery_campaign28': types.SimpleNamespace(verify_official=official)}))
            result = parent.select_parent(campaign, queue)
            assert result['ready'] and not result['f_training_authorized']
            assert result['selected']['arm'] == 'fixed50'
            assert result['selected']['selected_epoch'] == 16
            assert result['selected']['official_metrics'] == dict(
                complete=dict(synthetic=116), miss=dict(synthetic=1116))
            assert len(calls) == 60
            path = campaign / 'fixed60_formal_controller/official_completed.json'
            original = read(path)
            mutations = [
                ('partial_scope', dict(original, frames_per_evaluation=1)),
                ('wrong_selected_epoch', dict(original, selected_epoch=17)),
                ('test_selected', dict(original, official_test_selection_used=True)),
                ('missing_evaluation', dict(original, rows=original['rows'][:-1])),
                ('duplicate_condition', dict(original, rows=[original['rows'][0]] * 10)),
            ]
            for name, key, value in (
                    ('wrong_checkpoint', 'checkpoint_sha256', 'changed'),
                    ('wrong_checkpoint_path', 'checkpoint', 'changed'),
                    ('wrong_selected_flag', 'development_selected', False),
                    ('unverified_scores', 'independently_recomputed', False),
                    ('wrong_output', 'output', 'changed'),
                    ('wrong_frame_budget', 'expected_frames', 1)):
                altered = json.loads(json.dumps(original))
                altered['rows'][0][key] = value
                mutations.append((name, altered))
            for name, altered in mutations:
                save(path, altered)
                with tester.assertRaises(AssertionError):
                    parent.select_parent(campaign, queue)
                rejected.append(name)
            save(path, original)
            names = root / 'runs/fixed60/input_manifest.jsonl'
            names.write_text('changed\n')
            with tester.assertRaises(AssertionError):
                parent.select_parent(campaign, queue)
            rejected.append('changed_training_evidence')
    assert not torch.cuda.is_initialized()
    report = dict(status='passed', positive='Six synthetic arms; sixty stubbed evaluations',
        selected_arm='fixed50', selected_epoch=16, official_metrics_used_for_selection=False,
        rejected=rejected, cuda_initialized=False, f_training_authorized=False,
        source_sha256={str(path.resolve()): sha(path) for path in
                       (Path(__file__), Path(parent.__file__))},
        scope='Synthetic gate contracts only. Official and paired verifiers stubbed; '
              'no actual P parent, training, images, GPU job or benchmark result.')
    output = Path(args.output)
    assert not output.exists()
    save(output, report)
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
