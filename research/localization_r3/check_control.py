"""CPU controller fixtures: processes, GPUs, scores and review are synthetic."""
import json
from pathlib import Path
import unittest
from unittest.mock import patch, Mock

from research.localization_r3 import check_training as training_checks
from research.localization_r3 import control, collect as collection
from research.protocol_r.common import read, save, sha


class ControlChecks(unittest.TestCase):
    setUp = training_checks.TrainingChecks.setUp
    prepare = training_checks.TrainingChecks.prepare

    def config(self, arm='complete_only'):
        return read(self.campaign / (arm + '.json'))

    def review(self):
        path = self.campaign / 'fixture_review.json'
        save(path, dict(status='passed', blocking_issues=[],
            review_independence='same-family', acceptance_status='provisional',
            model='gpt-6-astra', reasoning_effort='max',
            scope='Synthetic fixture; no model reviewer was called'))
        return path

    def complete_training(self, config):
        root = Path(config['output'])
        root.mkdir(parents=True)
        save(root / 'run_config.json', config)
        save(root / 'initial_state.json', dict(scope='Synthetic initialization'))
        save(root / 'development_history.json', [
            dict(epoch=epoch, updates=2500 * epoch, samples=128,
                 task_loss={'11': 1., '01': 2., '10': 3.}, mean_task_loss=2.)
            for epoch in range(1, 21)])
        weights = []
        for epoch in (10, 16, 17, 18, 19, 20):
            path = root / ('epoch%02d.pth.tar' % epoch)
            path.write_bytes(b'fixture; not a trained checkpoint')
            if epoch != 10:
                weights.append(dict(epoch=epoch, path=str(path), sha256=sha(path)))
        (root / 'input_manifest.jsonl').write_text('synthetic closed manifest\n')
        with (root / 'steps.jsonl').open('w') as stream:
            for microstep in range(100000):
                row = dict(arm=config['arm'], microstep=microstep,
                           epoch=microstep // 5000 + 1, loss=1.)
                if (microstep + 1) % 2 == 0:
                    row.update(update=(microstep + 1) // 2, gradient_norm=.1)
                stream.write(json.dumps(row) + '\n')
        save(root / 'completed.json', dict(status='complete', epochs=20,
            updates=50000, early_stopped=False, all_checkpoints_retained=True,
            official_test_selection_used=False, selected_epoch=16,
            selection_rule=config['development_selection'],
            primary_development_epoch20_complete_loss=1., final_five=weights,
            input_manifest_sha256=sha(root / 'input_manifest.jsonl'),
            resume_supported=False))

    def test_absent_review_never_queries_gpu_or_starts_a_process(self):
        self.prepare()
        with patch.object(control, 'check_evidence'), \
                patch.object(control, 'gpu_ready') as gpu, \
                patch.object(control, 'run') as run:
            result = control.execute(self.config(), 'smoke', 2, self.campaign / 'missing')
        self.assertEqual(result['stage'], 'WAITING_FOR_INDEPENDENT_REVIEW')
        gpu.assert_not_called()
        run.assert_not_called()
        self.assertFalse((self.campaign / 'complete_only_smoke_launch.json').exists())

    def test_formal_waits_for_both_smokes_and_rejects_blocking_review(self):
        self.prepare()
        review = self.review()
        with patch.object(control, 'check_evidence'), \
                patch.object(control, 'check_smokes') as smokes:
            result = control.readiness(self.config(), 'train', review)
        self.assertEqual(result['stage'], 'WAITING_FOR_PAIRED_IMAGE_SMOKES')
        self.assertEqual(len(result['missing']), 2)
        smokes.assert_not_called()
        record = read(review)
        record['blocking_issues'] = ['Synthetic unresolved issue']
        save(review, record)
        with patch.object(control, 'check_evidence'):
            with self.assertRaises(AssertionError):
                control.readiness(self.config(), 'smoke', review)

    def test_own_terminal_training_scan_and_extra_step(self):
        self.prepare()
        config = self.config()
        self.complete_training(config)
        with patch.object(control, 'check_evidence'):
            result = control.verify_training(config)
        self.assertEqual(result['selected_epoch'], 16)
        path = Path(config['output']) / 'steps.jsonl'
        with path.open('a') as stream:
            stream.write(json.dumps(dict(arm=config['arm'], microstep=100000,
                                        epoch=21, loss=1.)) + '\n')
        with patch.object(control, 'check_evidence'):
            with self.assertRaises(AssertionError):
                control.verify_training(config)

    def test_two_controllers_and_collector_cover_exactly_twenty_full_jobs(self):
        self.prepare()
        review, results = self.review(), self.root / 'results'
        baseline = self.root / 'baseline.json'
        save(baseline, {'LasHeR': dict(PR=0., NPR=0., SR=0.),
                        'RGBT234': dict(MPR=0., MSR=0.)})
        for arm in control.ARMS:
            root = Path(self.config(arm)['output'] + '_smoke')
            root.mkdir(parents=True)
            save(root / 'smoke_verified.json', dict(status='passed', updates=4))

        def run(command, env, snapshot, log_path, status_path, stage, **extra):
            if stage == 'TRAIN':
                self.complete_training(read(command[command.index('--config') + 1]))
                return
            self.assertEqual(stage, 'OFFICIAL')
            self.assertNotIn('--limit', command)
            self.assertEqual(command[command.index('--workers-per-gpu') + 1], '4')
            self.assertIn(env['CUDA_VISIBLE_DEVICES'], ('2', '3'))
            destination = Path(command[command.index('--output') + 1])
            destination.mkdir(parents=True)
            condition = command[command.index('--condition') + 1]
            suffix = '-Miss' if condition == 'miss' else ''
            metrics = {
                'LasHeR' + suffix: dict(sequences=245, frames=220703, PR=1., NPR=1., SR=1.),
                'RGBT234' + suffix: dict(sequences=234, frames=116650, MPR=1., MSR=1.)}
            save(destination / 'metrics.json', metrics)
            save(destination / 'completed.json', dict(sequences=479, frames=337353))
            save(destination / 'run.json', dict(scope='Synthetic evaluation fixture'))
            save(destination / 'all_receipts.json', [])

        with patch.object(control, 'check_evidence'), \
                patch.object(control, 'check_smokes'), \
                patch.object(control, 'gpu_ready', return_value=dict(memory_mib=1)), \
                patch.object(control, 'run', side_effect=run) as process, \
                patch('verify_recovery_campaign28.verify_official',
                      side_effect=lambda output, checkpoint, condition, metrics: metrics) as score:
            for arm, gpu in zip(control.ARMS, (2, 3)):
                terminal = control.execute(self.config(arm), 'train', gpu, review, results)
                self.assertEqual(terminal['full_evaluations'], 10)
                jobs = terminal['rows']
                self.assertEqual([(job['epoch'], job['condition']) for job in jobs],
                    [(epoch, condition) for epoch in range(16, 21)
                     for condition in ('complete', 'miss')])
                with self.assertRaises(FileExistsError):
                    control.execute(self.config(arm), 'train', gpu, review, results)
            self.assertEqual(process.call_count, 22)
            self.assertEqual(score.call_count, 20)
            pair = dict(status='passed', phase='r3_training_pair',
                        selected_epochs=dict.fromkeys(control.ARMS, 16),
                        scope='Synthetic pair verification fixture')
            with patch.object(collection, 'verify_pair', return_value=pair), \
                    patch.object(collection, 'check_evidence'):
                report = collection.collect(self.campaign, results, baseline)
            self.assertEqual(score.call_count, 40)
        self.assertEqual(report['full_evaluations_recomputed'], 20)
        self.assertEqual(report['frames_verified'], 6747060)
        self.assertEqual(set(report['selected']), set(control.ARMS))
        for chosen in report['selected'].values():
            self.assertEqual(chosen['epoch'], 16)
            self.assertEqual(set(chosen['metrics']), {'complete', 'miss'})
        receipt = self.campaign / 'complete_only_official_completed.json'
        changed = read(receipt)
        changed['rows'][0]['condition'] = 'miss'
        save(receipt, changed)
        with patch.object(collection, 'verify_pair', return_value=pair), \
                patch.object(collection, 'check_evidence'):
            with self.assertRaises(AssertionError):
                collection.collect(self.campaign, results, baseline)

    def test_goal_assessment_uses_strict_thresholds_and_one_selected_weight(self):
        baseline = {'LasHeR': dict(PR=70., NPR=65., SR=55.),
                    'RGBT234': dict(MPR=85., MSR=60.)}
        complete = {'LasHeR': dict(PR=72., NPR=67., SR=57.),
                    'RGBT234': dict(MPR=87., MSR=62.)}
        miss = {'LasHeR-Miss': dict(PR=69.3, SR=54.4),
                'RGBT234-Miss': dict(MPR=88.1, MSR=64.3)}
        selected = {'complete_only': dict(epoch=16,
                        metrics=dict(complete=complete, miss=miss))}
        report = collection.assess_selected(selected, baseline)['complete_only']
        self.assertFalse(report['all_miss_targets_met'])
        self.assertTrue(all(value for values in report['complete_at_least2pp_reference'].values()
                            for value in values.values()))
        for dataset in miss.values():
            for key in dataset:
                dataset[key] += .01
        report = collection.assess_selected(selected, baseline)['complete_only']
        self.assertTrue(report['all_miss_targets_met'])

    def test_failed_child_records_failure_and_does_not_retry(self):
        self.prepare()
        child = Mock(returncode=1, pid=123)
        child.poll.return_value = 1
        status = self.campaign / 'failure_status.json'
        with patch.object(control.subprocess, 'Popen', return_value=child) as spawn, \
                patch.object(control.time, 'sleep') as sleep:
            with self.assertRaises(AssertionError):
                control.run(['synthetic-command'], {}, self.snapshot,
                    self.campaign / 'failure.log', status, 'TRAIN')
        spawn.assert_called_once()
        sleep.assert_not_called()
        self.assertEqual(read(status)['returncode'], 1)
        self.assertEqual(read(status)['stage'], 'TRAIN_EXITED')

    def test_collector_waits_before_any_pair_or_metric_work(self):
        self.prepare()
        with patch.object(collection, 'verify_pair') as pair:
            result = collection.collect(self.campaign)
        self.assertEqual(result['stage'], 'WAITING_FOR_TWENTY_FULL_RESULTS')
        pair.assert_not_called()


if __name__ == '__main__':
    unittest.main()
