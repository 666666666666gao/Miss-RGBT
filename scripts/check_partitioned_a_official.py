"""CPU outer-audit fixtures; full-dataset metric recomputation is mocked."""
import copy
from contextlib import ExitStack
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

from scripts import verify_partitioned_a_official as audit


class AuditChecks(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.queue, self.campaign = self.root / 'queue', self.root / 'campaign'
        self.queue.mkdir()
        self.campaign.mkdir()
        protocol = audit.protocol
        for key, value in dict(ROOT=self.root, QUEUE=self.queue, CAMPAIGN=self.campaign,
                               RESULTS=self.root / 'results').items():
            self.stack.enter_context(patch.object(protocol, key, value))
        self.read, self.save, self.sha = protocol.read, protocol.save, protocol.sha
        self.source = self.root / 'source.py'
        self.source.write_text('fixture source\n')
        self.save(self.queue / 'source_sha256.json',
                  {str(self.source): self.sha(self.source)})
        self.measured = {}
        self.scorer = Mock(side_effect=lambda path, digest, condition, expected:
                           self.measured[str(path)])
        self.stack.enter_context(patch.dict('sys.modules', {
            'verify_recovery_campaign28': types.SimpleNamespace(verify_official=self.scorer)}))
        trained = dict(status='passed', phase='formal', arms={})
        terminal = dict(status='complete', arms={}, official_evaluations=20,
            trajectories_verified=9580, frames_verified=6747060,
            official_test_selection_used=False, all_checkpoints_retained=True)
        for arm in ('utility', 'random'):
            root = self.root / 'runs/server28_partitioned_a_20261102_formal' / arm
            root.mkdir(parents=True)
            history = [dict(epoch=epoch, mean_task_loss=float(abs(epoch - 3)))
                       for epoch in range(1, 6)]
            self.save(root / 'development_history.json', history)
            weights = []
            for epoch in range(1, 6):
                path = root / ('epoch%02d.pth.tar' % epoch)
                path.write_bytes(('fixture %s %s' % (arm, epoch)).encode())
                weights.append(dict(epoch=epoch, path=str(path), sha256=self.sha(path)))
            done = dict(status='complete', epochs=5, final_five=weights,
                selected_epoch=3, selected_sha256=weights[2]['sha256'],
                all_checkpoints_retained=True, official_test_selection_used=False)
            self.save(root / 'completed.json', done)
            trained['arms'][arm] = dict(completed_sha256=self.sha(root / 'completed.json'))
            jobs = protocol.jobs_for(arm, done)
            self.save(self.queue / (arm + '_plan.json'), jobs)
            rows = []
            for job in jobs:
                output = Path(job['output'])
                output.mkdir(parents=True)
                metrics = dict(synthetic=job['epoch'], condition=job['condition'])
                self.measured[str(output)] = metrics
                for name in ('run.json', 'completed.json', 'metrics.json', 'all_receipts.json'):
                    self.save(output / name, metrics)
                name = '%s_epoch%02d_%s_exit.json' % (arm, job['epoch'], job['condition'])
                self.save(self.queue / name, dict(returncode=0))
                rows.append(dict(job, metrics=metrics, independently_recomputed=True))
            child = dict(status='complete', metrics=rows, selected_epoch=3,
                selected_sha256=weights[2]['sha256'], all_checkpoints_retained=True,
                official_test_selection_used=False)
            self.save(self.queue / (arm + '_completed.json'), child)
            self.save(self.queue / (arm + '_verified_metrics.json'), rows)
            terminal['arms'][arm] = child
        self.save(self.campaign / 'verification_v2.json', trained)
        self.save(self.queue / 'completed.json', terminal)

    def save_child(self, arm, child):
        self.save(self.queue / (arm + '_completed.json'), child)
        self.save(self.queue / (arm + '_verified_metrics.json'), child['metrics'])
        terminal = self.read(self.queue / 'completed.json')
        terminal['arms'][arm] = child
        self.save(self.queue / 'completed.json', terminal)

    def test_terminal_audit_recomputes_twenty_and_preserves_selection(self):
        result = audit.verify()
        self.assertEqual(result['status'], 'passed')
        self.assertEqual(result['phase'], 'formal_official_queue')
        self.assertEqual(self.scorer.call_count, 20)
        self.assertEqual(result['selected']['utility']['epoch'], 3)
        self.assertEqual(result['selected']['random']['epoch'], 3)
        self.assertFalse(result['dispatch_authorized'])
        self.assertFalse((self.queue / 'verification_v2.json').exists())

    def test_missing_terminal_waits_without_scoring(self):
        (self.queue / 'completed.json').unlink()
        result = audit.verify()
        self.assertEqual(result['status'], 'waiting')
        self.scorer.assert_not_called()

    def test_changed_jobs_scores_selection_and_exit_are_rejected(self):
        path = self.queue / 'utility_completed.json'
        original = self.read(path)
        alterations = [
            lambda c: c.update(selected_epoch=4),
            lambda c: c['metrics'][0].update(expected_frames=1),
            lambda c: c['metrics'][0].update(development_selected=True),
            lambda c: c['metrics'][0].update(metrics=dict(changed=True)),
            lambda c: c['metrics'].pop(),
            lambda c: c.update(official_test_selection_used=True),
        ]
        for index, change in enumerate(alterations):
            with self.subTest(index=index):
                child = copy.deepcopy(original)
                change(child)
                self.save_child('utility', child)
                with self.assertRaises(AssertionError):
                    audit.verify()
        self.save_child('utility', original)
        self.save(self.queue / 'utility_epoch01_complete_exit.json', dict(returncode=1))
        with self.assertRaises(AssertionError):
            audit.verify()

    def test_changed_sources_and_training_completion_are_rejected(self):
        self.source.write_text('changed')
        with self.assertRaises(AssertionError):
            audit.verify()
        self.source.write_text('fixture source\n')
        path = self.root / 'runs/server28_partitioned_a_20261102_formal/utility/completed.json'
        done = self.read(path)
        done['epochs'] = 6
        self.save(path, done)
        with self.assertRaises(AssertionError):
            audit.verify()

    def test_missing_retained_checkpoint_is_rejected(self):
        path = self.root / 'runs/server28_partitioned_a_20261102_formal/random/epoch01.pth.tar'
        path.unlink()
        with self.assertRaises(AssertionError):
            audit.verify()


if __name__ == '__main__':
    unittest.main()
