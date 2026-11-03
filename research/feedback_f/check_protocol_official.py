"""CPU orchestration fixtures; no real evaluator, image, model or GPU is run."""
from contextlib import ExitStack
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from research.feedback_f import protocol_official as official
from research.feedback_f.check_protocol_official_plan import OfficialPlanTests
from research.protocol_r.common import read, save, sha


class OfficialControllerTests(unittest.TestCase):
    def exercise(self, root, failure=None):
        config, done = OfficialPlanTests().fixture()
        output = root / 'run'
        output.mkdir()
        config.update(output=str(output),
                      snapshot=str(Path(official.__file__).resolve().parents[2]))
        for row in done['final_five']:
            weight = output / ('epoch%02d.pth.tar' % row['epoch'])
            weight.write_bytes(('fixture weight %s' % row['epoch']).encode())
            row.update(path=str(weight), sha256=sha(weight))
        save(output / 'completed.json', done)
        source = root / 'frozen.py'
        source.write_bytes(b'fixture source')
        verified = dict(status='passed', selected_epoch=18,
            completed_sha256=sha(output / 'completed.json'),
            evidence_sha256={str(output / 'completed.json'): sha(output / 'completed.json')},
            checkpoints_sha256={row['path']: row['sha256'] for row in done['final_five']})
        queue = root / 'controller'
        calls, guards = [], []

        def guard(local):
            self.assertEqual(local, config)
            guards.append(True)

        def training(local, parents, smokes):
            parents(local)
            smokes(local)
            return verified

        def child(command, environment, snapshot, log, status, job):
            calls.append(job)
            self.assertEqual(environment['CUDA_VISIBLE_DEVICES'], '3')
            self.assertEqual(command[-4:], ['--gpus', '3', '--workers-per-gpu', '4'])
            if failure == 'child' and len(calls) == 3:
                raise AssertionError('fixture evaluator failure')
            if failure == 'source':
                source.write_bytes(b'altered source')
            destination = Path(job['output'])
            destination.mkdir(parents=True)
            save(destination / 'completed.json', dict(sequences=479, frames=337353))

        def verify(destination, digest, condition):
            self.assertEqual(digest, sha(calls[-1]['checkpoint']))
            self.assertEqual(condition, calls[-1]['condition'])
            return dict(fixture_only=True)

        with ExitStack() as stack:
            stack.enter_context(patch.object(official, 'verify_run', side_effect=training))
            stack.enter_context(patch.object(official, 'official_sources',
                return_value={str(source): sha(source)}))
            gpu = stack.enter_context(patch.object(official, 'gpu_ready',
                return_value=dict(memory_mib=0, temperature_c=40, power_limit_w=250)))
            stack.enter_context(patch.object(official, 'run_child', side_effect=child))
            scorer = stack.enter_context(patch(
                'research.feedback_f.verify_official.verify', side_effect=verify))
            if failure:
                with self.assertRaises(AssertionError):
                    official.run(config, 3, root / 'results', queue, guard, guard)
                self.assertFalse((queue / 'official_completed.json').exists())
                self.assertEqual(len(calls), 3 if failure == 'child' else 1)
                self.assertEqual(scorer.call_count, 2 if failure == 'child' else 1)
            else:
                result = official.run(config, 3, root / 'results', queue, guard, guard)
                self.assertEqual(len(calls), 10)
                self.assertEqual(gpu.call_count, 10)
                self.assertEqual(scorer.call_count, 10)
                self.assertEqual(len(guards), 22)
                self.assertEqual(result, read(queue / 'official_completed.json'))
                self.assertFalse(result['four_arm_pairing_verified'])
                self.assertEqual(read(queue / 'status.json')['stage'], 'COMPLETE')
                self.assertEqual(len([job for job in calls if job['development_selected']]), 2)
                self.assertTrue(all(sha(path) == digest for path, digest in
                    read(queue / 'official_sources.json').items()))

    def test_full_arm(self):
        with tempfile.TemporaryDirectory() as directory:
            self.exercise(Path(directory))

    def test_failure_prevents_remaining_dispatch(self):
        for failure in ('child', 'source'):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as directory:
                self.exercise(Path(directory), failure)

    def test_nonzero_child_exit_is_recorded(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.object(official.subprocess, 'Popen') as popen:
                popen.return_value.pid = 123
                popen.return_value.poll.return_value = 7
                popen.return_value.returncode = 7
                with self.assertRaises(AssertionError):
                    official.run_child(['fixture'], {}, root, root / 'job.log',
                                       root / 'status.json', dict(fixture=True))
            self.assertEqual(read(root / 'job.exit.json')['returncode'], 7)
            self.assertEqual(read(root / 'status.json')['returncode'], 7)

    def test_native_runtime_must_match_training_snapshot(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runtime = root / 'runtime/task_utility_20261018'
            snapshot = root / 'snapshot'
            for folder, name in [('lib', 'model.py'), ('miss_rgbt', 'model.py'),
                                 ('experiments/cadtrack', 'cadtrack.yaml')]:
                for base in (runtime, snapshot):
                    path = base / folder / name
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(b'matched fixture')
            for path in [root / 'evaluate_candidate_official28.py',
                         runtime / 'reports/e0/inventory.json',
                         root / 'manifest.json'] + [
                    snapshot / 'research/feedback_f' / name for name in
                    ('evaluate.py', 'verify_official.py', 'tracker.py', 'model.py')]:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b'fixture source')
            config = dict(snapshot=str(snapshot), source_manifest=str(root / 'manifest.json'))
            with patch.object(official, 'ROOT', root):
                sources = official.official_sources(config)
                self.assertTrue(all(sha(path) == digest for path, digest in sources.items()))
                (runtime / 'experiments/cadtrack/cadtrack.yaml').write_bytes(b'changed')
                with self.assertRaises(AssertionError):
                    official.official_sources(config)


if __name__ == '__main__':
    unittest.main()
