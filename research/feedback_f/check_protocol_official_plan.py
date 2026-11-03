"""Check new F planning without datasets, torch, CUDA, or subprocesses."""
import copy
from pathlib import Path
import unittest

from research.feedback_f.protocol_official_plan import (
    CURRICULA, MODES, command_for, jobs_for,
)


class OfficialPlanTests(unittest.TestCase):
    def fixture(self, curriculum='paced+norm', mode='deep'):
        config = dict(init='ostrack', epochs=20, curriculum_arm=curriculum,
            feedback=dict(mode=mode, width=128), output='/fixture/run',
            development_selection='fixture internal task loss')
        completed = dict(status='complete', epochs=20, early_stopped=False,
            all_checkpoints_retained=True, official_test_selection_used=False,
            selection_rule=config['development_selection'], selected_epoch=18,
            final_five=[dict(epoch=epoch,
                path=str(Path(config['output']) / ('epoch%02d.pth.tar' % epoch)),
                sha256=('%064x' % epoch)) for epoch in range(16, 21)])
        return config, completed

    def test_all_curricula_modes_and_gpus(self):
        destinations = set()
        for curriculum in CURRICULA:
            for mode in MODES:
                config, completed = self.fixture(curriculum, mode)
                jobs = jobs_for(config, completed, '/fixture/results')
                self.assertEqual([(job['epoch'], job['condition']) for job in jobs],
                    [(epoch, condition) for epoch in range(16, 21)
                     for condition in ('complete', 'miss')])
                selected = [job for job in jobs if job['development_selected']]
                self.assertEqual(len(selected), 2)
                self.assertEqual(selected[0]['checkpoint'], selected[1]['checkpoint'])
                for job in jobs:
                    self.assertNotIn(job['output'], destinations)
                    destinations.add(job['output'])
                    for gpu in range(4):
                        command = command_for(job, gpu, '/fixture/snapshot', '/python')
                        self.assertEqual(command[2],
                            str(Path('/fixture/snapshot/research/feedback_f/evaluate.py')))
                        self.assertEqual(command[-4:],
                            ['--gpus', str(gpu), '--workers-per-gpu', '4'])
                        self.assertNotIn('--limit', command)
        self.assertEqual(len(destinations), 240)

    def test_invalid_training_or_checkpoint_evidence(self):
        config, completed = self.fixture()
        mutations = [('epochs', 19), ('status', 'running'), ('early_stopped', True),
            ('all_checkpoints_retained', False), ('official_test_selection_used', True),
            ('selection_rule', 'official maximum'), ('selected_epoch', 15)]
        for key, value in mutations:
            with self.subTest(key=key), self.assertRaises(AssertionError):
                jobs_for(config, dict(completed, **{key: value}), '/fixture/results')
        for field, value in [('epoch', 15), ('path', '/other/checkpoint'),
                             ('sha256', 'missing')]:
            altered = copy.deepcopy(completed)
            altered['final_five'][0][field] = value
            with self.subTest(field=field), self.assertRaises(AssertionError):
                jobs_for(config, altered, '/fixture/results')
        with self.assertRaises(AssertionError):
            jobs_for(dict(config, init='e0'), completed, '/fixture/results')
        job = jobs_for(config, completed, '/fixture/results')[0]
        with self.assertRaises(AssertionError):
            command_for(job, 4, '/snapshot', '/python')
        with self.assertRaises(AssertionError):
            command_for(dict(job, expected_sequences=1), 0, '/snapshot', '/python')


if __name__ == '__main__':
    unittest.main()
