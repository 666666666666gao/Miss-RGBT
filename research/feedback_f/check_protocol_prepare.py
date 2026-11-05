"""CPU preparation fixtures; P*/M1 decisions and source trees are synthetic."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from research.curriculum_p.engine import ARMS as CURRICULA
from research.feedback_f import protocol_prepare as preparation
from research.protocol_r.common import read, save, sha


class PreparationChecks(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.source = self.root / 'source'
        self.project = self.root / 'project'
        self.native = self.project / 'runtime/task_utility_20261018'
        self.campaign = self.root / 'campaign'
        self.snapshot = self.root / 'snapshot'
        self.runs = self.root / 'runs'
        self.weight = self.root / 'weight'
        self.weight.write_bytes(b'fixture initializer; not trained weights')
        self.cache = self.root / 'cache'
        self.cache.write_bytes(b'fixture development cache; not images')
        files = ['lib/train/data_specs/lasher_train.txt', 'miss_rgbt/model.py',
                 'experiments/cadtrack/cadtrack.yaml']
        for name in files:
            path = self.native / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('synthetic source\n')
        for folder in ('feedback_f', 'curriculum_p', 'protocol_r'):
            path = self.source / 'research' / folder / 'fixture.py'
            path.parent.mkdir(parents=True)
            path.write_text('synthetic source\n')
        self.names = self.native / files[0]
        self.parent_config = dict(seed=2027, dataset='/fixture/dataset',
            curriculum_arm='fixed50', init='ostrack', stage='A', epochs=20,
            early_stop=False, checkpoint_epochs=[10, 16, 17, 18, 19, 20],
            freeze_all_bn=True, repair=False,
            task_utility=dict(selection='utility', selection_seed=2027),
            utility_probe_interval=4, utility_weight=.1, patch_mask_probability=.1,
            forward_weight=.1, reverse_weight=0., samples_per_epoch=20000,
            workers=2, microbatch=4, accumulation=2, learning_rate=.0001,
            backbone_multiplier=.1, warmup_updates=500, lr_drop_epoch=10,
            clip_grad_norm=.1, official_test_selection_used=False, resume_supported=False,
            development_selection='fixture final-five internal mean',
            checkpoint=str(self.weight), initializer_sha256=sha(self.weight),
            development_cache=str(self.cache), development_cache_sha256=sha(self.cache),
            train_names_sha256=sha(self.names))
        self.config_path = self.root / 'parent.json'
        save(self.config_path, self.parent_config)
        self.parent = dict(ready=True, stage='P_PARENT_VERIFIED',
            selected=dict(arm='fixed50', config=str(self.config_path),
                          config_sha256=sha(self.config_path)),
            parents_sha256={str(self.config_path): sha(self.config_path)})
        self.m1 = dict(ready=True, stage='M1_DECISION_VERIFIED',
            m1_decision=preparation.CLOSED_M1, requires_ohcas_composition=False,
            parents_sha256={str(self.weight): sha(self.weight)})

    def gate(self, parent=None, m1=None):
        with patch.object(preparation, 'select_parent',
                          return_value=self.parent if parent is None else parent), \
                patch.object(preparation, 'verify_decision',
                             return_value=self.m1 if m1 is None else m1):
            return preparation.preparation_gate('report', 'receipt', 'config')

    def prepare(self):
        gate = self.gate()
        with patch.object(preparation, 'ROOT', self.project), \
                patch.object(preparation, '__file__',
                    str(self.source / 'research/feedback_f/protocol_prepare.py')):
            return preparation.prepare(gate, self.campaign, self.snapshot, self.runs)

    def verify(self, config):
        with patch.object(preparation, '__file__',
                          str(self.snapshot / 'research/feedback_f/protocol_prepare.py')):
            preparation.check_evidence(config)

    def test_waiting_parent_never_reads_m1_or_writes(self):
        with patch.object(preparation, 'select_parent',
                          return_value=dict(ready=False)), \
                patch.object(preparation, 'verify_decision') as m1:
            gate = preparation.preparation_gate('report', 'receipt', 'config')
            result = preparation.prepare(
                gate, self.campaign, self.snapshot, self.runs)
        self.assertEqual(result['stage'], 'WAITING_FOR_P')
        m1.assert_not_called()
        self.assertFalse(self.campaign.exists())
        self.assertFalse(result['f_training_authorized'])

    def test_m1_wait_retry_and_accepted_composition_cannot_prepare(self):
        for m1 in (dict(ready=False, stage='WAITING_FOR_M1'),
                   dict(ready=False, stage='M1_RETRY_REQUIRED'),
                   dict(self.m1, m1_decision='accepted_block3',
                        requires_ohcas_composition=True),
                   dict(self.m1, m1_decision='accepted_embedding',
                        requires_ohcas_composition=True)):
            with self.subTest(m1=m1):
                gate = self.gate(m1=m1)
                self.assertFalse(gate['ready'])
                self.assertFalse(gate['f_training_authorized'])
                preparation.prepare(gate, self.campaign, self.snapshot, self.runs)
                self.assertFalse(self.snapshot.exists())
                self.assertFalse(self.campaign.exists())

    def test_all_curricula_have_five_matched_native_configs(self):
        for arm in CURRICULA:
            gate = self.gate()
            gate['parent']['selected']['arm'] = arm
            parent = dict(self.parent_config, curriculum_arm=arm)
            original = copy.deepcopy(parent)
            configs = preparation.make_configs(
                parent, gate, self.campaign, self.snapshot, self.runs)
            self.assertEqual(parent, original)
            self.assertEqual([c['feedback']['mode'] for c in configs],
                             list(preparation.MODES))
            comparable = [{k: v for k, v in c.items() if k not in ('feedback', 'output')}
                          for c in configs]
            self.assertTrue(all(c == comparable[0] for c in comparable))
            self.assertEqual(configs[0]['checkpoint'], str(self.weight))
            self.assertEqual(configs[0]['seed'], 2030)
            self.assertEqual(configs[0]['mask_seed'], 65403)
            self.assertEqual(configs[0]['microbatch'] * configs[0]['accumulation'], 4)
            self.assertEqual(configs[0]['learning_rate'], 5e-5)
            self.assertEqual(configs[0]['development_cache'], str(self.cache))
            self.parent['selected']['arm'] = 'fixed50'

    def test_isolated_freeze_then_verify_every_config(self):
        result = self.prepare()
        self.assertEqual(result['status'], 'prepared_not_launched')
        self.assertEqual(result['expected_full_evaluations'], 50)
        self.assertEqual(result['updates_per_arm'], 100000)
        self.assertFalse(result['f_training_authorized'])
        self.assertFalse(self.runs.exists())
        self.assertEqual((self.snapshot / 'lib/train/data_specs/lasher_train.txt')
                         .read_bytes(), self.names.read_bytes())
        for mode in preparation.MODES:
            config = read(self.campaign / (mode + '.json'))
            self.verify(config)
            changed = dict(config, learning_rate=.0001)
            with self.assertRaises(AssertionError):
                self.verify(changed)

    def test_changed_parent_and_frozen_sources_are_rejected(self):
        self.prepare()
        config = read(self.campaign / 'deep.json')
        original = self.config_path.read_bytes()
        self.config_path.write_text('{}')
        with self.assertRaises(AssertionError):
            self.verify(config)
        self.config_path.write_bytes(original)
        (self.snapshot / 'miss_rgbt/model.py').write_text('changed')
        with self.assertRaises(AssertionError):
            self.verify(config)

    def test_existing_preparation_is_never_overwritten(self):
        self.prepare()
        before = sha(self.campaign / 'deep.json')
        with self.assertRaises(AssertionError):
            self.prepare()
        self.assertEqual(sha(self.campaign / 'deep.json'), before)


if __name__ == '__main__':
    unittest.main()
