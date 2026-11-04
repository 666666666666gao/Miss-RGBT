"""CPU fixture checks of actual R3 preparation and four-update trainer code."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from research.curriculum_p.check_engine import StochasticModel, inputs
from research.localization_r3 import prepare as preparation
from research.localization_r3 import train as trainer
from research.protocol_r.common import read, save, sha, tensor_sha


class TinyModel(StochasticModel):
    def __init__(self):
        super().__init__()
        self.bn = torch.nn.BatchNorm2d(1)

    def forward(self, *args):
        if self.training:
            outputs = super().forward(*args)
        else:
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(1)
                outputs = super().forward(*args)
        for output in outputs:
            output.update(size_map=output['score_map'], offset_map=output['score_map'])
        return outputs


class Loader:
    def __len__(self):
        return 5000

    def __iter__(self):
        templates, searches, boxes = inputs()
        for _ in range(8):
            yield dict(template_images=torch.stack(templates),
                search_images=torch.stack(searches), search_anno=boxes,
                frame_identity=torch.tensor([[index] * 4 for index in range(6)]))


class TrainingChecks(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.campaign = self.root / 'campaign'
        self.snapshot = self.root / 'snapshot'
        self.runs = self.root / 'runs'
        native = self.root / 'runtime/task_utility_20261018'
        for folder, name, text in [
                ('lib/train/data_specs', 'lasher_train.txt', 'sequence\n'),
                ('miss_rgbt', 'fixture.py', '# synthetic native fixture\n'),
                ('experiments/cadtrack', 'cadtrack.yaml', 'MODEL: {}\n')]:
            target = native / folder
            target.mkdir(parents=True, exist_ok=True)
            (target / name).write_text(text)
        self.initializer = self.root / 'initializer'
        self.initializer.write_bytes(b'fixture, not model weights')
        cache = self.root / 'cache'
        cache.write_bytes(b'fixture, not image tensors')
        report = self.root / 'r_report'
        report.mkdir()
        save(report / 'r1.json', dict(dataset='synthetic_dataset',
            development_cache=str(cache), development_cache_sha256=sha(cache)))
        self.audit = self.root / 'audit.json'
        save(self.audit, dict(status='passed', phase='formal_official_queue',
            official_evaluations_recomputed=20, sequences_per_evaluation=479,
            frames_per_evaluation=337353, all_checkpoints_retained=True,
            official_test_selection_used=False, evidence_sha256={},
            scope='Synthetic fixture; not an actual A audit'))
        for name, value in [('ROOT', self.root), ('REPORT', report),
                            ('INITIALIZER', self.initializer),
                            ('INITIALIZER_SHA', sha(self.initializer))]:
            mock = patch.object(preparation, name, value)
            mock.start()
            self.addCleanup(mock.stop)

    def prepare(self):
        return preparation.prepare(self.audit, self.campaign, self.snapshot, self.runs)

    def test_preparation_keeps_pair_budgets_and_does_not_launch(self):
        result = self.prepare()
        configs = [read(self.campaign / (arm + '.json')) for arm in preparation.ARMS]
        for config in configs:
            preparation.check_config(config)
        self.assertEqual(
            {k: v for k, v in configs[0].items() if k not in ('arm', 'output')},
            {k: v for k, v in configs[1].items() if k not in ('arm', 'output')})
        self.assertEqual(result['optimizer_updates_per_arm'], 50000)
        self.assertEqual(result['full_evaluations'], 20)
        self.assertFalse(result['dispatch_authorized'])
        self.assertFalse(self.runs.exists())
        with self.assertRaises(AssertionError):
            self.prepare()

    def test_missing_or_incomplete_a_audit_never_prepares(self):
        missing = preparation.prepare(self.root / 'missing.json',
            self.campaign, self.snapshot, self.runs)
        self.assertEqual(missing['status'], 'waiting')
        self.assertFalse(self.campaign.exists())
        data = read(self.audit)
        data['official_evaluations_recomputed'] = 19
        save(self.audit, data)
        with self.assertRaises(AssertionError):
            self.prepare()
        self.assertFalse(self.campaign.exists())

    def test_frozen_evidence_and_registered_config_are_checked(self):
        self.prepare()
        config = read(self.campaign / 'complete_only.json')
        own_file = str(self.snapshot / 'research/localization_r3/prepare.py')
        with patch.object(preparation, '__file__', own_file):
            preparation.check_evidence(config)
            changed = copy.deepcopy(config)
            changed['learning_rate'] *= 2
            with self.assertRaises(AssertionError):
                preparation.check_evidence(changed)
            source = self.snapshot / 'miss_rgbt/fixture.py'
            source.write_text('# changed synthetic fixture\n')
            with self.assertRaises(AssertionError):
                preparation.check_evidence(config)

    def test_actual_trainer_smoke_loop_pairs_inputs_and_masks_on_cpu(self):
        self.prepare()
        names = ['sequence'] + ['sequence%d' % index for index in range(881)]
        (self.snapshot / 'lib/train/data_specs/lasher_train.txt').write_text('\n'.join(names))
        models = []

        def build(_):
            torch.manual_seed(77)
            model = TinyModel()
            models.append(model)
            return model, dict(bn_sha256=trainer.bn_state(model),
                parameters_sha256={name: tensor_sha(value)
                                   for name, value in model.named_parameters()})

        for arm in preparation.ARMS:
            config = read(self.campaign / (arm + '.json'))
            with patch.object(trainer, 'check_evidence'), \
                    patch.object(trainer, 'build', build), \
                    patch.object(trainer, 'update_config_from_file'), \
                    patch.object(trainer, 'provenance_loader', return_value=(Loader(), names)), \
                    patch.object(torch.nn.Module, 'cuda', lambda self: self), \
                    patch.object(torch.Tensor, 'cuda', lambda self: self), \
                    patch.object(torch.cuda, 'max_memory_allocated', return_value=0):
                # Load the real CADTrack task-loss configuration, not fixture YAML.
                from lib.config.cadtrack.config import update_config_from_file
                update_config_from_file('experiments/cadtrack/cadtrack.yaml')
                trainer.train(config, max_steps=4)
            root = Path(config['output'] + '_smoke')
            receipt = read(root / 'smoke_verified.json')
            self.assertTrue(receipt['strict_reload_bitwise_equal'])
            self.assertTrue(receipt['bn_unchanged'])
            self.assertEqual(receipt['updates'], 4)
            saved = torch.load(str(root / 'smoke.pth.tar'), map_location='cpu')
            self.assertEqual(saved['microsteps'], 8)
            self.assertNotEqual(float(saved['model']['weight']), .25)
        self.assertEqual(len(models[0].calls), 10)
        self.assertEqual(len(models[1].calls), 18)
        with patch.object(trainer, 'check_evidence'):
            trainer.check_smokes(read(self.campaign / 'complete_only.json'))
        smoke = Path(str(self.runs / 'complete_missing') + '_smoke')
        with (smoke / 'input_manifest.jsonl').open('a') as stream:
            stream.write('{}\n')
        with patch.object(trainer, 'check_evidence'):
            with self.assertRaises(AssertionError):
                trainer.check_smokes(read(self.campaign / 'complete_only.json'))

    def test_config_rejects_objective_or_budget_drift(self):
        self.prepare()
        config = read(self.campaign / 'complete_missing.json')
        for key, value in [('arm', 'other'), ('microbatch', 8), ('epochs', 19),
                           ('freeze_all_bn', False), ('repair', True),
                           ('repair_width', 64), ('repair_layer', 3),
                           ('task_utility', {}), ('seed', 2027),
                           ('stochastic_seed', 1), ('early_stop', True)]:
            changed = dict(config)
            changed[key] = value
            with self.assertRaises(AssertionError):
                preparation.check_config(changed)

    @classmethod
    def tearDownClass(cls):
        assert not torch.cuda.is_initialized()


if __name__ == '__main__':
    unittest.main()
