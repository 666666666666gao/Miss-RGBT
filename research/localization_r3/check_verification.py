"""Synthetic terminal records test the full R3 verifier, never real metrics."""
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from research.localization_r3 import check_training as training_checks
from research.localization_r3 import verify as verification
from research.protocol_r.common import read, save, sha


class VerificationChecks(unittest.TestCase):
    setUp = training_checks.TrainingChecks.setUp
    prepare = training_checks.TrainingChecks.prepare

    def test_unfinished_pair_does_not_require_or_read_a_manifest(self):
        self.prepare()
        result = verification.verify(self.campaign)
        self.assertEqual(result['status'], 'waiting')
        self.assertEqual(len(result['missing']), 2)
        self.assertFalse(self.runs.exists())

    def test_full_budget_fixture_and_changed_step(self):
        self.prepare()
        configs = [read(self.campaign / (arm + '.json')) for arm in verification.ARMS]
        roots = [Path(config['output']) for config in configs]
        for index, (root, config) in enumerate(zip(roots, configs)):
            root.mkdir(parents=True)
            save(root / 'run_config.json', config)
            save(root / 'initial_state.json', dict(scope='Synthetic initialization'))
            value = 1. + .2 * index
            history = [dict(epoch=epoch, updates=2500 * epoch, samples=128,
                task_loss={'11': value, '01': 2., '10': 3.},
                mean_task_loss=(value + 5.) / 3) for epoch in range(1, 21)]
            save(root / 'development_history.json', history)
            weights = []
            for epoch in (10, 16, 17, 18, 19, 20):
                path = root / ('epoch%02d.pth.tar' % epoch)
                path.write_bytes(b'fixture, not a checkpoint')
                if epoch != 10:
                    weights.append(dict(epoch=epoch, path=str(path), sha256=sha(path)))
            with (root / 'input_manifest.jsonl').open('w') as inputs, \
                    (root / 'steps.jsonl').open('w') as steps:
                for microstep in range(100000):
                    epoch = microstep // 5000 + 1
                    inputs.write(json.dumps(dict(epoch=epoch, index=microstep % 5000,
                        identities=[dict(sequence='fixture')] * 4,
                        tensors_sha256=dict(search='fixture'))) + '\n')
                    row = dict(arm=verification.ARMS[index], microstep=microstep,
                        epoch=epoch, model_forwards=index + 1,
                        mask_applied_to_model=bool(index), task_c=1.,
                        task_m=1. if index else None,
                        kd_forward=0. if index else None, loss=1. + index,
                        input_sha256=dict(search='fixture'),
                        candidate_masks_sha256=['fixture'],
                        seeds=dict(complete=74101 + 3 * microstep,
                            missing=74102 + 3 * microstep, masks=74103 + 3 * microstep))
                    if (microstep + 1) % 2 == 0:
                        row.update(update=(microstep + 1) // 2,
                            gradient_norm=.1, update_loss_mean=1. + index)
                    steps.write(json.dumps(row) + '\n')
            save(root / 'completed.json', dict(status='complete', epochs=20,
                updates=50000, early_stopped=False, all_checkpoints_retained=True,
                official_test_selection_used=False, selected_epoch=16,
                selection_rule=config['development_selection'],
                primary_development_epoch20_complete_loss=value, final_five=weights,
                input_manifest_sha256=sha(root / 'input_manifest.jsonl')))
        with patch.object(verification, 'check_evidence'):
            result = verification.verify(self.campaign)
        self.assertEqual(result['status'], 'passed')
        self.assertEqual(result['paired_accepted_tuples'], 400000)
        self.assertEqual(result['model_forwards']['complete_missing'], 200000)
        self.assertAlmostEqual(result['primary_development_contrast'], .2)
        self.assertFalse(result['dispatch_authorized'])
        with (roots[1] / 'steps.jsonl').open('a') as stream:
            stream.write('{}\n')
        with patch.object(verification, 'check_evidence'):
            with self.assertRaises(AssertionError):
                verification.verify(self.campaign)


if __name__ == '__main__':
    unittest.main()
