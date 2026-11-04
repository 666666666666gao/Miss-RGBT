"""CPU stochastic/gradient checks of R3 objectives, with a synthetic tiny model."""
import copy
import unittest

import torch

from lib.config.cadtrack.config import cfg, update_config_from_file
from miss_rgbt.objectives import task_loss, training_loss
from research.curriculum_p.check_engine import StochasticModel, inputs
from research.localization_r3.objective import forward_objective


class ObjectiveChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        update_config_from_file('experiments/cadtrack/cadtrack.yaml')

    @classmethod
    def tearDownClass(cls):
        assert not torch.cuda.is_initialized()

    def test_c_only_has_one_forward_and_native_c_loss_and_gradient(self):
        templates, searches, boxes = inputs()
        model = StochasticModel()
        reference = copy.deepcopy(model)
        actual = forward_objective(
            model, templates, searches, boxes, cfg, 'complete_only', 3)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(74101 + 3 * 3)
            expected_outputs = reference(templates, searches)
        expected = task_loss(expected_outputs, boxes, cfg).mean()
        self.assertEqual(actual[0].item(), expected.item())
        actual[0].backward()
        expected.backward()
        self.assertTrue(torch.equal(model.weight.grad, reference.weight.grad))
        self.assertEqual(len(model.calls), 1)
        self.assertIsNone(actual[3])
        self.assertFalse(actual[1]['mask_applied_to_model'])
        self.assertIsNone(actual[1]['task_m'])

    def test_c_plus_m_is_exact_existing_native_objective(self):
        templates, searches, boxes = inputs()
        model = StochasticModel()
        loss, status, complete, missing, available, patch = forward_objective(
            model, templates, searches, boxes, cfg, 'complete_missing', 2)
        expected, expected_status = training_loss(complete, missing, boxes, cfg, .1, 0., .1)
        self.assertEqual(loss.item(), expected.item())
        for name, value in expected_status.items():
            self.assertEqual(status[name], value)
        self.assertEqual(len(model.calls), 2)
        self.assertTrue(model.calls[0]['complete'])
        self.assertFalse(model.calls[1]['complete'])
        self.assertTrue(status['mask_applied_to_model'])
        loss.backward()
        self.assertTrue(torch.isfinite(model.weight.grad))

    def test_pairing_and_outer_rng_hold_across_repeated_microsteps(self):
        templates, searches, boxes = inputs()
        models = [StochasticModel(), StochasticModel()]
        torch.manual_seed(74901)
        before = torch.get_rng_state().clone()
        seeds = set()
        for index in range(8):
            left = forward_objective(
                models[0], templates, searches, boxes, cfg, 'complete_only', index)
            right = forward_objective(
                models[1], templates, searches, boxes, cfg, 'complete_missing', index)
            self.assertEqual(left[1]['seeds'], right[1]['seeds'])
            seeds.update(left[1]['seeds'].values())
            for first, second in zip(left[2], right[2]):
                for key in ('score_map', 'pred_boxes'):
                    self.assertTrue(torch.equal(first[key], second[key]))
            for first, second in zip(left[4] + left[5], right[4] + right[5]):
                self.assertTrue(torch.equal(first, second))
            self.assertTrue(torch.equal(before, torch.get_rng_state()))
        self.assertEqual(len(models[0].calls), 8)
        self.assertEqual(len(models[1].calls), 16)
        self.assertEqual(len(seeds), 24)


if __name__ == '__main__':
    unittest.main()
