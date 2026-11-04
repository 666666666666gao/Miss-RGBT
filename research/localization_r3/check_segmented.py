"""Synthetic CPU equivalence checks; not native image or GPU-memory evidence."""
import copy
import unittest

import torch

from lib.config.cadtrack.config import cfg, update_config_from_file
from research.curriculum_p.check_engine import inputs
from research.localization_r3.objective import forward_objective
from research.localization_r3.segmented import backward_segmented


class Model(torch.nn.Module):
    def __init__(self, freeze_bn):
        super().__init__()
        self.linear = torch.nn.Linear(2, 4)
        self.bn = torch.nn.BatchNorm1d(4)
        self.train()
        if freeze_bn:
            self.bn.eval()

    def forward(self, templates, searches, availability=None, patch=None):
        outputs = []
        for index, image in enumerate(searches):
            values = image
            if availability is not None:
                values = values * availability[len(templates) + index]
            features = self.bn(self.linear(values))
            noisy = features + torch.rand_like(features) * .1
            score = torch.sigmoid(noisy[:, :1, None, None]).expand(-1, 1, 16, 16)
            boxes = (.2 + .3 * torch.sigmoid(noisy)).unsqueeze(1)
            outputs.append(dict(score_map=score, pred_boxes=boxes))
        return outputs


class SegmentedChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        update_config_from_file('experiments/cadtrack/cadtrack.yaml')

    def compare(self, freeze_bn, accumulation):
        torch.manual_seed(8191)
        joint = Model(freeze_bn)
        split = copy.deepcopy(joint)
        optimizers = [torch.optim.AdamW(model.parameters(), lr=1e-4)
                      for model in (joint, split)]
        data = inputs()
        outer = torch.get_rng_state().clone()
        for step in range(accumulation):
            loss, expected, _, _, masks, patches = forward_objective(
                joint, *data, cfg, 'complete_missing', step)
            (loss / accumulation).backward()
            actual_loss, actual, actual_masks, actual_patches = backward_segmented(
                split, *data, cfg, step, accumulation=accumulation)
            self.assertAlmostEqual(actual_loss, float(loss.detach()), places=4)
            for key in actual:
                self.assertAlmostEqual(expected[key], actual[key], places=5)
            for left, right in zip(masks + patches, actual_masks + actual_patches):
                self.assertTrue(torch.equal(left, right))
            self.assertTrue(torch.equal(outer, torch.get_rng_state()))
        for left, right in zip(joint.parameters(), split.parameters()):
            torch.testing.assert_close(left.grad, right.grad, rtol=2e-5, atol=2e-5)
        for left, right in zip(joint.buffers(), split.buffers()):
            self.assertTrue(torch.equal(left, right))
        for model, optimizer in zip((joint, split), optimizers):
            torch.nn.utils.clip_grad_norm_(model.parameters(), .1)
            optimizer.step()
        for left, right in zip(joint.parameters(), split.parameters()):
            torch.testing.assert_close(left, right, rtol=1e-6, atol=1e-7)
        for left, right in zip(optimizers[0].state.values(), optimizers[1].state.values()):
            for key in left:
                torch.testing.assert_close(left[key], right[key], rtol=2e-5, atol=2e-6)
        self.assertFalse(torch.cuda.is_initialized())

    def test_frozen_bn_single_microbatch(self):
        self.compare(True, 1)

    def test_frozen_bn_accumulation(self):
        self.compare(True, 2)

    def test_updating_bn_order(self):
        self.compare(False, 2)


if __name__ == '__main__':
    unittest.main()
