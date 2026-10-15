"""Synthetic CPU checks of module contracts, not tracking performance."""
import copy
import unittest

import torch
from torch.nn import functional as F

from miss_rgbt.task_utility import TaskUtilityRepair, utility_targets


class TaskUtilityTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(2027)
        torch.set_num_threads(1)
        self.rgb = torch.randn(2, 385, 8)
        self.tir = torch.randn(2, 385, 8)
        self.available = torch.ones(2, 385, 2)
        self.available[1, -256:, 0] = 0
        self.available[1, -16:] = 0

    def module(self, **kwargs):
        return TaskUtilityRepair(8, 16, 8, width=12, **kwargs)

    def test_zero_initialization_and_region_partition(self):
        module = self.module()
        rgb, tir, info = module(self.rgb, self.tir, self.available)
        self.assertTrue(torch.equal(rgb, self.rgb))
        self.assertTrue(torch.equal(tir, self.tir))
        self.assertTrue(torch.equal(module.region_tokens.flatten().sort().values, torch.arange(256)))
        self.assertEqual(tuple(info['utility_tokens'].shape), (2, 64))

    def test_only_selected_search_tokens_change_with_observed_refinement(self):
        module = self.module()
        with torch.no_grad():
            module.delta.bias.fill_(1)
        rgb, tir, info = module(self.rgb, self.tir, self.available)
        selected = torch.zeros(2, 385, dtype=torch.bool)
        selected.scatter_(1, info['utility_tokens'] + 129, True)
        self.assertTrue(torch.equal(rgb[~selected], self.rgb[~selected]))
        self.assertTrue(torch.equal(tir[~selected], self.tir[~selected]))
        self.assertTrue((rgb[selected] > self.rgb[selected]).all())
        self.assertTrue((tir[selected] > self.tir[selected]).all())
        self.assertTrue(torch.equal(rgb[:, :129], self.rgb[:, :129]))
        self.assertTrue(torch.equal(tir[:, :129], self.tir[:, :129]))
        self.assertTrue((rgb[0, -256:] != self.rgb[0, -256:]).any())

    def test_zero_refinement_limit_preserves_observed_positions(self):
        module = self.module(refinement_limit=0.)
        with torch.no_grad():
            module.delta.bias.fill_(1)
        rgb, tir, _ = module(self.rgb, self.tir, self.available)
        self.assertTrue(torch.equal(rgb[0], self.rgb[0]))
        self.assertTrue(torch.equal(tir[0], self.tir[0]))
        self.assertTrue((rgb[1] != self.rgb[1]).any())

    def test_correction_mlp_receives_actual_selected_budget(self):
        for mode in ['utility', 'similarity', 'random', 'all']:
            module = self.module(selection=mode)
            received = []
            hook = module.correction.register_forward_pre_hook(lambda _, args: received.append(args[0].shape))
            _, _, info = module(self.rgb, self.tir, self.available)
            hook.remove()
            count = 256 if mode == 'all' else 64
            self.assertEqual(tuple(received[0]), (2, count, 20))
            self.assertEqual(info['utility_tokens'].shape[1], count)
            for indices in info['utility_tokens']:
                self.assertEqual(indices.unique().numel(), count)

    def test_random_selection_owns_rng_and_strict_reload_restores_next_draw(self):
        module = self.module(selection='random')
        with torch.no_grad():
            module.delta.bias.fill_(1)
        initial = copy.deepcopy(module.state_dict())
        global_rng = torch.get_rng_state().clone()
        rgb, tir, first = module(self.rgb, self.tir, self.available)
        self.assertTrue(torch.equal(global_rng, torch.get_rng_state()))
        self.assertFalse(torch.equal(initial['selection_rng_state'], module.selection_rng_state))
        restored = self.module(selection='random')
        restored.load_state_dict(initial, strict=True)
        r2, t2, second = restored(self.rgb, self.tir, self.available)
        self.assertTrue(torch.equal(rgb, r2))
        self.assertTrue(torch.equal(tir, t2))
        self.assertTrue(torch.equal(first['utility_regions'], second['utility_regions']))
        module.reset_selection()
        _, _, reset = module(self.rgb, self.tir, self.available)
        self.assertTrue(torch.equal(first['utility_regions'], reset['utility_regions']))

    def test_signed_benefit_targets_stop_gradients(self):
        off = torch.tensor([2., 2., 0.], requires_grad=True)
        on = torch.tensor([1., 3., 0.], requires_grad=True)
        target = utility_targets(off, on)
        self.assertTrue(torch.equal(target, torch.tensor([.5, -.5, 0.])))
        self.assertFalse(target.requires_grad)

    def test_utility_auxiliary_does_not_reshape_backbone_context(self):
        module = self.module()
        rgb = self.rgb.clone().requires_grad_()
        tir = self.tir.clone().requires_grad_()
        _, _, info = module(rgb, tir, self.available)
        F.smooth_l1_loss(info['utility_logits'][:, 0], torch.tensor([.5, -.5])).backward()
        self.assertIsNone(rgb.grad)
        self.assertIsNone(tir.grad)
        self.assertIsNone(module.norm.weight.grad)
        self.assertIsNone(module.delta.weight.grad)
        self.assertGreater(float(module.selector[-1].weight.grad.abs().sum()), 0.)

    def test_real_adamw_updates_correction_and_supervised_selector(self):
        for depth in [1, 2]:
            module = self.module(depth=depth)
            optimizer = torch.optim.AdamW(module.parameters(), lr=1e-3)
            delta = module.delta.weight.detach().clone()
            selector = module.selector[-1].weight.detach().clone()
            rgb, tir, info = module(self.rgb, self.tir, self.available)
            target = torch.tensor([.5, -.5])
            loss = rgb.square().mean() + tir.square().mean() + F.smooth_l1_loss(info['utility_logits'][:, 0], target)
            loss.backward()
            self.assertGreater(float(module.delta.weight.grad.abs().sum()), 0.)
            self.assertGreater(float(module.selector[-1].weight.grad.abs().sum()), 0.)
            self.assertTrue(all(torch.isfinite(p.grad).all() for p in module.parameters() if p.grad is not None))
            optimizer.step()
            self.assertFalse(torch.equal(delta, module.delta.weight))
            self.assertFalse(torch.equal(selector, module.selector[-1].weight))


if __name__ == '__main__':
    unittest.main()
