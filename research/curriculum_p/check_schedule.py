"""Synthetic mask checks only; no dataset, model training or benchmark access."""
import argparse
import json
import random
from pathlib import Path

import torch

from schedule import ARMS, epoch_config, sample_masks, schedule
from miss_rgbt.train import sample_masks as native_sample_masks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    random.seed(65102)
    before = random.getstate()
    for arm in ARMS:
        assert len(schedule(arm)) == 20
        assert all(0. <= p <= 1. for p in schedule(arm))
    assert random.getstate() == before
    expected = [.2] * 5 + [.5] * 10 + [.8] * 5
    assert schedule('progressive') == expected
    assert sorted(schedule('shuffled')) == sorted(expected)
    assert schedule('shuffled') != expected
    assert abs(sum(expected) / 20. - .5) < 1e-12
    assert schedule('shuffled') == schedule('shuffled')
    templates = [torch.empty(64, 1), torch.empty(64, 1)]
    searches = [torch.empty(64, 1) for _ in range(3)]
    observations = {}
    for epoch in [1, 5, 6, 15, 16, 20]:
        paired_patches = []
        paired_states = []
        for arm in ARMS:
            config = dict(curriculum_arm=arm)
            local = epoch_config(config, epoch)
            assert config == dict(curriculum_arm=arm)
            assert abs(sum(local['complete_rgb_missing_tir_missing_probability']) - 1.) < 1e-12
            assert local['patch_mask_probability'] == .1
            torch.manual_seed(65103)
            actual = sample_masks(templates, searches, config, epoch)
            after = torch.get_rng_state().clone()
            torch.manual_seed(65103)
            reference = native_sample_masks(templates, searches, local)
            assert torch.equal(after, torch.get_rng_state())
            assert all(torch.equal(a, b) for aa, bb in zip(actual, reference)
                       for a, b in zip(aa, bb))
            available, patch = actual
            assert len(available) == 5 and len(patch) == 3
            assert torch.equal(available[0], torch.ones(64, 2))
            assert all((value.sum(-1) >= 1.).all() for value in available)
            assert all(value.shape == (64, 256, 2) for value in patch)
            assert all(((value == 0.) | (value == 1.)).all() for value in patch)
            paired_patches.append(patch)
            paired_states.append(after)
        assert all(torch.equal(paired_states[0], state) for state in paired_states)
        assert all(torch.equal(a, b) for patches in paired_patches[1:]
                   for a, b in zip(paired_patches[0], patches))
    for arm in ARMS:
        config = dict(curriculum_arm=arm)
        observations[arm] = schedule(arm)
        for epoch in range(1, 21):
            local = epoch_config(config, epoch)
            assert abs(1. - local['complete_rgb_missing_tir_missing_probability'][0]
                       - schedule(arm)[epoch - 1]) < 1e-12
    result = dict(status='PASS', scope='Synthetic CPU mask checks; no training',
                  schedules=observations, shared_patch_masks=True,
                  native_sampler_parity=True, shared_rng_consumption=True)
    with Path(args.output).open('x') as stream:
        stream.write(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
