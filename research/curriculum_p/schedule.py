"""Whole-modality missing-frequency controls; patch masking stays fixed."""
import random

ARMS = ('fixed60', 'fixed50', 'progressive', 'shuffled')
SHUFFLE_SEED = 65101


def schedule(arm):
    assert arm in ARMS
    if arm == 'fixed60':
        return [.6] * 20
    if arm == 'fixed50':
        return [.5] * 20
    values = [.2] * 5 + [.5] * 10 + [.8] * 5
    if arm == 'shuffled':
        random.Random(SHUFFLE_SEED).shuffle(values)
    return values


def epoch_config(config, epoch):
    assert 1 <= epoch <= 20
    probability = schedule(config['curriculum_arm'])[epoch - 1]
    return dict(config, complete_rgb_missing_tir_missing_probability=[
        1. - probability, probability / 2., probability / 2.],
        patch_mask_probability=.1)


def sample_masks(templates, searches, config, epoch):
    from miss_rgbt.train import sample_masks as native_sample_masks
    return native_sample_masks(templates, searches, epoch_config(config, epoch))
