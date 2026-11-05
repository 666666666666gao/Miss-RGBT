"""Synthetic history/step/round evidence contracts; no image-run claims."""
import argparse
import copy
import json
from pathlib import Path
import tempfile
import unittest

import verify
from prepare import sha


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    output = Path(args.output)
    assert not output.exists()
    test = unittest.TestCase()
    config = dict(feedback=dict(mode='deep', width=128), curriculum_arm='progressive',
        epochs=20, samples_per_epoch=8, microbatch=2, accumulation=2,
        development_samples=4, mask_seed=65303, learning_rate=5e-5,
        backbone_multiplier=.1, lr_drop_epoch=10)
    history = [dict(epoch=epoch, updates=epoch*2, samples=4,
        task_loss=dict.fromkeys(['11', '01', '10'], float(21-epoch)),
        mean_task_loss=float(21-epoch)) for epoch in range(1, 21)]
    completed = dict(status='complete', epochs=20, early_stopped=False,
        official_test_selection_used=False, all_checkpoints_retained=True,
        feedback=config['feedback'], curriculum_arm=config['curriculum_arm'],
        updates=40, selected_epoch=20, final_five=[dict(epoch=e) for e in range(16, 21)])
    assert verify.check_history(config, history, completed)['epoch'] == 20
    rejected = []
    for key, value in [('epochs', 19), ('selected_epoch', 16),
                       ('official_test_selection_used', True), ('updates', 39)]:
        changed = dict(completed, **{key: value})
        with test.assertRaises(AssertionError):
            verify.check_history(config, history, changed)
        rejected.append(key)
    status = dict(task_c=1., task_m=2., kd_forward=.5, kd_reverse=0.)
    for arm in ['single', 'none', 'shallow', 'deep']:
        local = dict(config, feedback=dict(mode=arm, width=128))
        for index in [1, 41, 77]:
            epoch = index//4+1
            factor = .1**((epoch-1)//10)
            step = dict(microstep=index, epoch=epoch, mask_seed=65303+index,
                availability_probability=verify.trainer.epoch_config(
                    config, epoch)['complete_rgb_missing_tir_missing_probability'],
                masks_sha256=['synthetic']*8, loss=3.,
                update=(index+1)//2, grad_norm=.1,
                learning_rates=[5e-6*factor, 5e-5*factor])
            step.update(status if arm == 'single' else dict(round1=status, round2=status))
            verify.check_step(local, index, step)
            changed = dict(step, mask_seed=0)
            with test.assertRaises(AssertionError):
                verify.check_step(local, index, changed)
            changed = copy.deepcopy(step)
            (changed if arm == 'single' else changed['round2'])['task_m'] = float('nan')
            with test.assertRaises(AssertionError):
                verify.check_step(local, index, changed)
    rejected += ['wrong_mask_seed', 'nonfinite_each_arm_task']
    with tempfile.TemporaryDirectory(prefix='miss_rgbt_f_evidence_') as directory:
        root = Path(directory)
        first, final = [1., 2., 3., 4.], [2., 1., 3., 2.]
        raw = {label: [dict(batch_index=0, round1=first, round2=final)]
               for label in ['11', '01', '10']}
        path = root/'development_round_samples_epoch01.json'
        path.write_text(json.dumps(raw))
        summary = dict(samples=4, round1_mean=2.5, round2_mean=2., mean_delta=-.5,
            lower_loss=2, higher_loss=1, equal_loss=1,
            lower_loss_fraction=.5, higher_loss_fraction=.25, equal_loss_fraction=.25)
        row = dict(epoch=1, task_loss=dict.fromkeys(raw, 2.),
                   round_comparison={key: dict(summary) for key in raw})
        verify.check_rounds(root, config, [row])
        row['round_comparison']['11']['lower_loss'] = 3
        with test.assertRaises(AssertionError):
            verify.check_rounds(root, config, [row])
        rejected.append('wrong_round_counts')
    report = dict(status='PASS', rejected=rejected,
        single_and_nested_round_status=True, lr_epoch_boundaries=True,
        source_sha256={str(path.resolve()): sha(path) for path in [
            Path(__file__), Path(verify.__file__)]},
        scope='Synthetic helper evidence only; no full saved-run/weight/image verification')
    output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
