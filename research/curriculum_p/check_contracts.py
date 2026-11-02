"""Exercise P verifier rejection paths on synthetic records, never weights."""
import argparse
import copy
import json
from pathlib import Path
import unittest

from prepare import sha
from schedule import epoch_config, schedule
from verify import check_history, check_pair, check_step
from official import jobs_for
from partitioned_a_official_queue_v2 import command_for


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    tester = unittest.TestCase()
    config = dict(curriculum_arm='progressive', epochs=20, samples_per_epoch=20,
        microbatch=5, accumulation=2, development_samples=8, learning_rate=.000125,
        backbone_multiplier=.1, lr_drop_epoch=10)
    history = [dict(epoch=epoch, samples=8, updates=epoch*2,
        missing_probability=schedule('progressive')[epoch-1],
        task_loss=dict.fromkeys(['11', '01', '10'], 1. if epoch < 16 else 2.),
        mean_task_loss=1. if epoch < 16 else 2.) for epoch in range(1, 21)]
    completed = dict(epochs=20, updates=40, early_stopped=False, all_checkpoints_retained=True,
        official_test_selection_used=False, curriculum_arm='progressive',
        schedule=schedule('progressive'), selected_epoch=16,
        final_five=[dict(epoch=epoch) for epoch in range(16, 21)])
    assert check_history(config, history, completed)['epoch'] == 16
    rejected = []
    for kind in ['partial_course', 'early_stopped', 'selection_outside_final5', 'nonfinite_loss',
                 'wrong_probability', 'wrong_final5', 'wrong_updates']:
        rows, terminal = copy.deepcopy(history), copy.deepcopy(completed)
        if kind == 'partial_course':
            rows = rows[:14]
            terminal['epochs'] = 14
        elif kind == 'early_stopped':
            terminal['early_stopped'] = True
        elif kind == 'selection_outside_final5':
            terminal['selected_epoch'] = 1
        elif kind == 'nonfinite_loss':
            rows[0]['task_loss']['11'] = float('nan')
        elif kind == 'wrong_probability':
            rows[15]['missing_probability'] = .5
        elif kind == 'wrong_final5':
            terminal['final_five'][0]['epoch'] = 15
        else:
            rows[0]['updates'] = 3
        with tester.assertRaises(AssertionError):
            check_history(config, rows, terminal)
        rejected.append(kind)
    pair = dict(input_sha256=dict(search='image'), patch_sha256=['p1', 'p2', 'p3'],
        masks_sha256=['state', 'p1', 'p2', 'p3'], microstep=0, epoch=1, missing_probability=.5)
    check_pair(pair, copy.deepcopy(pair))
    different_frequency = dict(pair, missing_probability=.2, masks_sha256=['other', 'p1', 'p2', 'p3'])
    check_pair(pair, different_frequency)
    for key, value in [('input_sha256', {}), ('patch_sha256', ['wrong']),
                       ('masks_sha256', ['wrong']), ('microstep', 1)]:
        bad = dict(pair, **{key: value})
        with tester.assertRaises(AssertionError):
            check_pair(pair, bad)
        rejected.append('pair_'+key)
    for epoch in [1, 10, 11, 16, 20]:
        index = (epoch-1)*4+1
        factor = .1**((epoch-1)//10)
        step = dict(pair, microstep=index, epoch=epoch, update=(index+1)//2,
            missing_probability=schedule('progressive')[epoch-1],
            availability_probability=epoch_config(config, epoch)['complete_rgb_missing_tir_missing_probability'],
            loss=1., task_c=1., task_m=1., kd_forward=0., kd_reverse=0., grad_norm=1.,
            learning_rates=[.000125*.1*factor, .000125*factor])
        check_step(config, index, step)
        for key, value in [('grad_norm', float('nan')), ('learning_rates', [0., 0.])]:
            with tester.assertRaises(AssertionError):
                check_step(config, index, dict(step, **{key: value}))
    official_terminal = dict(completed, status='complete',
        final_five=[dict(epoch=epoch, path='/synthetic/epoch%02d.pth.tar' % epoch,
                        sha256='synthetic-%d' % epoch) for epoch in range(16, 21)])
    all_outputs = []
    for arm in ['fixed60', 'fixed50', 'progressive', 'shuffled']:
        jobs = jobs_for(arm, official_terminal)
        assert len(jobs) == 10
        assert [row['condition'] for row in jobs] == ['complete', 'miss']*5
        assert [row['epoch'] for row in jobs] == [epoch for epoch in range(16, 21) for _ in range(2)]
        selected_jobs = [row for row in jobs if row['development_selected']]
        assert len(selected_jobs) == 2
        assert len({row['checkpoint_sha256'] for row in selected_jobs}) == 1
        for job in jobs:
            command = command_for(job, 0)
            assert '--limit' not in command and 'val97' not in ' '.join(command)
            assert command[command.index('--workers-per-gpu')+1] == '4'
            assert job['expected_sequences'] == 479 and job['expected_frames'] == 337353
            all_outputs.append(job['output'])
    assert len(all_outputs) == len(set(all_outputs)) == 40
    with tester.assertRaises(AssertionError):
        jobs_for('progressive', dict(official_terminal, final_five=official_terminal['final_five'][:-1]))
    result = dict(status='PASS', rejected=rejected,
        tested_lr_epochs=[1, 10, 11, 16, 20], same_probability_requires_same_masks=True,
        different_probability_may_change_modality_masks=True,
        full_official_jobs_planned=40, common_selected_checkpoint_for_both_conditions=True,
        source_sha256={str(path): sha(path) for path in [
            Path(__file__).resolve(), Path(__file__).with_name('verify.py').resolve(),
            Path(__file__).with_name('official.py').resolve()]},
        scope='Synthetic verifier contracts only; no images, checkpoints or training evidence')
    with Path(args.output).open('x') as stream:
        stream.write(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
