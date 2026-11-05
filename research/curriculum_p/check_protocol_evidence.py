"""CPU fixtures for saved-evidence helpers and the new full-official controller."""
import argparse
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import torch

from research.curriculum_p.engine import ARMS, CurriculumState
from research.curriculum_p.pacing import schedule
from research.curriculum_p.protocol_prepare import make_config, REPORT
from research.curriculum_p import protocol_launch as launch
from research.curriculum_p import protocol_queue as queue
from research.curriculum_p import protocol_verify as verify
from research.protocol_r.common import read, sha, tensor_sha


def step_for(config, index):
    arm = config['curriculum_arm']
    batches = config['samples_per_epoch'] // config['microbatch']
    epoch = index // batches + 1
    probability = schedule('paced' if arm == 'paced+norm' else arm)[epoch - 1]
    low = max(.2, probability - .3) if arm.startswith('paced') else probability
    row = dict(epoch=epoch, microstep=index + 1, curriculum_microstep=index + 1,
        curriculum_arm=arm, state_norm=arm == 'paced+norm', p_epoch=probability,
        p_hard=low, hard_fraction=.25,
        mean_missing_probability=probability + .25 * (low - probability),
        actual_search_missing_fraction=.5, masks_sha256=['state'] * 5 + ['patch'] * 3,
        loss=3., task_c=1., task_m=2., kd_forward=0., kd_reverse=0.,
        ema_c=1., reverse_fraction=0., input_sha256=dict(image='synthetic'))
    if arm == 'paced+norm':
        row.update(task_m_normalized=2., ema_m_11=1., ema_m_01=2., ema_m_10=3.)
    if index % 4 == 0:
        row.update(probe_view='C' if index // 4 % 2 == 0 else 'M',
            utility=dict(search_index=index // 4 // 2 % 3,
                regions=[0] * config['microbatch'], loss_off=1., loss_on=1.,
                target_mean=0., beneficial_fraction=0., prediction_mean=0., utility_loss=0.))
    if (index + 1) % config['accumulation'] == 0:
        update = (index + 1) // config['accumulation']
        factor = min(update / config['warmup_updates'], 1.)
        factor *= .1 if epoch > config['lr_drop_epoch'] else 1.
        row.update(update=update, gradient_norm=10., seconds=1., peak_gpu_bytes=100,
            learning_rates=[config['learning_rate'] * config['backbone_multiplier'] * factor,
                            config['learning_rate'] * factor])
    return row


def terminal_for(config):
    updates = config['samples_per_epoch'] // (config['microbatch'] * config['accumulation'])
    history = [dict(epoch=epoch, updates=epoch * updates, samples=128, seconds=epoch,
        task_loss=dict.fromkeys(('11', '01', '10'), 1. if epoch < 16 else 2.),
        mean_task_loss=1. if epoch < 16 else 2.) for epoch in range(1, 21)]
    done = dict(status='complete', epochs=20, updates=20 * updates,
        early_stopped=False, official_test_selection_used=False, all_checkpoints_retained=True,
        resume_supported=False, selection_rule=config['development_selection'], selected_epoch=16,
        final_five=[dict(epoch=epoch, path='/synthetic/epoch%02d.pth.tar' % epoch,
                        sha256='synthetic-%d' % epoch) for epoch in range(16, 21)])
    return history, done


def checkpoint_for(config, epoch):
    batches = config['samples_per_epoch'] // config['microbatch']
    updates = epoch * batches // config['accumulation']
    names = ('baseline.backbone.weight', 'baseline.box_head.weight', 'utility_repair.hidden.weight')
    initial = dict(parameters_sha256={name: tensor_sha(torch.zeros(1)) for name in names},
                   bn_sha256={'baseline.running_mean': tensor_sha(torch.zeros(1))})
    model = {name: torch.ones(1) for name in names}
    model['baseline.running_mean'] = torch.zeros(1)
    last = step_for(config, epoch * batches - 1)
    state = CurriculumState(config['curriculum_arm'], 'cpu', config['seed']).state_dict()
    state.update(microsteps=epoch * batches, ema_c=torch.tensor(last['ema_c']),
                 ema_m=torch.tensor([1., 2., 3.] if config['curriculum_arm'] == 'paced+norm'
                                    else [1., 1., 1.]))
    optimizer = dict(param_groups=[
        dict(params=[0], lr=last['learning_rates'][0], weight_decay=config['weight_decay']),
        dict(params=[1, 2], lr=last['learning_rates'][1], weight_decay=config['weight_decay'])],
        state={index: dict(step=torch.tensor(updates if index < 2 else updates // 2),
                           exp_avg=torch.ones(1), exp_avg_sq=torch.ones(1)) for index in range(3)})
    saved = dict(config=config, epoch=epoch, updates=updates, model=model, optimizer=optimizer,
        curriculum=state, resume_supported=False,
        loader_position=dict(epoch=epoch, consumed_batches=batches, batches_per_epoch=batches),
        rng=dict(python=__import__('random').getstate(),
                 numpy=__import__('numpy').random.get_state(), torch_cpu=torch.get_rng_state(),
                 torch_cuda=[torch.get_rng_state()], utility_probe=torch.get_rng_state()))
    stream = io.BytesIO()
    torch.save(saved, stream)
    stream.seek(0)
    return torch.load(stream, map_location='cpu'), initial, last


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    tester = unittest.TestCase()
    parent = read(REPORT / 'r1.json')
    configs = [make_config(arm, parent) for arm in ARMS]
    rejected, steps_checked, checkpoints_checked = [], 0, 0
    for config in configs:
        batches = config['samples_per_epoch'] // config['microbatch']
        history, done = terminal_for(config)
        assert verify.check_history(config, history, done)['epoch'] == 16
        for epoch in (1, 5, 6, 10, 11, 15, 16, 20):
            for offset in (0, 1, 4, 5, 999, 1001):
                index = (epoch - 1) * batches + offset
                verify.check_step(config, index, step_for(config, index))
                steps_checked += 1
        for epoch in config['checkpoint_epochs']:
            saved, initial, last = checkpoint_for(config, epoch)
            assert verify.check_checkpoint(saved, config, initial, epoch, last) == 3
            checkpoints_checked += 1
        for key, value in [('early_stopped', True), ('selected_epoch', 1),
                           ('resume_supported', True), ('official_test_selection_used', True)]:
            with tester.assertRaises(AssertionError):
                verify.check_history(config, history, dict(done, **{key: value}))
            rejected.append(config['curriculum_arm'] + ':' + key)
    config = configs[2]
    for index in (0, 1, 4, 5, 50001, 75001):
        row = step_for(config, index)
        mutations = [('microstep', 0), ('p_epoch', .99), ('ema_c', float('nan')),
                     ('masks_sha256', []), ('mean_missing_probability', .99)]
        if 'update' in row:
            mutations += [('gradient_norm', float('nan')), ('learning_rates', [0., 0.])]
        else:
            mutations += [('update', 1)]
        for key, value in mutations:
            with tester.assertRaises(AssertionError):
                verify.check_step(config, index, dict(row, **{key: value}))
            rejected.append('step%d:%s' % (index, key))
    saved, initial, last = checkpoint_for(configs[-1], 20)
    for kind in ('bn_changed', 'wrong_ema', 'wrong_loader', 'invalid_optimizer', 'nonfinite_model'):
        bad = copy.deepcopy(saved)
        if kind == 'bn_changed':
            bad['model']['baseline.running_mean'].add_(1)
        elif kind == 'wrong_ema':
            bad['curriculum']['ema_m'][0] += 1
        elif kind == 'wrong_loader':
            bad['loader_position']['consumed_batches'] -= 1
        elif kind == 'invalid_optimizer':
            bad['optimizer']['state'][0]['step'] += 1
        else:
            bad['model']['baseline.backbone.weight'][0] = float('nan')
        with tester.assertRaises(AssertionError):
            verify.check_checkpoint(bad, configs[-1], initial, 20, last)
        rejected.append(kind)
    control = step_for(config, 25000)
    verify.check_pair(control, copy.deepcopy(control))
    paced = step_for(configs[-1], 25000)
    paced['masks_sha256'][:5] = ['different'] * 5
    verify.check_pair(control, paced)
    for kind in ('inputs', 'patch', 'same_probability_control_mask'):
        bad = copy.deepcopy(control)
        if kind == 'inputs':
            bad['input_sha256'] = {}
        elif kind == 'patch':
            bad['masks_sha256'][-1] = 'wrong'
        else:
            bad['masks_sha256'][0] = 'wrong'
        with tester.assertRaises(AssertionError):
            verify.check_pair(control, bad)
        rejected.append(kind)
    outputs, commands = [], 0
    for config in configs:
        _, done = terminal_for(config)
        jobs = launch.jobs_for(config['curriculum_arm'], done)
        assert len(jobs) == 10
        assert [job['condition'] for job in jobs] == ['complete', 'miss'] * 5
        chosen = [job for job in jobs if job['development_selected']]
        assert len(chosen) == 2 and len({job['checkpoint_sha256'] for job in chosen}) == 1
        for job in jobs:
            for gpu in range(4):
                command = launch.command_for(job, gpu)
                assert '--limit' not in command and 'val97' not in ' '.join(command)
                assert command[command.index('--gpus') + 1] == str(gpu)
                assert command[command.index('--workers-per-gpu') + 1] == '4'
                commands += 1
            outputs.append(job['output'])
    assert len(outputs) == len(set(outputs)) == 60 and commands == 240
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        for code in (0, 2):
            child = Mock(pid=123, returncode=code)
            child.poll.side_effect = [None, code]
            log = root / ('child%d.log' % code)
            with patch.object(launch.subprocess, 'Popen', return_value=child) as popen, \
                    patch.object(launch.time, 'sleep') as sleep:
                if code:
                    with tester.assertRaises(AssertionError):
                        launch.run_child(['synthetic'], {}, log, root / 'status.json', 'FIXTURE')
                else:
                    launch.run_child(['synthetic'], {}, log, root / 'status.json', 'FIXTURE')
                popen.assert_called_once()
                assert popen.call_args[1]['cwd'] == str(launch.SNAPSHOT)
                sleep.assert_called_once_with(300)
            assert read(log.with_suffix('.exit.json'))['returncode'] == code
    for phase in ('smoke', 'formal'):
        with tempfile.TemporaryDirectory() as directory:
            children = [Mock(pid=100 + index) for index in range(6)]
            for child in children:
                child.poll.return_value = 0

            def fixture_read(path):
                if path.parent == queue.CAMPAIGN:
                    return dict(output=str(Path(directory) / path.stem))
                return dict(status='passed' if phase == 'smoke' else 'complete')

            with patch.object(queue, 'QUEUE', Path(directory)), \
                    patch.object(queue, 'read', side_effect=fixture_read), \
                    patch.object(queue, 'free_gpus', return_value=[0, 1, 2, 3]), \
                    patch.object(queue.subprocess, 'Popen', side_effect=children) as popen, \
                    patch.object(queue.time, 'sleep') as sleep:
                queue.dispatch(phase, {}, {})
                queue_commands = [call[0][0] for call in popen.call_args_list]
                assert len(queue_commands) == 6
                assert [command[command.index('--arm') + 1] for command in queue_commands] == list(ARMS)
                assert [command[command.index('--gpu') + 1] for command in queue_commands] == [
                    '0', '1', '2', '3', '0', '1']
                assert all(call[1]['cwd'] == str(queue.SNAPSHOT) for call in popen.call_args_list)
                assert sleep.call_count == 2
                assert all(call[0] == (300,) for call in sleep.call_args_list)
            status = read(Path(directory) / 'status.json')
            assert status['finished'] == list(ARMS) and not status['active'] and not status['pending']
    with tempfile.TemporaryDirectory() as directory:
        child = Mock(pid=123)
        child.poll.return_value = 2
        with patch.object(queue, 'QUEUE', Path(directory)), \
                patch.object(queue, 'free_gpus', return_value=[2]), \
                patch.object(queue.subprocess, 'Popen', return_value=child) as popen, \
                patch.object(queue.time, 'sleep'):
            with tester.assertRaises(AssertionError):
                queue.dispatch('smoke', {}, {})
            popen.assert_called_once()
        assert read(Path(directory) / 'fixed60_smoke_exit.json')['returncode'] == 2
    rows = '0, 1, 40, 250\n1, 20000, 80, 250\n2, 1, 85, 250\n3, 1, 84, 250\n'
    with patch.object(queue.subprocess, 'check_output', return_value=rows):
        assert queue.free_gpus() == [0, 3]
    with patch.object(queue.subprocess, 'check_output', return_value=rows.replace('250', '300')):
        with tester.assertRaises(AssertionError):
            queue.free_gpus()
    assert not torch.cuda.is_initialized()
    report = dict(status='PASS', cuda_initialized=False, step_fixtures=steps_checked,
        serialized_tiny_checkpoint_fixtures=checkpoints_checked, rejected=rejected,
        full_evaluations_planned=60, single_gpu_command_fixtures=commands,
        controller_wait_seconds=300, child_exit_records_checked=True,
        scheduler_dispatch_fixtures=12, scheduler_failed_child_stops_dispatch=True,
        scheduler_excludes_busy_or_hot_gpus=True,
        full_saved_run_verification_executed=False, real_official_execution=False,
        scope='Synthetic CPU helpers only; no native images, trained weights, GPU or efficacy evidence',
        source_sha256={str(path.resolve()): sha(path) for path in (
            Path(__file__), Path(verify.__file__), Path(launch.__file__), Path(queue.__file__),
            Path(queue.__file__).with_name('protocol_prepare.py'))})
    with Path(args.output).open('x') as stream:
        stream.write(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
