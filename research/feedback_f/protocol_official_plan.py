"""Pure final-five job planning for OSTrack F; grants no launch permission."""
from pathlib import Path


MODES = ('single', 'none', 'shallow', 'deep-no-response', 'deep')
CURRICULA = ('fixed60', 'fixed50', 'progressive', 'shuffled', 'paced', 'paced+norm')


def jobs_for(config, completed, results_root):
    """Plan paired Complete/Miss reports after caller-owned evidence verification."""
    mode = config['feedback']['mode']
    curriculum = config['curriculum_arm']
    assert mode in MODES and curriculum in CURRICULA
    assert config['init'] == 'ostrack'
    assert config['epochs'] == completed['epochs'] == 20
    assert completed['status'] == 'complete'
    assert completed['early_stopped'] is False
    assert completed['all_checkpoints_retained'] is True
    assert completed['official_test_selection_used'] is False
    assert completed['selection_rule'] == config['development_selection']
    assert [row['epoch'] for row in completed['final_five']] == list(range(16, 21))
    assert completed['selected_epoch'] in range(16, 21)
    run = Path(config['output'])
    jobs = []
    for row in completed['final_five']:
        epoch = row['epoch']
        assert Path(row['path']) == run / ('epoch%02d.pth.tar' % epoch)
        digest = row['sha256']
        assert isinstance(digest, str) and len(digest) == 64
        assert all(character in '0123456789abcdef' for character in digest)
        for condition in ('complete', 'miss'):
            destination = Path(results_root) / curriculum / mode / (
                'epoch%02d_%s' % (epoch, condition))
            jobs.append(dict(arm=mode, curriculum_arm=curriculum, epoch=epoch,
                checkpoint=row['path'], checkpoint_sha256=digest,
                condition=condition, output=str(destination),
                development_selected=epoch == completed['selected_epoch'],
                expected_sequences=479, expected_frames=337353))
    return jobs


def command_for(job, gpu, snapshot, python):
    """Return the existing F full evaluator invocation; never execute it."""
    assert gpu in (0, 1, 2, 3)
    assert job['condition'] in ('complete', 'miss')
    assert job['expected_sequences'] == 479 and job['expected_frames'] == 337353
    return [str(python), '-u',
        str(Path(snapshot) / 'research/feedback_f/evaluate.py'),
        '--checkpoint', job['checkpoint'], '--output', job['output'],
        '--condition', job['condition'], '--gpus', str(gpu),
        '--workers-per-gpu', '4']
