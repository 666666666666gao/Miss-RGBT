"""Prepare an isolated OSTrack Utility curriculum after the actual R gate."""
import argparse
from pathlib import Path
import shutil

from research.curriculum_p.engine import ARMS
from research.protocol_r.common import (
    ROOT, REPORT, INITIALIZER, INITIALIZER_SHA, read, save, sha,
)

CAMPAIGN = ROOT / 'reports/server28_rppm_ostrack_20261103'
SNAPSHOT = ROOT / 'runtime/rppm_ostrack_20261103'
RUNS = ROOT / 'runs/server28_rppm_ostrack_20261103'
BASELINE = ROOT / 'runtime/task_utility_20261018/reports/e0_complete/metrics.json'


def protocol_gate():
    """Recompute all twenty full R evaluations before accepting the protocol."""
    required = [REPORT / (arm + suffix) for arm in ('r1', 'r2')
                for suffix in ('_official_completed.json', '_training_verified.json')]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        return dict(ready=False, missing=missing,
                    reason='R1/R2 full training and official audits are unfinished')
    from research.protocol_r.common import check_sources
    from verify_recovery_campaign28 import verify_official
    check_sources()
    parents = {str(path): sha(path) for path in required}
    parents[str(REPORT / 'source_sha256.json')] = sha(REPORT / 'source_sha256.json')
    parents[str(ROOT / 'verify_recovery_campaign28.py')] = sha(
        ROOT / 'verify_recovery_campaign28.py')
    selected_scores = None
    for arm in ('r1', 'r2'):
        config_path = REPORT / (arm + '.json')
        config = read(config_path)
        run = Path(config['output'])
        done = read(run / 'completed.json')
        history = read(run / 'development_history.json')
        verified = read(REPORT / (arm + '_training_verified.json'))
        official = read(REPORT / (arm + '_official_completed.json'))
        assert read(run / 'run_config.json') == config
        assert done['status'] == official['status'] == 'complete'
        assert verified['status'] == 'passed'
        assert verified['completed_sha256'] == sha(run / 'completed.json')
        assert done['epochs'] == config['epochs'] == 20
        assert done['updates'] == 20 * config['samples_per_epoch'] // (
            config['microbatch'] * config['accumulation'])
        assert [row['epoch'] for row in history] == list(range(1, 21))
        assert all(row['samples'] == 128 for row in history)
        selected = min(history[-5:], key=lambda x: (x['mean_task_loss'], x['epoch']))
        assert done['selected_epoch'] == official['selected_epoch'] == selected['epoch']
        assert not done['official_test_selection_used']
        assert not official['official_test_selection_used']
        assert done['all_checkpoints_retained']
        assert official['full_evaluations'] == 10
        assert official['sequences_per_evaluation'] == 479
        assert official['frames_per_evaluation'] == 337353
        assert [row['epoch'] for row in official['rows']] == [16, 17, 18, 19, 20]
        assert [row['epoch'] for row in done['final_five']] == [16, 17, 18, 19, 20]
        checkpoints = {item['epoch']: item for item in done['final_five']}
        for row in official['rows']:
            checkpoint = checkpoints[row['epoch']]
            assert sha(checkpoint['path']) == checkpoint['sha256'] == row['checkpoint_sha256']
            assert row['development_selected'] == (row['epoch'] == selected['epoch'])
            assert set(row['conditions']) == {'complete', 'miss'}
            for condition, item in row['conditions'].items():
                destination = ROOT / 'results/r1r2_20261103' / arm / (
                    'epoch%02d_%s' % (row['epoch'], condition))
                assert Path(item['output']) == destination
                assert item['independently_recomputed']
                assert sha(destination / 'completed.json') == item['completed_sha256']
                actual = verify_official(destination, checkpoint['sha256'],
                                         condition, item['metrics'])
                assert actual == item['metrics']
                for name in ('run.json', 'completed.json', 'metrics.json',
                             'all_receipts.json'):
                    parents[str(destination / name)] = sha(destination / name)
                if arm == 'r1' and row['development_selected'] and condition == 'complete':
                    selected_scores = actual
        for path in (config_path, run / 'completed.json',
                     run / 'development_history.json', run / 'run_config.json'):
            parents[str(path)] = sha(path)
    baseline = read(BASELINE)
    deltas = {dataset: {key: selected_scores[dataset][key] - baseline[dataset][key]
                       for key in keys}
              for dataset, keys in [('LasHeR', ('PR', 'NPR', 'SR')),
                                    ('RGBT234', ('MPR', 'MSR'))]}
    assert all(value >= -.1 for scores in deltas.values() for value in scores.values()), (
        'R1 failed the registered gate; R3 needs a separate registered protocol', deltas)
    parents[str(BASELINE)] = sha(BASELINE)
    return dict(ready=True, protocol_arm='r1', parents_sha256=parents,
                selected_complete_delta_pp=deltas,
                scope='Actual full official R audit; R1 preregistered tolerance -0.1pp')


def make_config(arm, parent):
    assert arm in ARMS and parent['arm'] == 'r1' and parent['init'] == 'ostrack'
    keys = ('seed', 'dataset', 'microbatch', 'accumulation', 'samples_per_epoch',
            'epochs', 'workers', 'learning_rate', 'backbone_multiplier',
            'weight_decay', 'lr_drop_epoch', 'warmup_updates', 'clip_grad_norm',
            'development_cache', 'development_cache_sha256', 'development_selection')
    config = {key: parent[key] for key in keys}
    config.update(
        stage='A', init='ostrack', curriculum_arm=arm, checkpoint=str(INITIALIZER),
        initializer_sha256=INITIALIZER_SHA, repair=False, repair_layer=6,
        forward_weight=.1, reverse_weight=0., reverse_margin=.1,
        patch_mask_probability=.1, freeze_all_bn=True,
        bn_policy='Freeze all BN for existing Utility counterfactual probes in every arm',
        bn_difference_from_r='R freezes only head BN; this is Utility protocol, not an R replica',
        task_utility=dict(width=128, depth=1, region_grid=4, selected_regions=4,
                          refinement_limit=.1, selection='utility',
                          selection_seed=parent['seed']),
        utility_probe_interval=4, utility_weight=.1,
        checkpoint_epochs=[10, 16, 17, 18, 19, 20],
        output=str(RUNS / arm), snapshot=str(SNAPSHOT),
        gate=str(CAMPAIGN / 'protocol_gate.json'),
        gate_sha256=None, source_manifest=str(CAMPAIGN / 'source_sha256.json'),
        source_manifest_sha256=None, early_stop=False, official_test_selection_used=False,
        resume_supported=False)
    return config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--check-gate-only', action='store_true')
    args = parser.parse_args()
    gate = protocol_gate()
    if args.check_gate_only:
        print(__import__('json').dumps(gate, indent=2), flush=True)
        return
    assert gate['ready'], gate
    assert sha(INITIALIZER) == INITIALIZER_SHA
    assert not CAMPAIGN.exists() and not SNAPSHOT.exists()
    CAMPAIGN.mkdir()
    SNAPSHOT.mkdir()
    frozen = ROOT / 'runtime/task_utility_20261018'
    for folder in ('lib', 'miss_rgbt', 'experiments/cadtrack'):
        shutil.copytree(str(frozen / folder), str(SNAPSHOT / folder),
                        ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    for folder in ('curriculum_p', 'protocol_r'):
        shutil.copytree(str(ROOT / 'research' / folder),
                        str(SNAPSHOT / 'research' / folder),
                        ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    save(CAMPAIGN / 'protocol_gate.json', gate)
    sources = {str(path): sha(path) for path in sorted(SNAPSHOT.rglob('*'))
               if path.is_file() and path.suffix in ('.py', '.yaml', '.txt')}
    save(CAMPAIGN / 'source_sha256.json', sources)
    parent = read(REPORT / 'r1.json')
    train_names = SNAPSHOT / 'lib/train/data_specs/lasher_train.txt'
    for arm in ARMS:
        config = make_config(arm, parent)
        config.update(gate_sha256=sha(CAMPAIGN / 'protocol_gate.json'),
                      source_manifest_sha256=sha(CAMPAIGN / 'source_sha256.json'),
                      train_names_sha256=sha(train_names))
        save(CAMPAIGN / (arm + '.json'), config)
    save(CAMPAIGN / 'registration.json', dict(status='prepared_not_launched',
        arms=list(ARMS), epochs=20, early_stop=False,
        initialization='OSTrack', protocol_arm='r1',
        smoke_updates_required=4, formal_gate='Each real smoke must pass first',
        official_epochs=[16, 17, 18, 19, 20], official_conditions=['complete', 'miss'],
        expected_full_evaluations=60, sequences_per_evaluation=479,
        frames_per_evaluation=337353, resume_supported=False,
        bn_difference_from_r='All BN frozen for Utility probes, shared across six arms'))
    print('PREPARED six arms; all real smokes required before formal training')


if __name__ == '__main__':
    main()
