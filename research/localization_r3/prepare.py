"""Prepare an isolated R3 pair after the real A official terminal audit."""
import argparse
import json
from pathlib import Path
import shutil

from research.localization_r3.objective import ARMS
from research.protocol_r.common import (
    ROOT, REPORT, INITIALIZER, INITIALIZER_SHA, read, save, sha,
)

CAMPAIGN = ROOT / 'reports/server28_localization_r3_20261104'
SNAPSHOT = ROOT / 'runtime/localization_r3_20261104'
RUNS = ROOT / 'runs/server28_localization_r3_20261104'
SELECTION = 'Minimum mean 11/01/10 task loss among epochs16-20; earlier tie'


def make_config(arm, parent, campaign, snapshot, runs, audit):
    assert arm in ARMS
    return dict(arm=arm, init='ostrack', checkpoint=str(INITIALIZER),
        initializer_sha256=INITIALIZER_SHA, seed=2031, stochastic_seed=74101,
        repair=False, repair_width=128, repair_layer=6,
        task_utility=None, freeze_all_bn=True,
        dataset=parent['dataset'], microbatch=4, accumulation=2,
        epochs=20, samples_per_epoch=20000, workers=2, early_stop=False,
        learning_rate=1e-4, backbone_multiplier=.1, weight_decay=1e-4,
        warmup_updates=500, lr_drop_epoch=10, clip_grad_norm=.1,
        development_cache=parent['development_cache'],
        development_cache_sha256=parent['development_cache_sha256'],
        development_selection=SELECTION,
        checkpoint_epochs=[10, 16, 17, 18, 19, 20],
        snapshot=str(snapshot), output=str(Path(runs) / arm),
        source_manifest=str(Path(campaign) / 'source_sha256.json'),
        source_manifest_sha256=None, a_audit=str(audit), a_audit_sha256=None,
        train_names_sha256=None, resume_supported=False,
        official_test_selection_used=False)


def check_config(config):
    expected = dict(seed=2031, stochastic_seed=74101, init='ostrack',
        repair=False, repair_width=128, repair_layer=6,
        task_utility=None, freeze_all_bn=True, microbatch=4,
        accumulation=2, epochs=20, samples_per_epoch=20000, workers=2,
        early_stop=False, learning_rate=1e-4, backbone_multiplier=.1,
        weight_decay=1e-4, warmup_updates=500, lr_drop_epoch=10,
        clip_grad_norm=.1, development_selection=SELECTION,
        checkpoint_epochs=[10, 16, 17, 18, 19, 20],
        resume_supported=False, official_test_selection_used=False)
    assert config['arm'] in ARMS
    assert all(config[key] == value for key, value in expected.items())
    assert config['checkpoint'] == str(INITIALIZER)
    assert config['initializer_sha256'] == INITIALIZER_SHA


def check_audit(path):
    audit = read(path)
    assert audit['status'] == 'passed' and audit['phase'] == 'formal_official_queue'
    assert audit['official_evaluations_recomputed'] == 20
    assert audit['sequences_per_evaluation'] == 479
    assert audit['frames_per_evaluation'] == 337353
    assert audit['all_checkpoints_retained']
    assert audit['official_test_selection_used'] is False
    return audit


def prepare(audit_path, campaign=CAMPAIGN, snapshot=SNAPSHOT, runs=RUNS):
    """Freeze sources/configs only; no GPU, loader, smoke or formal launch."""
    audit_path = Path(audit_path)
    if not audit_path.is_file():
        return dict(status='waiting', stage='WAITING_FOR_A_TERMINAL_AUDIT',
                    missing=[str(audit_path)], dispatch_authorized=False)
    audit = check_audit(audit_path)
    for path, expected in audit['evidence_sha256'].items():
        assert sha(path) == expected, path
    campaign, snapshot, runs = map(Path, (campaign, snapshot, runs))
    assert not campaign.exists() and not snapshot.exists() and not runs.exists()
    assert sha(INITIALIZER) == INITIALIZER_SHA
    parent = read(REPORT / 'r1.json')
    assert sha(parent['development_cache']) == parent['development_cache_sha256']
    native = ROOT / 'runtime/task_utility_20261018'
    source = Path(__file__).resolve().parents[2]
    campaign.mkdir(parents=True)
    snapshot.mkdir(parents=True)
    for folder in ('lib', 'miss_rgbt', 'experiments/cadtrack'):
        shutil.copytree(str(native / folder), str(snapshot / folder),
                        ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    for folder in ('localization_r3', 'curriculum_p', 'protocol_r'):
        shutil.copytree(str(source / 'research' / folder),
                        str(snapshot / 'research' / folder),
                        ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    files = {str(path): sha(path) for path in sorted(snapshot.rglob('*'))
             if path.is_file() and path.suffix in ('.py', '.yaml', '.txt')}
    save(campaign / 'source_sha256.json', files)
    for arm in ARMS:
        config = make_config(arm, parent, campaign, snapshot, runs, audit_path)
        config.update(a_audit_sha256=sha(audit_path),
            source_manifest_sha256=sha(config['source_manifest']),
            train_names_sha256=sha(snapshot / 'lib/train/data_specs/lasher_train.txt'))
        check_config(config)
        save(campaign / (arm + '.json'), config)
    registration = dict(status='prepared_not_launched', arms=list(ARMS),
        epochs=20, samples_per_epoch=20000, optimizer_updates_per_arm=50000,
        seed=2031, stochastic_seed=74101, early_stop=False,
        primary_development_contrast='epoch20 complete_missing C loss minus complete_only C loss',
        development_selection=SELECTION, smoke_updates_per_arm=4,
        official_epochs=[16, 17, 18, 19, 20],
        official_conditions=['complete', 'miss'], full_evaluations=20,
        sequences_per_evaluation=479, frames_per_evaluation=337353,
        all_checkpoints_retained=True, official_test_selection_used=False,
        independent_review='REVIEW_UNAVAILABLE', dispatch_authorized=False,
        scope='Independent R3; does not replace failed R1 or authorize the old P queue')
    save(campaign / 'registration.json', registration)
    return registration


def check_evidence(config):
    check_config(config)
    snapshot = Path(config['snapshot'])
    assert snapshot == Path(__file__).resolve().parents[2]
    assert sha(config['source_manifest']) == config['source_manifest_sha256']
    for path, expected in read(config['source_manifest']).items():
        assert sha(path) == expected, path
    assert sha(config['a_audit']) == config['a_audit_sha256']
    check_audit(config['a_audit'])
    assert sha(config['checkpoint']) == config['initializer_sha256']
    assert sha(config['development_cache']) == config['development_cache_sha256']
    assert sha(snapshot / 'lib/train/data_specs/lasher_train.txt') == config['train_names_sha256']
    assert read(Path(config['source_manifest']).parent / (config['arm'] + '.json')) == config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--a-audit', required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.a_audit), indent=2), flush=True)


if __name__ == '__main__':
    main()
