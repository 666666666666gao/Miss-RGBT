"""Freeze five native-protocol F arms after actual P* and M1 decisions."""
import argparse
import copy
import json
from pathlib import Path
import shutil

from research.feedback_f.protocol_m1 import verify_decision
from research.feedback_f.protocol_parent import bind_manifest, select_parent
from research.protocol_r.common import ROOT, read, save, sha

CAMPAIGN = ROOT / 'reports/server28_feedback_f_protocol_20261104'
SNAPSHOT = ROOT / 'runtime/feedback_f_protocol_20261104'
RUNS = ROOT / 'runs/server28_feedback_f_protocol_20261104'
MODES = ('single', 'none', 'shallow', 'deep-no-response', 'deep')
CLOSED_M1 = 'closed_both_diagnostics_negative'


def preparation_gate(report, cache_receipt, diagnostic_config):
    parent = select_parent()
    if not parent['ready']:
        return dict(ready=False, stage='WAITING_FOR_P', parent=parent,
                    f_training_authorized=False)
    m1 = verify_decision(report, cache_receipt, diagnostic_config)
    if not m1['ready']:
        return dict(ready=False, stage=m1['stage'], parent=parent, m1=m1,
                    f_training_authorized=False)
    if m1['requires_ohcas_composition']:
        return dict(ready=False, stage='WAITING_FOR_OHCAS_COMPOSITION',
            parent=parent, m1=m1, f_training_authorized=False,
            reason='The native F builder has no accepted-OHCAS composition yet')
    assert m1['m1_decision'] == CLOSED_M1
    evidence = {}
    bind_manifest(parent['parents_sha256'], evidence)
    bind_manifest(m1['parents_sha256'], evidence)
    return dict(ready=True, stage='F_PREPARATION_READY', protocol_arm='r1',
        parent=parent, m1=m1, parents_sha256=evidence,
        f_training_authorized=False,
        remaining='Freeze five-arm sources and pass all five matched real-image smokes')


def make_configs(parent, gate, campaign, snapshot, runs):
    """Keep the selected curriculum/cache; start all F arms from OSTrack."""
    from research.feedback_f.protocol_runner import check_config
    assert gate['ready'] and gate['stage'] == 'F_PREPARATION_READY'
    assert gate['m1']['m1_decision'] == CLOSED_M1
    assert not gate['m1']['requires_ohcas_composition']
    assert parent['curriculum_arm'] == gate['parent']['selected']['arm']
    assert parent['init'] == 'ostrack'
    base = copy.deepcopy(parent)
    base.update(seed=2030, mask_seed=65403, microbatch=2, accumulation=2,
        learning_rate=5e-5, backbone_multiplier=.1, development_samples=128,
        gate=str(Path(campaign) / 'protocol_gate.json'), gate_sha256=None,
        snapshot=str(snapshot),
        source_manifest=str(Path(campaign) / 'source_sha256.json'),
        source_manifest_sha256=None)
    base['task_utility']['selection_seed'] = base['seed']
    configs = []
    for mode in MODES:
        config = copy.deepcopy(base)
        config.update(feedback=dict(mode=mode, width=128),
                      output=str(Path(runs) / mode))
        check_config(config)
        configs.append(config)
    return configs


def prepare(gate, campaign=CAMPAIGN, snapshot=SNAPSHOT, runs=RUNS):
    """Write a new isolated snapshot/config set; never launch a worker or model."""
    if not gate['ready']:
        return gate
    campaign, snapshot, runs = map(Path, (campaign, snapshot, runs))
    assert not campaign.exists() and not snapshot.exists() and not runs.exists()
    selected = gate['parent']['selected']
    assert sha(selected['config']) == selected['config_sha256']
    parent = read(selected['config'])
    bind_manifest(gate['parents_sha256'], {})
    configs = make_configs(parent, gate, campaign, snapshot, runs)
    source = Path(__file__).resolve().parents[2]
    native = ROOT / 'runtime/task_utility_20261018'
    campaign.mkdir(parents=True)
    snapshot.mkdir(parents=True)
    for folder in ('lib', 'miss_rgbt', 'experiments/cadtrack'):
        shutil.copytree(str(native / folder), str(snapshot / folder),
                        ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    for folder in ('feedback_f', 'curriculum_p', 'protocol_r'):
        shutil.copytree(str(source / 'research' / folder),
                        str(snapshot / 'research' / folder),
                        ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    save(campaign / 'protocol_gate.json', gate)
    sources = {str(path): sha(path) for path in sorted(snapshot.rglob('*'))
               if path.is_file() and path.suffix in ('.py', '.yaml', '.txt')}
    save(campaign / 'source_sha256.json', sources)
    for config in configs:
        config.update(gate_sha256=sha(config['gate']),
                      source_manifest_sha256=sha(config['source_manifest']))
        save(campaign / (config['feedback']['mode'] + '.json'), config)
    registration = dict(status='prepared_not_launched', arms=list(MODES),
        curriculum_arm=parent['curriculum_arm'], seed=2030, mask_seed=65403,
        epochs=20, early_stop=False, microbatch=2, accumulation=2,
        learning_rate=5e-5, backbone_learning_rate=5e-6,
        samples_per_epoch=20000, updates_per_arm=100000,
        development_samples=128, development_cache=parent['development_cache'],
        development_selection=parent['development_selection'],
        m1_decision=CLOSED_M1, smoke_updates_per_arm=4,
        formal_gate='All five matched real-image smokes must pass',
        official_epochs=list(range(16, 21)), official_conditions=['complete', 'miss'],
        expected_full_evaluations=50, sequences_per_evaluation=479,
        frames_per_evaluation=337353, all_checkpoints_retained=True,
        official_test_selection_used=False, f_training_authorized=False,
        initialization='OSTrack; P* selects curriculum, not continuation weights')
    save(campaign / 'registration.json', registration)
    return registration


def check_evidence(config):
    """Check the exact frozen preparation before a runner's caller uses it."""
    from research.feedback_f.protocol_runner import check_config
    check_config(config)
    snapshot = Path(config['snapshot'])
    assert snapshot == Path(__file__).resolve().parents[2]
    assert sha(config['source_manifest']) == config['source_manifest_sha256']
    bind_manifest(read(config['source_manifest']), {})
    assert sha(config['gate']) == config['gate_sha256']
    gate = read(config['gate'])
    assert gate['stage'] == 'F_PREPARATION_READY' and gate['ready']
    assert gate['f_training_authorized'] is False
    bind_manifest(gate['parents_sha256'], {})
    selected = gate['parent']['selected']
    assert sha(selected['config']) == selected['config_sha256']
    parent = read(selected['config'])
    campaign = Path(config['gate']).parent
    expected = make_configs(parent, gate, campaign, snapshot, Path(config['output']).parent)
    chosen = next(row for row in expected
                  if row['feedback']['mode'] == config['feedback']['mode'])
    chosen.update(gate_sha256=sha(config['gate']),
                  source_manifest_sha256=sha(config['source_manifest']))
    assert chosen == config
    assert sha(config['checkpoint']) == config['initializer_sha256']
    assert sha(config['development_cache']) == config['development_cache_sha256']
    assert sha(snapshot / 'lib/train/data_specs/lasher_train.txt') == config['train_names_sha256']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--m1-report', required=True)
    parser.add_argument('--m1-cache-receipt', required=True)
    parser.add_argument('--m1-config', required=True)
    parser.add_argument('--check-gate-only', action='store_true')
    args = parser.parse_args()
    gate = preparation_gate(args.m1_report, args.m1_cache_receipt, args.m1_config)
    result = gate if args.check_gate_only else prepare(gate)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
