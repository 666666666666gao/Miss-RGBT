"""Read-only P* selection for RCFR; this does not authorize F training."""
import argparse
import json
from pathlib import Path

from research.curriculum_p.engine import ARMS
from research.curriculum_p.protocol_prepare import CAMPAIGN
from research.curriculum_p.protocol_train import check_config
from research.curriculum_p.protocol_verify import check_history, verify_pairs
from research.protocol_r.common import ROOT, read, sha

QUEUE = ROOT / 'reports/server28_rppm_ostrack_20261103_queue'
SELECTION_RULE = (
    'Minimum final-five selected internal mean task loss across all six P arms; '
    'epoch ties choose earlier epoch; arm ties choose registered ARMS order. '
    'Official test metrics are not used for selection.')


def required_paths(campaign, queue):
    paths = [queue / 'completed.json', campaign / 'paired_training_verified.json']
    for arm in ARMS:
        paths.extend([campaign / (arm + '.json'),
                      campaign / (arm + '_training_verified.json'),
                      campaign / (arm + '_formal_controller') / 'official_completed.json'])
    return paths


def bind_manifest(manifest, evidence):
    assert manifest
    for name, digest in manifest.items():
        assert sha(name) == digest, name
        if name in evidence:
            assert evidence[name] == digest, name
        evidence[name] = digest


def select_parent(campaign=CAMPAIGN, queue=QUEUE):
    """Verify terminal P evidence; never open images, construct models or launch."""
    paths = required_paths(campaign, queue)
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        return dict(ready=False, stage='WAITING_FOR_P', missing=missing,
                    selection_rule=SELECTION_RULE, f_training_authorized=False)
    # Imported only for a completed campaign; missing receipts require no evaluator.
    from verify_recovery_campaign28 import verify_official
    terminal = read(queue / 'completed.json')
    paired = read(campaign / 'paired_training_verified.json')
    assert terminal['status'] == 'complete' and terminal['arms'] == list(ARMS)
    assert terminal['full_evaluations'] == 60 and terminal['pairing_verified'] is True
    assert paired['status'] == 'passed' and set(paired['arms']) == set(ARMS)
    evidence = {}
    bind_manifest({str(path): sha(path) for path in paths}, evidence)
    bind_manifest(read(queue / 'source_sha256.json'), evidence)
    evidence[str(queue / 'source_sha256.json')] = sha(queue / 'source_sha256.json')
    rows = []
    for arm in ARMS:
        config_path = campaign / (arm + '.json')
        config = read(config_path)
        check_config(config)
        assert config['curriculum_arm'] == arm
        root = Path(config['output'])
        assert read(root / 'run_config.json') == config
        verified = read(campaign / (arm + '_training_verified.json'))
        assert verified == paired['arms'][arm]
        assert verified['status'] == 'passed' and verified['arm'] == arm
        verifier = Path(config['snapshot']) / 'research/curriculum_p/protocol_verify.py'
        assert verified['verifier_sha256'] == sha(verifier)
        assert sha(verifier) == sha(
            Path(__file__).parents[1] / 'curriculum_p/protocol_verify.py')
        bind_manifest(verified['evidence_sha256'], evidence)
        for path in (config_path, root / 'completed.json', root / 'run_config.json',
                     root / 'development_history.json', root / 'input_manifest.jsonl'):
            assert verified['evidence_sha256'][str(path)] == sha(path)
        assert sha(config['source_manifest']) == config['source_manifest_sha256']
        bind_manifest(read(config['source_manifest']), evidence)
        evidence[config['source_manifest']] = config['source_manifest_sha256']
        gate_path = Path(config['gate'])
        assert sha(gate_path) == config['gate_sha256']
        gate = read(gate_path)
        assert gate['ready'] is True and gate['protocol_arm'] == 'r1'
        bind_manifest(gate['parents_sha256'], evidence)
        evidence[str(gate_path)] = sha(gate_path)
        for name, digest in ((config['checkpoint'], config['initializer_sha256']),
                             (config['development_cache'], config['development_cache_sha256'])):
            bind_manifest({name: digest}, evidence)
        done = read(root / 'completed.json')
        history = read(root / 'development_history.json')
        selected = check_history(config, history, done)
        assert verified['completed_sha256'] == sha(root / 'completed.json')
        assert verified['selected_epoch'] == selected['epoch']
        controller = campaign / (arm + '_formal_controller')
        official = read(controller / 'official_completed.json')
        assert official['status'] == 'complete' and official['full_evaluations'] == 10
        assert official['sequences_per_evaluation'] == 479
        assert official['frames_per_evaluation'] == 337353
        assert official['selected_epoch'] == selected['epoch']
        assert official['official_test_selection_used'] is False
        assert official['all_checkpoints_retained'] is True
        bind_manifest(read(controller / 'official_sources.json'), evidence)
        evidence[str(controller / 'official_sources.json')] = sha(
            controller / 'official_sources.json')
        assert [(item['epoch'], item['condition']) for item in official['rows']] == [
            (epoch, condition) for epoch in range(16, 21) for condition in ('complete', 'miss')]
        checkpoints = {item['epoch']: item for item in done['final_five']}
        chosen = {}
        for item in official['rows']:
            checkpoint = checkpoints[item['epoch']]
            assert Path(checkpoint['path']) == root / ('epoch%02d.pth.tar' % item['epoch'])
            assert item['checkpoint'] == checkpoint['path']
            assert item['checkpoint_sha256'] == checkpoint['sha256']
            bind_manifest({checkpoint['path']: checkpoint['sha256']}, evidence)
            assert item['development_selected'] == (item['epoch'] == selected['epoch'])
            assert item['independently_recomputed'] is True
            assert item['expected_sequences'] == 479 and item['expected_frames'] == 337353
            destination = ROOT / 'results/rppm_ostrack_20261103' / arm / (
                'epoch%02d_%s' % (item['epoch'], item['condition']))
            assert Path(item['output']) == destination
            assert sha(destination / 'completed.json') == item['completed_sha256']
            measured = verify_official(destination, checkpoint['sha256'],
                                       item['condition'], item['metrics'])
            assert measured == item['metrics']
            for name in ('run.json', 'completed.json', 'metrics.json', 'all_receipts.json'):
                evidence[str(destination / name)] = sha(destination / name)
            if item['development_selected']:
                chosen[item['condition']] = measured
        assert set(chosen) == {'complete', 'miss'}
        rows.append(dict(arm=arm, selected_epoch=selected['epoch'],
            internal_mean_task_loss=selected['mean_task_loss'],
            config=str(config_path), config_sha256=sha(config_path),
            checkpoint=checkpoints[selected['epoch']]['path'],
            checkpoint_sha256=checkpoints[selected['epoch']]['sha256'],
            official_metrics=chosen))
    assert verify_pairs() == paired['pairing']
    winner = min(enumerate(rows), key=lambda pair: (
        pair[1]['internal_mean_task_loss'], pair[0]))[1]
    evidence[str(Path(__file__).resolve())] = sha(__file__)
    return dict(ready=True, stage='P_PARENT_VERIFIED', selection_rule=SELECTION_RULE,
                selected=winner, arms=rows, parents_sha256=evidence,
                full_evaluations_recomputed=60, f_training_authorized=False,
                remaining='Actual M1 decision, accepted-OHCAS composition if needed, '
                          'frozen F preparation and five matched real-image smokes')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    print(json.dumps(select_parent(), indent=2), flush=True)


if __name__ == '__main__':
    main()
