"""Freeze P configs after both original formal input manifests are terminal."""
import argparse
import hashlib
import json
from pathlib import Path

from schedule import ARMS, schedule

PROJECT = Path('/home/gaob/Miss-RGBT')


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', required=True, choices=['smoke', 'formal'])
    args = parser.parse_args()
    parents, manifests = {}, {}
    for phase in ['smoke', 'formal']:
        for selector in ['utility', 'random']:
            root = PROJECT/'runs'/('server28_partitioned_a_20261102_'+phase)/selector
            path = root/'completed.json'
            completed = json.loads(path.read_text())
            assert completed['status'] == 'complete'
            parents[str(path)] = sha(path)
            manifest = root/'input_manifest.jsonl'
            manifests[str(manifest)] = sha(manifest)
    if args.phase == 'formal':
        smoke = PROJECT/'reports/server28_curriculum_p_20261102_smoke/verification.json'
        assert json.loads(smoke.read_text())['status'] == 'PASS'
        for arm in ARMS:
            root = PROJECT/'runs/server28_curriculum_p_20261102_smoke'/arm
            path = root/'completed.json'
            completed = json.loads(path.read_text())
            assert completed['status'] == 'complete' and completed['epochs'] == 20
            parents[str(path)] = sha(path)
            manifest = root/'input_manifest.jsonl'
            manifests[str(manifest)] = sha(manifest)
    stem = 'server28_curriculum_p_20261102_'+args.phase
    destination = PROJECT/'reports'/stem
    destination.mkdir(exist_ok=False)
    source = PROJECT/'reports/server28_partitioned_a_20261102_formal/utility.json'
    original = json.loads(source.read_text())
    records = []
    for arm in ARMS:
        config = dict(original)
        config.update(curriculum_arm=arm, epochs=20, seed=2029 if args.phase == 'formal' else 65201,
            development_seed=65102 if args.phase == 'formal' else 65202,
            development_samples=128 if args.phase == 'formal' else 8,
            samples_per_epoch=20000 if args.phase == 'formal' else 20,
            output=str(PROJECT/'runs'/stem/arm), parent_completion_sha256=parents,
            exclude_manifest_sha256=manifests, purpose='P missing-frequency control',
            phase=args.phase)
        config['task_utility'] = dict(original['task_utility'], selection='utility',
                                     selection_seed=config['seed'])
        for key in ['early_stop_min_epochs', 'early_stop_patience', 'early_stop_relative_delta',
                    'exclude_manifests']:
            config.pop(key, None)
        path = destination/(arm+'.json')
        path.write_text(json.dumps(config, indent=2)+'\n')
        records.append(dict(arm=arm, config=str(path), config_sha256=sha(path),
                            missing_frequency=schedule(arm)))
    (destination/'prepared.json').write_text(json.dumps(dict(status='PREPARED_NOT_LAUNCHED',
        parent_completion_sha256=parents, exclude_manifest_sha256=manifests,
        source_config_sha256=sha(source), arms=records), indent=2)+'\n')
    print(json.dumps(dict(status='PREPARED_NOT_LAUNCHED', destination=str(destination))))


if __name__ == '__main__':
    main()
