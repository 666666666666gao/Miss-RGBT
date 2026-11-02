"""Prepare matched F image-smoke configs after verified formal P training."""
import argparse
import copy
import hashlib
import json
from pathlib import Path

import train as trainer

PROJECT = trainer.PROJECT


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def configs(curriculum_arm):
    assert curriculum_arm in trainer.P_ARMS
    campaign = PROJECT/'reports/server28_curriculum_p_20261102_formal'
    verified_path = campaign/'verification.json'
    verified = json.loads(verified_path.read_text())
    prepared = json.loads((campaign/'prepared.json').read_text())
    entry = next(row for row in prepared['arms'] if row['arm'] == curriculum_arm)
    source = Path(entry['config'])
    original = json.loads(source.read_text())
    config = copy.deepcopy(original)
    config['parent_completion_sha256'] = dict(original['parent_completion_sha256'])
    config['exclude_manifest_sha256'] = dict(original['exclude_manifest_sha256'])
    config['parent_verification_sha256'] = {str(verified_path): sha(verified_path)}
    for arm in trainer.P_ARMS:
        root = PROJECT/'runs/server28_curriculum_p_20261102_formal'/arm
        config['parent_completion_sha256'][str(root/'completed.json')] = (
            verified['arms'][arm]['completed_sha256'])
        config['exclude_manifest_sha256'][str(root/'input_manifest.jsonl')] = (
            verified['arms'][arm]['input_manifest_sha256'])
    trainer.validate_parents(config)
    stem = 'server28_feedback_f_20261102_smoke_'+curriculum_arm
    config.update(phase='smoke', curriculum_arm=curriculum_arm, epochs=20,
        seed=65301, development_seed=65302, mask_seed=65303,
        development_samples=8, samples_per_epoch=8,
        microbatch=2, accumulation=2, learning_rate=5e-5,
        backbone_multiplier=.1, workers=2,
        purpose='Matched F real-image smoke; not formal or official evidence')
    config['task_utility'] = dict(original['task_utility'],
                                  selection='utility', selection_seed=config['seed'])
    for key in ['early_stop_min_epochs', 'early_stop_patience',
                'early_stop_relative_delta', 'exclude_manifests']:
        config.pop(key, None)
    arms = {}
    for arm in trainer.ARMS:
        local = copy.deepcopy(config)
        local.update(feedback=dict(mode=arm, width=128),
                     output=str(PROJECT/'runs'/stem/arm))
        arms[arm] = local
    return PROJECT/'reports'/stem, arms, source


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--curriculum-arm', required=True, choices=trainer.P_ARMS)
    args = parser.parse_args()
    destination, arms, source = configs(args.curriculum_arm)
    destination.mkdir(exist_ok=False)
    records = []
    for arm, config in arms.items():
        path = destination/(arm+'.json')
        path.write_text(json.dumps(config, indent=2)+'\n')
        records.append(dict(arm=arm, config=str(path), config_sha256=sha(path)))
    report = dict(status='PREPARED_NOT_LAUNCHED', phase='smoke',
        curriculum_arm=args.curriculum_arm, source_config=str(source),
        source_config_sha256=sha(source),
        parent_verification_sha256=arms['single']['parent_verification_sha256'],
        source_sha256={str(Path(__file__).resolve()): sha(__file__),
                       str(Path(trainer.__file__).resolve()): sha(trainer.__file__)},
        arms=records,
        scope='Four matched smoke configs only; no images, workers or weights')
    (destination/'prepared.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(dict(status=report['status'], destination=str(destination))))


if __name__ == '__main__':
    main()
