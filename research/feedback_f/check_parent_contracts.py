"""Exercise F parent validation on temporary synthetic metadata, never data."""
import argparse
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import train as trainer
from check_structure import sha


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2)+'\n')


def fixture(root):
    campaign = root/'reports/server28_curriculum_p_20261102_formal'
    prior = root/'prior/completed.json'
    manifest = root/'prior/input_manifest.jsonl'
    source = root/'synthetic_verifier.py'
    save(prior, dict(status='complete', scope='synthetic fixture'))
    save(manifest, dict(identities=[], scope='synthetic fixture'))
    source.write_text('# Synthetic metadata fixture only.\n')
    inherited = dict(parent_completion_sha256={str(prior): sha(prior)},
                     exclude_manifest_sha256={str(manifest): sha(manifest)})
    config = copy.deepcopy(inherited)
    prepared, arms = [], {}
    for arm in trainer.P_ARMS:
        run = root/'runs/server28_curriculum_p_20261102_formal'/arm
        parent_config = dict(inherited, curriculum_arm=arm)
        config_path = campaign/(arm+'.json')
        save(config_path, parent_config)
        save(run/'run_config.json', parent_config)
        completed = run/'completed.json'
        inputs = run/'input_manifest.jsonl'
        save(completed, dict(status='complete', epochs=20, scope='synthetic fixture'))
        save(inputs, dict(identities=[], scope='synthetic fixture'))
        config['parent_completion_sha256'][str(completed)] = sha(completed)
        config['exclude_manifest_sha256'][str(inputs)] = sha(inputs)
        arms[arm] = dict(completed_sha256=sha(completed), input_manifest_sha256=sha(inputs))
        prepared.append(dict(arm=arm, config=str(config_path), config_sha256=sha(config_path)))
    save(campaign/'prepared.json', dict(arms=prepared))
    verified = dict(status='PASS', phase='formal', arms=arms,
        source_sha256={str(source): sha(source)}, prepared_sha256=sha(campaign/'prepared.json'),
        scope='Synthetic metadata fixture, not P experiment evidence')
    verification = campaign/'verification.json'
    save(verification, verified)
    config['parent_verification_sha256'] = {str(verification): sha(verification)}
    return config, verification, prior, manifest, source


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    output = Path(args.output)
    assert not output.exists()
    tester = unittest.TestCase()
    cases = ['missing_verification', 'wrong_phase', 'missing_arm', 'changed_verifier',
             'changed_prepared', 'changed_config', 'changed_run_config',
             'incomplete_parent', 'changed_manifest', 'missing_inherited_completion',
             'missing_inherited_manifest', 'wrong_verification_hash']
    rejected = []
    for case in cases:
        with tempfile.TemporaryDirectory(prefix='miss_rgbt_f_parent_contract_') as directory:
            root = Path(directory)
            config, verification, prior, manifest, source = fixture(root)
            with patch.object(trainer, 'PROJECT', root):
                trainer.validate_parents(config)
                verified = json.loads(verification.read_text())
                campaign = verification.parent
                arm = trainer.P_ARMS[0]
                run = root/'runs/server28_curriculum_p_20261102_formal'/arm
                expected = AssertionError
                if case == 'missing_verification':
                    verification.unlink()
                    expected = FileNotFoundError
                elif case in ['wrong_phase', 'missing_arm', 'incomplete_parent']:
                    if case == 'wrong_phase':
                        verified['phase'] = 'smoke'
                    elif case == 'missing_arm':
                        del verified['arms'][arm]
                    else:
                        completed = run/'completed.json'
                        save(completed, dict(status='running', scope='synthetic fixture'))
                        verified['arms'][arm]['completed_sha256'] = sha(completed)
                        config['parent_completion_sha256'][str(completed)] = sha(completed)
                    save(verification, verified)
                    config['parent_verification_sha256'] = {str(verification): sha(verification)}
                elif case == 'changed_verifier':
                    source.write_text('# Changed fixture source.\n')
                elif case == 'changed_prepared':
                    save(campaign/'prepared.json', dict(arms=[]))
                elif case == 'changed_config':
                    save(campaign/(arm+'.json'), dict(changed=True))
                elif case == 'changed_run_config':
                    save(run/'run_config.json', dict(changed=True))
                elif case == 'changed_manifest':
                    save(run/'input_manifest.jsonl', dict(changed=True))
                elif case == 'missing_inherited_completion':
                    del config['parent_completion_sha256'][str(prior)]
                    expected = KeyError
                elif case == 'missing_inherited_manifest':
                    del config['exclude_manifest_sha256'][str(manifest)]
                    expected = KeyError
                else:
                    config['parent_verification_sha256'][str(verification)] = '0'*64
                with tester.assertRaises(expected):
                    trainer.validate_parents(config)
                assert not list(root.glob('runs/*feedback_f*'))
                rejected.append(case)
    sources = [Path(__file__), Path(trainer.__file__)]
    report = dict(status='PASS', scope='Temporary synthetic parent metadata only',
        positive_fixture_accepted=True, rejected=rejected, no_f_run_created=True,
        source_sha256={str(path.resolve()): sha(path) for path in sources},
        limitations=['Not real P receipts or actual positive-parent experiment validation',
                     'No dataset, GPU, image loop, saved-weight or official evaluation checks'])
    output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
