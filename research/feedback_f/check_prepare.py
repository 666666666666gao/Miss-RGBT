"""Check matched config generation on synthetic P metadata, never images."""
import argparse
import json
from pathlib import Path
import tempfile
from unittest.mock import patch

from check_parent_contracts import fixture, save
import prepare
import train as trainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    output = Path(args.output)
    assert not output.exists()
    with tempfile.TemporaryDirectory(prefix='miss_rgbt_f_prepare_') as directory:
        root = Path(directory)
        _, verification, _, _, _ = fixture(root)
        campaign = verification.parent
        prepared_path = campaign/'prepared.json'
        prepared = json.loads(prepared_path.read_text())
        for entry in prepared['arms']:
            path = Path(entry['config'])
            config = json.loads(path.read_text())
            config.update(task_utility=dict(selection='utility', selection_seed=2029),
                checkpoint='synthetic', patch_mask_probability=.1, reverse_weight=0.,
                freeze_all_bn=True, lr_drop_epoch=10, weight_decay=.0001,
                early_stop_min_epochs=14)
            save(path, config)
            save(root/'runs/server28_curriculum_p_20261102_formal'/
                 entry['arm']/'run_config.json', config)
            entry['config_sha256'] = prepare.sha(path)
        save(prepared_path, prepared)
        verified = json.loads(verification.read_text())
        verified['prepared_sha256'] = prepare.sha(prepared_path)
        save(verification, verified)
        with patch.object(prepare, 'PROJECT', root), patch.object(trainer, 'PROJECT', root):
            for curriculum in trainer.P_ARMS:
                destination, arms, source = prepare.configs(curriculum, 'smoke')
                assert not destination.exists()
                assert set(arms) == set(trainer.ARMS)
                common = []
                for arm, config in arms.items():
                    trainer.validate_parents(config)
                    assert config['feedback'] == dict(mode=arm, width=128)
                    assert config['curriculum_arm'] == curriculum
                    assert config['epochs'] == 20
                    assert config['samples_per_epoch'] == 8
                    assert config['samples_per_epoch']//(
                        config['microbatch']*config['accumulation'])*20 == 40
                    assert config['learning_rate'] == 5e-5
                    assert config['learning_rate']*config['backbone_multiplier'] == 5e-6
                    assert 'early_stop_min_epochs' not in config
                    common.append({key: value for key, value in config.items()
                                   if key not in ['feedback', 'output']})
                assert all(value == common[0] for value in common)
                assert source.exists()
                smoke_rows, smoke_results = [], {}
                for arm, config in arms.items():
                    path = destination/(arm+'.json')
                    save(path, config)
                    run = Path(config['output'])
                    save(run/'run_config.json', config)
                    save(run/'completed.json', dict(status='complete', scope='synthetic'))
                    save(run/'input_manifest.jsonl', dict(identities=[], scope='synthetic'))
                    smoke_rows.append(dict(arm=arm, config=str(path),
                                           config_sha256=prepare.sha(path)))
                    smoke_results[arm] = dict(
                        completed_sha256=prepare.sha(run/'completed.json'),
                        input_manifest_sha256=prepare.sha(run/'input_manifest.jsonl'))
                save(destination/'prepared.json', dict(arms=smoke_rows))
                check = destination/'verification.json'
                save(check, dict(status='PASS', phase='smoke', curriculum_arm=curriculum,
                    arms=smoke_results, source_sha256={},
                    prepared_sha256=prepare.sha(destination/'prepared.json')))
                formal_destination, formal, _ = prepare.configs(curriculum, 'formal')
                assert not formal_destination.exists()
                common = []
                for arm, config in formal.items():
                    trainer.validate_parents(config)
                    trainer.validate_smoke(config)
                    assert config['samples_per_epoch'] == 20000
                    assert config['development_samples'] == 128
                    assert config['seed'] == 2030 and config['mask_seed'] == 65403
                    assert len(config['exclude_manifest_sha256']) == 9
                    common.append({key: value for key, value in config.items()
                                   if key not in ['feedback', 'output']})
                assert all(value == common[0] for value in common)
                check.unlink()
                try:
                    prepare.configs(curriculum, 'formal')
                except FileNotFoundError:
                    pass
                else:
                    raise AssertionError('Formal preparation accepted absent F smoke')
            verification.unlink()
            try:
                prepare.configs('progressive', 'smoke')
            except FileNotFoundError:
                pass
            else:
                raise AssertionError('Missing formal P verification was accepted')
            assert not list(root.glob('reports/*feedback_f*formal*'))
    sources = [Path(__file__), Path(prepare.__file__), Path(trainer.__file__)]
    report = dict(status='PASS', curricula=list(trainer.P_ARMS),
        arms=list(trainer.ARMS), updates_per_arm=40,
        formal_configs_matched=True, formal_missing_smoke_rejected=True,
        matched_configs=True, missing_parent_rejected_before_output=True,
        source_sha256={str(path.resolve()): prepare.sha(path) for path in sources},
        scope='Synthetic parent metadata/config arithmetic only; no real P/F training')
    output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
