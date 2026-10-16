"""Full-budget BN running-stat controls; requires the attribution queue to end."""
import datetime
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

import rgbt
import torch

from lib.config.cadtrack.config import cfg, update_config_from_file
from miss_rgbt.model import load_pretrained
import run_campaign as campaign
from package_e0_availability import sha


ROOT = campaign.ROOT
campaign.RUN = ROOT/'logs/bn_freeze_full20_v2'


def state(stage):
    campaign.write('status.json', dict(stage=stage, controller_pid=os.getpid(),
        updated_utc=datetime.datetime.utcnow().isoformat()+'Z'))


def data_seals(training_root):
    package = Path(rgbt.__file__).resolve().parent
    paths = [ROOT/'reports/e0/inventory.json', ROOT/'env-spec.json']
    paths += sorted((ROOT/'lib/train/data_specs').glob('*.txt'))
    train_names = (ROOT/'lib/train/data_specs/lasher_train.txt').read_text().splitlines()
    assert len(train_names) == len(set(train_names)) == 882
    paths += [Path(training_root)/name/'init.txt' for name in train_names]
    paths += [path for path in sorted(package.rglob('*'))
        if path.is_file() and '__pycache__' not in path.parts and path.suffix != '.pyc']
    inventory = json.loads((ROOT/'reports/e0/inventory.json').read_text())
    for spec in inventory['datasets'].values():
        mask = Path(spec['mask'])
        assert sha(mask) == spec['mask_sha256']
        paths.append(mask)
        for row in spec['sequence_details']:
            for recorded, expected in row['gt_sha256'].items():
                actual = package/'gt_file'/recorded.split('/gt_file/', 1)[1]
                assert sha(actual) == expected
    return {str(path):sha(path) for path in paths}


def verify_training(stage, config):
    folder = Path(config['output'])
    complete = json.loads((folder/'completed.json').read_text())
    assert complete['epochs'] == 20 and complete['updates'] == 50000
    saved = torch.load(str(folder/'epoch20.pth.tar'), map_location='cpu')
    assert saved['epoch'] == 20 and saved['updates'] == 50000 and saved['config'] == config
    update_config_from_file(str(ROOT/'experiments/cadtrack/cadtrack.yaml'))
    model = load_pretrained(cfg, config['checkpoint'], repair=config['repair'],
        width=config['repair_width'], repair_layer=config['repair_layer'])
    initial = model.state_dict()
    buffers = [name for name in initial if name.endswith(('running_mean','running_var','num_batches_tracked'))]
    assert len(buffers) == 39 and all(torch.equal(initial[name], saved['model'][name]) for name in buffers)
    model.load_state_dict(saved['model'], strict=True)
    campaign.write(stage+'_training_verified.json', dict(completed=complete,
        checkpoint=str(folder/'epoch20.pth.tar'), checkpoint_sha256=sha(folder/'epoch20.pth.tar'),
        all_bn_buffers_equal_initial=len(buffers), strict_reload=True))


def main():
    os.chdir(str(ROOT))
    torch.set_num_threads(1)
    phase1 = ROOT/'logs/attribution_20261015_v2'
    assert json.loads((phase1/'status.json').read_text())['stage'] == 'CAMPAIGN_COMPLETE'
    assert (phase1/'completed.json').is_file()
    campaign.RUN.mkdir(parents=True, exist_ok=False)
    configs = {}
    for stage in ['e3','e4']:
        original = json.loads((ROOT/'configs'/(stage+'_full20.json')).read_text())
        config = json.loads((ROOT/'configs'/(stage+'_allbn_full20.json')).read_text())
        assert {key for key in set(config) | set(original) if config.get(key) != original.get(key)} == {'freeze_all_bn','output','source_version_file'}
        assert config['freeze_all_bn'] is True and config['epochs'] == 20 and config['samples_per_epoch'] == 20000
        assert config['microbatch'] * config['accumulation'] == 8 and config['seed'] == 2027
        assert config['checkpoint_epochs'] == [10,20]
        assert not Path(config['output']).exists()
        assert shutil.disk_usage(str(Path(config['output']).parent)).free > 10 * 1024 ** 3
        configs[stage] = config
    version = (ROOT/'reports/BN_FREEZE_SOURCE_VERSION.txt').read_text().strip()
    assert len(version) == 40 and int(version, 16) > 0
    hashes = {str(path.relative_to(ROOT)):sha(path)
        for folder in ['miss_rgbt','lib','scripts','configs','experiments']
        for path in sorted((ROOT/folder).rglob('*')) if path.is_file() and path.suffix in ['.py','.json','.yaml']}
    campaign.write('source_sha256.json', hashes)
    assert configs['e3']['dataset'] == configs['e4']['dataset']
    campaign.write('data_sha256.json', data_seals(configs['e3']['dataset']))
    assert sha(Path(configs['e3']['checkpoint'])) == 'd20c8641e4263df3ceadf9e430d0ac96c181c13ad482bca28c2dddef592a8bb8'
    assert configs['e3']['checkpoint'] == configs['e4']['checkpoint']
    rows = subprocess.check_output(['nvidia-smi','-i','2,3','--query-gpu=index,memory.used',
        '--format=csv,noheader,nounits'],universal_newlines=True).splitlines()
    assert len(rows) == 2
    for row in rows:
        gpu, used = [int(value.strip()) for value in row.split(',')]
        assert gpu in [2,3] and used < 300, row
    assert shutil.disk_usage(str(ROOT)).free > 1024 ** 3
    campaign.write('campaign.json',dict(source_version=version,configs=configs,
        started_utc=datetime.datetime.utcnow().isoformat()+'Z',
        formal_budget_per_model=dict(epochs=20,sampled_tuples_per_epoch=20000,effective_batch=8,optimizer_updates=50000,seed=2027),
        evaluation_epoch=20,selection='Fixed final epoch; same weight for complete/Miss; no test selection.',
        bn_policy='All 13 BatchNorm2d running statistics fixed; affine remains trainable; other module training modes unchanged.',
        admission_gpus=rows))
    started = time.time()
    state('PREFLIGHT_RUNNING')
    checks = {stage:campaign.launch(stage+'_bn_preflight',['scripts/validate_bn_training.py','--stage',stage],gpu)
        for stage,gpu in [('e3','2'),('e4','3')]}
    for stage in ['e3','e4']:
        campaign.finish(stage+'_bn_preflight',checks[stage])
        report = json.loads((campaign.RUN/('preflight_'+stage)/'completed.json').read_text())
        assert report['updates'] == 2 and report['all_bn_buffers_unchanged'] == 39 and report['strict_reload']
    for name, expected in hashes.items():
        assert sha(ROOT/name) == expected, name
    state('E3_E4_ALLBN_FULL20_RUNNING')
    trains = {stage:campaign.launch(stage+'_allbn_full20_train',['-m','miss_rgbt.train',
        '--config','configs/'+stage+'_allbn_full20.json'],gpu) for stage,gpu in [('e3','2'),('e4','3')]}
    for stage in ['e3','e4']:
        campaign.finish(stage+'_allbn_full20_train',trains[stage])
        verify_training(stage,configs[stage])
    for stage in ['e3','e4']:
        for condition in ['complete','miss']:
            label = stage+'_allbn_full20_'+condition
            state(label.upper()+'_RUNNING')
            folder = ROOT/'results'/(label+'_v1')
            process = campaign.launch(label,['scripts/evaluate_model.py','--checkpoint',
                str(Path(configs[stage]['output'])/'epoch20.pth.tar'),'--output',str(folder),
                '--condition',condition,'--gpus','2,3'],'2,3')
            campaign.finish(label,process)
            campaign.collect(label,folder)
    state('CAMPAIGN_COMPLETE')
    campaign.write('completed.json',dict(seconds=time.time()-started,research_goal='Requires full metric comparison and attribution; no automatic efficacy claim.'))
    print('CAMPAIGN_COMPLETE',flush=True)


if __name__ == '__main__':
    main()
