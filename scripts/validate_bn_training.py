"""Two genuine task updates for the BN-only control, after phase1 ends."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import torch

from lib.config.cadtrack.config import cfg, update_config_from_file
from miss_rgbt.model import load_pretrained


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=['e3', 'e4'], required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    phase1 = root/'logs/attribution_20261015_v2'
    assert json.loads((phase1/'status.json').read_text())['stage'] == 'CAMPAIGN_COMPLETE'
    assert (phase1/'completed.json').is_file()
    gpu = '2' if args.stage == 'e3' else '3'
    assert os.environ['CUDA_VISIBLE_DEVICES'] == gpu
    torch.set_num_threads(1)
    config = json.loads((root/'configs'/(args.stage+'_allbn_full20.json')).read_text())
    assert config['freeze_all_bn'] is True
    config['output'] = '/data2/gaob/Miss-RGBT/checkpoints/'+args.stage+'_bn_preflight_seed2027'
    logroot = root/'logs/bn_freeze_full20_v2'/('preflight_'+args.stage)
    logroot.mkdir(parents=True, exist_ok=False)
    path = logroot/'config.json'
    path.write_text(json.dumps(config, indent=2))
    with (logroot/'training.log').open('w') as log:
        subprocess.run([sys.executable, '-m', 'miss_rgbt.train', '--config', str(path), '--max-steps', '2'],
            cwd=str(root), stdout=log, stderr=subprocess.STDOUT, check=True)
    output = Path(config['output'])
    saved = torch.load(str(output/'smoke.pth.tar'), map_location='cpu')
    assert saved['epoch'] == 1 and saved['updates'] == 2 and saved['config'] == config
    update_config_from_file(str(root/'experiments/cadtrack/cadtrack.yaml'))
    model = load_pretrained(cfg, config['checkpoint'], repair=config['repair'],
        width=config['repair_width'], repair_layer=config['repair_layer'])
    initial = model.state_dict()
    buffers = [name for name in initial if name.endswith(('running_mean', 'running_var', 'num_batches_tracked'))]
    assert len(buffers) == 39 and all(torch.equal(initial[name], saved['model'][name]) for name in buffers)
    changed = [name for name in initial if not torch.equal(initial[name], saved['model'][name])]
    assert changed and any(name.startswith('baseline.decode_fuse_search.1.') for name in changed)
    assert all(torch.isfinite(value).all() for value in saved['model'].values())
    model.load_state_dict(saved['model'], strict=True)
    rows = [json.loads(line) for line in (output/'steps.jsonl').read_text().splitlines()]
    assert [row['update'] for row in rows] == [1, 2]
    source = ['miss_rgbt/train.py','miss_rgbt/model.py','miss_rgbt/objectives.py','scripts/validate_bn_training.py']
    record = dict(stage=args.stage, physical_gpu=int(gpu), updates=2,
        all_bn_buffers_unchanged=len(buffers), changed_state_keys=len(changed),
        fusion_bn_affine_updated=True, strict_reload=True,
        peak_gpu_bytes=max(row['peak_gpu_bytes'] for row in rows),
        seconds=rows[-1]['seconds'], source_sha256={name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in source},
        scope='Two actual task updates; all 39 BN running buffers unchanged and affine parameters still train. '
            'Fresh E0 initialization; this smoke checkpoint is not used for formal training or benchmark results.')
    (logroot/'completed.json').write_text(json.dumps(record, indent=2))
    print(json.dumps(record), flush=True)


if __name__ == '__main__':
    main()
