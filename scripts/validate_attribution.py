"""Real two-update and single-card inference witness for phase1 controls."""
import argparse
import json
import multiprocessing as mp
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch

from lib.config.cadtrack.config import cfg, update_config_from_file
from miss_rgbt.model import load_pretrained
from evaluate_model import worker_init, sequence


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=['e0_ft', 'e2'], required=True)
    args = parser.parse_args()
    gpu = '2' if args.stage == 'e0_ft' else '3'
    assert os.environ['CUDA_VISIBLE_DEVICES'] == gpu
    torch.set_num_threads(1)
    root = Path(__file__).resolve().parents[1]
    config = json.loads((root / 'configs' / (args.stage + '_full20.json')).read_text())
    config['output'] = 'checkpoints/' + args.stage + '_attribution_smoke_seed2027'
    config['source_version_file'] = 'reports/ATTRIBUTION_PREFLIGHT_SOURCE_VERSION.txt'
    logroot = root / 'logs/attribution_preflight_20261015_v1' / args.stage
    logroot.mkdir(parents=True, exist_ok=False)
    config_path = logroot / 'config.json'
    config_path.write_text(json.dumps(config, indent=2))
    with (logroot / 'training.log').open('w') as log:
        subprocess.run([sys.executable, '-m', 'miss_rgbt.train', '--config', str(config_path), '--max-steps', '2'],
            cwd=str(root), stdout=log, stderr=subprocess.STDOUT, check=True)
    output = root / config['output']
    saved = torch.load(str(output / 'smoke.pth.tar'), map_location='cpu')
    assert saved['epoch'] == 1 and saved['updates'] == 2 and saved['config'] == config
    update_config_from_file(str(root / 'experiments/cadtrack/cadtrack.yaml'))
    model = load_pretrained(cfg, config['checkpoint'], repair=False, width=128, repair_layer=6)
    initial = model.state_dict()
    bn = 'baseline.decode_fuse_search.1.num_batches_tracked'
    increment = int(saved['model'][bn] - initial[bn])
    assert increment == (12 if args.stage == 'e0_ft' else 24), increment
    head = [name for name in initial if name.startswith('baseline.box_head.') and
        name.endswith(('running_mean', 'running_var', 'num_batches_tracked'))]
    assert len(head) == 36 and all(torch.equal(initial[name], saved['model'][name]) for name in head)
    changed = [name for name in initial if not torch.equal(initial[name], saved['model'][name])]
    assert changed and all(torch.isfinite(value).all() for value in saved['model'].values())
    model.load_state_dict(saved['model'], strict=True)
    steps = [json.loads(line) for line in (output / 'steps.jsonl').read_text().splitlines()]
    assert [row['update'] for row in steps] == [1, 2]
    if args.stage == 'e0_ft':
        assert all('task_m' not in row and row['loss'] == row['task_c'] for row in steps)
    else:
        assert all(abs(row['loss'] - row['task_c'] - row['task_m']) < 2e-6 for row in steps)
    del model, initial, saved
    name = (root / 'lib/train/data_specs/lasher_train.txt').read_text().splitlines()[0]
    folder = Path(config['dataset']) / name
    rgb = sorted(str(path) for path in (folder / 'visible').glob('*.jpg'))[:3]
    tir = sorted(str(path) for path in (folder / 'infrared').glob('*.jpg'))[:3]
    box = np.loadtxt(str(folder / 'init.txt'), delimiter=',')[0].tolist()
    infer = logroot / 'inference'
    dataset = 'LasHeR-Miss'
    (infer / 'bbox' / dataset).mkdir(parents=True)
    (infer / 'receipts' / dataset).mkdir(parents=True)
    jobs = [dict(dataset=dataset, sequence='train_' + name + '_' + str(i), rgb=rgb, tir=tir,
        frames=3, mask=[[1, 1], state, state], init_bbox=box)
        for i, state in enumerate([[1, 1], [0, 1], [1, 0]])]
    mp.set_start_method('spawn', force=True)
    with mp.Pool(1, initializer=worker_init, initargs=(str(root / 'checkpoints/e0_availability/epoch20.pth.tar'),
        str(infer), [int(gpu)])) as pool:
        receipts = list(pool.imap_unordered(sequence, jobs))
    assert len(receipts) == 3 and all(row['physical_gpu'] == int(gpu) and row['frames'] == 3 for row in receipts)
    record = dict(stage=args.stage, updates=2, optimizer_update_and_strict_reload=True,
        changed_state_keys=len(changed), fusion_bn_count_increment=increment,
        unchanged_head_bn_buffers=36, peak_gpu_bytes=max(row['peak_gpu_bytes'] for row in steps),
        last_training_seconds=steps[-1]['seconds'], physical_gpu=int(gpu), inference_receipts=receipts,
        scope='Two real updates plus complete/RGB-missing/TIR-missing train-sequence inference on one physical card; not formal training or benchmark.')
    (logroot / 'completed.json').write_text(json.dumps(record, indent=2))
    print(json.dumps(record), flush=True)


if __name__ == '__main__':
    main()
