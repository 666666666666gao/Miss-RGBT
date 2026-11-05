"""Replay fixed original development tensors for E0/epoch3 loss diagnosis."""
import argparse
from collections import defaultdict
import hashlib
import io
import json
from pathlib import Path
import time

import cv2
import torch

import train_partitioned_a as base
from partitioned_a_official_queue_v2 import gpu_ready


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--gpu', required=True, type=int, choices=[0, 1])
    args = parser.parse_args()
    output = Path(args.output)
    assert not output.exists()
    preflight = gpu_ready(args.gpu)
    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    base.update_config_from_file('experiments/cadtrack/cadtrack.yaml')
    run = base.PROJECT/'runs/server28_partitioned_a_20261102_formal/utility'
    config = json.loads((run/'run_config.json').read_text())
    boundary = json.loads((run/'data_boundary.json').read_text())
    source_hashes = json.loads((run/'source_sha256.json').read_text())
    assert all(base.prior.common.sha256(path) == digest for path, digest in source_hashes.items())
    used = defaultdict(set, {name: set(frames)
                            for name, frames in boundary['excluded_frames'].items()})
    expected = []
    with (run/'input_manifest.jsonl').open() as stream:
        for line in stream:
            row = json.loads(line)
            if row['phase'] != 'development':
                break
            expected.append(row)
    assert len(expected) == config['development_samples']//4 == 32
    base.seed(config['development_seed'])
    loader = base.loader_for(config, boundary['partitions']['development'],
        config['development_samples'], 4, 0, used)
    batches = []
    for index, data in enumerate(loader):
        record = base.record_batch(data, boundary['partitions']['development'],
                                   'development', 0, index, io.StringIO())
        assert record == expected[index], index
        batches.append(data)
    assert len(batches) == len(expected)
    base.seed(config['seed'])
    assert base.prior.common.sha256(config['checkpoint']) == base.prior.common.CHECKPOINT_SHA
    model = base.load_pretrained(base.cfg, config['checkpoint'], repair=False,
                                 task_utility=config['task_utility']).cuda()
    base.fix_bn(model, True)
    initial = json.loads((run/'initial_state.json').read_text())
    actual = base.prior.common.digest(dict(model.named_parameters()))
    assert actual == initial['parameters_sha256']
    started = time.time()
    before = base.validate(model, batches)
    checkpoint = run/'epoch03.pth.tar'
    checkpoint_sha = base.prior.common.sha256(checkpoint)
    saved = torch.load(str(checkpoint), map_location='cpu')
    assert saved['epoch'] == 3 and saved['config'] == config
    model.load_state_dict(saved['model'], strict=True)
    del saved
    after = base.validate(model, batches)
    recorded = json.loads((run/'development_history.json').read_text())[2]
    assert recorded['epoch'] == 3
    for key in ['11', '01', '10']:
        assert abs(after['task_loss'][key]-recorded['task_loss'][key]) < 1e-6
    assert abs(after['mean_task_loss']-recorded['mean_task_loss']) < 1e-6
    report = dict(status='PASS', preflight=preflight,
        tensors_and_identities_exact=True, initial_parameters_exact=True,
        epoch3_record_reproduced=True, samples=config['development_samples'],
        initial=before, epoch3=after,
        delta={key: after['task_loss'][key]-before['task_loss'][key]
               for key in ['11', '01', '10']},
        mean_delta=after['mean_task_loss']-before['mean_task_loss'],
        seconds=time.time()-started, peak_gpu_bytes=torch.cuda.max_memory_allocated(),
        checkpoint_sha256=checkpoint_sha,
        source_sha256={str(Path(__file__).resolve()): base.prior.common.sha256(__file__),
                       str(Path(base.__file__).resolve()): base.prior.common.sha256(base.__file__)},
        input_records_sha256=hashlib.sha256(
            json.dumps(expected, sort_keys=True).encode()).hexdigest(),
        scope='Same fixed train882 development task loss; not official tracking accuracy, '
              'convergence, a new training run or a checkpoint-selection change')
    base.save(output, report)
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
