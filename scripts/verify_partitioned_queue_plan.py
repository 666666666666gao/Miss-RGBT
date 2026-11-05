"""Check queue planning and strict inference on synthetic images, never test data."""
import hashlib
import json
import os
from pathlib import Path
import subprocess

import cv2
import numpy as np
import torch
from lib.config.cadtrack.config import cfg, update_config_from_file
from miss_rgbt.tracker import MissingTracker
import partitioned_a_official_queue as queue


def parameter_hash(model):
    digest = hashlib.sha256()
    for name, value in model.named_parameters():
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def main():
    assert os.environ['CUDA_VISIBLE_DEVICES'] == '0'
    queue.gpu_ready(0)
    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    update_config_from_file(str(queue.RUNTIME/'experiments/cadtrack/cadtrack.yaml'))
    torch.manual_seed(64101)
    torch.cuda.manual_seed_all(64101)
    yy, xx = np.indices((320, 320))
    image = np.stack([(xx*(channel+1)+yy*(7-channel)) % 256
                      for channel in range(6)], axis=-1).astype(np.uint8)
    records = {}
    for selector in ['utility', 'random']:
        root = queue.ROOT/'runs/server28_partitioned_a_20261102_smoke'/selector
        completed = queue.read(root/'completed.json')
        jobs = queue.jobs_for(selector, completed)
        assert [job['epoch'] for job in jobs] == [epoch for epoch in range(2, 7) for _ in range(2)]
        assert [job['condition'] for job in jobs] == ['complete', 'miss']*5
        for job in jobs:
            assert queue.sha(job['checkpoint']) == job['checkpoint_sha256']
            command = queue.command_for(job, 2 if selector == 'utility' else 3)
            assert '--limit' not in command and '--workers-per-gpu' in command
            assert 'val97' not in ' '.join(command)
            assert job['expected_sequences'] == 479 and job['expected_frames'] == 337353
            assert not Path(job['output']).exists()
        selected = [job for job in jobs if job['development_selected']]
        assert len(selected) == 2 and len({job['checkpoint_sha256'] for job in selected}) == 1
        tracker = MissingTracker(cfg, completed['selected_checkpoint'])
        before = parameter_hash(tracker.network)
        tracker.initialize(image, dict(init_bbox=[120., 110., 40., 50.]))
        outputs = []
        for flags in [[1, 1], [0, 1], [1, 0]]:
            observed = image if flags == [1, 1] else np.concatenate(
                [image[:, :, 3:]]*2 if flags == [0, 1] else [image[:, :, :3]]*2, axis=-1)
            result = tracker.track(observed, flags)
            assert np.isfinite(result['target_bbox']).all() and np.isfinite(result['best_score'])
            outputs.append(dict(flags=flags, output=result))
        assert parameter_hash(tracker.network) == before
        assert all(value.grad is None for value in tracker.network.parameters())
        records[selector] = dict(plan_jobs=len(jobs), full_scope_only=True,
            common_selected_checkpoint=True, strict_checkpoint_load=True,
            checkpoint_sha256=completed['selected_sha256'], synthetic_outputs=outputs,
            parameters_unchanged=True, gradients_absent=True)
        del tracker
        torch.cuda.empty_cache()
    # Check live and terminal process identities without waiting for real training.
    process = subprocess.Popen(['sleep', '10'])
    fixture = dict(pid=process.pid, command=['sleep', '', 'sleep', '10'])
    assert queue.training_alive(fixture)
    process.terminate()
    process.wait()
    assert not queue.training_alive(fixture)
    result = dict(status='passed', arms=records, process_identity_checks=True,
        checkpoint_plan_jobs=20, benchmark_jobs_executed=0, physical_gpu=0,
        source_sha256={str(path.resolve()): queue.sha(path) for path in [
            Path(__file__), Path(queue.__file__)]},
        scope='Queue planning and synthetic-image inference only; no official/development metrics.')
    path = queue.ROOT/'reports/server28_partitioned_a_20261102_queue_checks.json'
    with path.open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
