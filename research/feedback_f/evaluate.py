"""Full 479-sequence F OPE using the frozen native dataset and sequence runner."""
import argparse
import json
import multiprocessing as mp
import os
from pathlib import Path
import time

import numpy as np
import rgbt

import evaluate_candidate_official28 as native

def worker_init(checkpoint, output, physical_gpus):
    import cv2
    import torch
    from lib.config.cadtrack.config import cfg, update_config_from_file
    from lib.train.dataset.depth_utils import get_x_frame
    from tracker import FeedbackTracker

    assert os.environ['CUDA_VISIBLE_DEVICES'] == ','.join(str(gpu) for gpu in physical_gpus)
    device = (mp.current_process()._identity[0]-1) % len(physical_gpus)
    torch.cuda.set_device(device)
    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    update_config_from_file(str(native.PROJECT/'experiments/cadtrack/cadtrack.yaml'))
    native.TRACKER = FeedbackTracker(cfg, checkpoint)
    native.READ_FRAME, native.TORCH, native.ROOT = get_x_frame, torch, Path(output)
    native.PHYSICAL_GPUS = physical_gpus
    print('WORKER_READY pid={} physical_gpu={}'.format(
        os.getpid(), physical_gpus[device]), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--condition', required=True, choices=['complete', 'miss'])
    parser.add_argument('--gpus', required=True, choices=['0', '1', '2', '3', '0,1', '2,3'])
    parser.add_argument('--workers-per-gpu', type=int, default=4)
    args = parser.parse_args()
    assert args.workers_per_gpu >= 1
    checkpoint = str(Path(args.checkpoint).resolve())
    checkpoint_sha = native.sha(checkpoint)
    jobs, inventory = native.prepare(args.condition)
    assert len(jobs) == 479 and sum(job['frames'] for job in jobs) == 337353
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    physical_gpus = [int(gpu) for gpu in args.gpus.split(',')]
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    workers = args.workers_per_gpu*len(physical_gpus)
    runtime_sources = {
        str(path.relative_to(native.PROJECT)): native.sha(path)
        for folder in ['miss_rgbt', 'lib', 'scripts']
        for path in sorted((native.PROJECT/folder).rglob('*.py'))}
    extra = [Path(__file__), Path(__file__).with_name('tracker.py'),
             Path(__file__).with_name('model.py'), Path(native.__file__)]
    native.save(output/'run.json', dict(checkpoint=checkpoint, checkpoint_sha256=checkpoint_sha,
        standalone_evaluator_sha256=native.sha(__file__),
        native_evaluator_sha256=native.sha(native.__file__),
        feedback_source_sha256={str(path.resolve()): native.sha(path) for path in extra},
        condition=args.condition, workers=workers, physical_gpus=physical_gpus,
        source_sha256=runtime_sources, workers_per_gpu=args.workers_per_gpu,
        frames=337353, sequences=479, limit=None,
        protocol='Full published inventory; native mask/first-frame/both-missing handling; '
                 'inherited crop/decode/template update; F round2 or single control'))
    for dataset in inventory['datasets']:
        (output/'bbox'/dataset).mkdir(parents=True)
        (output/'receipts'/dataset).mkdir(parents=True)
    started = time.time()
    mp.set_start_method('spawn', force=True)
    with mp.Pool(workers, initializer=worker_init,
                 initargs=(checkpoint, str(output), physical_gpus)) as pool:
        receipts = list(pool.imap_unordered(native.sequence, jobs, chunksize=1))
    assert len(receipts) == 479 and sum(row['frames'] for row in receipts) == 337353
    assert len({(row['dataset'], row['sequence']) for row in receipts}) == 479
    assert native.sha(checkpoint) == checkpoint_sha
    native.save(output/'all_receipts.json', receipts)
    metrics = {}
    for dataset, cls, keys in [('LasHeR-Miss', rgbt.LasHeR, ['PR', 'NPR', 'SR']),
                               ('RGBT234-Miss', rgbt.RGBT234, ['MPR', 'MSR'])]:
        expected = {row['name']: row['frames']
                    for row in inventory['datasets'][dataset]['sequence_details']}
        scorer = cls()
        assert set(scorer.seqs_name) == set(expected)
        for sequence, frames in expected.items():
            boxes = np.loadtxt(str(output/'bbox'/dataset/(sequence+'.txt')), ndmin=2)
            assert boxes.shape == (frames, 4) and np.isfinite(boxes).all()
        scorer('model', str(output/'bbox'/dataset), bbox_type='ltwh')
        label = dataset if args.condition == 'miss' else dataset.replace('-Miss', '')
        metrics[label] = dict(sequences=len(expected), frames=sum(expected.values()))
        curves = {}
        for key in keys:
            score, curve = getattr(scorer, key)('model')
            metrics[label][key] = float(score*100)
            curves[key] = curve.tolist()
        native.save(output/(label+'_curves.json'), curves)
    native.save(output/'metrics.json', metrics)
    native.save(output/'completed.json', dict(sequences=479, frames=337353,
                                            seconds=time.time()-started))
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == '__main__':
    main()
