"""Full paired complete/Miss OPE with a strictly loaded trained single model."""
import argparse
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import time

import numpy as np
import rgbt


PROJECT = Path(__file__).resolve().parents[1]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2), encoding='utf-8')


def prepare(condition):
    inventory = json.loads((PROJECT / 'reports/e0/inventory.json').read_text())
    jobs = []
    gt_root = Path(rgbt.__file__).parent / 'gt_file'
    for name, spec in inventory['datasets'].items():
        assert sha(spec['mask']) == spec['mask_sha256']
        masks = json.loads(Path(spec['mask']).read_text())
        assert len(masks) == spec['sequences']
        for item in spec['sequence_details']:
            seq, n = item['name'], item['frames']
            folder = Path(spec['data_root']) / seq
            rgb = sorted(str(p) for p in (folder / 'visible').glob('*.jpg'))
            tir = sorted(str(p) for p in (folder / 'infrared').glob('*.jpg'))
            gt_files = [gt_root / 'LasHeR/lasher_gt' / (seq + '.txt')] if name == 'LasHeR-Miss' else [
                gt_root / 'RGBT234/rgbt234_gt' / seq / f for f in ['visible.txt', 'infrared.txt']]
            expected_gt = list(item['gt_sha256'].values())
            for gt_file, expected in zip(gt_files, expected_gt):
                assert sha(gt_file) == expected, gt_file
                gt = np.loadtxt(str(gt_file), delimiter=',')
                assert gt.shape == (n, 4), (name, seq, gt.shape)
            first_gt = np.loadtxt(str(gt_files[0]), delimiter=',')[0].tolist()
            mask = np.asarray(masks[seq]['data'], dtype=np.uint8)
            assert len(rgb) == len(tir) == n and mask.shape == (n, 2)
            assert np.isin(mask, [0, 1]).all()
            if condition == 'complete':
                mask = np.ones((n, 2), dtype=np.uint8)
            jobs.append(dict(dataset=name, sequence=seq, rgb=rgb, tir=tir,
                             mask=mask.tolist(), frames=n, init_bbox=first_gt))
    assert len(jobs) == 479 and sum(j['frames'] for j in jobs) == 337353
    return sorted(jobs, key=lambda x: x['frames'], reverse=True), inventory


def worker_init(checkpoint, output):
    global TRACKER, READ_FRAME, TORCH, ROOT
    assert os.environ['CUDA_VISIBLE_DEVICES'] == '2,3'
    import cv2
    import torch
    from lib.config.cadtrack.config import cfg, update_config_from_file
    from lib.train.dataset.depth_utils import get_x_frame
    from miss_rgbt.tracker import MissingTracker
    device = (mp.current_process()._identity[0] - 1) % 2
    torch.cuda.set_device(device)
    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    update_config_from_file(str(PROJECT / 'experiments/cadtrack/cadtrack.yaml'))
    TRACKER = MissingTracker(cfg, checkpoint)
    READ_FRAME, TORCH, ROOT = get_x_frame, torch, Path(output)
    print('WORKER_READY pid={} physical_gpu={}'.format(os.getpid(), device + 2), flush=True)


def sequence(job):
    start = time.time()
    boxes = np.zeros((job['frames'], 4), dtype=np.float64)
    boxes[0] = job['init_bbox']
    TRACKER.initialize(READ_FRAME(job['rgb'][0], job['tir'][0], dtype='rgbrgb'), dict(init_bbox=job['init_bbox']))
    for i in range(1, job['frames']):
        r, t = job['mask'][i]
        if not r and not t:
            boxes[i] = boxes[i - 1]
            continue
        rgb = job['rgb'][i] if r else job['tir'][i]
        tir = job['tir'][i] if t else job['rgb'][i]
        boxes[i] = TRACKER.track(READ_FRAME(rgb, tir, dtype='rgbrgb'), [r, t])['target_bbox']
    TORCH.cuda.synchronize()
    assert np.isfinite(boxes).all()
    path = ROOT / 'bbox' / job['dataset'] / (job['sequence'] + '.txt')
    np.savetxt(str(path), boxes)
    receipt = dict(dataset=job['dataset'], sequence=job['sequence'], frames=job['frames'],
                   seconds=time.time() - start, physical_gpu=TORCH.cuda.current_device() + 2,
                   bbox_sha256=sha(path))
    save(ROOT / 'receipts' / job['dataset'] / (job['sequence'] + '.json'), receipt)
    print('DONE ' + json.dumps(receipt), flush=True)
    return receipt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--condition', choices=['complete', 'miss'], required=True)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    jobs, inventory = prepare(args.condition)
    checkpoint = str(Path(args.checkpoint).resolve())
    hashes = {str(p.relative_to(PROJECT)): sha(p) for folder in ['miss_rgbt', 'lib', 'scripts']
              for p in sorted((PROJECT / folder).rglob('*.py'))}
    save(output / 'run.json', dict(checkpoint=checkpoint, checkpoint_sha256=sha(checkpoint),
        condition=args.condition, workers=8, physical_gpus=[2, 3], source_sha256=hashes,
        frames=337353, sequences=479, protocol='fixed published masks; first frame complete; both missing hold; trained single-view inference; all frames scored'))
    for name in inventory['datasets']:
        (output / 'bbox' / name).mkdir(parents=True)
        (output / 'receipts' / name).mkdir(parents=True)
    started = time.time()
    mp.set_start_method('spawn', force=True)
    with mp.Pool(8, initializer=worker_init, initargs=(checkpoint, str(output))) as pool:
        receipts = list(pool.imap_unordered(sequence, jobs, chunksize=1))
    assert len(receipts) == 479
    save(output / 'all_receipts.json', receipts)
    metrics = {}
    for name, cls, keys in [('LasHeR-Miss', rgbt.LasHeR, ['PR', 'NPR', 'SR']),
                            ('RGBT234-Miss', rgbt.RGBT234, ['MPR', 'MSR'])]:
        expected = {s['name']: s['frames'] for s in inventory['datasets'][name]['sequence_details']}
        dataset = cls()
        assert set(dataset.seqs_name) == set(expected)
        for seq, n in expected.items():
            boxes = np.loadtxt(str(output / 'bbox' / name / (seq + '.txt')))
            assert boxes.shape == (n, 4) and np.isfinite(boxes).all()
        dataset('model', str(output / 'bbox' / name), bbox_type='ltwh')
        label = name if args.condition == 'miss' else name.replace('-Miss', '')
        metrics[label] = dict(sequences=len(expected), frames=sum(expected.values()))
        curves = {}
        for key in keys:
            score, curve = getattr(dataset, key)('model')
            metrics[label][key] = float(score * 100)
            curves[key] = curve.tolist()
        save(output / (label + '_curves.json'), curves)
    save(output / 'metrics.json', metrics)
    save(output / 'completed.json', dict(sequences=479, frames=337353, seconds=time.time() - started))
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == '__main__':
    main()
