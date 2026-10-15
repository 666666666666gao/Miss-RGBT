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


def worker_init(checkpoint, output, physical_gpus, template_policy, update_interval,
                update_threshold, update_audit):
    global TRACKER, READ_FRAME, TORCH, ROOT, PHYSICAL_GPUS, UPDATE_AUDIT
    PHYSICAL_GPUS = physical_gpus
    assert os.environ['CUDA_VISIBLE_DEVICES'] == ','.join(str(gpu) for gpu in physical_gpus)
    import cv2
    import torch
    from lib.config.cadtrack.config import cfg, update_config_from_file
    from lib.train.dataset.depth_utils import get_x_frame
    from miss_rgbt.tracker import MissingTracker
    device = (mp.current_process()._identity[0] - 1) % len(physical_gpus)
    torch.cuda.set_device(device)
    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    update_config_from_file(str(PROJECT / 'experiments/cadtrack/cadtrack.yaml'))
    cfg.TEST.UPDATE_INTERVALS = update_interval
    cfg.TEST.UPDATE_THRESHOLD = update_threshold
    TRACKER = MissingTracker(cfg, checkpoint, template_policy=template_policy, update_audit=update_audit)
    UPDATE_AUDIT = update_audit
    READ_FRAME, TORCH, ROOT = get_x_frame, torch, Path(output)
    print('WORKER_READY pid={} physical_gpu={}'.format(os.getpid(), physical_gpus[device]), flush=True)


def sequence(job):
    start = time.time()
    boxes = np.zeros((job['frames'], 4), dtype=np.float64)
    boxes[0] = job['init_bbox']
    TRACKER.initialize(READ_FRAME(job['rgb'][0], job['tir'][0], dtype='rgbrgb'), dict(init_bbox=job['init_bbox']))
    update_events = []
    last_write_video_frame = 0
    for i in range(1, job['frames']):
        r, t = job['mask'][i]
        if not r and not t:
            boxes[i] = boxes[i - 1]
            if UPDATE_AUDIT:
                update_events.append(dict(video_frame=i, track_call=TRACKER.frame_id,
                    available=[0, 0], both_missing_hold=True, template_written=False,
                    confidence=None, video_template_age_before=i - last_write_video_frame,
                    template_age_before=TRACKER.frame_id - TRACKER.last_template_write,
                    previous_template_available=list(TRACKER.last_template_available)))
            continue
        rgb = job['rgb'][i] if r else job['tir'][i]
        tir = job['tir'][i] if t else job['rgb'][i]
        boxes[i] = TRACKER.track(READ_FRAME(rgb, tir, dtype='rgbrgb'), [r, t])['target_bbox']
        if UPDATE_AUDIT:
            event = dict(TRACKER.last_update_event, video_frame=i, both_missing_hold=False,
                         video_template_age_before=i - last_write_video_frame)
            update_events.append(event)
            if event['template_written']:
                last_write_video_frame = i
    TORCH.cuda.synchronize()
    assert np.isfinite(boxes).all()
    path = ROOT / 'bbox' / job['dataset'] / (job['sequence'] + '.txt')
    np.savetxt(str(path), boxes)
    receipt = dict(dataset=job['dataset'], sequence=job['sequence'], frames=job['frames'],
                   seconds=time.time() - start, physical_gpu=PHYSICAL_GPUS[TORCH.cuda.current_device()],
                   bbox_sha256=sha(path))
    if UPDATE_AUDIT:
        trace = ROOT / 'template_updates' / job['dataset'] / (job['sequence'] + '.json')
        save(trace, dict(indexing='zero-based video_frame; initialization at frame 0; track_call excludes both-missing holds',
            policy=TRACKER.template_policy, interval=TRACKER.cfg.TEST.UPDATE_INTERVALS,
            threshold=TRACKER.cfg.TEST.UPDATE_THRESHOLD, initial_template_slots=TRACKER.num_template,
            events=update_events))
        receipt['template_updates_sha256'] = sha(trace)
        receipt['template_writes'] = sum(event['template_written'] for event in update_events)
    save(ROOT / 'receipts' / job['dataset'] / (job['sequence'] + '.json'), receipt)
    print('DONE ' + json.dumps(receipt), flush=True)
    return receipt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--condition', choices=['complete', 'miss'], required=True)
    parser.add_argument('--gpus', choices=['2', '3', '2,3'], default='2,3')
    parser.add_argument('--template-policy', choices=['off', 'current', 'complete_only'], default='current')
    parser.add_argument('--update-interval', type=int)
    parser.add_argument('--update-threshold', type=float)
    parser.add_argument('--update-audit', action='store_true')
    args = parser.parse_args()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    jobs, inventory = prepare(args.condition)
    checkpoint = str(Path(args.checkpoint).resolve())
    from lib.config.cadtrack.config import cfg, update_config_from_file
    update_config_from_file(str(PROJECT / 'experiments/cadtrack/cadtrack.yaml'))
    update_interval = cfg.TEST.UPDATE_INTERVALS if args.update_interval is None else args.update_interval
    update_threshold = cfg.TEST.UPDATE_THRESHOLD if args.update_threshold is None else args.update_threshold
    assert update_interval > 0 and 0 <= update_threshold <= 1
    hashes = {str(p.relative_to(PROJECT)): sha(p) for folder in ['miss_rgbt', 'lib', 'scripts']
              for p in sorted((PROJECT / folder).rglob('*.py'))}
    hashes['experiments/cadtrack/cadtrack.yaml'] = sha(PROJECT / 'experiments/cadtrack/cadtrack.yaml')
    physical_gpus = [int(gpu) for gpu in args.gpus.split(',')]
    workers = 4 * len(physical_gpus)
    save(output / 'run.json', dict(checkpoint=checkpoint, checkpoint_sha256=sha(checkpoint),
        condition=args.condition, workers=workers, physical_gpus=physical_gpus, source_sha256=hashes,
        frames=337353, sequences=479, template_policy=args.template_policy,
        update_interval=update_interval, update_threshold=update_threshold, update_audit=args.update_audit,
        protocol='fixed published masks; first frame complete; both missing hold; trained single-view inference; all frames scored'))
    for name in inventory['datasets']:
        (output / 'bbox' / name).mkdir(parents=True)
        (output / 'receipts' / name).mkdir(parents=True)
        if args.update_audit:
            (output / 'template_updates' / name).mkdir(parents=True)
    started = time.time()
    mp.set_start_method('spawn', force=True)
    with mp.Pool(workers, initializer=worker_init, initargs=(checkpoint, str(output), physical_gpus,
            args.template_policy, update_interval, update_threshold, args.update_audit)) as pool:
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
