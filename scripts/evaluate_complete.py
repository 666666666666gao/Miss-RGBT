"""Complete-input E0, same checkpoint, ground truth and toolkit as recorded Miss E0."""
import argparse
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys
import time

import numpy as np

SOURCE = Path(__file__).resolve().parents[1]
CHECKPOINT = SOURCE / 'checkpoints/pretrained/CADTrack_ep0020.pth.tar'
MASK_ROOT = Path('/data2/wangwj/Modality-missing-RGBT-Tracking-main')
ROOT = SOURCE / 'results/e0_complete_20261013'
import rgbt
TOOLKIT_GT = Path(rgbt.__file__).parent / 'gt_file'
DATASETS = {
    'LasHeR': {
        'root': '/data2/wangwj/dataset/LasHeR-0428/testingset/testingset',
        'mask': str(MASK_ROOT / 'LasHeR245-Miss/missing_results_lasher245.json'),
        'sequences': 245,
    },
    'RGBT234': {
        'root': '/data2/wangwj/dataset/RGBT234/RGB_T234',
        'mask': str(MASK_ROOT / 'RGBT234-Miss/missing_results_rgbt234.json'),
        'sequences': 234,
    },
}


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def save_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding='utf-8')


def preflight():
    ROOT.mkdir(parents=True, exist_ok=True)
    jobs = []
    report = {'checkpoint': str(CHECKPOINT), 'checkpoint_sha256': sha256(CHECKPOINT),
              'checkpoint_bytes': CHECKPOINT.stat().st_size, 'datasets': {},
              'gpu_ids': [2, 3], 'protocol': 'OPE; complete paired input on every frame; toolkit GT identical to recorded Miss E0; no model changes'}
    source_hashes = {str(p.relative_to(SOURCE)): sha256(p)
                     for p in sorted((SOURCE / 'lib').rglob('*.py'))}
    source_hashes['experiments/cadtrack/cadtrack.yaml'] = sha256(SOURCE / 'experiments/cadtrack/cadtrack.yaml')
    save_json(ROOT / 'source_sha256.json', source_hashes)
    for name, config in DATASETS.items():
        masks = json.loads(Path(config['mask']).read_text())
        assert len(masks) == config['sequences'], (name, len(masks))
        sequences = []
        states = np.zeros(4, dtype=np.int64)
        for seq in sorted(masks):
            folder = Path(config['root']) / seq
            rgb = sorted(str(p) for p in (folder / 'visible').glob('*.jpg'))
            tir = sorted(str(p) for p in (folder / 'infrared').glob('*.jpg'))
            mask = np.asarray(masks[seq]['data'], dtype=np.uint8)
            if name == 'LasHeR':
                gt_path = TOOLKIT_GT / 'LasHeR/lasher_gt' / (seq + '.txt')
                gt = np.loadtxt(str(gt_path), delimiter=',')
                gt_tir = gt
                gt_hashes = {str(gt_path): sha256(gt_path)}
            else:
                gt_path = TOOLKIT_GT / 'RGBT234/rgbt234_gt' / seq
                gt = np.loadtxt(str(gt_path / 'visible.txt'), delimiter=',')
                gt_tir = np.loadtxt(str(gt_path / 'infrared.txt'), delimiter=',')
                gt_hashes = {str(gt_path / f): sha256(gt_path / f) for f in ['visible.txt', 'infrared.txt']}
            n = masks[seq]['frames']
            assert len(rgb) == len(tir) == len(gt) == len(gt_tir) == n, (name, seq, len(rgb), len(tir), len(gt), len(gt_tir), n)
            assert mask.shape == (n, 2) and np.isin(mask, [0, 1]).all(), (name, seq)
            mask = np.ones((n, 2), dtype=np.uint8)
            counts = np.bincount(mask[:, 0] * 2 + mask[:, 1], minlength=4)
            states += counts
            sequences.append({'name': seq, 'frames': n, 'states_00_01_10_11': counts.tolist(), 'gt_sha256': gt_hashes})
            jobs.append({'dataset': name, 'sequence': seq, 'rgb': rgb, 'tir': tir,
                         'mask': mask.tolist(), 'init_bbox': gt[0].tolist(), 'frames': n})
        report['datasets'][name] = {'data_root': config['root'], 'mask': config['mask'],
                                    'mask_sha256': sha256(config['mask']),
                                    'sequences': len(sequences), 'frames': int(states.sum()),
                                    'states_00_01_10_11': states.tolist(), 'sequence_details': sequences}
    save_json(ROOT / 'inventory.json', report)
    save_json(ROOT / 'jobs.json', sorted(jobs, key=lambda x: x['frames'], reverse=True))
    print(json.dumps({k: {a: b for a, b in v.items() if a != 'sequence_details'}
                      for k, v in report['datasets'].items()}, indent=2), flush=True)


def init_worker(device=None):
    global TRACKER, CV2, TORCH, READ_FRAME
    assert os.environ['CUDA_VISIBLE_DEVICES'] == '2,3'
    import cv2
    import torch
    sys.path.insert(0, str(SOURCE))
    from lib.config.cadtrack.config import cfg, update_config_from_file
    from lib.test.utils import TrackerParams
    from lib.test.tracker.cadtrack import CADTrack
    from lib.train.dataset.depth_utils import get_x_frame
    if device is None:
        device = (mp.current_process()._identity[0] - 1) % 2
    torch.cuda.set_device(device)
    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    update_config_from_file(str(SOURCE / 'experiments/cadtrack/cadtrack.yaml'))
    params = TrackerParams()
    params.cfg = cfg
    params.checkpoint = str(CHECKPOINT)
    params.template_factor = cfg.TEST.TEMPLATE_FACTOR
    params.template_size = cfg.TEST.TEMPLATE_SIZE
    params.search_factor = cfg.TEST.SEARCH_FACTOR
    params.search_size = cfg.TEST.SEARCH_SIZE
    params.save_all_boxes = False
    params.debug = 0
    TRACKER = CADTrack(params)
    CV2, TORCH, READ_FRAME = cv2, torch, get_x_frame
    print('WORKER_READY pid={} physical_gpu={} strict_checkpoint={}'.format(os.getpid(), device + 2, CHECKPOINT), flush=True)


def run_sequence(job):
    started = time.time()
    boxes = np.zeros((job['frames'], 4), dtype=np.float64)
    boxes[0] = job['init_bbox']
    image = READ_FRAME(job['rgb'][0], job['tir'][0], dtype='rgbrgb')
    TRACKER.initialize(image, {'init_bbox': job['init_bbox']})
    for frame in range(1, job['frames']):
        rgb_ok, tir_ok = job['mask'][frame]
        if not rgb_ok and not tir_ok:
            boxes[frame] = boxes[frame - 1]
            continue
        rgb = job['rgb'][frame] if rgb_ok else job['tir'][frame]
        tir = job['tir'][frame] if tir_ok else job['rgb'][frame]
        image = READ_FRAME(rgb, tir, dtype='rgbrgb')
        boxes[frame] = TRACKER.track(image)['target_bbox']
    TORCH.cuda.synchronize()
    assert np.isfinite(boxes).all(), (job['dataset'], job['sequence'])
    result_dir = ROOT / 'results' / job['dataset']
    np.savetxt(str(result_dir / (job['sequence'] + '.txt')), boxes)
    seconds = time.time() - started
    receipt = {'dataset': job['dataset'], 'sequence': job['sequence'], 'frames': job['frames'],
               'elapsed_seconds': seconds, 'fps_including_skipped_frames': job['frames'] / seconds,
               'physical_gpu': TORCH.cuda.current_device() + 2,
               'result_sha256': sha256(result_dir / (job['sequence'] + '.txt'))}
    save_json(ROOT / 'receipts' / job['dataset'] / (job['sequence'] + '.json'), receipt)
    print('DONE ' + json.dumps(receipt), flush=True)
    return receipt


def evaluate(result_root, output_path):
    from rgbt import LasHeR, RGBT234
    summary = {}
    inventory = json.loads((ROOT / 'inventory.json').read_text())
    for name, cls, metrics in [('LasHeR', LasHeR, ['PR', 'NPR', 'SR']),
                                ('RGBT234', RGBT234, ['MPR', 'MSR'])]:
        dataset = cls()
        expected = {s['name']: s['frames'] for s in inventory['datasets'][name]['sequence_details']}
        assert set(dataset.seqs_name) == set(expected), name
        directory = Path(result_root) / name
        for seq, n in expected.items():
            bbox = np.loadtxt(str(directory / (seq + '.txt')))
            assert bbox.shape == (n, 4) and np.isfinite(bbox).all(), (name, seq, bbox.shape)
        dataset(tracker_name='CADTrack_ep0020', result_path=str(directory), bbox_type='ltwh')
        summary[name] = {'sequences': len(expected), 'frames': sum(expected.values())}
        curves = {}
        for metric in metrics:
            value, curve = getattr(dataset, metric)(tracker_name='CADTrack_ep0020')
            summary[name][metric] = float(value * 100)
            curves[metric] = curve.tolist()
        save_json(ROOT / (name + '_curves.json'), curves)
    save_json(output_path, summary)
    print(json.dumps(summary, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['preflight', 'smoke', 'run', 'evaluate'])
    args = parser.parse_args()
    if args.mode == 'preflight':
        preflight()
    elif args.mode == 'smoke':
        init_worker(0)
        jobs = json.loads((ROOT / 'jobs.json').read_text())
        job = jobs[0]
        image = READ_FRAME(job['rgb'][0], job['tir'][0], dtype='rgbrgb')
        TRACKER.initialize(image, {'init_bbox': job['init_bbox']})
        for state in [[1, 1]]:
            rgb = job['rgb'][1] if state[0] else job['tir'][1]
            tir = job['tir'][1] if state[1] else job['rgb'][1]
            bbox = TRACKER.track(READ_FRAME(rgb, tir, dtype='rgbrgb'))['target_bbox']
            assert np.isfinite(bbox).all()
            print('SMOKE state={} bbox={}'.format(state, bbox), flush=True)
        print('SMOKE_PASS', flush=True)
    elif args.mode == 'run':
        for name in DATASETS:
            (ROOT / 'results' / name).mkdir(parents=True, exist_ok=True)
            (ROOT / 'receipts' / name).mkdir(parents=True, exist_ok=True)
        jobs = json.loads((ROOT / 'jobs.json').read_text())
        receipts = [json.loads(p.read_text()) for p in (ROOT / 'receipts').glob('*/*.json')]
        completed = {(r['dataset'], r['sequence']) for r in receipts}
        jobs = [j for j in jobs if (j['dataset'], j['sequence']) not in completed]
        started = time.time()
        mp.set_start_method('spawn', force=True)
        with mp.Pool(processes=8, initializer=init_worker) as pool:
            receipts.extend(pool.imap_unordered(run_sequence, jobs, chunksize=1))
        save_json(ROOT / 'run_receipts.json', receipts)
        evaluate(ROOT / 'results', ROOT / 'metrics.json')
        save_json(ROOT / 'completed.json', {'sequences': len(receipts), 'elapsed_seconds': time.time() - started})
    else:
        evaluate(ROOT / 'results', ROOT / 'metrics.json')


if __name__ == '__main__':
    main()
