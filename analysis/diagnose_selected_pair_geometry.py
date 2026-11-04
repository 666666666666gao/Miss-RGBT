"""Full closed-pair box geometry; no tracker, state subsets or weight selection."""
import argparse
import json
from pathlib import Path

import numpy as np

from analysis.analyze_paired_baseline import errors_and_iou, sha


def read(path):
    return json.loads(Path(path).read_text())


def closed_pair(complete, missing):
    checkpoint = None
    for condition, folder in [('complete', complete), ('miss', missing)]:
        folder = Path(folder)
        run, done = read(folder / 'run.json'), read(folder / 'completed.json')
        assert run['condition'] == condition and run['limit'] is None
        assert run['sequences'] == done['sequences'] == 479
        assert run['frames'] == done['frames'] == 337353
        assert (folder / 'metrics.json').is_file()
        assert (folder / 'all_receipts.json').is_file()
        if checkpoint is None:
            checkpoint = run['checkpoint_sha256']
        assert run['checkpoint_sha256'] == checkpoint
    return checkpoint


def geometry(prediction, target):
    error, iou = errors_and_iou(prediction, target)
    target_center = target[:, :2] + target[:, 2:] / 2
    prediction_center = prediction[:, :2] + prediction[:, 2:] / 2
    centered = prediction.copy()
    centered[:, :2] = target_center - prediction[:, 2:] / 2
    resized = prediction.copy()
    resized[:, 2:] = target[:, 2:]
    resized[:, :2] = prediction_center - target[:, 2:] / 2
    _, center_iou = errors_and_iou(centered, target)
    _, size_iou = errors_and_iou(resized, target)
    size_error = np.abs(prediction[:, 2:] / target[:, 2:] - 1) * 100
    return np.column_stack((error, error / np.sqrt(target[:, 2:].prod(1)),
                            iou, center_iou, size_iou, size_error))


def summarize(values):
    means = values.mean(0)
    return dict(valid_frames=len(values),
        center_error_px_median=float(np.median(values[:, 0])),
        center_error_px_p90=float(np.percentile(values[:, 0], 90)),
        normalized_center_error_median=float(np.median(values[:, 1])),
        normalized_center_error_p90=float(np.percentile(values[:, 1], 90)),
        mean_iou_pct=float(100 * means[2]),
        recentered_mean_iou_pct=float(100 * means[3]),
        resized_mean_iou_pct=float(100 * means[4]),
        recentered_gain_pp=float(100 * (means[3] - means[2])),
        resized_gain_pp=float(100 * (means[4] - means[2])),
        width_mape_pct=float(means[5]), height_mape_pct=float(means[6]))


def diagnose(complete, missing, inventory_path, gt_root):
    complete, missing, gt_root = map(Path, (complete, missing, gt_root))
    checkpoint = closed_pair(complete, missing)
    inventory_path = Path(inventory_path)
    inventory = read(inventory_path)['datasets']
    assert set(inventory) == {'LasHeR-Miss', 'RGBT234-Miss'}
    inputs = {str(path): sha(path) for path in
              [Path(__file__), Path(__file__).with_name('analyze_paired_baseline.py'),
               inventory_path]}
    receipts = {}
    for condition, folder in [('complete', complete), ('miss', missing)]:
        rows = read(folder / 'all_receipts.json')
        receipts[condition] = {(row['dataset'], row['sequence']): row for row in rows}
        assert len(rows) == len(receipts[condition]) == 479
        assert sum(row['frames'] for row in rows) == 337353
        assert set(receipts[condition]) == {(dataset, item['name'])
            for dataset, spec in inventory.items() for item in spec['sequence_details']}
        for name in ('run.json', 'completed.json', 'metrics.json', 'all_receipts.json'):
            inputs[str(folder / name)] = sha(folder / name)
    groups, counts = {}, {}
    sequences = frames = gt_files = box_files = 0
    for dataset, spec in inventory.items():
        expected = (245, 220703) if dataset == 'LasHeR-Miss' else (234, 116650)
        assert (len(spec['sequence_details']), spec['frames']) == expected
        assert sum(item['frames'] for item in spec['sequence_details']) == spec['frames']
        for item in spec['sequence_details']:
            name, n = item['name'], item['frames']
            paths = ([gt_root / 'LasHeR/lasher_gt' / (name + '.txt')]
                     if dataset == 'LasHeR-Miss' else [
                         gt_root / 'RGBT234/rgbt234_gt' / name / filename
                         for filename in ('visible.txt', 'infrared.txt')])
            references = ['visible_GT'] if len(paths) == 1 else ['visible_GT', 'infrared_GT']
            assert len(paths) == len(item['gt_sha256'])
            predictions = {}
            for condition, folder in [('complete', complete), ('miss', missing)]:
                path = folder / 'bbox' / dataset / (name + '.txt')
                receipt = receipts[condition][dataset, name]
                assert receipt['frames'] == n and sha(path) == receipt['bbox_sha256']
                prediction = np.loadtxt(str(path), ndmin=2)
                assert prediction.shape == (n, 4) and np.isfinite(prediction).all()
                assert (prediction[:, 2:] > 0).all()
                predictions[condition] = prediction
                box_files += 1
            for reference, path, expected_sha in zip(references, paths, item['gt_sha256'].values()):
                assert sha(path) == expected_sha
                target = np.loadtxt(str(path), delimiter=',', ndmin=2)
                assert target.shape == (n, 4) and np.isfinite(target).all()
                valid = (target > 0).all(1)
                valid[0] = False
                key = (dataset.replace('-Miss', ''), reference)
                counts.setdefault(key, dict(sequences=0, total_frames=0, excluded_first_frames=0,
                                            excluded_other_nonpositive_gt_frames=0))
                counts[key]['sequences'] += 1
                counts[key]['total_frames'] += n
                counts[key]['excluded_first_frames'] += 1
                counts[key]['excluded_other_nonpositive_gt_frames'] += int((~valid[1:]).sum())
                for condition, prediction in predictions.items():
                    groups.setdefault(key + (condition,), []).append(geometry(prediction[valid], target[valid]))
                gt_files += 1
            sequences += 1
            frames += n
    assert sequences == 479 and frames == 337353 and box_files == 958 and gt_files == 713
    rows = [dict(dataset=key[0], reference=key[1], condition=key[2],
                 **counts[key[:2]], **summarize(np.concatenate(values)))
            for key, values in sorted(groups.items())]
    assert len(rows) == 6
    return dict(scope='Full closed same-checkpoint pair geometry only; all sequences, no state/sequence subsets, '
        'no tracker/images/checkpoint tensors or neural execution. First frames and GT with any nonpositive '
        'coordinate excluded as in the existing localization diagnostic. Continuous boxes, frame-weighted; '
        'not official sequence-averaged metrics. Each sensor GT reported separately, never oracle-maxed. '
        'Recentering preserves predicted size; resizing preserves predicted center. These use GT and are '
        'geometric oracle descriptions, not implementable scores, causal or additive error decomposition, '
        'module gain predictions, development evidence, parameter selection or independent semantic review.',
        checkpoint_sha256=checkpoint, sequences_per_condition=sequences, frames_per_condition=frames,
        prediction_files_verified=box_files, gt_files_verified=gt_files, rows=rows, inputs_sha256=inputs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--complete', required=True)
    parser.add_argument('--missing', required=True)
    parser.add_argument('--inventory', required=True)
    parser.add_argument('--gt-root', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    result = diagnose(args.complete, args.missing, args.inventory, args.gt_root)
    with Path(args.output).open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write(chr(10))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
