"""Offline paired full/Miss diagnostics; never trains or runs a tracker."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def errors_and_iou(prediction, target):
    assert np.isfinite(prediction).all() and (prediction[:, 2:] > 0).all()
    error = np.linalg.norm(prediction[:, :2] + prediction[:, 2:] / 2 - target[:, :2] - target[:, 2:] / 2, axis=1)
    lower = np.maximum(prediction[:, :2], target[:, :2])
    upper = np.minimum(prediction[:, :2] + prediction[:, 2:], target[:, :2] + target[:, 2:])
    intersection = np.maximum(upper - lower, 0).prod(1)
    union = prediction[:, 2:].prod(1) + target[:, 2:].prod(1) - intersection
    return error, intersection / union


def describe(values):
    values = np.asarray(values)
    full, missing = values[:, 0] <= 20, values[:, 1] <= 20
    return dict(frames=len(values), complete_precision20_pct=float(full.mean() * 100),
                missing_precision20_pct=float(missing.mean() * 100),
                delta_precision20_pp=float((missing.astype(float) - full).mean() * 100),
                complete_mean_iou_pct=float(values[:, 2].mean() * 100),
                missing_mean_iou_pct=float(values[:, 3].mean() * 100),
                delta_mean_iou_pp=float((values[:, 3] - values[:, 2]).mean() * 100),
                lost_correct_frames=int((full & ~missing).sum()),
                gained_correct_frames=int((~full & missing).sum()))


def history_labels(mask):
    states, histories = [], []
    seen_missing, complete_run = False, 0
    for index, available in enumerate(mask):
        state = ''.join(str(int(x)) for x in available)
        states.append(state)
        if index == 0:
            histories.append('initialization')
            continue
        if state != '11':
            seen_missing, complete_run = True, 0
            histories.append('current_' + state)
        elif not seen_missing:
            histories.append('11_before_first_missing')
        else:
            complete_run += 1
            label = '1' if complete_run == 1 else ('2_5' if complete_run <= 5 else ('6_20' if complete_run <= 20 else '21_plus'))
            histories.append('11_return_' + label)
    return np.asarray(states), np.asarray(histories)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--complete-root', required=True)
    parser.add_argument('--missing-root', required=True)
    parser.add_argument('--gt-root', required=True)
    parser.add_argument('--mask-root', required=True)
    parser.add_argument('--inventory', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    complete_root, missing_root = Path(args.complete_root), Path(args.missing_root)
    inventory = json.loads(Path(args.inventory).read_text())
    full_inventory = json.loads((complete_root / 'inventory.json').read_text())
    assert full_inventory['checkpoint_sha256'] == inventory['checkpoint_sha256']
    report = dict(scope='paired descriptive test-set diagnostics only; no NN; first frames excluded; '
                  'only GT coordinates strictly positive in every annotation; continuous bbox and frame-weighted, '
                  'not official sequence-averaged PR/SR; not used to tune frozen E3/E4',
                  checkpoint_sha256=inventory['checkpoint_sha256'], results={}, inputs={})
    report['inputs'][str(Path(args.inventory))] = sha(args.inventory)
    report['inputs'][str(complete_root / 'inventory.json')] = sha(complete_root / 'inventory.json')
    rows = []
    total_frames = 0
    for dataset, spec in inventory['datasets'].items():
        full_name = dataset.replace('-Miss', '')
        masks_file = Path(args.mask_root) / Path(spec['mask']).name
        assert sha(masks_file) == spec['mask_sha256']
        report['inputs'][str(masks_file)] = spec['mask_sha256']
        masks = json.loads(masks_file.read_text())
        by_state, by_history = {}, {}
        excluded = 0
        prefix_max_difference = 0.
        prefix_frames = 0
        for sequence in spec['sequence_details']:
            name, n = sequence['name'], sequence['frames']
            full_file = complete_root / 'results' / full_name / (name + '.txt')
            miss_file = missing_root / 'results' / dataset / (name + '.txt')
            full, missing = np.loadtxt(str(full_file)), np.loadtxt(str(miss_file))
            assert full.shape == missing.shape == (n, 4)
            for path, receipts, label in [(full_file, complete_root, full_name), (miss_file, missing_root, dataset)]:
                receipt = json.loads((receipts / 'receipts' / label / (name + '.json')).read_text())
                assert sha(path) == receipt['result_sha256']
                report['inputs'][str(path)] = receipt['result_sha256']
            gt_paths = [Path(args.gt_root) / 'LasHeR/lasher_gt' / (name + '.txt')] if full_name == 'LasHeR' else [
                Path(args.gt_root) / 'RGBT234/rgbt234_gt' / name / f for f in ['visible.txt', 'infrared.txt']]
            targets = []
            for gt_file, expected in zip(gt_paths, sequence['gt_sha256'].values()):
                assert sha(gt_file) == expected
                target = np.loadtxt(str(gt_file), delimiter=',')
                assert target.shape == (n, 4)
                targets.append(target)
                report['inputs'][str(gt_file)] = expected
            valid = np.stack([(target > 0).all(1) for target in targets]).all(0)
            valid[0] = False
            excluded += int((~valid[1:]).sum())
            mask = np.asarray(masks[name]['data'])
            assert mask.shape == (n, 2)
            mask[0] = [1, 1]  # Execution always initialized with both sensors.
            states, histories = history_labels(mask)
            prefix = histories == '11_before_first_missing'
            if prefix.any():
                prefix_max_difference = max(prefix_max_difference, float(np.abs(full[prefix] - missing[prefix]).max()))
                prefix_frames += int(prefix.sum())
            pairs = [errors_and_iou(prediction[valid], target[valid])
                     for prediction in [full, missing] for target in targets]
            k = len(targets)
            values = np.column_stack([np.stack([v[0] for v in pairs[:k]]).min(0),
                                      np.stack([v[0] for v in pairs[k:]]).min(0),
                                      np.stack([v[1] for v in pairs[:k]]).max(0),
                                      np.stack([v[1] for v in pairs[k:]]).max(0)])
            for labels, groups in [(states[valid], by_state), (histories[valid], by_history)]:
                for label in np.unique(labels):
                    groups.setdefault(str(label), []).extend(values[labels == label].tolist())
            stats = describe(values)
            rows.append(dict(dataset=dataset, sequence=name, **stats))
            total_frames += n
        report['results'][dataset] = dict(
            sequences=len(spec['sequence_details']), total_frames=spec['frames'],
            excluded_first_frames=len(spec['sequence_details']), excluded_invalid_other_frames=excluded,
            identical_input_prefix_frames=prefix_frames, identical_input_prefix_max_bbox_difference=prefix_max_difference,
            states={key: describe(value) for key, value in sorted(by_state.items())},
            histories={key: describe(value) for key, value in sorted(by_history.items())})
    assert len(rows) == 479 and total_frames == 337353
    (output / 'paired_diagnostics.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    with (output / 'sequence_deltas.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(dict(scope=report['scope'], results=report['results']), indent=2))


if __name__ == '__main__':
    main()
