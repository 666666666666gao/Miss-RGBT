"""Closed-trajectory box-quality and restored-crop diagnosis; no neural execution."""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import tarfile

import numpy as np

from analyze_paired_baseline import errors_and_iou, history_labels, sha
from analyze_search_support import support


def archive_data(path, inputs):
    inputs[str(path)] = sha(path)
    data = {}
    with tarfile.open(str(path), 'r|gz') as archive:
        for member in archive:
            if member.isfile() and (member.name.endswith('.txt') or
                    ('/receipts/' in '/' + member.name and member.name.endswith('.json'))):
                data[member.name] = archive.extractfile(member).read()
    return data


def add(groups, key, values):
    values = np.asarray(values, dtype=np.float64)
    if key not in groups:
        groups[key] = np.zeros(values.shape[1] + 1, dtype=np.float64)
    groups[key][0] += len(values)
    groups[key][1:] += values.sum(0)


def box_values(prediction, target):
    error, iou = errors_and_iou(prediction, target)
    hit = error <= 20
    size_error = np.abs(prediction[:, 2:] / target[:, 2:] - 1) * 100
    return hit, iou, size_error


def quality_row(key, values):
    dataset, reference, condition, model, group = key
    n, hits, iou, hit_iou, width, height = values
    return dict(dataset=dataset, reference=reference, condition=condition, model=model, group=group,
        frames=int(n), center_hits=int(hits), precision20_frame_pct=100 * hits / n if n else None,
        mean_iou_pct=100 * iou / n if n else None,
        mean_iou_on_own_hits_pct=100 * hit_iou / hits if hits else None,
        width_mape_on_own_hits_pct=width / hits if hits else None,
        height_mape_on_own_hits_pct=height / hits if hits else None)


def crop_row(values):
    n, full_center, missing_center, full_overlap, missing_overlap, lost_center, lost_overlap = values
    return dict(frames=int(n), full_center_inside_frames=int(full_center),
        missing_center_inside_frames=int(missing_center), lost_center_support_frames=int(lost_center),
        lost_center_support_pct=100 * lost_center / n if n else None,
        full_box_overlap_frames=int(full_overlap), missing_box_overlap_frames=int(missing_overlap),
        lost_all_gt_box_overlap_frames=int(lost_overlap),
        lost_all_gt_box_overlap_pct=100 * lost_overlap / n if n else None,
        missing_no_gt_overlap_pct=100 * (n - missing_overlap) / n if n else None)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project-root', required=True)
    parser.add_argument('--raw-root', required=True)
    parser.add_argument('--gt-root', required=True)
    parser.add_argument('--mask-root', required=True)
    parser.add_argument('--e0-complete-archive', required=True)
    parser.add_argument('--e0-miss-archive', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    root, raw, output = Path(args.project_root), Path(args.raw_root), Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    inputs = {str(Path(__file__)): sha(__file__)}
    for name in ['analyze_paired_baseline.py', 'analyze_search_support.py']:
        path = Path(__file__).parent / name
        inputs[str(path)] = sha(path)
    config = raw / 'source/experiments/cadtrack/cadtrack.yaml'
    assert '  SEARCH_FACTOR: 4.0\n' in config.read_text(encoding='utf-8')
    inputs[str(config)] = sha(config)
    inventory_path = root / 'reports/e0/inventory.json'
    inventory = json.loads(inventory_path.read_text(encoding='utf-8'))
    inputs[str(inventory_path)] = sha(inventory_path)
    baseline = {condition: archive_data(Path(path), inputs) for condition, path in
        [('complete', args.e0_complete_archive), ('miss', args.e0_miss_archive)]}
    receipts = {}
    for model in ['e3', 'e4']:
        for condition in ['complete', 'miss']:
            path = raw / 'results' / (model + '_full20_' + condition + '_20261013') / 'all_receipts.json'
            rows = json.loads(path.read_text())
            assert len(rows) == 479 and sum(row['frames'] for row in rows) == 337353
            inputs[str(path)] = sha(path)
            receipts[model, condition] = {(row['dataset'], row['sequence']): row for row in rows}
            assert len(receipts[model, condition]) == 479
    quality, joint, crops, prefixes = {}, {}, {}, {}
    sequences = 0
    for dataset, spec in inventory['datasets'].items():
        base_name = dataset.replace('-Miss', '')
        masks_path = Path(args.mask_root) / Path(spec['mask']).name
        assert sha(masks_path) == spec['mask_sha256']
        inputs[str(masks_path)] = spec['mask_sha256']
        masks = json.loads(masks_path.read_text())
        assert set(masks) == {row['name'] for row in spec['sequence_details']}
        for sequence in spec['sequence_details']:
            name, n = sequence['name'], sequence['frames']
            targets = [Path(args.gt_root) / 'LasHeR/lasher_gt' / (name + '.txt')] if base_name == 'LasHeR' else [
                Path(args.gt_root) / 'RGBT234/rgbt234_gt' / name / filename for filename in ['visible.txt', 'infrared.txt']]
            references = ['visible_GT'] if base_name == 'LasHeR' else ['visible_GT', 'infrared_GT']
            ground_truth = []
            for path, expected in zip(targets, sequence['gt_sha256'].values()):
                assert sha(path) == expected
                inputs[str(path)] = expected
                ground_truth.append(np.loadtxt(str(path), delimiter=','))
            predictions = {}
            for condition in ['complete', 'miss']:
                prefix = 'e0_complete_20261013/' if condition == 'complete' else ''
                label = base_name if condition == 'complete' else dataset
                key = prefix + 'results/' + label + '/' + name + '.txt'
                receipt = json.loads(baseline[condition][prefix + 'receipts/' + label + '/' + name + '.json'])
                blob = baseline[condition][key]
                assert hashlib.sha256(blob).hexdigest() == receipt['result_sha256']
                predictions['e0', condition] = np.loadtxt(io.BytesIO(blob))
                for model in ['e3', 'e4']:
                    path = raw / 'results' / (model + '_full20_' + condition + '_20261013') / 'bbox' / dataset / (name + '.txt')
                    expected = receipts[model, condition][dataset, name]
                    assert expected['frames'] == n and sha(path) == expected['bbox_sha256']
                    inputs[str(path)] = expected['bbox_sha256']
                    predictions[model, condition] = np.loadtxt(str(path))
            assert all(p.shape == (n, 4) and np.isfinite(p).all() and (p[:, 2:] > 0).all() for p in predictions.values())
            assert all(np.array_equal(p[0], ground_truth[0][0]) for p in predictions.values())
            mask = np.asarray(masks[name]['data']).copy()
            assert mask.shape == (n, 2) and np.isin(mask, [0, 1]).all()
            mask[0] = [1, 1]
            states, histories = history_labels(mask)
            restored = np.char.startswith(histories, '11_return_')
            for model in ['e0', 'e3', 'e4']:
                prefix_mask = histories == '11_before_first_missing'
                key = (base_name, model)
                counts = prefixes.setdefault(key, dict(frames=0, max_abs_bbox_difference=0.))
                counts['frames'] += int(prefix_mask.sum())
                if prefix_mask.any():
                    counts['max_abs_bbox_difference'] = max(counts['max_abs_bbox_difference'],
                        float(np.abs(predictions[model, 'complete'][prefix_mask] - predictions[model, 'miss'][prefix_mask]).max()))
            for reference, gt in zip(references, ground_truth):
                valid = (gt > 0).all(1)
                valid[0] = False
                groups = {'all': valid, 'restored_current11': valid & restored}
                groups.update({'state_' + state: valid & (states == state) for state in ['11', '01', '10', '00']})
                for condition in ['complete', 'miss']:
                    measured = {model: box_values(predictions[model, condition][valid], gt[valid]) for model in ['e0', 'e3', 'e4']}
                    selected_groups = groups if condition == 'miss' else {'all': valid}
                    for group, selector in selected_groups.items():
                        selected = selector[valid]
                        for model, (hit, iou, error) in measured.items():
                            values = np.column_stack([hit, iou, hit * iou, hit * error[:, 0], hit * error[:, 1]])
                            add(quality, (base_name, reference, condition, model, group), values[selected])
                        for a, b in [('e0', 'e3'), ('e0', 'e4'), ('e3', 'e4')]:
                            ha, ia, ea = measured[a]
                            hb, ib, eb = measured[b]
                            common = selected & ha & hb
                            values = np.column_stack([ia, ib, ea, eb, ib > ia, ib < ia])[common]
                            add(joint, (base_name, reference, condition, a, b, group), values)
            valid = np.stack([(gt > 0).all(1) for gt in ground_truth]).all(0) & mask.any(1)
            valid[0] = False
            indices = np.flatnonzero(valid)
            groups = {'all_observed': np.ones(len(indices), dtype=bool), 'restored_current11': restored[indices]}
            groups.update({str(label): histories[indices] == label for label in np.unique(histories[indices])})
            for model in ['e0', 'e3', 'e4']:
                pairs = [support(predictions[model, condition][indices - 1], gt[indices], 4.)
                    for condition in ['complete', 'miss'] for gt in ground_truth]
                k = len(ground_truth)
                full_center = np.stack([pair[0] for pair in pairs[:k]]).any(0)
                miss_center = np.stack([pair[0] for pair in pairs[k:]]).any(0)
                full_overlap = np.stack([pair[1] for pair in pairs[:k]]).any(0)
                miss_overlap = np.stack([pair[1] for pair in pairs[k:]]).any(0)
                values = np.column_stack([full_center, miss_center, full_overlap, miss_overlap,
                    full_center & ~miss_center, full_overlap & ~miss_overlap])
                for group, selector in groups.items():
                    add(crops, (base_name, model, group), values[selector])
            sequences += 1
    assert sequences == 479
    quality_rows = [quality_row(key, values) for key, values in sorted(quality.items())]
    joint_rows = []
    for key, values in sorted(joint.items()):
        dataset, reference, condition, a, b, group = key
        n, ia, ib, wa, ha, wb, hb, improved, harmed = values
        joint_rows.append(dict(dataset=dataset, reference=reference, condition=condition, model_a=a, model_b=b,
            group=group, joint_center_hits=int(n), model_a_mean_iou_pct=100 * ia / n if n else None,
            model_b_mean_iou_pct=100 * ib / n if n else None, delta_mean_iou_pp=100 * (ib - ia) / n if n else None,
            model_a_width_mape_pct=wa / n if n else None, model_b_width_mape_pct=wb / n if n else None,
            model_a_height_mape_pct=ha / n if n else None, model_b_height_mape_pct=hb / n if n else None,
            model_b_iou_improved_frames=int(improved), model_b_iou_harmed_frames=int(harmed)))
    crop_rows = [dict(dataset=key[0], model=key[1], group=key[2], **crop_row(values)) for key, values in sorted(crops.items())]
    previous_path = root / 'reports/e0_paired_diagnostics_v1/search_support.json'
    previous = json.loads(previous_path.read_text())['results']
    inputs[str(previous_path)] = sha(previous_path)
    for row in crop_rows:
        if row['model'] == 'e0' and row['group'] == 'restored_current11':
            expected = previous[row['dataset'] + '-Miss']['restored_currently_complete']
            assert all(abs(row[key] - value) < 1e-12 for key, value in expected.items()), row
    report = dict(scope='Offline descriptive frame-weighted diagnostics, continuous boxes, no model/images or parameter selection. '
        'First frame and nonpositive GT excluded. Width/height use each fixed sensor GT separately. '
        'Own-hit cohorts differ by model; joint-hit comparisons fix the same frames for both models. '
        'Crop rows exclude both-missing hold frames, require all annotation GT valid, and accept support from either annotation. '
        'Crop/GT intersection is geometric support, not visible pixels or reliable identity. '
        'Test trajectories previously used for failure diagnosis; these are not additional benchmark scores or causal module interventions.',
        sequences=sequences, models=['e0', 'e3', 'e4'], seed=2027, trained_checkpoint='fixed epoch20 for E3/E4',
        quality=quality_rows, joint_center_hits=joint_rows, crop_support=crop_rows,
        identical_input_prefix=[dict(dataset=k[0], model=k[1], **v) for k,v in sorted(prefixes.items())], inputs=inputs)
    (output / 'diagnostics.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    for filename, rows in [('box_quality.csv', quality_rows), ('joint_center_hits.csv', joint_rows), ('crop_support.csv', crop_rows)]:
        with (output / filename).open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print(json.dumps(dict(sequences=sequences, quality_rows=len(quality_rows), joint_rows=len(joint_rows),
        restored_crops=[row for row in crop_rows if row['group']=='restored_current11'], inputs=len(inputs)), indent=2))


if __name__ == '__main__':
    main()
