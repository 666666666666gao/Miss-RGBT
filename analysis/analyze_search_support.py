"""Infer search-crop geometry from closed E0 trajectories; no NN or image reads."""
import argparse
import json
from pathlib import Path

import numpy as np

from analyze_paired_baseline import history_labels, sha


def support(previous, target, factor):
    side = np.ceil(np.sqrt(previous[:, 2] * previous[:, 3]) * factor)
    lower = np.rint(previous[:, :2] + previous[:, 2:] / 2 - side[:, None] / 2)
    upper = lower + side[:, None]
    center = target[:, :2] + target[:, 2:] / 2
    center_inside = ((center >= lower) & (center < upper)).all(1)
    overlap = (np.minimum(target[:, :2] + target[:, 2:], upper) - np.maximum(target[:, :2], lower) > 0).all(1)
    return center_inside, overlap


def counts(values):
    a = np.asarray(values, dtype=bool)
    lost_center = a[:, 0] & ~a[:, 1]
    lost_overlap = a[:, 2] & ~a[:, 3]
    return dict(frames=len(a), full_center_inside_frames=int(a[:, 0].sum()),
                missing_center_inside_frames=int(a[:, 1].sum()),
                lost_center_support_frames=int(lost_center.sum()), lost_center_support_pct=float(lost_center.mean() * 100),
                full_box_overlap_frames=int(a[:, 2].sum()), missing_box_overlap_frames=int(a[:, 3].sum()),
                lost_all_gt_box_overlap_frames=int(lost_overlap.sum()),
                lost_all_gt_box_overlap_pct=float(lost_overlap.mean() * 100))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--complete-root', required=True)
    parser.add_argument('--missing-root', required=True)
    parser.add_argument('--gt-root', required=True)
    parser.add_argument('--mask-root', required=True)
    parser.add_argument('--inventory', required=True)
    parser.add_argument('--paired-report', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    inventory = json.loads(Path(args.inventory).read_text())
    verified_inputs = json.loads(Path(args.paired_report).read_text())['inputs']
    config_path = Path(__file__).resolve().parents[1] / 'experiments/cadtrack/cadtrack.yaml'
    # Frozen actual TEST.SEARCH_FACTOR is 4.0, also used by the native parameter loader.
    assert '  SEARCH_FACTOR: 4.0\n' in config_path.read_text()
    factor = 4.
    report = dict(scope='offline geometric search-support diagnosis; no NN/images; first/invalid GT excluded; '
                  'no current search crop is counted for both-missing hold frames; '
                  'crop/GT intersection is not a guarantee of visible target pixels; not official metrics',
                  native_formula='sample_target: ceil(sqrt(previous_w*previous_h)*4), round(previous_center-side/2)',
                  config_sha256=sha(config_path), paired_input_manifest_sha256=sha(args.paired_report), results={})
    for dataset, spec in inventory['datasets'].items():
        full_name = dataset.replace('-Miss', '')
        masks = json.loads((Path(args.mask_root) / Path(spec['mask']).name).read_text())
        assert sha(Path(args.mask_root) / Path(spec['mask']).name) == spec['mask_sha256']
        by_state, by_history, restored = {}, {}, []
        for seq in spec['sequence_details']:
            name = seq['name']
            full_path = Path(args.complete_root) / 'results' / full_name / (name + '.txt')
            missing_path = Path(args.missing_root) / 'results' / dataset / (name + '.txt')
            for path in [full_path, missing_path]:
                assert sha(path) == verified_inputs[str(path)]
            full, missing = np.loadtxt(str(full_path)), np.loadtxt(str(missing_path))
            paths = [Path(args.gt_root) / 'LasHeR/lasher_gt' / (name + '.txt')] if full_name == 'LasHeR' else [
                Path(args.gt_root) / 'RGBT234/rgbt234_gt' / name / f for f in ['visible.txt', 'infrared.txt']]
            targets = []
            for path, expected in zip(paths, seq['gt_sha256'].values()):
                assert sha(path) == expected
                targets.append(np.loadtxt(str(path), delimiter=','))
            mask = np.asarray(masks[name]['data'])
            mask[0] = [1, 1]
            states, histories = history_labels(mask)
            valid = np.stack([(gt > 0).all(1) for gt in targets]).all(0)
            valid &= mask.any(1)
            valid[0] = False
            indices = np.flatnonzero(valid)
            pairs = [support(prediction[indices - 1], target[indices], factor)
                     for prediction in [full, missing] for target in targets]
            k = len(targets)
            values = np.column_stack([np.stack([v[0] for v in pairs[:k]]).any(0),
                                      np.stack([v[0] for v in pairs[k:]]).any(0),
                                      np.stack([v[1] for v in pairs[:k]]).any(0),
                                      np.stack([v[1] for v in pairs[k:]]).any(0)])
            for labels, groups in [(states[valid], by_state), (histories[valid], by_history)]:
                for label in np.unique(labels):
                    groups.setdefault(str(label), []).extend(values[labels == label].tolist())
            restored.extend(values[np.char.startswith(histories[valid], '11_return_')].tolist())
        report['results'][dataset] = dict(states={key: counts(value) for key, value in sorted(by_state.items())},
            histories={key: counts(value) for key, value in sorted(by_history.items())},
            restored_currently_complete=counts(restored))
    Path(args.output).write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({name: data['restored_currently_complete'] for name, data in report['results'].items()}, indent=2))


if __name__ == '__main__':
    main()
