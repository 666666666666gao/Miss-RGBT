"""Descriptive frame diagnostics, separate from official sequence-averaged metrics."""
import json
from pathlib import Path

import numpy as np
import rgbt


original = Path('/home/gaob/CADTrack_miss_ep20_20261013')
inventory = json.loads((original / 'inventory.json').read_text())
gt_root = Path(rgbt.__file__).parent / 'gt_file'
summary = {}
for dataset, spec in inventory['datasets'].items():
    masks = json.loads(Path(spec['mask']).read_text())
    grouped = {state: [] for state in ['00', '01', '10', '11']}
    transition = dict(changed=[], stable=[])
    for sequence in spec['sequence_details']:
        name = sequence['name']
        predicted = np.loadtxt(str(original / 'results' / dataset / (name + '.txt')))
        paths = [gt_root / 'LasHeR/lasher_gt' / (name + '.txt')] if dataset == 'LasHeR-Miss' else [
            gt_root / 'RGBT234/rgbt234_gt' / name / f for f in ['visible.txt', 'infrared.txt']]
        errors, overlaps = [], []
        for path in paths:
            target = np.loadtxt(str(path), delimiter=',')
            assert target.shape == predicted.shape
            errors.append(np.linalg.norm(predicted[:, :2] + predicted[:, 2:] / 2 - target[:, :2] - target[:, 2:] / 2, axis=1))
            lower = np.maximum(predicted[:, :2], target[:, :2])
            upper = np.minimum(predicted[:, :2] + predicted[:, 2:], target[:, :2] + target[:, 2:])
            intersection = np.maximum(upper - lower, 0).prod(1)
            union = predicted[:, 2:].prod(1) + target[:, 2:].prod(1) - intersection
            overlaps.append(np.divide(intersection, union, out=np.zeros_like(union), where=union > 0))
        error = np.stack(errors).min(0)
        iou = np.stack(overlaps).max(0)
        mask = np.asarray(masks[name]['data'])
        assert len(mask) == len(predicted)
        # Initialization is always complete; exclude it from state diagnostics.
        for i in range(1, len(mask)):
            state = ''.join(str(int(x)) for x in mask[i])
            grouped[state].append((error[i], iou[i]))
            changed = not np.array_equal(mask[i], mask[i - 1])
            transition['changed' if changed else 'stable'].append((error[i], iou[i]))
    def describe(values):
        a = np.asarray(values)
        return dict(frames=len(a), frame_precision_at_20_pct=float((a[:, 0] <= 20).mean() * 100),
                    frame_mean_iou_pct=float(a[:, 1].mean() * 100),
                    frame_success_at_0_5_pct=float((a[:, 1] > .5).mean() * 100))
    summary[dataset] = dict(states={k: describe(v) for k, v in grouped.items()},
                            transitions={k: describe(v) for k, v in transition.items()})
report = dict(scope='descriptive raw continuous bbox diagnostics; first frames excluded; frame-weighted, not official PR/SR; test data not a training target',
              results=summary)
Path('reports/e0/missing_state_diagnostics.json').write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
