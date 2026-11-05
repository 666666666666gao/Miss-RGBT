"""Inventory audit metadata without images, predictions or model updates."""
import json
from pathlib import Path

import numpy as np
from lib.train.dataset import LasHeR
import train_causal_crop_exposure as training


def main():
    root = training.PROJECT / 'reports/server28_causal_crop_training_20261102'
    assert not (root / 'audit_clips.json').exists()
    assert not (root / 'report.json').exists()
    names = Path('lib/train/data_specs/lasher_train.txt').read_text().splitlines()
    used, _ = training.causal.excluded_frames()
    prior = json.loads((training.PROJECT /
        'reports/server28_causal_crop_exposure_20261102/report.json').read_text())
    for clip in prior['clips']:
        used[clip['sequence']].update(clip['frame_ids'])
    dataset = LasHeR(root=training.burst.DATASET, split='train')
    rows = []
    for name in names[::5][88:]:
        info = dataset.get_sequence_info(dataset.sequence_list.index(name))
        boxes = info['bbox'].numpy()
        valid = (np.isfinite(boxes).all(1) & (boxes[:, 2:] > 0).all(1)
                 & info['visible'].numpy().astype(bool))
        counts = {}
        for length in [96, 64, 48]:
            counts[str(length)] = int(sum(
                valid[start:start + length].all()
                and not used[name].intersection(range(start, start + length))
                for start in range(len(boxes) - length + 1)))
        rows.append(dict(sequence=name, frames=len(boxes), eligible_starts=counts))
    result = dict(scope='Metadata only; no audit images or predictions inspected.',
        rows=rows, eligible_sequences={str(length): sum(
            row['eligible_starts'][str(length)] > 0 for row in rows)
            for length in [96, 64, 48]})
    training.save(root / 'audit_inventory_v2.json', result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
