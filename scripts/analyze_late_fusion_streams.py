"""Independently recompute late-fusion stream ablation evidence."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np

PRIOR_REPORTS = [
    'server28_fusion_adapter_pilot_20261102/report.json',
    'server28_head_supervision_20261102/report.json',
    'server28_modality_heads_20261102/report.json',
    'server28_response_geometry_20261102/report.json',
    'server28_absent_search_kv_20261102_v2/report.json',
    'server28_absent_position_20261102_v4/report.json',
]


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def clustered(rows, key, seed):
    groups = defaultdict(list)
    for row in rows:
        groups[row['identity']['sequence']].append(row[key])
    means = np.asarray([np.mean(groups[name]) for name in sorted(groups)])
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(means), size=(10000, len(means)))
    low, high = np.quantile(means[draws].mean(axis=1), [.025, .975])
    return dict(mean=float(means.mean()), ci95_low=float(low), ci95_high=float(high),
                sequences=len(means), rows=len(rows))


def prior_tuple_ids(report_root):
    identities = set()
    for filename in PRIOR_REPORTS:
        report = json.loads((report_root / filename).read_text())
        cache_manifests = report.get('cache_manifests')
        if cache_manifests:
            manifests = cache_manifests.values()
        elif 'manifest' in report:
            manifests = [report['manifest']]
        else:
            assert filename in (
                'server28_absent_search_kv_20261102_v2/report.json',
                'server28_absent_position_20261102_v4/report.json')
            for row in report['rows']:
                item = row['identity']
                identities.add((item['sequence'], tuple(item['template_frames']),
                                tuple(item['search_frames'])))
            continue
        for manifest in manifests:
            for row in manifest['input_manifest']:
                item = row['identity']
                identities.add((item['sequence'], tuple(item['template_frames']),
                                tuple(item['search_frames'])))
    return identities


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('directory')
    parser.add_argument('--verify-server-files', action='store_true')
    args = parser.parse_args()
    root = Path(args.directory)
    report_path = root / 'report.json'
    report = json.loads(report_path.read_text())
    completed = json.loads((root / 'completed.json').read_text())
    assert report['status'] == completed['status'] == 'complete'
    assert sha256(report_path) == completed['report_sha256']
    launch_path = root.parent / (root.name + '.launch.json')
    launch = json.loads(launch_path.read_text())
    protocol_snapshot = root.parent / Path(launch['protocol_snapshot']).name
    assert sha256(protocol_snapshot) == launch['protocol_sha256']
    assert launch['pid'] == report['pid']
    assert report['source_sha256'][launch['command'][2]] == launch['script_sha256']
    assert report['seed'] == 46101
    assert report['exact_previous_tuple_overlap'] is False
    assert report['baseline_state_unchanged'] and report['baseline_gradients_absent']
    assert report['complete_equal'] and len(report['sequence_names']) == 705
    assert len(set(report['sequence_names'])) == 705
    assert set(report['prior_report_sha256']) == set(PRIOR_REPORTS)
    prior_root = root.parent
    for filename in PRIOR_REPORTS:
        assert sha256(prior_root / filename) == report['prior_report_sha256'][filename]
    rows = report['rows']
    assert len(rows) == 128 * 9 * 2
    keyed = defaultdict(dict)
    for row in rows:
        identity = row['identity']
        assert identity['sequence'] in report['sequence_names']
        key = (identity['sequence'], tuple(identity['template_frames']),
               tuple(identity['search_frames']))
        slot = (row['cell'], row['direction'])
        assert slot not in keyed[key]
        keyed[key][slot] = row
        assert np.isfinite(row['loss_delta']) and np.isfinite(row['iou_delta'])
    assert len(keyed) == 128
    assert not (set(keyed) & prior_tuple_ids(prior_root))
    cells = {a + '_' + b for a in ['11', '01', '10'] for b in ['11', '01', '10']}
    slots = {(cell, direction) for cell in cells for direction in ['target', 'sham']}
    assert all(set(item) == slots for item in keyed.values())

    effects = {}
    for state in ['01', '10']:
        selected = [row for row in rows if row['cell'].endswith('_' + state)]
        target = [row for row in selected if row['direction'] == 'target']
        sham = [row for row in selected if row['direction'] == 'sham']
        paired = []
        for values in keyed.values():
            for cell in cells:
                if cell.endswith('_' + state):
                    treatment = values[(cell, 'target')]
                    control = values[(cell, 'sham')]
                    paired.append(dict(identity=treatment['identity'],
                        iou_delta=treatment['iou_delta'] - control['iou_delta']))
        effects[state] = dict(
            target_iou=clustered(target, 'iou_delta', 46102),
            target_loss=clustered(target, 'loss_delta', 46103),
            target_vs_sham_iou=clustered(paired, 'iou_delta', 46104))
    assert effects == report['effects']
    complete = [row for row in rows
                if row['direction'] == 'target' and row['cell'].endswith('_11')]
    assert len(complete) == 384 and all(row['complete_equal'] for row in complete)
    assert all(row['iou_delta'] == row['loss_delta'] == 0 for row in complete)
    advance = all(effects[state]['target_iou']['mean'] >= 0.01 and
                  effects[state]['target_iou']['ci95_low'] > 0 and
                  effects[state]['target_vs_sham_iou']['ci95_low'] > 0
                  for state in ['01', '10'])
    assert advance == report['advance'] == completed['advance']
    if args.verify_server_files:
        for filename, expected in report['source_sha256'].items():
            assert sha256(filename) == expected, filename
    print(json.dumps(dict(status='verified', report_sha256=completed['report_sha256'],
        effects=effects, advance=advance), indent=2))


if __name__ == '__main__':
    main()
