"""Recompute descriptive response/geometry statistics from saved evidence."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def clustered(rows, key, bootstrap_seed=39102):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row['identity']['sequence']].append(row[key])
    means = np.array([np.mean(grouped[name]) for name in sorted(grouped)])
    rng = np.random.default_rng(bootstrap_seed)
    draws = rng.integers(0, len(means), size=(10000, len(means)))
    low, high = np.quantile(means[draws].mean(1), [.025, .975])
    return dict(mean=float(means.mean()), ci95_low=float(low), ci95_high=float(high),
                sequences=len(means), rows=len(rows))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('directory')
    parser.add_argument('--verify-server-files', action='store_true')
    args = parser.parse_args()
    root = Path(args.directory)
    report = json.loads((root / 'report.json').read_text())
    done = json.loads((root / 'completed.json').read_text())
    assert report['status'] == done['status'] == 'complete'
    assert sha(root / 'report.json') == done['report_sha256']
    assert report['seed'] == 39101 and report['no_exact_prior_pilot_tuple_overlap']
    assert report['baseline_state_unchanged'] and report['baseline_gradients_absent']
    manifest = report['manifest']
    assert manifest['samples'] == 128 and manifest['rows'] == 1152
    assert manifest['native_head_replay_bitwise'] is True
    assert len(manifest['sequence_names']) == len(set(manifest['sequence_names'])) == 705
    assert len(manifest['input_manifest']) == 128
    rows = report['rows']
    assert len(rows) == 1152
    paired = defaultdict(dict)
    for row in rows:
        identity = row['identity']
        assert identity['sequence'] in manifest['sequence_names']
        key = (identity['sequence'], tuple(identity['template_frames']),
               tuple(identity['search_frames']))
        assert row['cell'] not in paired[key]
        paired[key][row['cell']] = row
        assert np.isclose(row['regression'] + row['focal'], row['total'], atol=1e-6)
        assert np.isclose(row['oracle_iou'] - row['iou'], row['oracle_gain'], atol=1e-6)
        assert row['gradient_conflict'] == float(row['gradient_cosine'] < 0)
        assert row['regression_gradient_norm'] > 0 and row['focal_gradient_norm'] > 0
        if row['peak_matches']:
            assert row['peak_error_cells'] == 0
            assert row['oracle_iou'] == row['iou']
    assert len(paired) == 128
    expected = {a + '_' + b for a in ['11', '01', '10'] for b in ['11', '01', '10']}
    assert all(set(item) == expected for item in paired.values())
    keys = ['regression', 'focal', 'total', 'iou', 'oracle_iou', 'oracle_gain',
            'peak_error_cells', 'peak_matches', 'gradient_cosine', 'gradient_conflict']
    cells = {cell: {key: clustered([row for row in rows if row['cell'] == cell], key)
                    for key in keys} for cell in sorted(expected)}
    assert cells == report['cells']
    # Paired input-state contrasts, exploratory and not an intervention on trained adapters.
    contrasts = {}
    for cell in sorted(expected - {'11_11'}):
        changes = []
        for sample in paired.values():
            native, changed = sample['11_11'], sample[cell]
            changes.append(dict(identity=native['identity'],
                iou_drop=native['iou'] - changed['iou'],
                peak_error_increase=changed['peak_error_cells'] - native['peak_error_cells'],
                oracle_gain_increase=changed['oracle_gain'] - native['oracle_gain'],
                conflict_increase=changed['gradient_conflict'] - native['gradient_conflict']))
        contrasts[cell] = {key: clustered(changes, key) for key in
            ['iou_drop', 'peak_error_increase', 'oracle_gain_increase', 'conflict_increase']}
    if args.verify_server_files:
        for path, digest in report['source_sha256'].items():
            assert sha(path) == digest, path
        assert sha(report['checkpoint']) == report['checkpoint_sha256']
        assert sha(root / 'fresh/features.pth') == manifest['cache_sha256']
    result = dict(status='saved_evidence_verified', report_sha256=done['report_sha256'],
        server_file_hashes_checked=args.verify_server_files,
        target_peak_counts={str(value): sum(row['target_peak_count'] == value for row in rows)
                            for value in sorted({row['target_peak_count'] for row in rows})},
        cells=cells, paired_vs_complete=contrasts,
        scope='Exploratory saved arithmetic/hash checks. No independent neural replay; '
              'GT peak oracle is not deployable, no training promotion.')
    with (root / 'analysis.json').open('x', newline='\n') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps(dict(status=result['status'], report_sha256=done['report_sha256'],
                          target_peak_counts=result['target_peak_counts']), indent=2))


if __name__ == '__main__':
    main()
