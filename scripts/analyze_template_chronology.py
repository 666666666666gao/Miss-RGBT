"""Recompute saved temporal-mask diagnostic box evidence and decision."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np

from analyze_cross_stream_context import box_iou, clustered, identity_key, sha256
from analyze_candidate_quality import prior_tuples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('directory')
    parser.add_argument('--server-files', action='store_true')
    parser.add_argument('--output')
    args = parser.parse_args()
    root = Path(args.directory)
    report = json.loads((root / 'report.json').read_text())
    done = json.loads((root / 'completed.json').read_text())
    launch = json.loads((root.parent / (root.name + '.launch.json')).read_text())
    assert report['status'] == done['status'] == 'complete'
    assert report['pid'] == launch['pid'] and report['seed'] == 52101
    assert sha256(root / 'report.json') == done['report_sha256']
    assert sha256(root.parent / (root.name + '.preregistered_protocol.md')) == launch['protocol_sha256']
    assert report['source_sha256'][launch['command'][2]] == launch['script_sha256']
    assert report['baseline_unchanged'] and report['baseline_gradients_absent']
    excluded, prior_hashes = prior_tuples(root.parent)
    for name in ['server28_candidate_quality_20261102_v2', 'server28_quality_ranker_20261102']:
        path = root.parent / name / 'report.json'
        record = json.loads(path.read_text())
        prior_hashes[name] = sha256(path)
        manifests = (record['cache_manifests'].values() if 'cache_manifests' in record
                     else [dict(input_manifest=record['input_manifest'])])
        for manifest in manifests:
            excluded.update(identity_key(item['identity']) for item in manifest['input_manifest'])
    manifest = report['input_manifest']
    assert len(manifest) == 128 and len(set(report['fitting_names'])) == 705
    ids = {identity_key(item['identity']) for item in manifest}
    assert len(ids) == 128
    for item in manifest:
        identity = item['identity']
        assert identity['sequence'] in report['fitting_names']
        assert identity_key(identity) not in excluded
        newer, older = identity['template_frames']
        assert older < newer < min(identity['search_frames'])
        assert item['age_gap'] == newer - older and len(item['tensors_sha256']) == 64
    rows = report['rows']
    assert len(rows) == 1152
    expected = {(a, b) for a in ['11', '01', '10'] for b in ['11', '01', '10']}
    keyed = defaultdict(set)
    for row in rows:
        key = identity_key(row['identity'])
        cell = (row['dynamic'], row['search'])
        assert key in ids and cell not in keyed[key]
        keyed[key].add(cell)
        for mode in ['native', 'sham', 'chronological']:
            assert np.isfinite(row[mode + '_iou'])
            assert np.isclose(box_iou(row[mode + '_box'], row['target']), row[mode + '_iou'],
                              atol=2e-6, rtol=1e-6)
        if row['dynamic'] == '11':
            assert row['sham_box'] == row['chronological_box']
            assert row['sham_iou'] == row['chronological_iou']
    assert len(keyed) == 128 and all(cells == expected for cells in keyed.values())
    effects = {}
    for state in ['11', '01', '10']:
        selected = [dict(identity=row['identity'], gain=row['chronological_iou'] - row['native_iou'])
                    for row in rows if row['dynamic'] != '11' and row['search'] == state]
        assert len(selected) == 256
        effects[state] = clustered(selected, 'gain', 52102)
    shams = [dict(identity=row['identity'], gain=row['sham_iou'] - row['native_iou']) for row in rows]
    sham = clustered(shams, 'gain', 52102)
    small = max(abs(row['sham_iou'] - row['native_iou']) for row in rows) <= 1e-4
    material = any(abs(value['mean']) >= .01 and
        (value['ci95_low'] > 0 or value['ci95_high'] < 0) for value in effects.values())
    assert effects == report['effects'] and sham == report['sham_effect']
    assert small == report['sham_small'] and material == report['material_temporal_mask_effect']
    supported = material and small
    assert supported == report['diagnostic_supported'] == done['diagnostic_supported']
    if args.server_files:
        assert sha256(report['checkpoint']) == report['checkpoint_sha256']
        for filename, expected_hash in report['source_sha256'].items():
            assert sha256(filename) == expected_hash, filename
        names = Path('lib/train/data_specs/lasher_train.txt').read_text().splitlines()
        assert report['fitting_names'] == [name for index, name in enumerate(names) if index % 5]
    result = dict(status='verified', report_sha256=done['report_sha256'], effects=effects,
        sham_effect=sham, diagnostic_supported=supported, prior_reports_sha256=prior_hashes,
        server_file_hashes_checked=args.server_files, neural_replay=False)
    if args.output:
        with Path(args.output).open('x', newline='\n') as stream:
            json.dump(result, stream, indent=2)
            stream.write('\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
