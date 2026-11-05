"""Independently check fixed-candidate evidence and optionally replay native maps."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np

from analyze_cross_stream_context import box_iou, clustered, identity_key, previous_ids, sha256


def prior_tuples(folder):
    excluded, hashes = previous_ids(folder)
    for name in ['server28_cross_stream_context_20261102_v2',
                 'server28_tail_lowrank_20261102']:
        path = folder / name / 'report.json'
        record = json.loads(path.read_text())
        hashes[name] = sha256(path)
        for manifest in record['cache_manifests'].values():
            excluded.update(identity_key(item['identity']) for item in manifest['input_manifest'])
    return excluded, hashes


def replay_maps(report):
    import torch
    cache = torch.load(report['map_cache']['path'], map_location='cpu')
    score = cache['score_map'].numpy()[:, 0].reshape(-1, 256)
    size = cache['size_map'].numpy().reshape(-1, 2, 256)
    offset = cache['offset_map'].numpy().reshape(-1, 2, 256)
    window = cache['window'].numpy().reshape(256)
    weighted = score * window[None]
    locations = np.arange(256)
    boxes = np.stack([
        ((locations % 16).astype(np.float32)[None] + offset[:, 0]) / 16,
        ((locations // 16).astype(np.float32)[None] + offset[:, 1]) / 16,
        size[:, 0], size[:, 1]], axis=-1)
    assert len(boxes) == len(report['rows']) == 1152
    for index, row in enumerate(report['rows']):
        target = cache['boxes'][index].numpy()
        assert target.tolist() == row['target']
        raw_index = int(score[index].argmax())
        order = np.argsort(-weighted[index], kind='mergesort')[:8]
        assert raw_index == row['raw_index']
        assert order.tolist() == [candidate['index'] for candidate in row['candidates']]
        assert np.array_equal(boxes[index, raw_index],
                              cache['pred_boxes'][index, 0].numpy())
        overlaps = np.asarray([box_iou(box, target) for box in boxes[index]])
        for candidate in row['candidates']:
            location = candidate['index']
            assert boxes[index, location].tolist() == candidate['box']
            assert float(score[index, location]) == candidate['raw_score']
            assert float(weighted[index, location]) == candidate['hann_score']
            assert np.isclose(overlaps[location], candidate['iou'], atol=2e-6, rtol=1e-6)
        raw_top8 = np.argsort(-score[index], kind='mergesort')[:8]
        assert np.isclose(overlaps[raw_top8].max(), row['raw_top8_oracle_iou'],
                          atol=2e-6, rtol=1e-6)
        assert np.isclose(overlaps.max(), row['dense_oracle_iou'], atol=2e-6, rtol=1e-6)
        assert np.isclose(overlaps[row['dense_oracle_index']], overlaps.max(),
                          atol=2e-6, rtol=1e-6)
    return dict(independent_numpy_map_replay=True, rows=1152,
                dense_boxes_checked=1152 * 256, neural_inference_replayed=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('directory')
    parser.add_argument('--server-files', action='store_true')
    parser.add_argument('--replay-maps', action='store_true')
    parser.add_argument('--output')
    args = parser.parse_args()
    root = Path(args.directory)
    report = json.loads((root / 'report.json').read_text())
    completed = json.loads((root / 'completed.json').read_text())
    launch = json.loads((root.parent / (root.name + '.launch.json')).read_text())
    assert report['status'] == completed['status'] == 'complete'
    assert report['pid'] == launch['pid'] and report['seed'] == 50101
    assert sha256(root / 'report.json') == completed['report_sha256']
    snapshot = root.parent / (root.name + '.preregistered_protocol.md')
    assert sha256(snapshot) == launch['protocol_sha256']
    assert report['source_sha256'][launch['command'][2]] == launch['script_sha256']
    assert set(report['controls'].values()) == {True}
    assert len(set(report['fitting_names'])) == 705
    excluded, prior_hashes = prior_tuples(root.parent)
    manifest = report['input_manifest']
    assert len(manifest) == 128
    fresh_ids = {identity_key(item['identity']) for item in manifest}
    assert len(fresh_ids) == 128
    for item in manifest:
        assert item['identity']['sequence'] in report['fitting_names']
        assert identity_key(item['identity']) not in excluded
        assert len(item['tensors_sha256']) == 64
    expected_cells = {a + '_' + b for a in ['11', '01', '10'] for b in ['11', '01', '10']}
    by_tuple = defaultdict(set)
    assert len(report['rows']) == 1152
    for row in report['rows']:
        key = identity_key(row['identity'])
        assert key in fresh_ids and row['cell'] not in by_tuple[key]
        by_tuple[key].add(row['cell'])
        candidates = row['candidates']
        assert len(candidates) == len({candidate['index'] for candidate in candidates}) == 8
        assert all(0 <= candidate['index'] < 256 for candidate in candidates)
        assert candidates == sorted(candidates, key=lambda candidate:
                                     (-candidate['hann_score'], candidate['index']))
        for candidate in candidates:
            assert all(np.isfinite(candidate[field]) for field in ['raw_score', 'hann_score', 'iou'])
            assert np.isclose(box_iou(candidate['box'], row['target']),
                              candidate['iou'], atol=2e-6, rtol=1e-6)
        assert candidates[0]['index'] == row['hann_index']
        assert candidates[0]['iou'] == row['hann_iou']
        assert max(candidate['iou'] for candidate in candidates) == row['top8_oracle_iou']
        assert row['top8_oracle_gain'] == row['top8_oracle_iou'] - row['hann_iou']
        assert row['top8_oracle_gain'] >= 0
        assert row['top8_uniform_iou'] == float(np.mean([candidate['iou'] for candidate in candidates]))
        assert row['hann_minus_raw'] == row['hann_iou'] - row['raw_iou']
        assert row['dense_oracle_iou'] >= row['top8_oracle_iou']
        assert row['raw_top8_oracle_iou'] >= row['raw_iou']
        assert np.isclose(box_iou(row['raw_box'], row['target']), row['raw_iou'],
                          atol=2e-6, rtol=1e-6)
        assert np.isclose(box_iou(row['dense_oracle_box'], row['target']),
                          row['dense_oracle_iou'], atol=2e-6, rtol=1e-6)
    assert len(by_tuple) == 128 and all(cells == expected_cells for cells in by_tuple.values())
    categories = dict(
        complete=[row for row in report['rows'] if row['cell'] == '11_11'],
        rgb_missing=[row for row in report['rows'] if row['cell'].endswith('_01')],
        tir_missing=[row for row in report['rows'] if row['cell'].endswith('_10')])
    assert [len(values) for values in categories.values()] == [128, 384, 384]
    effects = {name: clustered(values, 'top8_oracle_gain', 50102)
               for name, values in categories.items()}
    advance = all(value['mean'] >= .02 and value['ci95_low'] > 0
                  for value in effects.values())
    assert effects == report['effects'] and advance == report['advance'] == completed['advance']
    descriptive = {name: {field: clustered(values, field, 50102)
        for field in ['raw_iou', 'hann_iou', 'top8_oracle_iou', 'top8_uniform_iou',
                      'raw_top8_oracle_iou', 'dense_oracle_iou', 'hann_minus_raw']}
        for name, values in categories.items()}
    if args.server_files:
        assert sha256(report['checkpoint']) == report['checkpoint_sha256']
        for filename, expected in report['source_sha256'].items():
            assert sha256(filename) == expected, filename
        assert sha256(report['map_cache']['path']) == report['map_cache']['sha256']
        names = Path('lib/train/data_specs/lasher_train.txt').read_text().splitlines()
        assert report['fitting_names'] == [name for index, name in enumerate(names) if index % 5]
    result = dict(status='verified', report_sha256=completed['report_sha256'],
        effects=effects, advance=advance, descriptive=descriptive,
        prior_reports_sha256=prior_hashes, server_file_hashes_checked=args.server_files,
        map_replay=replay_maps(report) if args.replay_maps else None)
    if args.output:
        with Path(args.output).open('x', newline='\n') as stream:
            json.dump(result, stream, indent=2)
            stream.write('\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
