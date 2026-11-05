"""Recompute saved task/KD gradient evidence, with optional cached-head replay."""
import argparse
from collections import defaultdict
import contextlib
import io
import json
from pathlib import Path

import numpy as np

from analyze_candidate_quality import prior_tuples
from analyze_cross_stream_context import clustered, identity_key, sha256


def replay(report, root):
    import torch
    from lib.config.cadtrack.config import cfg, update_config_from_file
    from lib.models.layers.head import build_box_head
    from diagnose_teacher_reliability import measure

    torch.set_num_threads(1)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    update_config_from_file('experiments/cadtrack/cadtrack.yaml')
    native = torch.load(report['checkpoint'], map_location='cpu')['net']
    state = {key[len('box_head.'):]: value for key, value in native.items()
             if key.startswith('box_head.')}
    head = build_box_head(cfg, 768).cuda().eval()
    head.load_state_dict(state, strict=True)
    cache = torch.load(root / 'fitting/features.pth', map_location='cpu')
    with contextlib.redirect_stdout(io.StringIO()):
        actual = measure(head, cache)
    assert actual == report['rows']
    return dict(rows=len(actual), cached_head_gradients_replay_exact=True,
                shared_measurement_implementation=True, independent_image_replay=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('directory')
    parser.add_argument('--server-files', action='store_true')
    parser.add_argument('--replay', action='store_true')
    parser.add_argument('--output')
    args = parser.parse_args()
    root = Path(args.directory)
    report = json.loads((root / 'report.json').read_text())
    done = json.loads((root / 'completed.json').read_text())
    launch = json.loads((root.parent / (root.name + '.launch.json')).read_text())
    assert report['status'] == done['status'] == 'complete'
    assert report['pid'] == launch['pid'] and report['seed'] == 55101
    assert sha256(root / 'report.json') == done['report_sha256']
    assert sha256(root.parent / (root.name + '.preregistered_protocol.md')) == launch['protocol_sha256']
    assert report['source_sha256'][launch['command'][2]] == launch['script_sha256']
    assert report['baseline_unchanged'] and report['baseline_gradients_absent']
    assert report['copied_head_unchanged'] and report['projection_gradient_sum_verified']
    assert report['source_sha256'] == json.loads((root / 'source_sha256.json').read_text())
    excluded, hashes = prior_tuples(root.parent)
    for name in ['server28_candidate_quality_20261102_v2', 'server28_quality_ranker_20261102',
                 'server28_template_chronology_20261102', 'server28_chronology_training_20261102',
                 'server28_window_training_20261102_v2']:
        path = root.parent / name / 'report.json'
        prior = json.loads(path.read_text())
        hashes[name] = sha256(path)
        manifests = (prior['cache_manifests'].values() if 'cache_manifests' in prior
                     else [dict(input_manifest=prior['input_manifest'])])
        for manifest in manifests:
            excluded.update(identity_key(item['identity']) for item in manifest['input_manifest'])
    for name, expected in report['prior_reports_sha256'].items():
        assert hashes[name] == expected
    manifest = report['cache_manifest']
    assert manifest['samples'] == 128 and manifest['rows'] == 1152 and manifest['seed'] == 55101
    assert manifest['sequence_names'] == report['fitting_names']
    assert len(set(report['fitting_names'])) == 705 and manifest['native_head_replay_bitwise']
    assert len(manifest['input_manifest']) == 128
    ids = {identity_key(item['identity']) for item in manifest['input_manifest']}
    assert len(ids) == 128 and not ids.intersection(excluded)
    for item in manifest['input_manifest']:
        assert item['identity']['sequence'] in report['fitting_names']
        assert all(len(item[key]) == 64 for key in ['templates_sha256', 'search_sha256', 'boxes_sha256'])
    rows = report['rows']
    assert len(rows) == 1152
    expected_cells = {a + '_' + b for a in ['11', '01', '10'] for b in ['11', '01', '10']}
    seen = defaultdict(set)
    for row in rows:
        key = identity_key(row['identity'])
        assert key in ids and row['cell'] not in seen[key]
        seen[key].add(row['cell'])
        assert row['teacher_worse'] == (row['teacher_task'] >= row['task'])
        assert all(np.isfinite(row[name]) for name in ['task', 'teacher_task', 'kd',
            'fused_task_norm', 'fused_kd_norm', 'fused_dot', 'fused_cosine', 'projection_cosine'])
        first, second = np.asarray(row['projection_task']), np.asarray(row['projection_kd'])
        assert first.shape == second.shape == (33,)
        assert np.isfinite(first).all() and np.isfinite(second).all()
        projected_cosine = np.dot(first, second) / (
            max(np.linalg.norm(first), 1e-12) * max(np.linalg.norm(second), 1e-12))
        fused_cosine = row['fused_dot'] / (
            max(row['fused_task_norm'], 1e-12) * max(row['fused_kd_norm'], 1e-12))
        assert np.isclose(projected_cosine, row['projection_cosine'], atol=1e-7, rtol=1e-7)
        assert np.isclose(fused_cosine, row['fused_cosine'], atol=1e-7, rtol=1e-7)
        if row['cell'] == '11_11':
            assert row['teacher_task'] == row['task'] and abs(row['kd']) < 1e-6
    assert all(value == expected_cells for value in seen.values())
    missing = [row for row in rows if not row['cell'].endswith('_11')]
    assert len(missing) == 768
    groups = dict(unreliable=[row for row in missing if row['teacher_worse']],
                  reliable=[row for row in missing if not row['teacher_worse']])
    effects, counts, grouped = {}, {}, {}
    for group, values in groups.items():
        grouped[group] = defaultdict(list)
        for row in values:
            grouped[group][row['identity']['sequence']].append(row['fused_cosine'])
        counts[group] = dict(rows=len(values), sequences=len(grouped[group]))
        assert len(values) >= 32 and len(grouped[group]) >= 16
        for field in ['fused_cosine', 'projection_cosine']:
            effects[group + '_' + field] = clustered(values, field, 55301)
    shared = sorted(set(grouped['reliable']) & set(grouped['unreliable']))
    paired = [dict(identity=dict(sequence=name),
        gain=float(np.mean(grouped['reliable'][name]) - np.mean(grouped['unreliable'][name])))
        for name in shared]
    assert len(paired) >= 16
    effects['reliable_minus_unreliable_fused'] = clustered(paired, 'gain', 55302)
    bad, gap = effects['unreliable_fused_cosine'], effects['reliable_minus_unreliable_fused']
    supported = (bad['mean'] <= -.05 and bad['ci95_high'] < 0 and
                 gap['mean'] >= .1 and gap['ci95_low'] > 0 and
                 effects['unreliable_projection_cosine']['ci95_high'] < 0)
    assert counts == report['group_counts'] and effects == report['effects']
    assert supported == report['diagnostic_supported'] == done['diagnostic_supported']
    if args.server_files:
        names = Path('lib/train/data_specs/lasher_train.txt').read_text().splitlines()
        assert report['fitting_names'] == [name for index, name in enumerate(names) if index % 5]
        assert sha256(report['checkpoint']) == report['checkpoint_sha256']
        assert sha256(root / 'fitting/features.pth') == manifest['cache_sha256']
        for path, expected in report['source_sha256'].items():
            assert sha256(path) == expected, path
    result = dict(status='verified', report_sha256=done['report_sha256'],
        analyzer_sha256=sha256(__file__), effects=effects, group_counts=counts,
        diagnostic_supported=supported, prior_reports_sha256=hashes,
        server_files_checked=args.server_files, replay=replay(report, root) if args.replay else None)
    if args.output:
        with Path(args.output).open('x', newline='\n') as stream:
            json.dump(result, stream, indent=2)
            stream.write('\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
