"""Verify saved joint-task gradient Gram arithmetic and prospective criterion."""
import argparse
import json
from pathlib import Path

import numpy as np

from analyze_candidate_quality import prior_tuples
from analyze_cross_stream_context import identity_key, sha256


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
    assert report['pid'] == launch['pid'] and report['seed'] == 56101
    assert sha256(root / 'report.json') == done['report_sha256']
    assert sha256(root.parent / (root.name + '.preregistered_protocol.md')) == launch['protocol_sha256']
    assert report['source_sha256'][launch['command'][2]] == launch['script_sha256']
    assert report['baseline_unchanged'] and report['baseline_gradients_absent']
    assert len(set(report['fitting_names'])) == 705
    excluded, hashes = prior_tuples(root.parent)
    for name in ['server28_candidate_quality_20261102_v2', 'server28_quality_ranker_20261102',
                 'server28_template_chronology_20261102', 'server28_chronology_training_20261102',
                 'server28_window_training_20261102_v2', 'server28_teacher_reliability_20261102']:
        path = root.parent / name / 'report.json'
        prior = json.loads(path.read_text())
        hashes[name] = sha256(path)
        if 'cache_manifests' in prior:
            manifests = prior['cache_manifests'].values()
        elif 'cache_manifest' in prior:
            manifests = [prior['cache_manifest']]
        else:
            manifests = [dict(input_manifest=prior['input_manifest'])]
        for manifest in manifests:
            excluded.update(identity_key(item['identity']) for item in manifest['input_manifest'])
    assert all(hashes[name] == expected for name, expected in report['prior_reports_sha256'].items())
    manifest = report['input_manifest']
    assert len(manifest) == 64
    for item in manifest:
        assert identity_key(item['identity']) not in excluded
        assert item['identity']['sequence'] in report['fitting_names']
        assert len(item['identity']['template_frames']) == 2
        assert len(item['identity']['search_frames']) == 3
        assert len(item['tensors_sha256']) == 64
    rows = report['rows']
    assert len(rows) == 16 and [row['batch'] for row in rows] == list(range(16))
    for index, row in enumerate(rows):
        assert row['identities'] == [item['identity'] for item in manifest[index * 4:(index + 1) * 4]]
        assert np.isfinite([row['complete_loss'], row['missing_loss'], row['kd']]).all()
        assert len(row['mask_sha256']) == 64 and len(row['availability']) == 5
        assert row['availability'][0] == [[1., 1.]] * 4
        for flags in row['availability']:
            assert len(flags) == 4 and all(value in [[1., 1.], [0., 1.], [1., 0.]] for value in flags)
        counts = np.asarray(row['patch_keep_counts'])
        assert counts.shape == (3, 4) and ((counts >= 0) & (counts <= 512)).all()
        assert row['unused_parameters'] == rows[0]['unused_parameters']
        groups = row['groups']
        assert set(groups) == {'all', 'backbone', 'head_fusion'}
        for field in ['complete_sq', 'auxiliary_sq', 'dot', 'parameters']:
            assert np.isclose(groups['all'][field],
                groups['backbone'][field] + groups['head_fusion'][field], rtol=1e-10, atol=1e-10)
        for group in groups.values():
            assert np.isfinite(list(group.values())).all()
            assert group['complete_sq'] > 0 and group['auxiliary_sq'] > 0
            cosine = group['dot'] / np.sqrt(group['complete_sq'] * group['auxiliary_sq'])
            assert cosine == group['cosine'] and abs(cosine) <= 1 + 1e-10
    values = np.asarray([row['groups']['all']['cosine'] for row in rows])
    rng = np.random.default_rng(56301)
    draws = rng.integers(0, len(values), size=(10000, len(values)))
    low, high = np.quantile(values[draws].mean(1), [.025, .975])
    effects = dict(mean=float(values.mean()), ci95_low=float(low), ci95_high=float(high),
        negative_batches=int((values < 0).sum()), batches=len(values))
    supported = bool(effects['mean'] <= -.05 and high < 0 and (values < 0).mean() >= .25)
    assert effects == report['effects']
    assert supported == report['diagnostic_supported'] == done['diagnostic_supported']
    if args.server_files:
        names = Path('lib/train/data_specs/lasher_train.txt').read_text().splitlines()
        assert report['fitting_names'] == [name for index, name in enumerate(names) if index % 5]
        assert sha256(report['checkpoint']) == report['checkpoint_sha256']
        for path, expected in report['source_sha256'].items():
            assert sha256(path) == expected, path
    result = dict(status='verified', report_sha256=done['report_sha256'],
        analyzer_sha256=sha256(__file__), effects=effects, diagnostic_supported=supported,
        prior_reports_sha256=hashes, server_files_checked=args.server_files,
        neural_gradient_replay=False,
        limitation='Saved Gram arithmetic only; raw gradients/images not persisted.')
    if args.output:
        with Path(args.output).open('x', newline='\n') as stream:
            json.dump(result, stream, indent=2)
            stream.write('\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
