"""Audit saved paired pilot arithmetic and optionally server artifact hashes."""
import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def effect(rows, field, seed):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row['sequence']].append(row[field])
    values = np.array([np.mean(grouped[name]) for name in sorted(grouped)])
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(values), size=(10000, len(values)))
    low, high = np.quantile(values[draws].mean(1), [.025, .975])
    return dict(mean=float(values.mean()), ci95_low=float(low), ci95_high=float(high),
                sequences=len(values), rows=len(rows))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('directory')
    parser.add_argument('--verify-server-files', action='store_true')
    args = parser.parse_args()
    root = Path(args.directory)
    report_path = root / 'report.json'
    report = json.loads(report_path.read_text())
    completed = json.loads((root / 'completed.json').read_text())
    assert completed['status'] == 'complete'
    assert completed['report_sha256'] == sha(report_path)
    launch = report['launch']
    names = launch['split_names']
    assert [len(names[key]) for key in ['train', 'development', 'audit']] == [705, 88, 89]
    assert len(set(names['train'] + names['development'] + names['audit'])) == 882
    if args.verify_server_files:
        for path, digest in launch['source_sha256'].items():
            assert sha(path) == digest, path
        assert sha(launch['checkpoint']) == launch['checkpoint_sha256']
    for split, expected_samples, expected_rows in [('train', 256, 512),
            ('development', 128, 1152), ('audit', 128, 1152)]:
        manifest = report['cache_manifests'][split]
        assert manifest['sequence_names'] == names[split]
        assert manifest['samples'] == expected_samples and manifest['rows'] == expected_rows
        assert len(manifest['input_manifest']) == expected_samples
        assert manifest['native_head_replay_bitwise'] is True
        assert all(item['identity']['sequence'] in names[split]
                   for item in manifest['input_manifest'])
        if args.verify_server_files:
            assert sha(root / split / 'features.pth') == manifest['cache_sha256']
    for mode in ['local', 'global']:
        arm = report['arms'][mode]
        history = arm['history']
        assert [row['epoch'] for row in history] == list(range(1, len(history) + 1))
        assert 10 <= len(history) <= 20
        for index, row in enumerate(history):
            assert row['updates'] == (index + 1) * 8
            expected = np.mean([cell['adapted'] for name, cell in row['cells'].items()
                                if not name.endswith('_11')])
            assert expected == row['selection_score']
            assert all(np.isfinite(value) for cell in row['cells'].values()
                       for value in cell.values())
            assert np.isclose(row['lr'], 8e-4 if index < 6 else 8e-5)
        reference, stale = None, 0
        for row in history[6:]:
            if reference is None or row['selection_score'] < reference * .998:
                reference, stale = row['selection_score'], 0
            else:
                stale += 1
            assert stale == row['stale_epochs']
        assert len(history) == 20 or stale >= 3
        selected = min(history[-5:], key=lambda row: (row['selection_score'], row['epoch']))
        assert selected == arm['selected']
        assert arm['updates'] == len(history) * 8
        assert [item['epoch'] for item in arm['final_five']] == [row['epoch'] for row in history[-5:]]
        assert arm['complete_history_maps_bitwise_equal'] is True
        if args.verify_server_files:
            assert sha(arm['selected_path']) == arm['selected_sha256']
            for item in arm['final_five']:
                assert sha(root / mode / ('epoch%02d.pth' % item['epoch'])) == item['sha256']
    assert report['arms']['local']['parameter_count'] == report['arms']['global']['parameter_count']
    assert report['arms']['local']['initial_adapter_sha256'] == report['arms']['global']['initial_adapter_sha256']
    paired = []
    local_rows = report['audit']['local']['rows']
    global_rows = report['audit']['global']['rows']
    assert len(local_rows) == len(global_rows) == 1152
    for local, control in zip(local_rows, global_rows):
        assert local['identity'] == control['identity']
        assert local['identity']['sequence'] in names['audit']
        assert local['cell'] == control['cell']
        assert local['native_loss'] == control['native_loss']
        assert all(np.isfinite(row[key]) for row in [local, control]
                   for key in ['native_loss', 'adapted_loss'])
        if local['cell'] == '11_11':
            assert local['native_loss'] == local['adapted_loss'] == control['adapted_loss']
        paired.append(dict(sequence=local['identity']['sequence'], cell=local['cell'],
            native_loss=local['native_loss'],
            native_gain=local['native_loss'] - local['adapted_loss'],
            relative_gain=(local['native_loss'] - local['adapted_loss']) /
                          max(local['native_loss'], 1e-6),
            control_gain=control['adapted_loss'] - local['adapted_loss']))
    effects, gates = {}, []
    for state in ['01', '10']:
        rows = [row for row in paired if row['cell'].endswith('_' + state)]
        assert len(rows) == 384
        native = defaultdict(list)
        for row in rows:
            native[row['sequence']].append(row['native_loss'])
        denominator = float(np.mean([np.mean(values) for values in native.values()]))
        effects[state] = dict(native_gain=effect(rows, 'native_gain', 50101),
                             relative_gain=effect(rows, 'relative_gain', 50102))
        effects[state]['relative_macro_loss_improvement'] = effects[state]['native_gain']['mean'] / denominator
        gates.append(effects[state]['native_gain']['ci95_low'] > 0 and
                     effects[state]['relative_macro_loss_improvement'] >= .01)
    missing = [row for row in paired if not row['cell'].endswith('_11')]
    assert len(missing) == 768
    effects['local_vs_global'] = effect(missing, 'control_gain', 50103)
    gates.append(effects['local_vs_global']['ci95_low'] > 0)
    assert effects == report['effects']
    assert report['launch']['baseline_initial_state_sha256'] == report['baseline_final_state_sha256']
    for key in ['baseline_state_unchanged', 'baseline_gradients_absent',
                'complete_history_all_map_parity']:
        assert report[key] is True
    assert all(gates) == report['advance_to_fresh_train882_closed_loop']
    assert all(gates) == completed['advance_to_fresh_train882_closed_loop']
    result = dict(status='saved_evidence_verified', report_sha256=sha(report_path),
                  server_file_hashes_checked=args.verify_server_files,
                  effects=effects, advance_to_fresh_train882_closed_loop=all(gates),
                  scope='Saved arithmetic/hash verification; no independent neural replay or external scientific review.')
    with (root / 'analysis.json').open('x', newline='\n') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
