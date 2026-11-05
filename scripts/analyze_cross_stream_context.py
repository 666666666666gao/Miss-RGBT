"""Recompute saved pilot arithmetic and optionally replay selected adapters."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def identity_key(identity):
    return (identity['sequence'], tuple(identity['template_frames']),
            tuple(identity['search_frames']))


def previous_ids(folder):
    names = ['server28_fusion_adapter_pilot_20261102',
        'server28_head_supervision_20261102', 'server28_modality_heads_20261102',
        'server28_response_geometry_20261102', 'server28_absent_search_kv_20261102_v2',
        'server28_absent_position_20261102_v4', 'server28_late_fusion_streams_20261102']
    ids = set()
    hashes = {}
    for name in names:
        path = folder / name / 'report.json'
        record = json.loads(path.read_text())
        hashes[name] = sha256(path)
        if 'cache_manifests' in record:
            manifests = record['cache_manifests'].values()
        elif 'manifest' in record:
            manifests = [record['manifest']]
        else:
            for row in record['rows']:
                ids.add(identity_key(row['identity']))
            continue
        for manifest in manifests:
            ids.update(identity_key(row['identity']) for row in manifest['input_manifest'])
    return ids, hashes


def box_iou(box, target):
    box, target = np.asarray(box), np.asarray(target)
    first = np.concatenate([box[:2] - box[2:] / 2, box[:2] + box[2:] / 2])
    second = np.clip(np.concatenate([target[:2], target[:2] + target[2:]]), 0, 1)
    overlap = np.maximum(np.minimum(first[2:], second[2:]) -
                         np.maximum(first[:2], second[:2]), 0).prod()
    union = np.prod(first[2:] - first[:2]) + np.prod(second[2:] - second[:2]) - overlap
    return float(overlap / union)


def clustered(rows, field, seed):
    by_sequence = defaultdict(list)
    for row in rows:
        by_sequence[row['identity']['sequence']].append(row[field])
    means = np.array([np.mean(by_sequence[name]) for name in sorted(by_sequence)])
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(means), size=(10000, len(means)))
    lo, hi = np.quantile(means[draws].mean(1), [.025, .975])
    return dict(mean=float(means.mean()), ci95_low=float(lo), ci95_high=float(hi),
                sequences=len(means), rows=len(rows))


def replay(report):
    import torch
    from lib.config.cadtrack.config import cfg, update_config_from_file
    from miss_rgbt.model import load_pretrained
    from probe_cross_stream_context import ContextResidual, measure

    torch.set_num_threads(1)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    update_config_from_file('experiments/cadtrack/cadtrack.yaml')
    model = load_pretrained(cfg, report['checkpoint'], repair=False).cuda().eval().requires_grad_(False)
    audit_cache = torch.load(report['cache_manifests']['audit']['path'], map_location='cpu')
    for mode in ['aligned', 'shuffled']:
        saved = report['arms'][mode]
        adapter = ContextResidual(model.baseline.backbone.embed_dim,
                                  model.baseline.feat_len_s, mode).cuda()
        adapter.load_state_dict(torch.load(saved['selected_path'], map_location='cpu')['adapter'],
                                strict=True)
        actual = measure(adapter, model, audit_cache)
        assert actual == report['audit'][mode], mode
    return dict(selected_adapters_replayed=True, audit_rows=2304)


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
    assert report['pid'] == launch['pid']
    assert sha256(root / 'report.json') == done['report_sha256']
    assert sha256(root.parent / (root.name + '.preregistered_protocol.md')) == launch['protocol_sha256']
    assert report['source_sha256'][launch['command'][2]] == launch['script_sha256']
    assert report['baseline_unchanged'] and report['baseline_gradients_absent']
    assert set(report['control_checks'].values()) == {True}
    partitions = report['partitions']
    assert [len(partitions[key]) for key in ['train', 'development', 'audit']] == [705, 88, 89]
    assert len(set(sum(partitions.values(), []))) == 882
    old_ids, prior_hashes = previous_ids(root.parent)
    for split, samples, sample_seed in [('train', 256, 48101),
                                       ('development', 128, 48102), ('audit', 128, 48103)]:
        manifest = report['cache_manifests'][split]
        assert manifest['samples'] == samples and manifest['seed'] == sample_seed
        assert manifest['sequence_names'] == partitions[split]
        assert len(manifest['input_manifest']) == samples
        assert manifest['rows'] == samples * (6 if split == 'train' else 9)
        assert manifest['native_map_replay_exact']
        for entry in manifest['input_manifest']:
            assert entry['identity']['sequence'] in partitions[split]
            assert len(entry['tensors_sha256']) == 64
            assert identity_key(entry['identity']) not in old_ids
    for mode in ['aligned', 'shuffled']:
        arm = report['arms'][mode]
        assert arm['parameter_count'] == 199104
        history = arm['history']
        assert 10 <= len(history) <= 20
        assert [row['epoch'] for row in history] == list(range(1, len(history) + 1))
        reference, stale = None, 0
        for row in history:
            epoch = row['epoch']
            assert row['updates'] == 48 * epoch
            assert row['lr'] == (4e-4 if epoch <= 6 else 4e-5)
            expected_score = float(np.mean([cell['loss'] for name, cell in
                row['development_cells'].items() if not name.endswith('_11')]))
            assert row['score'] == expected_score and np.isfinite(row['train_mean_loss'])
            if epoch >= 7:
                if reference is None or row['score'] < reference * .998:
                    reference, stale = row['score'], 0
                else:
                    stale += 1
            assert row['stale'] == stale
            if epoch >= 10 and stale >= 3:
                assert epoch == len(history)
        assert len(history) == 20 or stale >= 3
        assert arm['selected'] == min(history[-5:], key=lambda row: (row['score'], row['epoch']))
        assert [row['epoch'] for row in arm['final_five']] == [row['epoch'] for row in history[-5:]]
    assert report['arms']['aligned']['initial_state_sha256'] == report['arms']['shuffled']['initial_state_sha256']

    pairs = []
    audit_ids = {identity_key(entry['identity'])
                 for entry in report['cache_manifests']['audit']['input_manifest']}
    complete_cells = {state + '_11' for state in ['11', '01', '10']}
    all_cells = {a + '_' + b for a in ['11', '01', '10'] for b in ['11', '01', '10']}
    for mode in ['aligned', 'shuffled']:
        audit = report['audit'][mode]
        assert len(audit['rows']) == 1152 and audit['complete_maps_equal']
        keyed = defaultdict(set)
        for row in audit['rows']:
            key = identity_key(row['identity'])
            assert key in audit_ids and row['cell'] not in keyed[key]
            keyed[key].add(row['cell'])
            assert all(np.isfinite(row[field]) for field in ['loss', 'iou', 'native_loss', 'native_iou'])
            assert np.isclose(box_iou(row['box'], row['target']), row['iou'], atol=2e-6, rtol=1e-6)
            assert np.isclose(box_iou(row['native_box'], row['target']),
                              row['native_iou'], atol=2e-6, rtol=1e-6)
            if row['cell'] in complete_cells:
                assert row['box'] == row['native_box']
                assert row['loss'] == row['native_loss'] and row['iou'] == row['native_iou']
        assert len(keyed) == 128 and all(cells == all_cells for cells in keyed.values())
    for treatment, control in zip(report['audit']['aligned']['rows'],
                                  report['audit']['shuffled']['rows']):
        assert treatment['identity'] == control['identity'] and treatment['cell'] == control['cell']
        assert treatment['target'] == control['target']
        assert treatment['native_box'] == control['native_box']
        assert treatment['native_loss'] == control['native_loss']
        pairs.append(dict(identity=treatment['identity'], cell=treatment['cell'],
            gain=treatment['iou'] - treatment['native_iou'],
            control_gain=treatment['iou'] - control['iou']))
    effects = {state: clustered(
        [row for row in pairs if row['cell'].endswith('_' + state)], 'gain', 48301)
        for state in ['01', '10']}
    effects['vs_shuffled'] = clustered(
        [row for row in pairs if not row['cell'].endswith('_11')], 'control_gain', 48302)
    assert effects == report['effects']
    advance = all(effects[state]['mean'] >= .01 and effects[state]['ci95_low'] > 0
        for state in ['01', '10']) and effects['vs_shuffled']['ci95_low'] > 0
    assert advance == report['advance'] == done['advance']
    if args.server_files:
        assert sha256(report['checkpoint']) == report['checkpoint_sha256']
        for filename, expected in report['source_sha256'].items():
            assert sha256(filename) == expected, filename
        for manifest in report['cache_manifests'].values():
            assert sha256(manifest['path']) == manifest['cache_sha256']
        for arm in report['arms'].values():
            assert sha256(arm['selected_path']) == arm['selected_sha256']
            for checkpoint in arm['final_five']:
                assert sha256(checkpoint['path']) == checkpoint['sha256']
    result = dict(status='verified', report_sha256=done['report_sha256'],
                  effects=effects, advance=advance, prior_reports_sha256=prior_hashes,
                  server_file_hashes_checked=args.server_files,
                  replay=replay(report) if args.replay else None)
    if args.output:
        with Path(args.output).open('x') as stream:
            json.dump(result, stream, indent=2)
            stream.write('\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
