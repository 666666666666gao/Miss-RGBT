"""Verify matched ranker evidence and replay selected models from candidate caches."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np

from analyze_cross_stream_context import box_iou, clustered, identity_key, sha256
from analyze_candidate_quality import prior_tuples


def replay(report):
    import torch
    from probe_quality_ranker import Ranker, evaluate
    torch.set_num_threads(1)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    cache = torch.load(report['cache_manifests']['audit']['path'], map_location='cpu')
    for position, row in enumerate(report['audit']['quality']['rows']):
        assert cache['identities'][position] == row['identity']
        assert cache['labels'][position] == row['cell']
        for candidate in range(8):
            actual = box_iou(cache['boxes'][position, candidate].numpy(),
                             cache['target'][position].numpy())
            assert np.isclose(actual, float(cache['ious'][position, candidate]),
                              atol=2e-6, rtol=1e-6)
        assert cache['indices'][position].tolist() == sorted(
            cache['indices'][position].tolist(),
            key=lambda location: (-float(cache['prior'][position,
                cache['indices'][position].tolist().index(location)]), location))
    for mode in ['quality', 'response']:
        ranker = Ranker().cuda()
        state = torch.load(report['arms'][mode]['selected_path'], map_location='cpu')
        ranker.load_state_dict(state['ranker'], strict=True)
        assert evaluate(ranker, cache, mode) == report['audit'][mode], mode
    return dict(selected_ranker_replay_exact=True, rows=2304,
                audit_candidate_box_ious_checked=9216, neural_backbone_replayed=False)


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
    partitions = report['partitions']
    assert [len(partitions[key]) for key in ['train', 'development', 'audit']] == [705, 88, 89]
    assert len(set(sum(partitions.values(), []))) == 882
    excluded, prior_hashes = prior_tuples(root.parent)
    path = root.parent / 'server28_candidate_quality_20261102_v2/report.json'
    diagnostic = json.loads(path.read_text())
    assert diagnostic['advance']
    prior_hashes[path.parent.name] = sha256(path)
    excluded.update(identity_key(item['identity']) for item in diagnostic['input_manifest'])
    for split, samples, sample_seed in [('train', 512, 51101),
                                       ('development', 128, 51102), ('audit', 128, 51103)]:
        manifest = report['cache_manifests'][split]
        assert manifest['samples'] == samples and manifest['seed'] == sample_seed
        assert manifest['rows'] == samples * 9 and manifest['native_hann_box_exact']
        assert manifest['sequence_names'] == partitions[split]
        assert len(manifest['input_manifest']) == samples
        for item in manifest['input_manifest']:
            assert item['identity']['sequence'] in partitions[split]
            assert identity_key(item['identity']) not in excluded
            assert len(item['tensors_sha256']) == 64
    assert report['arms']['quality']['initial_state_sha256'] == report['arms']['response']['initial_state_sha256']
    expected_cells = {a + '_' + b for a in ['11', '01', '10'] for b in ['11', '01', '10']}
    audit_ids = {identity_key(item['identity']) for item in
                 report['cache_manifests']['audit']['input_manifest']}
    for mode in ['quality', 'response']:
        arm = report['arms'][mode]
        assert arm['parameter_count'] == 51585
        history = arm['history']
        assert 10 <= len(history) <= 20
        assert [row['epoch'] for row in history] == list(range(1, len(history) + 1))
        reference, stale = None, 0
        for row in history:
            epoch = row['epoch']
            assert row['updates'] == epoch * 18
            assert row['lr'] == (8e-4 if epoch <= 6 else 8e-5)
            assert row['score'] == float(np.mean(list(row['cells'].values())))
            assert set(row['cells']) == expected_cells
            assert np.isfinite(row['train_mean_loss'])
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
        rows = report['audit'][mode]['rows']
        assert len(rows) == 1152
        by_tuple = defaultdict(set)
        for row in rows:
            key = identity_key(row['identity'])
            assert key in audit_ids and row['cell'] not in by_tuple[key]
            by_tuple[key].add(row['cell'])
            assert 0 <= row['choice'] < 8 and 0 <= row['index'] < 256
            assert np.isclose(box_iou(row['box'], row['target']), row['iou'], atol=2e-6, rtol=1e-6)
            assert np.isclose(box_iou(row['native_box'], row['target']),
                              row['native_iou'], atol=2e-6, rtol=1e-6)
            assert row['oracle_iou'] >= row['iou']
        assert len(by_tuple) == 128 and all(cells == expected_cells for cells in by_tuple.values())
    effects = {}
    for category, suffix in [('complete', '11_11'), ('rgb_missing', '_01'), ('tir_missing', '_10')]:
        rows = [dict(identity=row['identity'], gain=row['iou'] - row['native_iou'])
                for row in report['audit']['quality']['rows'] if row['cell'].endswith(suffix)]
        effects[category] = clustered(rows, 'gain', 51301)
    paired = []
    for first, second in zip(report['audit']['quality']['rows'], report['audit']['response']['rows']):
        assert all(first[key] == second[key] for key in
                   ['identity', 'cell', 'target', 'native_iou', 'native_box', 'oracle_iou'])
        paired.append(dict(identity=first['identity'], gain=first['iou'] - second['iou']))
    effects['quality_minus_response'] = clustered(paired, 'gain', 51302)
    advance = all(effects[name]['mean'] >= .01 and effects[name]['ci95_low'] > 0
        for name in ['complete', 'rgb_missing', 'tir_missing'])
    advance = advance and effects['quality_minus_response']['ci95_low'] > 0
    assert effects == report['effects'] and advance == report['advance'] == done['advance']
    if args.server_files:
        import torch
        assert sha256(report['checkpoint']) == report['checkpoint_sha256']
        for filename, expected in report['source_sha256'].items():
            assert sha256(filename) == expected, filename
        names = Path('lib/train/data_specs/lasher_train.txt').read_text().splitlines()
        assert partitions == dict(train=[name for index, name in enumerate(names) if index % 5],
                                   development=names[::5][:88], audit=names[::5][88:])
        for manifest in report['cache_manifests'].values():
            assert sha256(manifest['path']) == manifest['cache_sha256']
        for arm in report['arms'].values():
            assert sha256(arm['selected_path']) == arm['selected_sha256']
            initial = json.loads((Path(arm['selected_path']).parent / 'initial.json').read_text())
            assert initial['native_selection_exact']
            for item in arm['final_five']:
                assert sha256(item['path']) == item['sha256']
                state = torch.load(item['path'], map_location='cpu')
                assert state['epoch'] == item['epoch']
                assert sum(value.numel() for value in state['ranker'].values()) == 51585
                assert all(torch.isfinite(value).all() for value in state['ranker'].values())
    result = dict(status='verified', report_sha256=done['report_sha256'], effects=effects,
        advance=advance, prior_reports_sha256=prior_hashes,
        server_file_hashes_checked=args.server_files, replay=replay(report) if args.replay else None)
    if args.output:
        with Path(args.output).open('x', newline='\n') as stream:
            json.dump(result, stream, indent=2)
            stream.write('\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
