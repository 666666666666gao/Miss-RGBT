"""Independent arithmetic/provenance checks for full-image crop training."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np

from analyze_causal_crop_exposure import exclusions, overlap
from analyze_cross_stream_context import sha256


def digest(state):
    import hashlib
    value = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        value.update(name.encode('utf-8'))
        value.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return value.hexdigest()


def clustered(rows, field):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row['sequence']].append(row[field])
    means = np.asarray([np.mean(grouped[name]) for name in sorted(grouped)])
    rng = np.random.default_rng(58301)
    draws = rng.integers(0, len(means), (10000, len(means)))
    low, high = np.quantile(means[draws].mean(1), [.025, .975])
    return dict(mean=float(means.mean()), ci95_low=float(low), ci95_high=float(high),
                sequences=len(means), rows=len(rows))


def check_rollout(result, clips):
    assert len(result['rows']) == sum((len(clip['frame_ids']) - 1) * 2 for clip in clips)
    expected = {}
    for clip in clips:
        for trajectory in ['complete', 'mixed']:
            for index, frame in enumerate(clip['frame_ids'][1:], 1):
                expected[(clip['sequence'], trajectory, frame)] = (
                    '11' if trajectory == 'complete' else clip['mixed_schedule'][index])
    seen = set()
    for row in result['rows']:
        key = row['sequence'], row['trajectory'], row['frame']
        assert key not in seen and row['state'] == expected[key]
        seen.add(key)
        assert np.isfinite(row['box'] + row['gt'] + [row['confidence'], row['iou']]).all()
        assert abs(overlap(row['box'], row['gt']) - row['iou']) < 1e-10
    for name in ['complete', 'missing']:
        values = [row for row in result['rows'] if (
            row['trajectory'] == 'complete' if name == 'complete'
            else row['trajectory'] == 'mixed' and row['state'] != '11')]
        means = {sequence: float(np.mean([row['iou'] for row in values if row['sequence'] == sequence]))
                 for sequence in sorted({row['sequence'] for row in values})}
        assert len(means) == len(clips)
        assert result['means'][name] == dict(mean=float(np.mean(list(means.values()))), sequences=means)
    assert result['score'] == float(np.mean([value['mean'] for value in result['means'].values()]))


def server_checks(report, root, replay):
    import cv2
    import torch
    from lib.config.cadtrack.config import cfg, update_config_from_file
    from lib.train.dataset import LasHeR
    from lib.train.data.processing_utils import transform_image_to_crop
    from miss_rgbt.tracker import MissingTracker

    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    update_config_from_file('experiments/cadtrack/cadtrack.yaml')
    names = Path('lib/train/data_specs/lasher_train.txt').read_text().splitlines()
    expected = dict(train=[name for index, name in enumerate(names) if index % 5],
                    development=names[::5][:88], audit=names[::5][88:])
    assert report['partitions'] == expected
    assert sha256(report['checkpoint']) == report['checkpoint_sha256']
    for path, value in report['source_sha256'].items():
        assert sha256(path) == value, path
    input_replay = json.loads((root / 'input_replay_first_clip.json').read_text())
    assert sha256(Path(__file__).with_name('verify_crop_training_inputs.py')) == input_replay['verifier_sha256']
    manifest = report['training_manifest']
    assert sha256(manifest['cache_path']) == manifest['cache_sha256']
    cache = torch.load(manifest['cache_path'], map_location='cpu')
    assert len(cache['native_search']) == manifest['samples']
    required = {'templates', 'native_search', 'causal_search', 'native_target',
                'causal_target', 'availability'}
    assert set(cache) == required
    for index, item in enumerate(manifest['accepted']):
        tensors = {key: value[index] for key, value in cache.items()}
        assert digest(tensors) == item['tensors_sha256']
        assert all(torch.isfinite(value).all() for value in tensors.values())
        assert tensors['templates'].shape == (2, 6, 128, 128)
        assert tensors['native_search'].shape == tensors['causal_search'].shape == (6, 256, 256)
        assert tensors['availability'].shape == (3, 2)
        assert tensors['availability'][0].tolist() == [1., 1.]
        assert tensors['availability'][-1].tolist() == {
            '11': [1., 1.], '01': [0., 1.], '10': [1., 0.]}[item['state']]
        for mode, box_key in [('native', 'jittered_crop_box'), ('causal', 'predicted_crop_box')]:
            extract = item[box_key]
            side = np.ceil(np.sqrt(extract[2] * extract[3]) * 4.)
            label = transform_image_to_crop(torch.tensor(item['gt']), torch.tensor(extract),
                256. / side, torch.tensor([256., 256.]), normalize=True)
            assert torch.equal(label, tensors[mode + '_target'])
            assert (label[:2] >= 0).all() and (label[:2] + label[2:] <= 1).all()
    baseline = torch.load(report['checkpoint'], map_location='cpu')['net']
    baseline = {'baseline.' + key: value for key, value in baseline.items()}
    assert digest(baseline) == report['original_state_sha256']
    bn = {key: value for key, value in baseline.items()
          if key.endswith(('running_mean', 'running_var', 'num_batches_tracked'))}
    finite_weights, selected_changes = 0, {}
    for mode, arm in report['arms'].items():
        assert sha256(arm['selected_path']) == arm['selected_sha256']
        for item in arm['final_five']:
            assert sha256(item['path']) == item['sha256']
            checkpoint = torch.load(item['path'], map_location='cpu')
            assert checkpoint['mode'] == mode and checkpoint['epoch'] == item['epoch']
            assert set(checkpoint['model']) == set(baseline)
            assert all(torch.isfinite(value).all() for value in checkpoint['model'].values())
            assert all(torch.equal(value, checkpoint['model'][key]) for key, value in bn.items())
            assert digest(checkpoint['model']) != report['original_state_sha256']
            if item['epoch'] == arm['selected']['epoch']:
                groups = {name: dict(changed_tensors=0, squared_delta=0.)
                          for name in ['backbone', 'head_fusion']}
                for key, value in checkpoint['model'].items():
                    if key in bn or torch.equal(value, baseline[key]):
                        continue
                    group = groups['backbone' if 'backbone' in key else 'head_fusion']
                    group['changed_tensors'] += 1
                    delta = value.double() - baseline[key].double()
                    group['squared_delta'] += float(delta.square().sum())
                assert all(group['changed_tensors'] > 0 and group['squared_delta'] > 0
                           for group in groups.values())
                selected_changes[mode] = groups
            finite_weights += 1
    dataset = LasHeR(root='/data/wangwj/dataset/LasHeR/trainingset', split='train')
    if 'audit_amendment' in report:
        used, _ = exclusions(root.parent)
        previous = json.loads((root.parent /
            'server28_causal_crop_exposure_20261102/report.json').read_text())
        for clip in previous['clips']:
            used[clip['sequence']].update(clip['frame_ids'])
        rng = np.random.default_rng(58103)
        reconstructed = []
        for offset in rng.permutation(len(expected['audit'])):
            name = expected['audit'][offset]
            sequence = dataset.sequence_list.index(name)
            info = dataset.get_sequence_info(sequence)
            boxes = info['bbox'].numpy()
            valid = (np.isfinite(boxes).all(1) & (boxes[:, 2:] > 0).all(1)
                     & info['visible'].numpy().astype(bool))
            starts = [start for start in range(len(boxes) - 63)
                      if valid[start:start + 64].all()
                      and not used[name].intersection(range(start, start + 64))]
            if starts:
                start = int(rng.choice(starts))
                schedule = ['11'] + rng.choice(['11', '01', '10'], 63, p=[.4, .3, .3]).tolist()
                reconstructed.append(dict(sequence=name, dataset_index=sequence,
                    frame_ids=list(range(start, start + 64)), mixed_schedule=schedule))
            if len(reconstructed) == 24:
                break
        assert reconstructed == report['audit_clips']
    for clips in [manifest['clips'], report['development_clips'], report['audit_clips']]:
        for clip in clips:
            info = dataset.get_sequence_info(dataset.sequence_list.index(clip['sequence']))
            assert info['visible'][clip['frame_ids']].all()
    for mode, actual in report['audits'].items():
        assert len(actual['image_manifest']) == len(report['audit_clips'])
        for clip, image in zip(report['audit_clips'], actual['image_manifest']):
            assert image['sequence'] == clip['sequence'] and image['frame_ids'] == clip['frame_ids']
            info = dataset.get_sequence_info(dataset.sequence_list.index(clip['sequence']))
            boxes = info['bbox'][clip['frame_ids']].tolist()
            assert image['gt'] == boxes
            positions = dict(zip(clip['frame_ids'], boxes))
            for row in actual['rows']:
                if row['sequence'] == clip['sequence']:
                    assert row['gt'] == positions[row['frame']]
    replayed = 0
    if replay:
        import hashlib
        clip = report['audit_clips'][0]
        info = dataset.get_sequence_info(clip['dataset_index'])
        frames, annotations, _ = dataset.get_frames(clip['dataset_index'], clip['frame_ids'], info)
        boxes = [value.tolist() for value in annotations['bbox']]
        image_hash = hashlib.sha256()
        for image in frames:
            image_hash.update(np.ascontiguousarray(image).tobytes())
        for mode in ['e0', 'native', 'causal']:
            path = ('/home/gaob/Miss-RGBT/checkpoints/e0_availability/epoch20.pth.tar'
                    if mode == 'e0' else report['arms'][mode]['selected_path'])
            tracker = MissingTracker(cfg, path)
            tracker.network.eval().requires_grad_(False)
            actual = report['audits'][mode]
            image = actual['image_manifest'][0]
            assert image['image_sha256'] == image_hash.hexdigest() and image['gt'] == boxes
            for trajectory in ['complete', 'mixed']:
                tracker.initialize(frames[0], dict(init_bbox=boxes[0]))
                rows = [row for row in actual['rows']
                        if row['sequence'] == clip['sequence'] and row['trajectory'] == trajectory]
                for index, row in enumerate(rows, 1):
                    flags = {'11': [1., 1.], '01': [0., 1.], '10': [1., 0.]}[row['state']]
                    result = tracker.track(frames[index], flags)
                    assert result['target_bbox'] == row['box']
                    assert result['best_score'] == row['confidence']
                    replayed += 1
            del tracker
    return dict(finite_final_five_checkpoints=finite_weights, image_cache_verified=True,
                selected_weight_changes=selected_changes,
                first_audit_clip_replayed_rows=replayed, shared_native_tracker=True,
                full_audit_image_replay=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('directory')
    parser.add_argument('--server-files', action='store_true')
    parser.add_argument('--replay-first-clip', action='store_true')
    parser.add_argument('--output')
    args = parser.parse_args()
    root = Path(args.directory)
    report = json.loads((root / 'report.json').read_text())
    completed = json.loads((root / 'completed.json').read_text())
    launch = json.loads((root.parent / (root.name + '.launch.json')).read_text())
    assert report['status'] == completed['status'] == 'complete'
    assert report['pid'] == launch['pid']
    assert sha256(root / 'report.json') == completed['report_sha256']
    assert sha256(root.parent / (root.name + '.preregistered_protocol.md')) == launch['protocol_sha256']
    assert report['source_sha256'][launch['command'][2]] == launch['script_sha256']
    assert report['source_sha256'] == json.loads((root / 'source_sha256.json').read_text())
    count, length = 16, 96
    if 'audit_amendment' in report:
        amendment = json.loads((root / 'audit_resume.launch.json').read_text())
        assert report['audit_amendment'] == amendment
        assert amendment['original_pid'] == launch['pid']
        assert (amendment['count'], amendment['length'], amendment['seed']) == (24, 64, 58103)
        assert report['original_protocol_completed'] is False
        assert sha256(root / 'audit_amendment.md') == amendment['protocol_sha256']
        assert sha256(root / 'audit_inventory_v2.json') == amendment['inventory_sha256']
        inventory = json.loads((root / 'audit_inventory_v2.json').read_text())
        assert inventory['eligible_sequences'] == {'96': 12, '64': 24, '48': 29}
        assert {clip['sequence'] for clip in report['audit_clips']} == {
            row['sequence'] for row in inventory['rows'] if row['eligible_starts']['64'] > 0}
        if args.server_files:
            assert sha256(amendment['script_path']) == amendment['script_sha256']
        for mode in ['native', 'causal']:
            assert sha256(root / mode / 'selection.json') == amendment['selection_sha256'][mode]
            assert report['arms'][mode] == json.loads((root / mode / 'selection.json').read_text())
        count, length = 24, 64
    partitions = report['partitions']
    assert [len(partitions[name]) for name in ['train', 'development', 'audit']] == [705, 88, 89]
    assert len(set(sum(partitions.values(), []))) == 882
    used, hashes = exclusions(root.parent)
    path = root.parent / 'server28_causal_crop_exposure_20261102/report.json'
    prior = json.loads(path.read_text())
    hashes[path.parent.name] = sha256(path)
    for clip in prior['clips']:
        used[clip['sequence']].update(clip['frame_ids'])
    assert all(hashes[name] == value for name, value in report['prior_reports_sha256'].items())
    manifest = report['training_manifest']
    input_replay = json.loads((root / 'input_replay_first_clip.json').read_text())
    assert input_replay['status'] == 'verified'
    assert input_replay['image_manifest_sha256'] == sha256(root / 'image_manifest.json')
    assert input_replay['image_cache_sha256'] == manifest['cache_sha256']
    assert input_replay['sequence'] == manifest['clips'][0]['sequence']
    assert input_replay['frame_ids'] == manifest['clips'][0]['frame_ids']
    assert input_replay['accepted_image_pairs_bitwise'] + input_replay['rejection_decisions_reproduced'] == 11
    assert input_replay['independent_numpy_target_max_error'] < 1e-6
    assert all(input_replay[key] for key in ['prediction_crop_boxes_exact', 'jitter_boxes_exact',
                                           'template_history_exact', 'model_unchanged', 'gradients_absent'])
    clip_sets = dict(train=manifest['clips'], development=report['development_clips'],
                     audit=report['audit_clips'])
    initial = json.loads((root / 'initial_clips.json').read_text())
    assert initial == {key: clip_sets[key] for key in ['train', 'development']}
    assert clip_sets['audit'] == json.loads((root / 'audit_clips.json').read_text())
    audit_count = count
    for split, count, length in [('train', 32, 96), ('development', 8, 48), ('audit', count, length)]:
        clips = clip_sets[split]
        assert len(clips) == len({clip['sequence'] for clip in clips}) == count
        for clip in clips:
            assert clip['sequence'] in partitions[split]
            ids = clip['frame_ids']
            assert len(ids) == length and ids == list(range(ids[0], ids[0] + length))
            assert not used[clip['sequence']].intersection(ids)
            assert len(clip['mixed_schedule']) == length
            assert clip['mixed_schedule'][0] == '11'
            assert set(clip['mixed_schedule']) <= {'11', '01', '10'}
    assert manifest['candidates'] == 32 * 11
    assert manifest['samples'] == len(manifest['accepted']) >= 128
    assert manifest['candidates'] == manifest['samples'] + len(manifest['rejected'])
    expected_candidates = {(clip['sequence'], clip['frame_ids'][index])
                           for clip in clip_sets['train'] for index in range(8, 96, 8)}
    seen = set()
    for item in manifest['accepted'] + manifest['rejected']:
        identity = item['identity']
        key = identity['sequence'], identity['frame']
        assert key in expected_candidates and key not in seen
        seen.add(key)
        if 'native_valid' in item:
            assert not (item['native_valid'] and item['causal_valid'])
        else:
            assert len(item['tensors_sha256']) == 64
    assert seen == expected_candidates
    sizing = report['batch_sizing']
    assert sizing == json.loads((root / 'batch_sizing.json').read_text())
    fixed = sizing['static_bytes'] + 3 * sizing['parameters_bytes']
    activation = max(sizing['peak_bytes'] - fixed, 1)
    batch = min(32, 4 * int((.8 * sizing['total_bytes'] - fixed) / (activation / 4 * 4)))
    assert sizing['batch'] == batch and batch >= 4
    assert sizing['lr'] == 1e-5 * batch / 8
    assert sizing['updates_per_epoch'] == manifest['samples'] // batch
    assert sizing['dropped_remainder'] == manifest['samples'] % batch
    for mode, arm in report['arms'].items():
        assert arm['initial_sha256'] == report['original_state_sha256']
        assert arm['bn_buffers_unchanged']
        history = arm['history']
        assert 8 <= len(history) <= 12
        assert [row['epoch'] for row in history] == list(range(1, len(history) + 1))
        reference, stale = None, 0
        for row in history:
            epoch = row['epoch']
            assert row['updates'] == epoch * sizing['updates_per_epoch']
            assert row['lr'] == sizing['lr'] * (1 if epoch < 5 else .1)
            assert np.isfinite(row['train_mean_loss']) and np.isfinite(row['score'])
            assert row['score'] == row['development']['score']
            assert row['score'] == float(np.mean([
                value['mean'] for value in row['development']['means'].values()]))
            if epoch >= 5:
                if reference is None or row['score'] > reference * 1.002:
                    reference, stale = row['score'], 0
                else:
                    stale += 1
            assert row['stale'] == stale
            if epoch >= 8 and stale >= 3:
                assert epoch == len(history)
        assert len(history) == 12 or stale >= 3
        assert arm['selected'] == max(history[-5:], key=lambda item: (item['score'], -item['epoch']))
        assert [item['epoch'] for item in arm['final_five']] == [item['epoch'] for item in history[-5:]]
    assert report['arms']['native']['parameter_count'] == report['arms']['causal']['parameter_count']
    for mode, result in report['audits'].items():
        assert result == json.loads((root / ('audit_' + mode + '.json')).read_text())
        check_rollout(result, clip_sets['audit'])
    paired = []
    for base, native, changed in zip(report['audits']['e0']['rows'],
            report['audits']['native']['rows'], report['audits']['causal']['rows']):
        assert all(base[key] == native[key] == changed[key]
                   for key in ['sequence', 'frame', 'trajectory', 'state', 'gt'])
        paired.append(dict(sequence=base['sequence'], trajectory=base['trajectory'], state=base['state'],
            gain=changed['iou'] - base['iou'], control=changed['iou'] - native['iou']))
    effects = {state: clustered([row for row in paired
        if row['trajectory'] == 'mixed' and row['state'] == state], 'gain') for state in ['01', '10']}
    effects['causal_minus_native'] = clustered([row for row in paired
        if row['trajectory'] == 'mixed' and row['state'] != '11'], 'control')
    effects['complete'] = clustered([row for row in paired if row['trajectory'] == 'complete'], 'gain')
    assert effects == report['effects']
    advance = (all(effects[state]['mean'] >= .01 and effects[state]['ci95_low'] > 0
                   and effects[state]['sequences'] == audit_count for state in ['01', '10'])
        and effects['causal_minus_native']['ci95_low'] > 0
        and effects['causal_minus_native']['sequences'] == audit_count
        and effects['complete']['mean'] >= 0 and effects['complete']['ci95_low'] > -.01
        and effects['complete']['sequences'] == audit_count)
    assert advance == report['advance'] == completed['advance']
    server = server_checks(report, root, args.replay_first_clip) if args.server_files else None
    result = dict(status='verified', report_sha256=completed['report_sha256'],
        analyzer_sha256=sha256(__file__), effects=effects, advance=advance,
        audit_protocol='amended24x64' if 'audit_amendment' in report else 'original16x96',
        original_protocol_completed=report.get('original_protocol_completed', True),
        first_training_clip_replay=input_replay,
        server_checks=server, server_files_checked=args.server_files,
        limitation='Finite offline full-image pilot; not official scores, full-dataset convergence, '
                   'online policy aggregation, independent reviewer acceptance or a validated module.')
    if args.output:
        with Path(args.output).open('x', newline='\n') as stream:
            json.dump(result, stream, indent=2)
            stream.write('\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
