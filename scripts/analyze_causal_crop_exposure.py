"""Recompute causal-crop evidence, with optional independent first-clip replay."""
import argparse
from collections import defaultdict
import json
import math
from pathlib import Path

import numpy as np

from analyze_candidate_quality import prior_tuples
from analyze_cross_stream_context import sha256


def overlap(first, second):
    first, second = np.asarray(first, dtype=np.float64), np.asarray(second, dtype=np.float64)
    intersection = np.maximum(0., np.minimum(first[:2] + first[2:], second[:2] + second[2:])
                              - np.maximum(first[:2], second[:2])).prod()
    union = first[2:].prod() + second[2:].prod() - intersection
    assert union > 0
    return float(intersection / union)


def bootstrap(rows, field):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row['sequence']].append(row[field])
    means = np.array([np.mean(grouped[name]) for name in sorted(grouped)])
    rng = np.random.default_rng(57301)
    draws = rng.integers(0, len(means), (10000, len(means)))
    low, high = np.quantile(means[draws].mean(1), [.025, .975])
    return dict(mean=float(means.mean()), ci95_low=float(low), ci95_high=float(high),
                sequences=len(means), rows=len(rows))


def exclusions(folder):
    identities, hashes = prior_tuples(folder)
    for name in ['server28_candidate_quality_20261102_v2', 'server28_quality_ranker_20261102',
                 'server28_template_chronology_20261102', 'server28_chronology_training_20261102',
                 'server28_window_training_20261102_v2', 'server28_teacher_reliability_20261102',
                 'server28_joint_gradients_20261102']:
        path = folder / name / 'report.json'
        record = json.loads(path.read_text())
        hashes[name] = sha256(path)
        if 'cache_manifests' in record:
            manifests = record['cache_manifests'].values()
        elif 'cache_manifest' in record:
            manifests = [record['cache_manifest']]
        else:
            manifests = [dict(input_manifest=record['input_manifest'])]
        for manifest in manifests:
            for item in manifest['input_manifest']:
                value = item['identity']
                identities.add((value['sequence'], tuple(value['template_frames']),
                                tuple(value['search_frames'])))
    used = defaultdict(set)
    for name, templates, searches in identities:
        used[name].update(templates + searches)
    path = folder / 'server28_burst_mask_rollout_20261102/report.json'
    record = json.loads(path.read_text())
    hashes[path.parent.name] = sha256(path)
    for clip in record['clips']:
        used[clip['sequence']].update(clip['frame_ids'])
    return used, hashes


def replay_first_clip(report):
    import hashlib
    import cv2
    import torch
    from lib.config.cadtrack.config import cfg, update_config_from_file
    from lib.train.dataset import LasHeR
    from lib.train.data.processing_utils import sample_target
    from lib.utils.box_ops import clip_box
    from miss_rgbt.tracker import MissingTracker

    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    update_config_from_file('experiments/cadtrack/cadtrack.yaml')
    clip = report['clips'][0]
    dataset = LasHeR(root='/data/wangwj/dataset/LasHeR/trainingset', split='train')
    sequence = dataset.sequence_list.index(clip['sequence'])
    frames, anno, _ = dataset.get_frames(sequence, clip['frame_ids'], dataset.get_sequence_info(sequence))
    boxes = [value.tolist() for value in anno['bbox']]
    assert boxes == clip['gt_boxes']
    digest = hashlib.sha256()
    for image in frames:
        digest.update(np.ascontiguousarray(image).tobytes())
    assert digest.hexdigest() == clip['image_sha256']
    tracker = MissingTracker(cfg, report['wrapper_checkpoint'])
    tracker.network.eval().requires_grad_(False)
    max_shadow_box_error = 0.
    checked = 0
    for trajectory in ['complete', 'mixed']:
        tracker.initialize(frames[0], dict(init_bbox=boxes[0]))
        rows = [row for row in report['rows'] if row['sequence'] == clip['sequence']
                and row['trajectory'] == trajectory]
        for index, row in enumerate(rows, 1):
            assert tracker.state == row['previous_native']
            assert [value.cpu().tolist() for value in tracker.availability] == row['template_availability']
            # Direct prediction from native templates; no shadow tracker update/restore.
            previous = boxes[index - 1]
            crop, factor, _ = sample_target(frames[index], previous, cfg.TEST.SEARCH_FACTOR,
                                           output_sz=cfg.TEST.SEARCH_SIZE)
            search = tracker.preprocessor.process(crop)
            present = {'11': [1., 1.], '01': [0., 1.], '10': [1., 0.]}[row['state']]
            observed = torch.tensor([present], dtype=torch.float32, device='cuda')
            with torch.no_grad():
                out = tracker.network(tracker.templates, [search], tracker.availability + [observed])[0]
                response = tracker.window * out['score_map']
                prediction = tracker.network.baseline.box_head.cal_bbox(
                    response, out['size_map'], out['offset_map'])
                cx, cy, width, height = (prediction.view(-1, 4).mean(0)
                                        * cfg.TEST.SEARCH_SIZE / factor).tolist()
                center_x, center_y = previous[0] + .5 * previous[2], previous[1] + .5 * previous[3]
                half = .5 * cfg.TEST.SEARCH_SIZE / factor
                shadow = clip_box([cx + center_x - half - .5 * width,
                    cy + center_y - half - .5 * height, width, height],
                    frames[index].shape[0], frames[index].shape[1], margin=10)
                confidence = float(response.max())
            error = float(np.abs(np.asarray(shadow) - row['shadow_box']).max())
            max_shadow_box_error = max(max_shadow_box_error, error)
            assert error <= 1e-4 and confidence == row['shadow_confidence']
            native = tracker.track(frames[index], present)
            assert native['target_bbox'] == row['native_box']
            assert native['best_score'] == row['native_confidence']
            checked += 1
    return dict(first_clip_rows=checked, image_hash_verified=True, native_replay_exact=True,
                direct_shadow_max_box_error=max_shadow_box_error,
                full_report_image_replay=False, shared_native_network_and_crop_utilities=True)


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
    assert report['seed'] == 57101 and report['pid'] == launch['pid']
    assert sha256(root / 'report.json') == completed['report_sha256']
    assert sha256(root.parent / (root.name + '.preregistered_protocol.md')) == launch['protocol_sha256']
    assert report['source_sha256'][launch['command'][2]] == launch['script_sha256']
    assert report['state_unchanged'] and report['gradients_absent']
    assert len(set(report['fitting_names'])) == 705
    used, hashes = exclusions(root.parent)
    assert all(hashes[name] == value for name, value in report['prior_reports_sha256'].items())
    clips, rows = report['clips'], report['rows']
    assert len(clips) == len({clip['sequence'] for clip in clips}) == 48
    assert len(rows) == 48 * 95 * 2
    selected = json.loads((root / 'selected_clips.json').read_text())
    assert [(item['sequence'], item['frame_ids']) for item in selected] == [
        (item['sequence'], item['frame_ids']) for item in clips]
    by_trajectory = defaultdict(list)
    for row in rows:
        by_trajectory[(row['sequence'], row['trajectory'])].append(row)
    for clip in clips:
        name, ids, boxes = clip['sequence'], clip['frame_ids'], clip['gt_boxes']
        assert name in report['fitting_names']
        assert len(ids) == 96 and ids == list(range(ids[0], ids[0] + 96))
        assert not used[name].intersection(ids)
        array = np.asarray(boxes)
        assert array.shape == (96, 4) and np.isfinite(array).all() and (array[:, 2:] > 0).all()
        assert len(clip['image_sha256']) == 64
        for trajectory in ['complete', 'mixed']:
            schedule = clip['schedules'][trajectory]
            assert len(schedule) == 96 and schedule[0] == '11'
            assert set(schedule) <= {'11', '01', '10'}
            if trajectory == 'complete':
                assert set(schedule) == {'11'}
            values = by_trajectory[(name, trajectory)]
            assert len(values) == 95
            for index, row in enumerate(values, 1):
                assert row['frame'] == ids[index] and row['previous_frame'] == ids[index - 1]
                assert row['state'] == schedule[index]
                assert row['gt'] == boxes[index] and row['previous_gt'] == boxes[index - 1]
                assert row['previous_native'] == (boxes[0] if index == 1 else values[index - 2]['native_box'])
                for arm in ['native', 'shadow']:
                    assert np.isfinite(row[arm + '_box'] + [row[arm + '_confidence']]).all()
                    assert abs(overlap(row[arm + '_box'], row['gt']) - row[arm + '_iou']) < 1e-10
                first, second = np.asarray(row['previous_native']), np.asarray(row['previous_gt'])
                error = np.linalg.norm(first[:2] + .5 * first[2:] - second[:2] - .5 * second[2:])
                error /= np.sqrt(second[2:].prod())
                assert np.isclose(error, row['prior_center_error'], rtol=1e-12, atol=1e-12)
                assert row['sham_checked'] == (index <= 3)
                if index <= 3:
                    assert row['native_box'] == row['sham_box']
                    assert row['native_confidence'] == row['sham_confidence']
                else:
                    assert row['sham_box'] is row['sham_confidence'] is None
                if index == 1:
                    assert row['native_box'] == row['shadow_box']
                    assert row['native_confidence'] == row['shadow_confidence']
                availability = row['template_availability']
                assert len(availability) == 2 and availability[0] == [[1., 1.]]
                assert availability[1] in [[[1., 1.]], [[0., 1.]], [[1., 0.]]]
                if trajectory == 'complete':
                    assert availability[1] == [[1., 1.]]
    categories = dict(
        complete=[row for row in rows if row['trajectory'] == 'complete'],
        mixed_complete=[row for row in rows if row['trajectory'] == 'mixed' and row['state'] == '11'],
        mixed_rgb_missing=[row for row in rows if row['trajectory'] == 'mixed' and row['state'] == '01'],
        mixed_tir_missing=[row for row in rows if row['trajectory'] == 'mixed' and row['state'] == '10'],
        mixed_missing=[row for row in rows if row['trajectory'] == 'mixed' and row['state'] != '11'])
    for row in rows:
        row['gain'] = row['shadow_iou'] - row['native_iou']
    effects = {key: bootstrap(values, 'gain') for key, values in categories.items()}
    assert effects == report['effects']
    effect = effects['mixed_missing']
    advance = bool(effect['mean'] >= .02 and effect['ci95_low'] > 0 and effect['sequences'] == 48)
    assert advance == report['advance'] == completed['advance']
    descriptive = {key: {field: bootstrap(values, field) for field in ['native_iou', 'shadow_iou']}
                   for key, values in categories.items()}
    error_bins = {}
    for name, lower, upper in [('below_quarter', 0., .25), ('quarter_to_one', .25, 1.),
                               ('at_least_one', 1., float('inf'))]:
        values = [row for row in categories['mixed_missing']
                  if lower <= row['prior_center_error'] < upper]
        if values:
            error_bins[name] = {field: bootstrap(values, field)
                               for field in ['native_iou', 'shadow_iou', 'gain']}
    for row in categories['mixed_missing']:
        previous, target = np.asarray(row['previous_native']), np.asarray(row['gt'])
        side = math.ceil(math.sqrt(previous[2:].prod()) * 4.)
        origin = np.array([round(previous[0] + .5 * previous[2] - .5 * side),
                           round(previous[1] + .5 * previous[3] - .5 * side)])
        visible = np.maximum(0., np.minimum(origin + side, target[:2] + target[2:])
                             - np.maximum(origin, target[:2])).prod()
        row['nominal_target_crop_fraction'] = float(visible / target[2:].prod())
    crop_support = {}
    for name, predicate in [
        ('no_nominal_overlap', lambda value: value <= 0.),
        ('partial_nominal_overlap', lambda value: 0. < value < 1. - 1e-10),
        ('full_nominal_overlap', lambda value: value >= 1. - 1e-10)]:
        values = [row for row in categories['mixed_missing']
                  if predicate(row['nominal_target_crop_fraction'])]
        if values:
            crop_support[name] = {field: bootstrap(values, field)
                                 for field in ['native_iou', 'shadow_iou', 'gain']}
    if args.server_files:
        names = Path('lib/train/data_specs/lasher_train.txt').read_text().splitlines()
        assert report['fitting_names'] == [name for index, name in enumerate(names) if index % 5]
        for path, expected in report['source_sha256'].items():
            assert sha256(path) == expected, path
        for prefix in ['original', 'wrapper']:
            assert sha256(report[prefix + '_checkpoint']) == report[prefix + '_checkpoint_sha256']
        from lib.train.dataset import LasHeR
        dataset = LasHeR(root='/data/wangwj/dataset/LasHeR/trainingset', split='train')
        for clip in clips:
            info = dataset.get_sequence_info(dataset.sequence_list.index(clip['sequence']))
            ids = clip['frame_ids']
            assert info['visible'][ids].all()
            assert info['bbox'][ids].tolist() == clip['gt_boxes']
    replay = replay_first_clip(report) if args.replay_first_clip else None
    result = dict(status='verified', report_sha256=completed['report_sha256'],
        analyzer_sha256=sha256(__file__), effects=effects, advance=advance,
        descriptive_iou=descriptive, exploratory_prior_error_bins=error_bins,
        exploratory_nominal_crop_support=crop_support,
        nominal_crop_support_note='Native factor4 rounded crop bounds before image padding; '
                                  'post-hoc description, not a promotion criterion.',
        server_files_checked=args.server_files, replay=replay,
        limitation='Previous-GT crop oracle under native histories; no trained policy or official scores. '
                   'State/gradient absence assertions are producer checks.')
    if args.output:
        with Path(args.output).open('x', newline='\n') as stream:
            json.dump(result, stream, indent=2)
            stream.write('\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
