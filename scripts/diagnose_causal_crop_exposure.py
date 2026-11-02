"""Frozen E0 native-history crop intervention on fresh train882 clips."""
import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import time

import cv2
import numpy as np
import torch

from lib.config.cadtrack.config import cfg, update_config_from_file
from lib.train.dataset import LasHeR
from miss_rgbt.model import load_pretrained
from miss_rgbt.tracker import MissingTracker
import diagnose_absent_position as common
import diagnose_burst_mask_rollout as burst
import diagnose_teacher_reliability as reliability

PROJECT = Path('/home/gaob/Miss-RGBT')
STATES = {'11': [1., 1.], '01': [0., 1.], '10': [1., 0.]}
CLIPS, LENGTH, SEED = 48, 96, 57101
WRAPPER = PROJECT / 'checkpoints/e0_availability/epoch20.pth.tar'


def save(path, value):
    with Path(path).open('x', newline='\n') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')


def excluded_frames():
    tuples, hashes = reliability.exclusions()
    for stem in ['server28_teacher_reliability_20261102',
                 'server28_joint_gradients_20261102']:
        path = PROJECT / 'reports' / stem / 'report.json'
        report = json.loads(path.read_text())
        manifest = (report['cache_manifest']['input_manifest'] if 'cache_manifest' in report
                    else report['input_manifest'])
        tuples.update(reliability.identity_key(item['identity']) for item in manifest)
        hashes[stem] = common.sha256(path)
    used = defaultdict(set)
    for name, templates, searches in tuples:
        used[name].update(templates + searches)
    path = PROJECT / 'reports/server28_burst_mask_rollout_20261102/report.json'
    report = json.loads(path.read_text())
    hashes[path.parent.name] = common.sha256(path)
    for clip in report['clips']:
        used[clip['sequence']].update(clip['frame_ids'])
    return used, hashes


def choose_clips(dataset, fitting, used, rng):
    clips = []
    for index in rng.permutation(len(fitting)):
        name = fitting[index]
        sequence = dataset.sequence_list.index(name)
        info = dataset.get_sequence_info(sequence)
        boxes = info['bbox'].numpy()
        valid = (np.isfinite(boxes).all(1) & (boxes[:, 2:] > 0).all(1)
                 & info['visible'].numpy().astype(bool))
        starts = [start for start in range(len(boxes) - LENGTH + 1)
                  if valid[start:start + LENGTH].all()
                  and not used[name].intersection(range(start, start + LENGTH))]
        if starts:
            start = int(rng.choice(starts))
            clips.append(dict(sequence=name, dataset_index=sequence,
                              frame_ids=list(range(start, start + LENGTH))))
        if len(clips) == CLIPS:
            break
    assert len(clips) == CLIPS
    return clips


def snapshot(tracker):
    return dict(state=list(tracker.state), templates=list(tracker.templates),
                availability=list(tracker.availability), frame_id=tracker.frame_id)


def restore(tracker, saved):
    tracker.state = list(saved['state'])
    tracker.templates = list(saved['templates'])
    tracker.availability = list(saved['availability'])
    tracker.frame_id = saved['frame_id']


def same_snapshot(tracker, saved):
    assert tracker.state == saved['state'] and tracker.frame_id == saved['frame_id']
    assert all(first is second for first, second in zip(tracker.templates, saved['templates']))
    assert all(first is second for first, second in zip(tracker.availability, saved['availability']))


def effects(rows):
    selections = {
        'complete': lambda row: row['trajectory'] == 'complete',
        'mixed_complete': lambda row: row['trajectory'] == 'mixed' and row['state'] == '11',
        'mixed_rgb_missing': lambda row: row['trajectory'] == 'mixed' and row['state'] == '01',
        'mixed_tir_missing': lambda row: row['trajectory'] == 'mixed' and row['state'] == '10',
        'mixed_missing': lambda row: row['trajectory'] == 'mixed' and row['state'] != '11',
    }
    output = {}
    for key, predicate in selections.items():
        values = [dict(identity=dict(sequence=row['sequence']),
                       gain=row['shadow_iou'] - row['native_iou']) for row in rows if predicate(row)]
        output[key] = common.clustered(values, 'gain', 57301)
    missing = output['mixed_missing']
    advance = missing['mean'] >= .02 and missing['ci95_low'] > 0 and missing['sequences'] == CLIPS
    return output, bool(advance)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(exist_ok=False)
    started = time.time()
    common.seed(SEED)
    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    update_config_from_file('experiments/cadtrack/cadtrack.yaml')
    assert common.sha256(common.CHECKPOINT) == common.CHECKPOINT_SHA
    baseline = load_pretrained(cfg, common.CHECKPOINT, repair=False)
    expected = common.digest(baseline.state_dict())
    del baseline
    tracker = MissingTracker(cfg, str(WRAPPER))
    tracker.network.eval().requires_grad_(False)
    assert common.digest(tracker.network.state_dict()) == expected
    train_file = Path('lib/train/data_specs/lasher_train.txt')
    names = train_file.read_text().splitlines()
    assert len(names) == len(set(names)) == 882
    fitting = [name for index, name in enumerate(names) if index % 5]
    dataset = LasHeR(root=burst.DATASET, split='train')
    used, prior_hashes = excluded_frames()
    rng = np.random.default_rng(SEED)
    clips = choose_clips(dataset, fitting, used, rng)
    rows, manifests = [], []
    save(output / 'selected_clips.json', clips)
    sources = [Path(__file__), Path(common.__file__), Path(burst.__file__),
               Path(reliability.__file__), train_file,
               Path('experiments/cadtrack/cadtrack.yaml'),
               Path('lib/train/dataset/lasher.py'), Path('lib/train/data/processing_utils.py')]
    sources += sorted(Path('miss_rgbt').rglob('*.py')) + sorted(Path('lib/models').rglob('*.py'))
    source_hashes = {str(path.resolve()): common.sha256(path) for path in sources}
    for clip_index, clip in enumerate(clips):
        info = dataset.get_sequence_info(clip['dataset_index'])
        frames, annotations, _ = dataset.get_frames(clip['dataset_index'], clip['frame_ids'], info)
        boxes = [value.tolist() for value in annotations['bbox']]
        mixed = ['11'] + rng.choice(list(STATES), size=LENGTH - 1, p=[.4, .3, .3]).tolist()
        schedules = dict(complete=['11'] * LENGTH, mixed=mixed)
        for trajectory, schedule in schedules.items():
            tracker.initialize(frames[0], dict(init_bbox=boxes[0]))
            for index in range(1, LENGTH):
                prior = snapshot(tracker)
                tracker.state = list(boxes[index - 1])
                shadow = tracker.track(frames[index], STATES[schedule[index]])
                restore(tracker, prior)
                same_snapshot(tracker, prior)
                sham = None
                if index <= 3:
                    sham = tracker.track(frames[index], STATES[schedule[index]])
                    restore(tracker, prior)
                    same_snapshot(tracker, prior)
                native = tracker.track(frames[index], STATES[schedule[index]])
                if sham is not None:
                    assert native == sham
                if index == 1:
                    assert native == shadow
                center_error = np.linalg.norm(
                    np.asarray(prior['state'][:2]) + .5 * np.asarray(prior['state'][2:])
                    - np.asarray(boxes[index - 1][:2]) - .5 * np.asarray(boxes[index - 1][2:]))
                scale = np.sqrt(np.prod(boxes[index - 1][2:]))
                rows.append(dict(sequence=clip['sequence'], trajectory=trajectory,
                    frame=clip['frame_ids'][index], previous_frame=clip['frame_ids'][index - 1],
                    state=schedule[index], previous_gt=boxes[index - 1], gt=boxes[index],
                    previous_native=prior['state'], prior_center_error=float(center_error / scale),
                    template_availability=[value.cpu().tolist() for value in prior['availability']],
                    native_box=list(native['target_bbox']), shadow_box=list(shadow['target_bbox']),
                    native_confidence=native['best_score'], shadow_confidence=shadow['best_score'],
                    native_iou=burst.box_iou(native['target_bbox'], boxes[index]),
                    shadow_iou=burst.box_iou(shadow['target_bbox'], boxes[index]),
                    sham_checked=sham is not None,
                    sham_box=None if sham is None else list(sham['target_bbox']),
                    sham_confidence=None if sham is None else sham['best_score']))
        manifests.append(dict(sequence=clip['sequence'], frame_ids=clip['frame_ids'],
            image_sha256=burst.sha_image_sequence(frames), schedules=schedules, gt_boxes=boxes))
        print(json.dumps(dict(stage='clip', completed=clip_index + 1, expected=CLIPS,
                              seconds=time.time() - started)), flush=True)
    assert len(rows) == CLIPS * (LENGTH - 1) * 2
    assert common.digest(tracker.network.state_dict()) == expected
    assert all(parameter.grad is None for parameter in tracker.network.parameters())
    assert all(np.isfinite([row['native_iou'], row['shadow_iou'],
        row['native_confidence'], row['shadow_confidence'], row['prior_center_error']]
        + row['native_box'] + row['shadow_box']).all() for row in rows)
    statistics, advance = effects(rows)
    report = dict(status='complete', pid=os.getpid(), seed=SEED, clips=manifests, rows=rows,
        effects=statistics, advance=advance, fitting_names=fitting, prior_reports_sha256=prior_hashes,
        original_checkpoint=common.CHECKPOINT, original_checkpoint_sha256=common.CHECKPOINT_SHA,
        wrapper_checkpoint=str(WRAPPER), wrapper_checkpoint_sha256=common.sha256(WRAPPER),
        baseline_state_sha256=expected, state_unchanged=True, gradients_absent=True,
        source_sha256=source_hashes, seconds=time.time() - started,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        scope='Train-only causal previous-GT crop oracle under native template histories; no official scores.')
    save(output / 'report.json', report)
    save(output / 'completed.json', dict(status='complete', report_sha256=common.sha256(output / 'report.json'),
                                        advance=advance, seconds=report['seconds']))
    print(json.dumps(dict(stage='complete', effects=statistics, advance=advance,
                          seconds=report['seconds'])), flush=True)


if __name__ == '__main__':
    main()
