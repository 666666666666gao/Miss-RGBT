"""Frozen same-native-history comparison of last-observed sensor templates."""
import argparse
import json
import os
from pathlib import Path
import time

import cv2
import numpy as np
import torch
from lib.config.cadtrack.config import cfg, update_config_from_file
from lib.train.dataset import LasHeR
from miss_rgbt.tracker import MissingTracker
import train_causal_crop_exposure as training

common = training.common
PROJECT = training.PROJECT
STATES = training.STATES
COUNT, LENGTH, SEED = 32, 128, 59101


def exclusions():
    used, hashes = training.causal.excluded_frames()
    path = PROJECT / 'reports/server28_causal_crop_exposure_20261102/report.json'
    report = json.loads(path.read_text())
    hashes[path.parent.name] = common.sha256(path)
    for clip in report['clips']:
        used[clip['sequence']].update(clip['frame_ids'])
    root = PROJECT / 'reports/server28_causal_crop_training_20261102'
    initial = json.loads((root / 'initial_clips.json').read_text())
    clips = initial['train'] + initial['development'] + json.loads((root / 'audit_clips.json').read_text())
    for clip in clips:
        used[clip['sequence']].update(clip['frame_ids'])
    for name in ['initial_clips.json', 'audit_clips.json']:
        hashes[root.name + '/' + name] = common.sha256(root / name)
    return used, hashes


def retain_observed(bank, candidate, flags):
    return torch.cat([candidate[:, :3] if flags[0] else bank[:, :3],
                      candidate[:, 3:] if flags[1] else bank[:, 3:]], 1)


def summaries(rows):
    effects = {}
    for state in ['01', '10', 'missing', 'complete']:
        chosen = [row for row in rows if (
            row['state'] != '11' if state == 'missing' else
            row['state'] == '11' if state == 'complete' else row['state'] == state)]
        values = [dict(identity=dict(sequence=row['sequence']),
            gain=row['retained_iou'] - row['native_iou'],
            control=row['retained_iou'] - row['complete_only_iou']) for row in chosen]
        effects[state] = {key: common.clustered(values, key, 59301)
                          for key in ['gain', 'control']}
    advance = (all(effects[state]['gain']['mean'] >= .01
                   and effects[state]['gain']['ci95_low'] > 0
                   and effects[state]['gain']['sequences'] == COUNT for state in ['01', '10'])
        and effects['missing']['control']['ci95_low'] > 0
        and effects['missing']['control']['sequences'] == COUNT
        and effects['complete']['gain']['mean'] >= 0
        and effects['complete']['gain']['ci95_low'] > -.01
        and effects['complete']['gain']['sequences'] == COUNT)
    return effects, bool(advance)


def run_clip(tracker, dataset, clip):
    frames, boxes = training.load_clip(dataset, clip)
    tracker.initialize(frames[0], dict(init_bbox=boxes[0]))
    assert len(tracker.templates) == 2
    retained = tracker.templates[1]
    complete_only = tracker.templates[1]
    first = clip['frame_ids'][0]
    retained_sources, complete_sources, native_sources = [first, first], [first, first], [first, first]
    rows, events = [], []
    with torch.no_grad():
        for index in range(1, LENGTH):
            frame = clip['frame_ids'][index]
            state = clip['mixed_schedule'][index]
            flags = STATES[state]
            prior = training.causal.snapshot(tracker)
            selected = index <= 3 or index % 4 == 0
            shadows, sham = {}, None
            parity = bool(prior['availability'][1].eq(1).all())
            provenance = dict(native_sources=list(native_sources),
                retained_sources=list(retained_sources), complete_sources=list(complete_sources))
            if selected:
                provenance.update(native_availability=prior['availability'][1].tolist()[0],
                    native_tensor_sha256=common.digest({'image': prior['templates'][1]}),
                    retained_tensor_sha256=common.digest({'image': retained}),
                    complete_only_tensor_sha256=common.digest({'image': complete_only}))
                if parity:
                    assert torch.equal(retained, prior['templates'][1])
                    assert torch.equal(complete_only, prior['templates'][1])
                for label, bank in [('retained', retained), ('complete_only', complete_only)]:
                    tracker.templates = [prior['templates'][0], bank]
                    tracker.availability = [prior['availability'][0], torch.ones(1, 2, device='cuda')]
                    shadows[label] = tracker.track(frames[index], flags)
                    training.causal.restore(tracker, prior)
                    training.causal.same_snapshot(tracker, prior)
                if index <= 3:
                    sham = tracker.track(frames[index], flags)
                    training.causal.restore(tracker, prior)
                    training.causal.same_snapshot(tracker, prior)
            native = tracker.track(frames[index], flags)
            if sham is not None:
                assert native == sham
            if selected:
                if parity:
                    assert all(result == native for result in shadows.values())
                row = dict(sequence=clip['sequence'], frame=frame, call=index, state=state,
                    prior_box=prior['state'], gt=boxes[index], provenance=provenance,
                    complete_history_parity=parity, sham_checked=sham is not None)
                for label, result in dict(native=native, **shadows).items():
                    row[label + '_box'] = list(result['target_bbox'])
                    row[label + '_confidence'] = result['best_score']
                    row[label + '_iou'] = training.burst.box_iou(result['target_bbox'], boxes[index])
                rows.append(row)
            wrote = tracker.templates[1] is not prior['templates'][1]
            assert wrote == (tracker.frame_id % cfg.TEST.UPDATE_INTERVALS == 0
                             and native['best_score'] > cfg.TEST.UPDATE_THRESHOLD)
            if wrote:
                candidate = tracker.templates[1]
                before_sources = list(retained_sources)
                retained = retain_observed(retained, candidate, flags)
                for stream in range(2):
                    if flags[stream]:
                        retained_sources[stream] = frame
                native_sources = [frame, frame]
                if state == '11':
                    complete_only, complete_sources = candidate, [frame, frame]
                # Unobserved candidate pixels cannot enter either retained slice.
                poisoned = candidate.clone()
                if not flags[0]:
                    poisoned[:, :3] = 12345.
                if not flags[1]:
                    poisoned[:, 3:] = 12345.
                assert torch.equal(retained, retain_observed(retained, poisoned, flags))
                events.append(dict(sequence=clip['sequence'], frame=frame, call=index, state=state,
                    confidence=native['best_score'], candidate_sha256=common.digest({'image': candidate}),
                    retained_before_sources=before_sources, retained_sources=list(retained_sources),
                    complete_sources=list(complete_sources),
                    retained_sha256=common.digest({'image': retained}),
                    complete_only_sha256=common.digest({'image': complete_only})))
    return dict(rows=rows, events=events,
        image_sha256=training.burst.sha_image_sequence(frames), gt=boxes)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    root = Path(args.output)
    root.mkdir(exist_ok=False)
    started = time.time()
    common.seed(SEED)
    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    update_config_from_file('experiments/cadtrack/cadtrack.yaml')
    assert common.sha256(common.CHECKPOINT) == common.CHECKPOINT_SHA
    names_path = Path('lib/train/data_specs/lasher_train.txt')
    names = names_path.read_text().splitlines()
    assert len(names) == len(set(names)) == 882
    fitting = [name for index, name in enumerate(names) if index % 5]
    used, prior_hashes = exclusions()
    dataset = LasHeR(root=training.burst.DATASET, split='train')
    clips = training.fresh_clips(dataset, fitting, COUNT, LENGTH, SEED, used)
    training.save(root / 'selected_clips.json', clips)
    sources = [Path(__file__), Path(training.__file__), Path(training.causal.__file__),
        Path(common.__file__), Path(training.burst.__file__), names_path,
        Path('experiments/cadtrack/cadtrack.yaml'), Path('lib/train/dataset/lasher.py'),
        Path('lib/train/data/processing_utils.py'), Path('lib/test/tracker/data_utils.py'),
        Path('lib/test/utils/hann.py'), Path('lib/utils/box_ops.py')]
    sources += sorted(Path('miss_rgbt').rglob('*.py')) + sorted(Path('lib/models').rglob('*.py'))
    hashes = {str(path.resolve()): common.sha256(path) for path in sources}
    training.save(root / 'source_sha256.json', hashes)
    baseline = training.load_pretrained(cfg, common.CHECKPOINT, repair=False)
    expected = common.digest(baseline.state_dict())
    del baseline
    tracker = MissingTracker(cfg, str(training.causal.WRAPPER))
    tracker.network.eval().requires_grad_(False)
    before = common.digest(tracker.network.state_dict())
    assert before == expected
    rows, events, manifests = [], [], []
    for index, clip in enumerate(clips):
        result = run_clip(tracker, dataset, clip)
        rows.extend(result['rows'])
        events.extend(result['events'])
        manifests.append(dict(clip, image_sha256=result['image_sha256'], gt=result['gt']))
        print(json.dumps(dict(stage='clip', clips=index + 1, rows=len(rows),
            writes=len(events), seconds=time.time() - started)), flush=True)
    assert len(rows) == COUNT * 34
    assert common.digest(tracker.network.state_dict()) == before
    assert all(value.grad is None for value in tracker.network.parameters())
    for row in rows:
        assert np.isfinite(sum([row[label + '_box'] + [row[label + '_confidence'],
            row[label + '_iou']] for label in ['native', 'retained', 'complete_only']], [])).all()
    effects, advance = summaries(rows)
    report = dict(status='complete', pid=os.getpid(), seed=SEED, clips=manifests,
        rows=rows, events=events, effects=effects, advance=advance,
        fitting_names=fitting, prior_reports_sha256=prior_hashes, source_sha256=hashes,
        checkpoint=common.CHECKPOINT, checkpoint_sha256=common.CHECKPOINT_SHA,
        wrapper_checkpoint=str(training.causal.WRAPPER),
        wrapper_sha256=common.sha256(training.causal.WRAPPER), baseline_state_sha256=before,
        state_unchanged=True, gradients_absent=True, seconds=time.time() - started,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        scope='Train882 same-native-history one-step template intervention, not official metrics.')
    training.save(root / 'report.json', report)
    training.save(root / 'completed.json', dict(status='complete', advance=advance,
        report_sha256=common.sha256(root / 'report.json'), seconds=report['seconds']))
    print(json.dumps(dict(stage='complete', effects=effects, advance=advance,
                          seconds=report['seconds'])), flush=True)


if __name__ == '__main__':
    main()
