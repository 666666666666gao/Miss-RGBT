"""Recompute saved last-observed template evidence and optional image replay."""
import argparse
import json
from pathlib import Path

import numpy as np
from analyze_causal_crop_exposure import exclusions as prior_exclusions, overlap
from analyze_cross_stream_context import sha256


def exclusions(folder):
    used, hashes = prior_exclusions(folder)
    path = folder / 'server28_causal_crop_exposure_20261102/report.json'
    hashes[path.parent.name] = sha256(path)
    for clip in json.loads(path.read_text())['clips']:
        used[clip['sequence']].update(clip['frame_ids'])
    root = folder / 'server28_causal_crop_training_20261102'
    initial = json.loads((root / 'initial_clips.json').read_text())
    for clip in initial['train'] + initial['development'] + json.loads((root / 'audit_clips.json').read_text()):
        used[clip['sequence']].update(clip['frame_ids'])
    for name in ['initial_clips.json', 'audit_clips.json']:
        hashes[root.name + '/' + name] = sha256(root / name)
    return used, hashes


def server_checks(report, root, replay):
    import cv2
    import torch
    from lib.config.cadtrack.config import cfg, update_config_from_file
    from lib.train.dataset import LasHeR
    from miss_rgbt.tracker import MissingTracker
    import diagnose_last_observed_template as worker

    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    update_config_from_file('experiments/cadtrack/cadtrack.yaml')
    assert cfg.TEST.UPDATE_INTERVALS == 5 and cfg.TEST.UPDATE_THRESHOLD == .65
    for path, value in report['source_sha256'].items():
        assert sha256(path) == value, path
    for name, key in [('checkpoint', 'checkpoint_sha256'), ('wrapper_checkpoint', 'wrapper_sha256')]:
        assert sha256(report[name]) == report[key]
    baseline = worker.training.load_pretrained(cfg, report['checkpoint'], repair=False)
    assert worker.common.digest(baseline.state_dict()) == report['baseline_state_sha256']
    del baseline
    names = Path('lib/train/data_specs/lasher_train.txt').read_text().splitlines()
    fitting = [name for index, name in enumerate(names) if index % 5]
    assert report['fitting_names'] == fitting
    dataset = LasHeR(root=worker.training.burst.DATASET, split='train')
    used, _ = exclusions(root.parent)
    rng = np.random.default_rng(59101)
    reconstructed = []
    for offset in rng.permutation(len(fitting)):
        name = fitting[offset]
        sequence = dataset.sequence_list.index(name)
        info = dataset.get_sequence_info(sequence)
        boxes = info['bbox'].numpy()
        valid = (np.isfinite(boxes).all(1) & (boxes[:, 2:] > 0).all(1)
                 & info['visible'].numpy().astype(bool))
        starts = [start for start in range(len(boxes) - 127)
                  if valid[start:start + 128].all()
                  and not used[name].intersection(range(start, start + 128))]
        if starts:
            start = int(rng.choice(starts))
            schedule = ['11'] + rng.choice(['11', '01', '10'], 127, p=[.4, .3, .3]).tolist()
            reconstructed.append(dict(sequence=name, dataset_index=sequence,
                frame_ids=list(range(start, start + 128)), mixed_schedule=schedule))
        if len(reconstructed) == 32:
            break
    selected = json.loads((root / 'selected_clips.json').read_text())
    assert reconstructed == selected
    for clip in report['clips']:
        info = dataset.get_sequence_info(clip['dataset_index'])
        assert info['visible'][clip['frame_ids']].all()
        assert info['bbox'][clip['frame_ids']].tolist() == clip['gt']
    replay_rows = replay_events = 0
    if replay:
        tracker = MissingTracker(cfg, report['wrapper_checkpoint'])
        tracker.network.eval().requires_grad_(False)
        before = worker.common.digest(tracker.network.state_dict())
        result = worker.run_clip(tracker, dataset, selected[0])
        name = selected[0]['sequence']
        assert result['rows'] == [row for row in report['rows'] if row['sequence'] == name]
        assert result['events'] == [row for row in report['events'] if row['sequence'] == name]
        assert result['image_sha256'] == report['clips'][0]['image_sha256']
        assert result['gt'] == report['clips'][0]['gt']
        assert worker.common.digest(tracker.network.state_dict()) == before == report['baseline_state_sha256']
        assert all(value.grad is None for value in tracker.network.parameters())
        replay_rows, replay_events = len(result['rows']), len(result['events'])
    return dict(sources_checkpoint_actual_gt_verified=True, seeded_clip_selection_reconstructed=True,
        first_clip_rows_replayed=replay_rows, first_clip_events_replayed=replay_events,
        shared_policy_worker_and_native_tracker=True, full_dataset_image_replay=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('directory')
    parser.add_argument('--server-files', action='store_true')
    parser.add_argument('--replay-first-clip', action='store_true')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    root = Path(args.directory)
    report = json.loads((root / 'report.json').read_text())
    completed = json.loads((root / 'completed.json').read_text())
    launch = json.loads((root.parent / (root.name + '.launch.json')).read_text())
    assert report['status'] == completed['status'] == 'complete'
    assert report['pid'] == launch['pid'] and report['seed'] == 59101
    assert sha256(root / 'report.json') == completed['report_sha256']
    assert sha256(root.parent / (root.name + '.preregistered_protocol.md')) == launch['protocol_sha256']
    assert report['source_sha256'][launch['command'][2]] == launch['script_sha256']
    assert report['source_sha256'] == json.loads((root / 'source_sha256.json').read_text())
    assert report['state_unchanged'] and report['gradients_absent']
    used, hashes = exclusions(root.parent)
    assert all(hashes[name] == value for name, value in report['prior_reports_sha256'].items())
    clips = report['clips']
    assert len(clips) == len({clip['sequence'] for clip in clips}) == 32
    assert len(report['fitting_names']) == len(set(report['fitting_names'])) == 705
    selected = json.loads((root / 'selected_clips.json').read_text())
    assert selected == [{key: clip[key] for key in [
        'sequence', 'dataset_index', 'frame_ids', 'mixed_schedule']} for clip in clips]
    seen, parity, shams, incomplete_writes = set(), 0, 0, 0
    for clip in clips:
        name, ids = clip['sequence'], clip['frame_ids']
        assert name in report['fitting_names']
        assert len(ids) == 128 and ids == list(range(ids[0], ids[0] + 128))
        assert not used[name].intersection(ids)
        schedule = clip['mixed_schedule']
        assert len(schedule) == 128 and schedule[0] == '11' and set(schedule) <= {'11', '01', '10'}
        assert len(clip['gt']) == 128
        values = {row['call']: row for row in report['rows'] if row['sequence'] == name}
        assert set(values) == set(range(4, 128, 4)) | {1, 2, 3}
        events = [event for event in report['events'] if event['sequence'] == name]
        assert events == sorted(events, key=lambda event: event['call'])
        assert len(events) == len({event['call'] for event in events})
        assert all(event['call'] % 5 == 0 and event['confidence'] > .65 for event in events)
        last_native = last_retained = last_complete = [ids[0], ids[0]]
        availability = [1., 1.]
        first = values[1]['provenance']
        native_hash = retained_hash = complete_hash = first['native_tensor_sha256']
        event_by_call = {event['call']: event for event in events}
        for call in range(1, 128):
            state = schedule[call]
            if call in values:
                row = values[call]
                assert row['frame'] == ids[call] and row['state'] == state and row['gt'] == clip['gt'][call]
                identity = name, row['frame']
                assert identity not in seen
                seen.add(identity)
                expected = dict(native_sources=last_native, retained_sources=last_retained,
                    complete_sources=last_complete, native_availability=availability,
                    native_tensor_sha256=native_hash, retained_tensor_sha256=retained_hash,
                    complete_only_tensor_sha256=complete_hash)
                assert row['provenance'] == expected
                assert all(source <= ids[call - 1] for source in last_retained + last_complete + last_native)
                assert row['complete_history_parity'] == (availability == [1., 1.])
                assert row['sham_checked'] == (call <= 3)
                shams += row['sham_checked']
                for label in ['native', 'retained', 'complete_only']:
                    assert np.isfinite(row[label + '_box'] + [
                        row[label + '_confidence'], row[label + '_iou']]).all()
                    assert abs(overlap(row[label + '_box'], row['gt']) - row[label + '_iou']) < 1e-10
                if row['complete_history_parity']:
                    assert native_hash == retained_hash == complete_hash
                    for label in ['retained', 'complete_only']:
                        assert row[label + '_box'] == row['native_box']
                        assert row[label + '_confidence'] == row['native_confidence']
                    parity += 1
            if call in event_by_call:
                event = event_by_call[call]
                assert event['frame'] == ids[call] and event['state'] == state
                assert event['retained_before_sources'] == last_retained
                last_native = [ids[call], ids[call]]
                availability = {'11': [1., 1.], '01': [0., 1.], '10': [1., 0.]}[state]
                last_retained = [ids[call] if flag else last_retained[index]
                                 for index, flag in enumerate(availability)]
                if state == '11':
                    last_complete = [ids[call], ids[call]]
                else:
                    incomplete_writes += 1
                assert event['retained_sources'] == last_retained and event['complete_sources'] == last_complete
                native_hash, retained_hash, complete_hash = [
                    event[key] for key in ['candidate_sha256', 'retained_sha256', 'complete_only_sha256']]
    assert len(seen) == len(report['rows']) == 1088 and shams == 96
    assert sum(len([event for event in report['events'] if event['sequence'] == clip['sequence']])
               for clip in clips) == len(report['events'])
    effects = {}
    for state in ['01', '10', 'missing', 'complete']:
        rows = [row for row in report['rows'] if (
            row['state'] != '11' if state == 'missing' else
            row['state'] == '11' if state == 'complete' else row['state'] == state)]
        values = [dict(sequence=row['sequence'], gain=row['retained_iou'] - row['native_iou'],
            control=row['retained_iou'] - row['complete_only_iou']) for row in rows]
        for field in ['gain', 'control']:
            grouped = {name: np.mean([row[field] for row in values if row['sequence'] == name])
                       for name in sorted({row['sequence'] for row in values})}
            means = np.asarray(list(grouped.values()))
            rng = np.random.default_rng(59301)
            draws = rng.integers(0, len(means), (10000, len(means)))
            low, high = np.quantile(means[draws].mean(1), [.025, .975])
            effects.setdefault(state, {})[field] = dict(mean=float(means.mean()),
                ci95_low=float(low), ci95_high=float(high), sequences=len(means), rows=len(values))
    assert effects == report['effects']
    advance = (all(effects[state]['gain']['mean'] >= .01 and effects[state]['gain']['ci95_low'] > 0
                   and effects[state]['gain']['sequences'] == 32 for state in ['01', '10'])
        and effects['missing']['control']['ci95_low'] > 0
        and effects['missing']['control']['sequences'] == 32
        and effects['complete']['gain']['mean'] >= 0
        and effects['complete']['gain']['ci95_low'] > -.01
        and effects['complete']['gain']['sequences'] == 32)
    assert advance == report['advance'] == completed['advance']
    server = server_checks(report, root, args.replay_first_clip) if args.server_files else None
    result = dict(status='verified', report_sha256=completed['report_sha256'],
        analyzer_sha256=sha256(__file__), effects=effects, advance=bool(advance),
        rows=1088, shams=shams, complete_history_parity_rows=parity,
        writes=len(report['events']), incomplete_writes=incomplete_writes, server_checks=server,
        limitation='Same-native-history shadow intervention; not closed-loop policy benefit, '
                   'official metrics, external scientific review or validated novelty.')
    with Path(args.output).open('x', newline='\n') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
