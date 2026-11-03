"""Synthetic-frame F tracker integration, no dataset or benchmark access."""
import argparse
import copy
import io
import json
from pathlib import Path
import time
from unittest.mock import patch

import numpy as np
import torch

from lib.config.cadtrack.config import cfg, update_config_from_file
from miss_rgbt.model import load_pretrained
import miss_rgbt.tracker as native_tracker
from model import FeedbackTrackerModel
from tracker import FeedbackTracker
from check_structure import sha


class CountList(list):
    def __init__(self, values):
        super().__init__(values)
        self.appends = 0
        self.pops = 0

    def append(self, value):
        self.appends += 1
        return super().append(value)

    def pop(self, index):
        self.pops += 1
        return super().pop(index)


def checkpoint(model, config):
    stream = io.BytesIO()
    torch.save(dict(model=model.state_dict(), config=config), stream)
    stream.seek(0)
    return stream


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    output = Path(args.output)
    assert not output.exists()
    expected_sha = 'd20c8641e4263df3ceadf9e430d0ac96c181c13ad482bca28c2dddef592a8bb8'
    assert sha(args.checkpoint) == expected_sha
    started = time.time()
    torch.set_num_threads(1)
    torch.manual_seed(65302)
    torch.cuda.manual_seed_all(65302)
    random = np.random.RandomState(65302)
    update_config_from_file('experiments/cadtrack/cadtrack.yaml')
    test_cfg = copy.deepcopy(cfg)
    test_cfg.TEST.UPDATE_INTERVALS = 1
    test_cfg.TEST.UPDATE_THRESHOLD = -1.
    utility = dict(width=128, depth=1, region_grid=4, selected_regions=4,
        refinement_limit=.1, selection='utility', selection_seed=65302)
    base = load_pretrained(test_cfg, args.checkpoint, task_utility=utility).eval()
    with torch.no_grad():
        base.utility_repair.delta.weight.normal_(std=.001)
    model = FeedbackTrackerModel(base.baseline, task_utility=utility).eval()
    model.utility_repair.load_state_dict(base.utility_repair.state_dict(), strict=True)
    base_config = dict(repair=False, repair_width=128, repair_layer=6, task_utility=utility)
    config = dict(base_config, feedback=dict(mode='deep', width=128))
    baseline = native_tracker.MissingTracker(test_cfg, checkpoint(base, base_config))
    tracker = FeedbackTracker(test_cfg, checkpoint(model, config))
    single = FeedbackTracker(test_cfg, checkpoint(base,
        dict(base_config, feedback=dict(mode='single', width=128))))
    assert FeedbackTracker.track is native_tracker.MissingTracker.track
    assert FeedbackTracker.initialize is native_tracker.MissingTracker.initialize
    checks = dict(inherited_native_initialize_and_track=True)
    for mode in ['none', 'shallow', 'deep-no-response', 'deep']:
        control = FeedbackTrackerModel(base.baseline, task_utility=utility, feedback_mode=mode).eval()
        control.utility_repair.load_state_dict(base.utility_repair.state_dict(), strict=True)
        loaded = FeedbackTracker(test_cfg, checkpoint(control,
            dict(base_config, feedback=dict(mode=mode, width=128))))
        assert loaded.network.feedback_mode == mode
        del loaded, control
    checks['all_feedback_checkpoint_modes_strict_load'] = True
    frame = random.randint(0, 256, (180, 240, 6), dtype=np.uint8)
    box = [80., 60., 35., 30.]
    baseline.initialize(frame, dict(init_bbox=box))
    tracker.initialize(frame, dict(init_bbox=box))
    single.initialize(frame, dict(init_bbox=box))
    parity = []
    for flags in [[1, 1], [0, 1], [1, 0]]:
        current = random.randint(0, 256, frame.shape, dtype=np.uint8)
        expected = baseline.track(current, flags)
        actual = tracker.track(current, flags)
        assert expected == actual
        assert single.track(current, flags) == expected
        assert single.frame_id == baseline.frame_id
        assert all(torch.equal(left, right) for left, right in
                   zip(single.templates, baseline.templates))
        assert all(torch.equal(left, right) for left, right in
                   zip(single.availability, baseline.availability))
        assert tracker.frame_id == baseline.frame_id
        assert all(torch.equal(left, right) for left, right in
                   zip(tracker.templates, baseline.templates))
        assert all(torch.equal(left, right) for left, right in
                   zip(tracker.availability, baseline.availability))
        parity.append(dict(availability=flags, frame_id=tracker.frame_id, exact=True))
    checks['zero_residual_native_multiframe_box_template_availability_parity'] = parity
    checks['single_checkpoint_native_multiframe_parity_exact'] = True
    with torch.no_grad():
        tracker.network.feedback.delta.weight.normal_(std=.01)
    rounds = []

    def capture(module, inputs, outputs):
        rounds.append(outputs[0])

    handle = tracker.network.register_forward_hook(capture)
    original_clip = native_tracker.clip_box
    update_cases = []
    for allow_update in [True, False]:
        tracker.cfg.TEST.UPDATE_THRESHOLD = -1. if allow_update else 2.
        tracker.initialize(frame, dict(init_bbox=box))
        tracker.templates = CountList(tracker.templates)
        tracker.availability = CountList(tracker.availability)
        old_template0 = tracker.templates[0]
        old_template1 = tracker.templates[1]
        observed = [0, 1]
        rounds.clear()
        with patch.object(native_tracker, 'clip_box', wraps=original_clip) as clipped, \
                patch.object(native_tracker, 'sample_target',
                             wraps=native_tracker.sample_target) as crops:
            actual = tracker.track(frame, observed)
            assert clipped.call_count == 1
            assert crops.call_count == (2 if allow_update else 1)
            # Independent decode and image mapping using only round2 output.
            prediction = rounds[0]
            response = tracker.window * prediction['score_map']
            predicted = tracker.network.baseline.box_head.cal_bbox(
                response, prediction['size_map'], prediction['offset_map'])
            _, factor, _ = native_tracker.sample_target(frame, box,
                tracker.cfg.TEST.SEARCH_FACTOR, output_sz=tracker.cfg.TEST.SEARCH_SIZE)
            cx, cy, width, height = (predicted.view(-1, 4).mean(0) *
                                     tracker.cfg.TEST.SEARCH_SIZE / factor).tolist()
            half = .5 * tracker.cfg.TEST.SEARCH_SIZE / factor
            expected_box = original_clip(
                [cx + box[0] + .5 * box[2] - half - .5 * width,
                 cy + box[1] + .5 * box[3] - half - .5 * height, width, height],
                frame.shape[0], frame.shape[1], margin=10)
            assert actual['target_bbox'] == expected_box
            assert actual['best_score'] == float(response.max())
            assert any(not torch.equal(prediction[key], prediction['round1'][key])
                       for key in ['score_map', 'size_map', 'offset_map'])
        assert tracker.frame_id == 1 and len(rounds) == 1
        assert tracker.templates[0] is old_template0
        assert len(tracker.templates) == len(tracker.availability) == 2
        assert tracker.templates.appends == tracker.templates.pops == int(allow_update)
        assert tracker.availability.appends == tracker.availability.pops == int(allow_update)
        if allow_update:
            assert tracker.templates[1] is not old_template1
            assert tracker.availability[1].tolist() == [[0., 1.]]
        else:
            assert tracker.templates[1] is old_template1
        update_cases.append(dict(allow_update=allow_update, network_calls=len(rounds),
            frame_id=tracker.frame_id, state_commits=clipped.call_count,
            template_appends=tracker.templates.appends,
            template_pops=tracker.templates.pops, round2_decode_exact=True))
    handle.remove()
    checks['single_state_and_conditional_template_commit'] = update_cases
    files = [Path(__file__), Path(__file__).with_name('tracker.py'),
             Path(__file__).with_name('model.py'), Path(__file__).with_name('check_structure.py'),
             Path('miss_rgbt/tracker.py'), Path('miss_rgbt/model.py')]
    report = dict(status='PASS', scope='Synthetic uint8 frames and actual E0 network; no dataset',
        seed=65302, checkpoint_sha256=expected_sha, checks=checks,
        source_sha256={str(path.resolve()): sha(path) for path in files},
        seconds=time.time()-started, peak_gpu_bytes=torch.cuda.max_memory_allocated(),
        limitations=['No real-image training or official metrics',
            'Synthetic initialized repair and residual, not learned F weights',
            'No image-training or spawned official evaluator execution',
            'No actual-workload latency comparison or correction/destruction rates'])
    output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
