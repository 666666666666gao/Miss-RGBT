"""Resume the prospectively amended audit; never repeat fitting."""
import json
import os
from pathlib import Path
import time

import cv2
import torch
from lib.config.cadtrack.config import cfg, update_config_from_file
from lib.train.dataset import LasHeR
from miss_rgbt.tracker import MissingTracker
import train_causal_crop_exposure as training


def main():
    started = time.time()
    root = training.PROJECT / 'reports/server28_causal_crop_training_20261102'
    receipt = json.loads((root / 'audit_resume.launch.json').read_text())
    assert receipt['pid'] == os.getpid()
    assert training.common.sha256(__file__) == receipt['script_sha256']
    assert training.common.sha256(root / 'audit_amendment.md') == receipt['protocol_sha256']
    inventory = json.loads((root / 'audit_inventory_v2.json').read_text())
    assert inventory['eligible_sequences'] == {'96': 12, '64': 24, '48': 29}
    assert training.common.sha256(root / 'audit_inventory_v2.json') == receipt['inventory_sha256']
    assert not any((root / name).exists() for name in ['report.json', 'audit_clips.json'])
    sources = json.loads((root / 'source_sha256.json').read_text())
    for path, value in sources.items():
        assert training.common.sha256(path) == value, path
    original_launch = json.loads((root.parent / (root.name + '.launch.json')).read_text())
    assert sources[original_launch['command'][2]] == original_launch['script_sha256']
    arms = {mode: json.loads((root / mode / 'selection.json').read_text())
            for mode in ['native', 'causal']}
    assert arms['native']['selected']['epoch'] == 11
    assert arms['causal']['selected']['epoch'] == 6
    for mode, arm in arms.items():
        assert training.common.sha256(root / mode / 'selection.json') == receipt['selection_sha256'][mode]
        assert training.common.sha256(arm['selected_path']) == arm['selected_sha256']
    assert arms['native']['initial_sha256'] == arms['causal']['initial_sha256']
    training.common.seed(58103)
    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    update_config_from_file('experiments/cadtrack/cadtrack.yaml')
    names = Path('lib/train/data_specs/lasher_train.txt').read_text().splitlines()
    partitions = dict(train=[name for index, name in enumerate(names) if index % 5],
                      development=names[::5][:88], audit=names[::5][88:])
    used, prior_hashes = training.causal.excluded_frames()
    previous = training.PROJECT / 'reports/server28_causal_crop_exposure_20261102/report.json'
    prior = json.loads(previous.read_text())
    prior_hashes[previous.parent.name] = training.common.sha256(previous)
    for clip in prior['clips']:
        used[clip['sequence']].update(clip['frame_ids'])
    dataset = LasHeR(root=training.burst.DATASET, split='train')
    clips = training.fresh_clips(dataset, partitions['audit'], 24, 64, 58103, used)
    assert {clip['sequence'] for clip in clips} == {
        row['sequence'] for row in inventory['rows'] if row['eligible_starts']['64'] > 0}
    training.save(root / 'audit_clips.json', clips)
    audits = {}
    for mode in ['e0', 'native', 'causal']:
        path = str(training.causal.WRAPPER) if mode == 'e0' else arms[mode]['selected_path']
        tracker = MissingTracker(cfg, path)
        tracker.network.eval().requires_grad_(False)
        before = training.common.digest(tracker.network.state_dict())
        audits[mode] = training.rollout(tracker, dataset, clips, mode, True)
        assert training.common.digest(tracker.network.state_dict()) == before
        assert all(value.grad is None for value in tracker.network.parameters())
        training.save(root / ('audit_' + mode + '.json'), audits[mode])
        print(json.dumps(dict(stage='audit', mode=mode, seconds=time.time() - started)), flush=True)
        del tracker
        torch.cuda.empty_cache()
    effects, original_gate = training.statistics(audits)
    assert original_gate is False
    advance = (all(effects[state]['mean'] >= .01 and effects[state]['ci95_low'] > 0
                   and effects[state]['sequences'] == 24 for state in ['01', '10'])
        and effects['causal_minus_native']['ci95_low'] > 0
        and effects['causal_minus_native']['sequences'] == 24
        and effects['complete']['mean'] >= 0 and effects['complete']['ci95_low'] > -.01
        and effects['complete']['sequences'] == 24)
    manifest = json.loads((root / 'image_manifest.json').read_text())
    initial = json.loads((root / 'initial_clips.json').read_text())
    report = dict(status='complete', pid=original_launch['pid'],
        checkpoint=training.common.CHECKPOINT, checkpoint_sha256=training.common.CHECKPOINT_SHA,
        original_state_sha256=arms['native']['initial_sha256'], partitions=partitions,
        training_manifest=manifest, development_clips=initial['development'],
        audit_clips=clips, batch_sizing=json.loads((root / 'batch_sizing.json').read_text()),
        arms=arms, audits=audits, effects=effects, advance=bool(advance),
        prior_reports_sha256=prior_hashes, source_sha256=sources,
        seconds=time.time() - started, seconds_scope='resumed audit only',
        audit_amendment=receipt, original_protocol_completed=False,
        scope='Prospectively amended24x64 train882 internal audit; not official metrics.')
    training.save(root / 'report.json', report)
    training.save(root / 'completed.json', dict(status='complete', advance=bool(advance),
        report_sha256=training.common.sha256(root / 'report.json'), seconds=report['seconds']))
    print(json.dumps(dict(stage='complete', effects=effects, advance=bool(advance),
                          seconds=report['seconds'])), flush=True)


if __name__ == '__main__':
    main()
