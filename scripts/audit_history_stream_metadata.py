"""Metadata-only history feasibility; never freeze active training manifests."""
import argparse
import json
from pathlib import Path
import time

from lib.train.dataset import LasHeR
import audit_partitioned_fresh_frames as previous
from research.history_stream.metadata import history_candidates

base = previous.base


def add_completed_replay(root, used, hashes):
    root = Path(root)
    completed_path, report_path = root / 'completed.json', root / 'report.json'
    completed = json.loads(completed_path.read_text())
    assert completed['status'] == 'complete'
    assert base.prior.common.sha256(report_path) == completed['report_sha256']
    report = json.loads(report_path.read_text())
    assert report['status'] == 'passed' and report['cuda_initialized'] is False
    assert report['first_update_hashes_matched'] is True
    assert report['epoch2_prefix_bitwise_and_identity_equal'] is True
    assert report['all_historical_frame_exclusions_verified'] is False
    for path, expected in report['source_sha256'].items():
        assert base.prior.common.sha256(path) == expected, path
        hashes[path] = expected
    for arm, expected in report['run_config_sha256'].items():
        path = base.PROJECT / 'runs/server28_protocol_r_20261103' / arm / 'run_config.json'
        assert base.prior.common.sha256(path) == expected
        hashes[str(path)] = expected
    manifest = root / 'epoch01_input_manifest.jsonl'
    assert base.prior.common.sha256(manifest) == report['epoch1_manifest_sha256']
    count, tuples = 0, 0
    with manifest.open() as stream:
        for count, line in enumerate(stream, 1):
            row = json.loads(line)
            assert row['epoch'] == 1 and row['index'] == count - 1
            assert len(row['identities']) == 4
            for identity in row['identities']:
                used[identity['sequence']].update(
                    identity['template_frames'] + identity['search_frames'])
                tuples += 1
    assert count == report['exhausted_native_epoch_microbatches'] == 5000
    assert tuples == report['returned_epoch1_tuples'] == 20000
    assert len(report['epoch2_prefix']) == report['epoch2_prefix_microbatches'] == 4
    for row in report['epoch2_prefix']:
        for identity in row['identities']:
            used[identity['sequence']].update(
                identity['template_frames'] + identity['search_frames'])
    for path in (completed_path, report_path, manifest):
        hashes[str(path)] = base.prior.common.sha256(path)
    return dict(completed_replay=str(root), epoch1_tuples=tuples,
                epoch2_prefix_tuples=16, full_original_r_provenance_verified=False)


def add_completed_epoch(receipt_path, used, hashes):
    """Apply one closed conditional replay manifest, never an active epoch."""
    receipt_path = Path(receipt_path).resolve()
    root = receipt_path.parent
    registration_path = root / 'registration.json'
    registration = json.loads(registration_path.read_text())
    receipt = json.loads(receipt_path.read_text())
    epoch = receipt['epoch']
    assert epoch in registration['epochs'] and 2 <= epoch <= 20
    assert receipt_path.name == 'epoch%02d_completed.json' % epoch
    assert registration['cpu_only'] is True
    assert registration['actual_original_inputs_independently_certified'] is False
    assert registration['ohcas_data_gate_passed'] is False
    witness = Path(registration['witness'])
    assert base.prior.common.sha256(witness / 'report.json') == (
        registration['witness_report_sha256'])
    witness_done = json.loads((witness / 'completed.json').read_text())
    assert witness_done['status'] == 'complete'
    assert witness_done['report_sha256'] == registration['witness_report_sha256']
    for name, expected in registration['source_sha256'].items():
        assert base.prior.common.sha256(name) == expected, name
        hashes[name] = expected
    for arm, expected in registration['run_config_sha256'].items():
        path = base.PROJECT / 'runs/server28_protocol_r_20261103' / arm / 'run_config.json'
        assert base.prior.common.sha256(path) == expected
        hashes[str(path)] = expected
    manifest = root / ('epoch%02d_input_manifest.jsonl' % epoch)
    assert Path(receipt['manifest']).resolve() == manifest
    assert base.prior.common.sha256(manifest) == receipt['manifest_sha256']
    count, tuples = 0, 0
    with manifest.open() as stream:
        for count, line in enumerate(stream, 1):
            row = json.loads(line)
            assert row['epoch'] == epoch and row['index'] == count - 1
            assert len(row['identities']) == 4
            for identity in row['identities']:
                used[identity['sequence']].update(
                    identity['template_frames'] + identity['search_frames'])
                tuples += 1
    assert count == receipt['microbatches'] == registration['microbatches_per_epoch'] == 5000
    assert tuples == receipt['tuples'] == 20000
    assert base.prior.common.sha256(manifest) == receipt['manifest_sha256']
    for path in (receipt_path, registration_path, manifest,
                 witness / 'report.json', witness / 'completed.json'):
        hashes[str(path)] = base.prior.common.sha256(path)
    return dict(epoch=epoch, tuples=tuples, receipt=str(receipt_path),
                manifest_sha256=receipt['manifest_sha256'],
                full_original_r_provenance_verified=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--completed-r-replay')
    parser.add_argument('--completed-r-epoch', action='append', default=[],
                        help='Closed per-epoch receipt; repeat for completed epochs only')
    args = parser.parse_args()
    started = time.time()
    output = Path(args.output)
    assert not output.exists()
    names_path = Path('lib/train/data_specs/lasher_train.txt')
    names = names_path.read_text().splitlines()
    assert len(names) == len(set(names)) == 882
    fitting = [name for index, name in enumerate(names) if index % 5]
    used, hashes = base.prior.exclusions()
    dense = base.PROJECT / 'reports/server28_dense_geometry_20261102/report.json'
    hashes[str(dense)] = base.prior.common.sha256(dense)
    for manifest in json.loads(dense.read_text())['cache_manifests'].values():
        for entry in manifest['input_manifest']:
            identity = entry['identity']
            used[identity['sequence']].update(
                identity['template_frames'] + identity['search_frames'])
    replay = (add_completed_replay(args.completed_r_replay, used, hashes)
              if args.completed_r_replay else None)
    replay_epochs = [add_completed_epoch(path, used, hashes)
                     for path in args.completed_r_epoch]
    assert len({row['epoch'] for row in replay_epochs}) == len(replay_epochs)
    verification = base.PROJECT / 'reports/server28_partitioned_a_20261102_formal/verification_v2.json'
    blockers, inventory = [], []
    if verification.exists():
        previous.add_verified_partitioned_a_exclusions(verification, used, hashes)
    else:
        blockers.append('Partitioned A formal verification absent; growing manifests not read.')
    for arm in ('utility', 'random'):
        root = base.PROJECT / 'runs/server28_partitioned_a_20261102_formal' / arm
        inventory.append(dict(campaign='partitioned_a', arm=arm,
            completed_exists=(root / 'completed.json').exists(),
            frame_manifest_exists=(root / 'input_manifest.jsonl').exists(),
            exclusions_applied=verification.exists()))
    for arm in ('r1', 'r2'):
        root = base.PROJECT / 'runs/server28_protocol_r_20261103' / arm
        manifest = root / 'input_manifest.jsonl'
        inventory.append(dict(campaign='protocol_r', arm=arm,
            completed_exists=(root / 'completed.json').exists(),
            frame_manifest_exists=manifest.exists(), exclusions_applied=False))
        blockers.append('Protocol %s full original frame history unresolved; '
                        'recorded tensor hashes and partial replay are insufficient.' % arm)
    config_path = base.PROJECT / 'reports/server28_partitioned_a_20261102_formal/utility.json'
    config = json.loads(config_path.read_text())
    dataset = LasHeR(root=config['dataset'], split='train', dtype='rgbrgb')
    dataset.sequence_list = fitting
    details = []
    for index, name in enumerate(fitting):
        info = dataset.get_sequence_info(index)
        visible = (info['visible'].bool() & info['valid'].bool()).tolist()
        result = history_candidates(visible, used[name])
        if result['first_candidate'] is not None:
            result['first_candidate']['sequence'] = name
        result.update(sequence=name, frames=len(visible),
                      excluded_frames=len(used[name]))
        details.append(result)
    sources = [Path(__file__), Path(previous.__file__),
        base.PROJECT / 'research/history_stream/metadata.py', names_path, config_path,
        Path('lib/train/dataset/lasher.py'),
        base.PROJECT / 'runtime/protocol_r_20261103/research/protocol_r/train.py']
    eligible = sum(row['eligible_searches'] > 0 for row in details)
    report = dict(status='metadata_complete', data_gate_passed=False,
        all_historical_frame_exclusions_verified=False,
        evaluated_at_unix=time.time(), seconds=time.time() - started,
        split='train882_fitting', sequences=len(fitting), history_max_age=64,
        eligible_sequences=eligible, required_samples=128,
        enough_distinct_sequences_under_partial_exclusions=eligible >= 128,
        eligible_searches=sum(row['eligible_searches'] for row in details),
        history_search_pairs=sum(row['history_search_pairs'] for row in details),
        exclusion_sources_sha256=hashes, unresolved_exclusions=blockers,
        campaign_inventory=inventory, sequence_details=details,
        completed_r_replay_exclusions=replay,
        completed_r_epoch_exclusions=replay_epochs,
        source_sha256={str(path.resolve()): base.prior.common.sha256(path) for path in sources},
        images_opened=0, gpu_allocations=0, cache_created=False,
        scope='Upper bound under completed historical diagnostic/cache exclusions '
              'and optional completed conditional R replay epochs. '
              'No growing manifests read; no historical-union or real-image acceptance claim. '
              'Candidate identities are examples, not a reserved or approved cache.')
    base.save(output, report)
    print(json.dumps({key: value for key, value in report.items()
                      if key not in ('sequence_details', 'exclusion_sources_sha256',
                                     'source_sha256')}), flush=True)


if __name__ == '__main__':
    main()
