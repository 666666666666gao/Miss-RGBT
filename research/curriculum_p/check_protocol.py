"""CPU contracts for the isolated trainer; no image, GPU or launch evidence."""
import argparse
import io
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from research.curriculum_p.engine import ARMS
from research.curriculum_p import protocol_prepare as prepare
from research.curriculum_p import protocol_train as train
from research.protocol_r.common import read, sha


def check_configs():
    parent = read(prepare.REPORT / 'r1.json')
    configs = [prepare.make_config(arm, parent) for arm in ARMS]
    for config in configs:
        train.check_config(config)
    reference = {k: v for k, v in configs[0].items()
                 if k not in ('curriculum_arm', 'output')}
    assert all({k: v for k, v in config.items()
                if k not in ('curriculum_arm', 'output')} == reference for config in configs)
    rejected = []
    for key, value in [('epochs', 14), ('freeze_all_bn', False),
                       ('early_stop', True), ('init', 'e0'),
                       ('official_test_selection_used', True),
                       ('checkpoint_epochs', [16, 17, 18, 19]),
                       ('resume_supported', True)]:
        invalid = dict(configs[0], **{key: value})
        with __import__('unittest').TestCase().assertRaises(AssertionError):
            train.check_config(invalid)
        rejected.append(key)
    return dict(six_arm_shared_config=True, rejected_mutations=rejected,
                all_bn_frozen_utility_policy=True, checkpoint_epochs=[10, 16, 17, 18, 19, 20])


def check_provenance():
    names = ['sequence_%04d' % i for i in range(882)]
    frames = []

    def get_frames(index, ids, anno=None):
        frames.append((index, list(ids)))

    dataset = SimpleNamespace(sequence_list=names, get_frames=get_frames)

    def getitem():
        # The first pair represents a rejected native crop; only the last is accepted.
        dataset.get_frames(1, [2, 1])
        dataset.get_frames(1, [3, 4, 5])
        dataset.get_frames(7, [20, 19])
        dataset.get_frames(7, [21, 22, 23])
        return dict(template_images=torch.rand(2, 1, 6, 2, 2),
                    search_images=torch.rand(3, 1, 6, 2, 2),
                    search_anno=torch.rand(3, 1, 4))

    sampled = SimpleNamespace(datasets=[dataset], getitem=getitem)
    loader = SimpleNamespace(dataset=sampled)
    torch.manual_seed(661103)
    expected = getitem()
    expected_rng = torch.get_rng_state().clone()
    frames.clear()
    torch.manual_seed(661103)
    with patch.object(train, 'make_loader', return_value=loader):
        instrumented, actual_names = train.provenance_loader(dict(
            dataset='synthetic', microbatch=1, samples_per_epoch=20000, workers=2))
        actual = instrumented.dataset.getitem()
    assert actual_names == names
    assert actual['frame_identity'].tolist() == [7, 20, 19, 21, 22, 23]
    assert torch.equal(torch.get_rng_state(), expected_rng)
    assert all(torch.equal(actual[key], expected[key]) for key in expected)
    actual['frame_identity'] = actual['frame_identity'].reshape(6, 1)
    stream = io.StringIO()
    record = train.record_batch(actual, names, 1, 0, stream)
    assert json.loads(stream.getvalue()) == record
    assert record['identities'] == [dict(sequence=names[7], template_frames=[20, 19],
                                        search_frames=[21, 22, 23])]
    return dict(accepted_native_tuple_identity=True, rejected_candidate_not_misattributed=True,
                tensor_and_rng_bitwise_unchanged=True, jsonl_record_read_back=True,
                scope='Synthetic sampler only; real native loader parity remains pending')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    configs = check_configs()
    provenance = check_provenance()
    with patch.object(prepare.Path, 'is_file', return_value=False):
        gate = prepare.protocol_gate()
    assert not gate['ready'] and len(gate['missing']) == 4
    actual_gate = prepare.protocol_gate()
    assert not actual_gate['ready'], 'This check expects the current unfinished R campaign'
    assert not torch.cuda.is_initialized()
    report = dict(status='PASS', scope='CPU contracts only; not real-image or full-run evidence',
        configs=configs, provenance=provenance, missing_parent_gate_rejected=True,
        actual_parent_gate=actual_gate, cuda_initialized=False,
        remaining=['Actual R full protocol gate', 'Real six-arm smoke and data parity',
                   'Formal saved-run verifier', 'Full official controller and sixty evaluations'],
        source_sha256={str(path.resolve()): sha(path) for path in
                       [Path(prepare.__file__), Path(train.__file__), Path(__file__)]})
    with Path(args.output).open('x') as stream:
        json.dump(report, stream, indent=2)
        stream.write('\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
