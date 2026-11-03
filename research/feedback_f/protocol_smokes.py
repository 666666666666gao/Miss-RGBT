"""Verify all four saved F smokes before caller-owned formal dispatch."""
from itertools import zip_longest
import math
from pathlib import Path

import torch

from research.feedback_f.protocol_runner import check_config
from research.feedback_f.protocol_training import ARMS
from research.feedback_f.protocol_verify import check_feedback
from research.curriculum_p.protocol_verify import records
from research.protocol_r.common import read, sha


def verify_smokes(config, configs, verify_evidence):
    """Require matched four-update receipts; parent/M1 checks remain mandatory."""
    assert [candidate['feedback']['mode'] for candidate in configs] == list(ARMS)
    assert config in configs
    reference = {key: value for key, value in config.items()
                 if key not in ('feedback', 'output')}
    assert len({candidate['output'] for candidate in configs}) == 4
    evidence, initializations, inputs, feedback_initial = {}, [], [], {}
    for candidate in configs:
        check_config(candidate)
        verify_evidence(candidate)
        assert {key: value for key, value in candidate.items()
                if key not in ('feedback', 'output')} == reference
        mode = candidate['feedback']['mode']
        root = Path(candidate['output'] + '_smoke')
        receipt = read(root / 'smoke_verified.json')
        assert read(root / 'run_config.json') == candidate
        assert receipt['status'] == 'passed' and receipt['updates'] == 4
        assert receipt['fresh_cpu_strict_load'] is True
        assert receipt['all_rounds_reload_bitwise_equal'] is True
        weight = root / 'smoke.pth.tar'
        assert sha(weight) == receipt['checkpoint_sha256']
        fingerprint = sha(root / 'input_manifest.jsonl')
        assert fingerprint == receipt['input_manifest_sha256']
        count, updates = 0, []
        for index, (identity, step) in enumerate(zip_longest(
                records(root / 'input_manifest.jsonl'), records(root / 'steps.jsonl'))):
            assert identity is not None and step is not None
            assert identity['epoch'] == step['epoch'] == 1
            assert identity['index'] == index and step['microstep'] == index + 1
            assert identity['tensors_sha256'] == step['input_sha256']
            assert math.isfinite(step['loss'])
            if (index + 1) % candidate['accumulation'] == 0:
                assert step['update'] == (index + 1) // candidate['accumulation']
                updates.append(step['update'])
            else:
                assert 'update' not in step
            count += 1
        assert count == 4 * candidate['accumulation'] and updates == [1, 2, 3, 4]
        inputs.append(fingerprint)
        initializations.append(read(root / 'initial_state.json'))
        initial = read(root / 'feedback_initial_state.json')
        feedback_initial[mode] = initial
        saved = torch.load(str(weight), map_location='cpu')
        assert saved['config'] == candidate and saved['updates'] == 4
        assert saved['epoch'] == 1 and saved['resume_supported'] is False
        batches = candidate['samples_per_epoch'] // candidate['microbatch']
        assert saved['loader_position'] == dict(epoch=1,
            consumed_batches=4 * candidate['accumulation'], batches_per_epoch=batches)
        assert saved['curriculum']['arm'] == candidate['curriculum_arm']
        assert saved['curriculum']['microsteps'] == 4 * candidate['accumulation']
        assert saved['model'] and all(torch.isfinite(value).all()
                                      for value in saved['model'].values())
        check_feedback(saved, candidate, initial)
        del saved
        for name in ('run_config.json', 'smoke_verified.json', 'smoke.pth.tar',
                     'input_manifest.jsonl', 'initial_state.json',
                     'feedback_initial_state.json', 'steps.jsonl'):
            evidence[str(root / name)] = sha(root / name)
    assert len(set(inputs)) == 1
    assert all(value == initializations[0] for value in initializations)
    assert feedback_initial['single'] == feedback_initial['none'] == {}
    assert feedback_initial['shallow'] == feedback_initial['deep']
    return dict(status='passed', arms=list(ARMS), updates_per_arm=4,
        native_initialization_and_inputs_exact=True, feedback_initialization_matched=True,
        evidence_sha256=evidence,
        scope='Saved four-arm smoke receipts only; no full training or official metrics')
