"""Audit saved OSTrack F runs after caller-owned parent and smoke gates."""
from itertools import zip_longest
import math
from pathlib import Path

import torch

from research.curriculum_p.protocol_verify import (
    check_checkpoint as check_native_checkpoint, check_history, check_pair,
    check_step as check_native_step, records,
)
from research.feedback_f.protocol_runner import check_config
from research.feedback_f.protocol_training import ARMS
from research.feedback_f.protocol_validation import compare_rounds
from research.protocol_r.common import MISSING, read, sha, tensor_sha


def check_step(config, index, row):
    """Use native scheduling checks without mistaking round logs for P logs."""
    mode = config['feedback']['mode']
    count = 1 if mode == 'single' else 2
    assert row['feedback_mode'] == mode and len(row['rounds']) == count
    normalized = config['curriculum_arm'] == 'paced+norm'
    fields = {'task_c', 'task_m', 'kd_forward', 'kd_reverse'}
    if not normalized:
        fields.add('reverse_fraction')
    for values in row['rounds']:
        assert set(values) == fields
        assert all(math.isfinite(value) for value in values.values())
        assert values['task_c'] >= 0 and values['task_m'] >= 0
        if not normalized:
            assert 0 <= values['reverse_fraction'] <= 1
    flat = dict(row)
    flat.update({key: sum(value[key] for value in row['rounds']) / count
                 for key in fields})
    if normalized:
        flat['reverse_fraction'] = 0.
        assert row['normalization_observations'] == count * 3 * config['microbatch']
        assert row['task_m_normalized'] >= 0
    check_native_step(config, index, flat)
    missing = row['task_m_normalized'] if normalized else flat['task_m']
    expected = flat['task_c'] + missing + config['forward_weight'] * flat['kd_forward']
    expected += config['reverse_weight'] * flat['kd_reverse']
    if 'utility' in row:
        assert row['utility']['utility_loss'] >= 0
        expected += config['utility_weight'] * row['utility']['utility_loss']
    # Logged terms were separately reduced in float32 before conversion to JSON.
    assert math.isclose(row['loss'], expected, rel_tol=2e-6, abs_tol=2e-6)


def check_diagnostic(config, summary, raw):
    assert summary == dict(
        {key: value for key, value in raw.items() if key != 'round_samples'},
        diagnostic_path=summary['diagnostic_path'],
        diagnostic_sha256=summary['diagnostic_sha256'])
    assert raw['samples'] == config['development_samples'] == 128
    labels = {'11', '01', '10'}
    assert set(raw['task_loss']) == labels
    if config['feedback']['mode'] == 'single':
        assert not {'round_samples', 'round_comparison', 'round_comparison_scope'}.intersection(raw)
        return
    assert set(raw['round_samples']) == set(raw['round_comparison']) == labels
    layouts = []
    for label in sorted(labels):
        first, final, losses, sizes = [], [], [], []
        for index, row in enumerate(raw['round_samples'][label]):
            assert row['batch_index'] == index
            assert set(row) == {'batch_index', 'round1_task_loss', 'round2_task_loss',
                                'round1_iou', 'round2_iou'}
            size = len(row['round1_task_loss'])
            assert size > 0 and len(row['round2_task_loss']) == size
            assert all(math.isfinite(value) and value >= 0 for key in
                       ('round1_task_loss', 'round2_task_loss') for value in row[key])
            sizes.append(size)
            losses.extend(row['round2_task_loss'])
            for key, target in (('round1_iou', first), ('round2_iou', final)):
                assert len(row[key]) == 3 and all(len(search) == size for search in row[key])
                target.extend(value for search in row[key] for value in search)
        assert sum(sizes) == 128 and len(first) == len(final) == 384
        layouts.append(sizes)
        assert raw['round_comparison'][label] == compare_rounds(first, final)
        assert math.isclose(sum(losses) / 128., raw['task_loss'][label],
                            rel_tol=2e-6, abs_tol=2e-6)
    assert layouts[0] == layouts[1] == layouts[2]


def check_feedback(saved, config, initial):
    mode = config['feedback']['mode']
    parameters = {name: value for name, value in saved['model'].items()
                  if name.startswith('feedback.')}
    assert set(parameters) == set(initial)
    if mode in ('single', 'none'):
        assert not initial
        return []
    assert initial
    for name, value in parameters.items():
        assert list(value.shape) == initial[name]['shape']
    changed = [name for name, value in parameters.items()
               if tensor_sha(value) != initial[name]['sha256']]
    assert any(name in changed for name in ('feedback.delta.weight', 'feedback.delta.bias'))
    return changed


def verify_run(config, verify_evidence, verify_smokes):
    """Library gate: callers must supply actual F parent/source/smoke checks."""
    check_config(config)
    verify_evidence(config)
    verify_smokes(config)
    root = Path(config['output'])
    assert read(root / 'run_config.json') == config
    assert read(root / 'source_sha256.json') == read(config['source_manifest'])
    names_path = Path(config['snapshot']) / 'lib/train/data_specs/lasher_train.txt'
    names = names_path.read_text().splitlines()
    assert len(names) == len(set(names)) == 882
    assert sha(names_path) == config['train_names_sha256']
    assert read(root / 'data_boundary.json') == dict(
        split='train882', train_names=names, development_split='val97',
        development_samples=128, accepted_tuple_identities=True,
        rejected_crop_reads_recorded=False, train_names_sha256=config['train_names_sha256'])
    done = read(root / 'completed.json')
    history = read(root / 'development_history.json')
    selected = check_history(config, history, done)
    assert sha(root / 'input_manifest.jsonl') == done['input_manifest_sha256']
    evidence = {str(root / name): sha(root / name) for name in (
        'run_config.json', 'source_sha256.json', 'data_boundary.json', 'completed.json',
        'initial_state.json', 'feedback_initial_state.json', 'development_history.json',
        'input_manifest.jsonl', 'steps.jsonl')}
    assert sorted(path.name for path in (root / 'round_diagnostics').glob('*.json')) == [
        'epoch%02d.json' % epoch for epoch in range(1, 21)]
    for row in history:
        path = root / 'round_diagnostics' / ('epoch%02d.json' % row['epoch'])
        assert Path(row['diagnostic_path']) == path
        assert sha(path) == row['diagnostic_sha256']
        check_diagnostic(config, row, read(path))
        evidence[str(path)] = sha(path)
    batches = config['samples_per_epoch'] // config['microbatch']
    names, total, last_steps = set(names), 0, {}
    for index, (identity, step) in enumerate(zip_longest(
            records(root / 'input_manifest.jsonl'), records(root / 'steps.jsonl'))):
        assert identity is not None and step is not None
        check_step(config, index, step)
        assert identity['epoch'] == index // batches + 1
        assert identity['index'] == index % batches
        assert identity['tensors_sha256'] == step['input_sha256']
        assert set(step['input_sha256']) == {'template_images', 'search_images', 'search_anno'}
        assert len(identity['identities']) == config['microbatch']
        for item in identity['identities']:
            assert item['sequence'] in names
            assert len(item['template_frames']) == 2 and len(item['search_frames']) == 3
            assert all(isinstance(frame, int) and frame >= 0
                       for frame in item['template_frames'] + item['search_frames'])
        if (index + 1) % batches == 0:
            last_steps[step['epoch']] = step
        total += 1
    assert total == 20 * batches
    initial = read(root / 'initial_state.json')
    feedback = read(root / 'feedback_initial_state.json')
    assert initial['missing_keys'] == MISSING and not initial['unexpected_keys']
    assert initial['initializer_sha256'] == config['initializer_sha256']
    smoke = Path(config['output'] + '_smoke')
    assert initial == read(smoke / 'initial_state.json')
    assert feedback == read(smoke / 'feedback_initial_state.json')
    assert sorted(path.name for path in root.glob('epoch*.pth.tar')) == [
        'epoch%02d.pth.tar' % epoch for epoch in config['checkpoint_epochs']]
    for row in done['final_five']:
        assert Path(row['path']) == root / ('epoch%02d.pth.tar' % row['epoch'])
        assert sha(row['path']) == row['sha256']
    changed, weights = {}, {}
    for epoch in config['checkpoint_epochs']:
        path = root / ('epoch%02d.pth.tar' % epoch)
        saved = torch.load(str(path), map_location='cpu')
        changed[str(epoch)] = dict(native=check_native_checkpoint(
            saved, config, initial, epoch, last_steps[epoch]),
            feedback=check_feedback(saved, config, feedback))
        weights[str(path)] = sha(path)
        del saved
    verify_evidence(config)
    verify_smokes(config)
    return dict(status='passed', feedback_mode=config['feedback']['mode'],
        selected_epoch=selected['epoch'], completed_sha256=sha(root / 'completed.json'),
        evidence_sha256=evidence, checkpoints_sha256=weights,
        changed_parameters=changed, microsteps=total, verifier_sha256=sha(__file__),
        official_evaluations_required=10, resume_supported=False,
        scope='Saved records and weights only; no independent image replay or official scores')


def verify_pairs(configs):
    assert [config['feedback']['mode'] for config in configs] == list(ARMS)
    reference = {key: value for key, value in configs[0].items()
                 if key not in ('feedback', 'output')}
    assert all({key: value for key, value in config.items()
                if key not in ('feedback', 'output')} == reference for config in configs)
    roots = [Path(config['output']) for config in configs]
    assert len(set(roots)) == len(ARMS)
    for name in ('initial_state.json', 'data_boundary.json', 'input_manifest.jsonl'):
        assert len({sha(root / name) for root in roots}) == 1, name
    assert read(roots[0] / 'feedback_initial_state.json') == {}
    assert read(roots[1] / 'feedback_initial_state.json') == {}
    feedback_initial = [read(root / 'feedback_initial_state.json') for root in roots[2:]]
    assert all(value == feedback_initial[0] for value in feedback_initial[1:])
    total = 0
    for index, rows in enumerate(zip_longest(
            *(records(root / 'steps.jsonl') for root in roots))):
        assert all(row is not None for row in rows)
        for config, row in zip(configs, rows):
            check_step(config, index, row)
        for left_index, left in enumerate(rows):
            for right in rows[left_index + 1:]:
                check_pair(left, right)
        total += 1
    assert total == 20 * reference['samples_per_epoch'] // reference['microbatch']
    return dict(microsteps=total, native_initialization_and_inputs_exact=True,
        feedback_initialization_matched=True, patch_masks_exact=True,
        equal_probability_control_masks_exact=True,
        paced_availability_masks_may_differ=True)
