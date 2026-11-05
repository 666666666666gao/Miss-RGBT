"""Frozen train882 diagnostic of task versus complete-response KD gradients."""
import argparse
from collections import defaultdict
import copy
import json
import os
from pathlib import Path
import time

import cv2
import numpy as np
import torch
from torch.nn import functional as F

from lib.config.cadtrack.config import cfg, update_config_from_file
from miss_rgbt.model import load_pretrained
from miss_rgbt.objectives import response_kd, task_loss
from scripts import diagnose_joint_utility_sets as utilities
import diagnose_absent_position as common
import diagnose_template_chronology as chronology
import probe_fusion_adapter as fusion


def save(path, value):
    with Path(path).open('x', newline='\n') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')


def identity_key(value):
    return value['sequence'], tuple(value['template_frames']), tuple(value['search_frames'])


def exclusions():
    excluded = chronology.old_ids()
    hashes = {}
    for name in ['server28_template_chronology_20261102',
                 'server28_chronology_training_20261102', 'server28_window_training_20261102_v2']:
        path = chronology.PROJECT / 'reports' / name / 'report.json'
        report = json.loads(path.read_text())
        hashes[name] = common.sha256(path)
        manifests = (report['cache_manifests'].values() if 'cache_manifests' in report
                     else [dict(input_manifest=report['input_manifest'])])
        for manifest in manifests:
            excluded.update(identity_key(item['identity']) for item in manifest['input_manifest'])
    return excluded, hashes


def cosine(first, second):
    return F.cosine_similarity(first.double(), second.double(), dim=1, eps=1e-12)


def measure(head, cache):
    head.eval().requires_grad_(False)
    projection = head.conv5_ctr
    assert projection.weight.shape[0] == 1 and projection.weight.shape[2:] == (1, 1)
    projection.weight.requires_grad_(True)
    projection.bias.requires_grad_(True)
    complete_indices = {identity_key(value): index for index, (value, label) in enumerate(
        zip(cache['identities'], cache['labels'])) if label == '11_11'}
    captured = {}

    def capture(module, inputs, output):
        captured['input'], captured['logits'] = inputs[0], output
        # Native CENTER uses sigmoid_(); preserve its pre-sigmoid gradient witness.
        return output.clone()

    handle = projection.register_forward_hook(capture)
    rows = []
    for start in range(0, len(cache['labels']), 16):
        end = start + 16
        indices = [complete_indices[identity_key(value)] for value in cache['identities'][start:end]]
        target = cache['boxes'][start:end].cuda()
        with torch.no_grad():
            teacher = fusion.head_output(head, cache['features'][indices].cuda())
            teacher_task = task_loss([teacher], target[None], cfg)
        features = cache['features'][start:end].cuda().requires_grad_(True)
        student = fusion.head_output(head, features)
        task = task_loss([student], target[None], cfg)
        kd = response_kd([student], [teacher])
        arguments = (features, captured['logits'], projection.weight, projection.bias)
        task_grad = torch.autograd.grad(task.sum(), arguments, retain_graph=True)
        kd_grad = torch.autograd.grad(kd.sum(), arguments)
        vectors = []
        for gradients in [task_grad, kd_grad]:
            weights = (gradients[1] * captured['input'].detach()).sum((2, 3))
            bias = gradients[1].sum((2, 3))
            assert torch.allclose(weights.sum(0), gradients[2].flatten(), atol=2e-5, rtol=2e-5)
            assert torch.allclose(bias.sum(0), gradients[3], atol=2e-5, rtol=2e-5)
            vectors.append(torch.cat([weights, bias], 1).detach())
        fused_task = task_grad[0].flatten(1).double()
        fused_kd = kd_grad[0].flatten(1).double()
        fused_cos = cosine(fused_task, fused_kd)
        projected_cos = cosine(*vectors)
        for index in range(len(target)):
            position = start + index
            if cache['labels'][position] == '11_11':
                assert all(torch.equal(student[key][index], teacher[key][index]) for key in common.MAPS)
                assert abs(float(kd[index])) < 1e-6
            row = dict(identity=cache['identities'][position], cell=cache['labels'][position],
                task=float(task[index]), teacher_task=float(teacher_task[index]), kd=float(kd[index]),
                teacher_worse=bool(teacher_task[index] >= task[index]),
                fused_task_norm=float(fused_task[index].norm()),
                fused_kd_norm=float(fused_kd[index].norm()),
                fused_dot=float((fused_task[index] * fused_kd[index]).sum()),
                fused_cosine=float(fused_cos[index]), projection_cosine=float(projected_cos[index]),
                projection_task=vectors[0][index].cpu().tolist(),
                projection_kd=vectors[1][index].cpu().tolist())
            assert all(np.isfinite(row[key]) for key in ['task', 'teacher_task', 'kd',
                'fused_task_norm', 'fused_kd_norm', 'fused_dot', 'fused_cosine', 'projection_cosine'])
            rows.append(row)
        print(json.dumps(dict(stage='gradient', rows=len(rows), expected=len(cache['labels']))), flush=True)
    handle.remove()
    projection.weight.requires_grad_(False)
    projection.bias.requires_grad_(False)
    return rows


def statistics(rows):
    missing = [row for row in rows if not row['cell'].endswith('_11')]
    groups = dict(unreliable=[row for row in missing if row['teacher_worse']],
                  reliable=[row for row in missing if not row['teacher_worse']])
    effects, counts = {}, {}
    sequences = {}
    for group, values in groups.items():
        sequences[group] = defaultdict(list)
        for row in values:
            sequences[group][row['identity']['sequence']].append(row['fused_cosine'])
        counts[group] = dict(rows=len(values), sequences=len(sequences[group]))
        assert len(values) >= 32 and len(sequences[group]) >= 16
        for field in ['fused_cosine', 'projection_cosine']:
            effects[group + '_' + field] = common.clustered(values, field, 55301)
    paired = [dict(identity=dict(sequence=name),
        gain=float(np.mean(sequences['reliable'][name]) - np.mean(sequences['unreliable'][name])))
        for name in sorted(set(sequences['reliable']) & set(sequences['unreliable']))]
    assert len(paired) >= 16
    effects['reliable_minus_unreliable_fused'] = common.clustered(paired, 'gain', 55302)
    bad = effects['unreliable_fused_cosine']
    gap = effects['reliable_minus_unreliable_fused']
    supported = (bad['mean'] <= -.05 and bad['ci95_high'] < 0 and
                 gap['mean'] >= .1 and gap['ci95_low'] > 0 and
                 effects['unreliable_projection_cosine']['ci95_high'] < 0)
    return effects, counts, supported


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    root = Path(args.output)
    root.mkdir(exist_ok=False)
    started = time.time()
    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    update_config_from_file('experiments/cadtrack/cadtrack.yaml')
    sources = [Path(__file__), Path(common.__file__), Path(fusion.__file__),
        Path(utilities.__file__), Path(chronology.__file__),
        Path('lib/train/data_specs/lasher_train.txt'), Path('experiments/cadtrack/cadtrack.yaml')]
    sources += sorted(Path('miss_rgbt').rglob('*.py')) + sorted(Path('lib/models').rglob('*.py'))
    source_sha256 = {str(path): common.sha256(path) for path in sources}
    save(root / 'source_sha256.json', source_sha256)
    assert common.sha256(common.CHECKPOINT) == common.CHECKPOINT_SHA
    model = load_pretrained(cfg, common.CHECKPOINT, repair=False).cuda().eval().requires_grad_(False)
    original = common.digest(model.state_dict())
    names = Path('lib/train/data_specs/lasher_train.txt').read_text().splitlines()
    assert len(names) == len(set(names)) == 882
    fitting = [name for index, name in enumerate(names) if index % 5]
    excluded, prior_hashes = exclusions()
    cache, manifest = fusion.build_cache(model, fitting, 128, 55101, False, root / 'fitting')
    assert all(identity_key(item['identity']) not in excluded for item in manifest['input_manifest'])
    head = copy.deepcopy(model.baseline.box_head).cuda().eval().requires_grad_(False)
    before = common.digest(head.state_dict())
    rows = measure(head, cache)
    assert before == common.digest(head.state_dict())
    assert all(parameter.grad is None for parameter in head.parameters())
    assert original == common.digest(model.state_dict())
    assert all(parameter.grad is None for parameter in model.parameters())
    effects, counts, supported = statistics(rows)
    report = dict(status='complete', pid=os.getpid(), checkpoint=common.CHECKPOINT,
        checkpoint_sha256=common.CHECKPOINT_SHA, fitting_names=fitting, seed=55101,
        cache_manifest=manifest, rows=rows, effects=effects, group_counts=counts,
        diagnostic_supported=supported, source_sha256=source_sha256, prior_reports_sha256=prior_hashes,
        projection_gradient_sum_verified=True, copied_head_unchanged=True,
        baseline_unchanged=True, baseline_gradients_absent=True, baseline_state_sha256=original,
        seconds=time.time() - started, scope='Frozen train882 task/KD gradient diagnostic; no optimizer or official scores.')
    save(root / 'report.json', report)
    save(root / 'completed.json', dict(status='complete', diagnostic_supported=supported,
        report_sha256=common.sha256(root / 'report.json'), seconds=report['seconds']))
    print(json.dumps(dict(stage='complete', effects=effects, group_counts=counts,
                          diagnostic_supported=supported, seconds=report['seconds'])), flush=True)


if __name__ == '__main__':
    main()
