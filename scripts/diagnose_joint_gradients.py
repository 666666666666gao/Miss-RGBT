"""Measure complete/auxiliary full-parameter gradients on fresh training batches."""
import argparse
import json
import os
from pathlib import Path
import time

import cv2
import numpy as np
import torch

from lib.config.cadtrack.config import cfg, update_config_from_file
from miss_rgbt.data import make_loader
from miss_rgbt.model import load_pretrained
from miss_rgbt.objectives import response_kd, task_loss
from miss_rgbt.train import fix_bn, sample_masks
import diagnose_absent_position as common
import diagnose_teacher_reliability as reliability


def save(path, value):
    with Path(path).open('x', newline='\n') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')


def summarize(rows):
    values = np.asarray([row['groups']['all']['cosine'] for row in rows])
    rng = np.random.default_rng(56301)
    draws = rng.integers(0, len(values), size=(10000, len(values)))
    low, high = np.quantile(values[draws].mean(1), [.025, .975])
    result = dict(mean=float(values.mean()), ci95_low=float(low), ci95_high=float(high),
        negative_batches=int((values < 0).sum()), batches=len(values))
    supported = result['mean'] <= -.05 and high < 0 and (values < 0).mean() >= .25
    return result, bool(supported)


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
    common.seed(56101)
    update_config_from_file('experiments/cadtrack/cadtrack.yaml')
    assert common.sha256(common.CHECKPOINT) == common.CHECKPOINT_SHA
    model = load_pretrained(cfg, common.CHECKPOINT, repair=False).cuda().train()
    fix_bn(model, True)
    before = common.digest(model.state_dict())
    names = Path('lib/train/data_specs/lasher_train.txt').read_text().splitlines()
    assert len(names) == len(set(names)) == 882
    fitting = [name for index, name in enumerate(names) if index % 5]
    excluded, prior_hashes = reliability.exclusions()
    teacher_path = reliability.chronology.PROJECT / 'reports/server28_teacher_reliability_20261102/report.json'
    prior = json.loads(teacher_path.read_text())
    excluded.update(reliability.identity_key(item['identity'])
                    for item in prior['cache_manifest']['input_manifest'])
    prior_hashes[teacher_path.parent.name] = common.sha256(teacher_path)
    loader = make_loader(cfg, '/data/wangwj/dataset/LasHeR/trainingset', 'train', 4, 64, 0)
    loader.dataset.datasets[0].sequence_list = fitting
    identities = common.capture_identities(loader)
    named = list(model.named_parameters())
    parameters = [value for _, value in named]
    rows, manifest = [], []
    masks_config = dict(complete_rgb_missing_tir_missing_probability=[.4, .3, .3],
                        patch_mask_probability=.1)
    for batch_index, data in enumerate(loader):
        templates = [value.cuda() for value in data['template_images']]
        searches = [value.cuda() for value in data['search_images']]
        boxes = data['search_anno'].cuda()
        assert len(searches) == 3 and len(templates) == 2
        batch_ids = identities[batch_index * 4:(batch_index + 1) * 4]
        assert len(batch_ids) == 4
        for index, identity in enumerate(batch_ids):
            assert reliability.identity_key(identity) not in excluded
            assert identity['sequence'] in fitting
            manifest.append(dict(identity=identity, tensors_sha256=common.digest(dict(
                templates=data['template_images'][:, index], searches=data['search_images'][:, index],
                boxes=data['search_anno'][:, index]))))
        common.seed(56200 + batch_index)
        available, patch = sample_masks(templates, searches, masks_config)
        mask_hash = common.digest({str(index): value for index, value in enumerate(available + patch)})
        complete = model(templates, searches)
        loss_c = task_loss(complete, boxes, cfg).mean()
        teacher = [{key: value.detach() for key, value in out.items() if key in common.MAPS}
                   for out in complete]
        complete_gradient = torch.autograd.grad(loss_c, parameters, allow_unused=True)
        complete_value = float(loss_c.detach())
        del complete, loss_c
        missing = model(templates, searches, available, patch)
        loss_m = task_loss(missing, boxes, cfg).mean()
        kd = response_kd(missing, teacher).mean()
        auxiliary_gradient = torch.autograd.grad(loss_m + .1 * kd, parameters, allow_unused=True)
        groups = {key: dict(complete_sq=0., auxiliary_sq=0., dot=0., parameters=0)
                  for key in ['all', 'backbone', 'head_fusion']}
        unused = []
        for (name, parameter), first, second in zip(named, complete_gradient, auxiliary_gradient):
            if first is None or second is None:
                assert first is second is None, name
                unused.append(name)
                continue
            assert torch.isfinite(first).all() and torch.isfinite(second).all()
            first, second = first.double(), second.double()
            c2, a2, dot = float(first.square().sum()), float(second.square().sum()), float((first * second).sum())
            for group in ['all', 'backbone' if 'backbone' in name else 'head_fusion']:
                groups[group]['complete_sq'] += c2
                groups[group]['auxiliary_sq'] += a2
                groups[group]['dot'] += dot
                groups[group]['parameters'] += parameter.numel()
        for group in groups.values():
            assert group['complete_sq'] > 0 and group['auxiliary_sq'] > 0
            group['cosine'] = group['dot'] / np.sqrt(group['complete_sq'] * group['auxiliary_sq'])
        row = dict(batch=batch_index, identities=batch_ids, complete_loss=complete_value,
            missing_loss=float(loss_m.detach()), kd=float(kd.detach()), groups=groups,
            unused_parameters=unused, mask_sha256=mask_hash,
            availability=[value.cpu().tolist() for value in available],
            patch_keep_counts=[value.sum((1, 2)).cpu().tolist() for value in patch])
        rows.append(row)
        print(json.dumps(dict(stage='batch', batch=batch_index + 1, expected=16,
                              groups=groups)), flush=True)
        del complete_gradient, auxiliary_gradient, missing, teacher, loss_m, kd
    assert len(rows) == 16 and len(manifest) == 64
    assert before == common.digest(model.state_dict())
    assert all(value.grad is None for value in parameters)
    effects, supported = summarize(rows)
    sources = [Path(__file__), Path(common.__file__), Path(reliability.__file__),
        Path('lib/train/data_specs/lasher_train.txt'), Path('experiments/cadtrack/cadtrack.yaml')]
    sources += sorted(Path('miss_rgbt').rglob('*.py')) + sorted(Path('lib/models').rglob('*.py'))
    report = dict(status='complete', pid=os.getpid(), seed=56101, checkpoint=common.CHECKPOINT,
        checkpoint_sha256=common.CHECKPOINT_SHA, fitting_names=fitting, input_manifest=manifest,
        rows=rows, effects=effects, diagnostic_supported=supported, prior_reports_sha256=prior_hashes,
        baseline_unchanged=True, baseline_gradients_absent=True, baseline_state_sha256=before,
        source_sha256={str(path): common.sha256(path) for path in sources},
        peak_gpu_bytes=torch.cuda.max_memory_allocated(), seconds=time.time() - started,
        scope='Train-mode complete versus missing+.1KD raw parameter gradients; no optimizer or official scores.')
    save(root / 'report.json', report)
    save(root / 'completed.json', dict(status='complete', report_sha256=common.sha256(root / 'report.json'),
        diagnostic_supported=supported, seconds=report['seconds']))
    print(json.dumps(dict(stage='complete', effects=effects, diagnostic_supported=supported,
                          seconds=report['seconds'])), flush=True)


if __name__ == '__main__':
    main()
