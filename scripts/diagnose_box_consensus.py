"""Fixed top-eight box averaging on fresh fitting crops; no learned selector."""
import argparse
import json
import os
from pathlib import Path
import time

import cv2
import numpy as np
import torch
from lib.config.cadtrack.config import cfg, update_config_from_file
from lib.test.utils.hann import hann2d
from miss_rgbt.data import make_loader
from miss_rgbt.model import load_pretrained
import diagnose_template_attention as prior

common, candidates, history = prior.common, prior.candidates, prior.history
PROJECT = prior.PROJECT


def exclusions():
    used, hashes = prior.exclusions()
    path = PROJECT / 'reports/server28_template_attention_20261102/report.json'
    hashes[path.parent.name + '/report.json'] = common.sha256(path)
    for item in json.loads(path.read_text())['input_manifest']:
        identity = item['identity']
        used[identity['sequence']].update(identity['template_frames'] + identity['search_frames'])
    return used, hashes


def summarize(rows):
    groups = dict(complete=[r for r in rows if r['cell'] == '11_11'],
        recovery01=[r for r in rows if r['cell'] == '01_11'],
        recovery10=[r for r in rows if r['cell'] == '10_11'],
        rgb_missing=[r for r in rows if r['cell'].endswith('_01')],
        tir_missing=[r for r in rows if r['cell'].endswith('_10')],
        missing=[r for r in rows if not r['cell'].endswith('_11')])
    effects = {name: {key: common.clustered(values, key, 61301)
        for key in ['weighted_gain', 'uniform_gain', 'weighted_minus_uniform']}
        for name, values in groups.items()}
    advance = (all(effects[name]['weighted_gain']['mean'] >= .01
        and effects[name]['weighted_gain']['ci95_low'] > 0
        and effects[name]['weighted_gain']['sequences'] >= 64
        for name in ['rgb_missing', 'tir_missing'])
        and effects['missing']['weighted_minus_uniform']['ci95_low'] > 0
        and effects['missing']['weighted_minus_uniform']['sequences'] >= 64
        and all(effects[name]['weighted_gain']['mean'] >= 0
            and effects[name]['weighted_gain']['ci95_low'] > -.005
            and effects[name]['weighted_gain']['sequences'] >= 64
            for name in ['complete', 'recovery01', 'recovery10']))
    return effects, bool(advance)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    root = Path(args.output)
    root.mkdir(exist_ok=False)
    started = time.time()
    common.seed(61101)
    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    update_config_from_file('experiments/cadtrack/cadtrack.yaml')
    assert common.sha256(common.CHECKPOINT) == common.CHECKPOINT_SHA
    used, previous_hashes = exclusions()
    model = load_pretrained(cfg, common.CHECKPOINT, repair=False).cuda().eval().requires_grad_(False)
    before = common.digest(model.state_dict())
    names_path = Path('lib/train/data_specs/lasher_train.txt')
    names = names_path.read_text().splitlines()
    assert len(names) == len(set(names)) == 882
    fitting = [name for index, name in enumerate(names) if index % 5]
    loader = make_loader(cfg, history.training.burst.DATASET, 'train', 16, 128, 0)
    dataset = loader.dataset.datasets[0]
    dataset.sequence_list = fitting
    original_info = dataset.get_sequence_info

    def excluded_info(index):
        info = dict(original_info(index))
        info['visible'] = info['visible'].clone()
        info['visible'][sorted(used[dataset.sequence_list[index]])] = False
        return info

    dataset.get_sequence_info = excluded_info
    identities = common.capture_identities(loader)
    window = hann2d(torch.tensor([16, 16]), centered=True).cuda()
    rows, manifests, count = [], [], 0
    maps = {key: [] for key in ['score_map', 'size_map', 'offset_map', 'target']}
    with torch.no_grad():
        for data in loader:
            templates = [value.cuda() for value in data['template_images']]
            search = data['search_images'][0].cuda()
            target = data['search_anno'][0].cuda()
            batch_ids = identities[count:count + len(search)]
            for index, identity in enumerate(batch_ids):
                assert identity['sequence'] in fitting
                assert not used[identity['sequence']].intersection(
                    identity['template_frames'] + identity['search_frames'])
                tensors = dict(templates=data['template_images'][:, index],
                    search=data['search_images'][0, index], target=data['search_anno'][0, index])
                manifests.append(dict(identity=identity, tensors_sha256=common.digest(tensors)))
            if count == 0:
                torch.save(dict(templates=data['template_images'].transpose(0, 1),
                    search=data['search_images'][0], target=data['search_anno'][0]),
                    str(root / 'first_images.pth'))
            for dynamic_name, dynamic in prior.STATES:
                for search_name, flags in prior.STATES:
                    cell = dynamic_name + '_' + search_name
                    output = model(templates, [search], [search.new_ones(len(search), 2),
                        search.new_tensor(dynamic).expand(len(search), -1),
                        search.new_tensor(flags).expand(len(search), -1)])[0]
                    inspections = candidates.inspect_output(model, output, target, window)
                    boxes = search.new_tensor([[item['box'] for item in row['candidates']]
                                               for row in inspections])
                    scores = search.new_tensor([[item['hann_score'] for item in row['candidates']]
                                                for row in inspections])
                    assert (scores > 0).all()
                    weighted = (boxes * (scores / scores.sum(-1, keepdim=True))[..., None]).sum(1)
                    uniform = boxes.mean(1)
                    assert (weighted[:, 2:] > 0).all() and torch.isfinite(weighted).all()
                    weighted_iou = candidates.dense_iou(weighted[:, None], target)[:, 0]
                    uniform_iou = candidates.dense_iou(uniform[:, None], target)[:, 0]
                    for index, (identity, inspected) in enumerate(zip(batch_ids, inspections)):
                        native = inspected['hann_iou']
                        wi, ui = float(weighted_iou[index]), float(uniform_iou[index])
                        rows.append(dict(identity=identity, cell=cell, target=inspected['target'],
                            candidates=inspected['candidates'], native_iou=native,
                            weighted_box=weighted[index].cpu().tolist(),
                            uniform_box=uniform[index].cpu().tolist(), weighted_iou=wi, uniform_iou=ui,
                            weighted_gain=wi-native, uniform_gain=ui-native,
                            weighted_minus_uniform=wi-ui))
                    for key in ['score_map', 'size_map', 'offset_map']:
                        maps[key].append(output[key].cpu())
                    maps['target'].append(target.cpu())
            count += len(search)
            print(json.dumps(dict(stage='batch', tuples=count, rows=len(rows),
                                  seconds=time.time()-started)), flush=True)
    assert count == 128 and len(rows) == 1152
    torch.save(dict({key: torch.cat(values) for key, values in maps.items()},
                    window=window.cpu()), str(root / 'maps.pth'))
    effects, advance = summarize(rows)
    assert common.digest(model.state_dict()) == before
    assert all(value.grad is None for value in model.parameters())
    sources = [Path(__file__), Path(prior.__file__), Path(history.__file__),
        Path(history.training.__file__), Path(common.__file__), Path(candidates.__file__),
        names_path, Path('experiments/cadtrack/cadtrack.yaml'),
        Path('lib/train/data/processing_utils.py'), Path('lib/train/data/sampler.py'),
        Path('lib/train/dataset/lasher.py'), Path('lib/test/utils/hann.py'),
        Path('lib/utils/box_ops.py')]
    sources += sorted(Path('miss_rgbt').rglob('*.py')) + sorted(Path('lib/models').rglob('*.py'))
    report = dict(status='complete', pid=os.getpid(), seed=61101, fitting_names=fitting,
        input_manifest=manifests, rows=rows, effects=effects, advance=advance,
        prior_reports_sha256=previous_hashes, checkpoint=common.CHECKPOINT,
        checkpoint_sha256=common.CHECKPOINT_SHA, baseline_state_sha256=before,
        controls=dict(native_hann_decode_exact=True, frozen_state=True, gradients_absent=True),
        caches={key: dict(path=str(root/(key+'.pth')), sha256=common.sha256(root/(key+'.pth')))
                for key in ['first_images', 'maps']},
        source_sha256={str(path.resolve()): common.sha256(path) for path in sources},
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        peak_reserved_bytes=torch.cuda.max_memory_reserved(), seconds=time.time()-started,
        scope='Fixed box averaging on train882 crops, not closed-loop or official tracking metrics.')
    history.training.save(root/'report.json', report)
    history.training.save(root/'completed.json', dict(status='complete', advance=advance,
        report_sha256=common.sha256(root/'report.json'), seconds=report['seconds']))
    print(json.dumps(dict(stage='complete', effects=effects, advance=advance,
                          seconds=report['seconds'])), flush=True)


if __name__ == '__main__':
    main()
