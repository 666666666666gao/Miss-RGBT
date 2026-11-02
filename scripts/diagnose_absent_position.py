"""Train882 diagnostic of absolute positions on unavailable search streams."""
import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import random
import time

import cv2
import numpy as np
import torch

from lib.config.cadtrack.config import cfg, update_config_from_file
from lib.utils.box_ops import box_cxcywh_to_xyxy, box_xywh_to_xyxy, generalized_box_iou
from miss_rgbt.data import make_loader
from miss_rgbt.model import load_pretrained
from miss_rgbt.objectives import task_loss

CHECKPOINT = '/home/gaob/Miss-RGBT/checkpoints/pretrained/CADTrack_ep0020.pth.tar'
CHECKPOINT_SHA = 'd20c8641e4263df3ceadf9e430d0ac96c181c13ad482bca28c2dddef592a8bb8'
MAPS = ['score_map', 'size_map', 'offset_map', 'pred_boxes']
REPORTS = Path('/home/gaob/Miss-RGBT/reports')


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def digest(state):
    value = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        value.update(name.encode('utf-8'))
        value.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return value.hexdigest()


def capture_identities(loader):
    sampler = loader.dataset
    dataset = sampler.datasets[0]
    original_getitem = sampler.getitem
    original_get_frames = dataset.get_frames
    identities = []

    def tracked_getitem():
        calls = []

        def tracked_get_frames(seq_id, frame_ids, anno=None):
            calls.append((seq_id, tuple(int(value) for value in frame_ids)))
            return original_get_frames(seq_id, frame_ids, anno)

        dataset.get_frames = tracked_get_frames
        sample = original_getitem()
        dataset.get_frames = original_get_frames
        template_call, search_call = calls[-2:]
        assert template_call[0] == search_call[0]
        identities.append(dict(sequence=dataset.sequence_list[template_call[0]],
            template_frames=list(template_call[1]), search_frames=list(search_call[1])))
        return sample

    sampler.getitem = tracked_getitem
    return identities


def clustered(rows, key, bootstrap_seed):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row['identity']['sequence']].append(row[key])
    means = np.asarray([np.mean(grouped[name]) for name in sorted(grouped)])
    rng = np.random.default_rng(bootstrap_seed)
    draws = rng.integers(0, len(means), size=(10000, len(means)))
    low, high = np.quantile(means[draws].mean(axis=1), [.025, .975])
    return dict(mean=float(means.mean()), ci95_low=float(low), ci95_high=float(high),
                sequences=len(means), rows=len(rows))


def save(path, value):
    with path.open('x', newline='\n') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')


def seed(value):
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    torch.cuda.manual_seed_all(value)


def without_search_position(model, templates, search, availability, stream):
    backbone = model.baseline.backbone
    start = 1 + len(templates) * backbone.pos_embed_z.shape[1]
    end = start + backbone.pos_embed_x.shape[1]
    position = backbone.pos_embed_x

    def remove_position(block, inputs):
        rgb, tir = inputs
        values = [rgb, tir]
        flags = availability[-1][:, stream].eq(0)
        selected = values[stream].clone()
        selected[:, start:end] -= position * flags[:, None, None]
        values[stream] = selected
        return tuple(values)

    handle = backbone.blocks[0].register_forward_pre_hook(remove_position)
    output = model(templates, [search], availability)[0]
    handle.remove()
    return output


def metrics(output, target):
    loss = task_loss([output], target.unsqueeze(0), cfg)
    _, iou = generalized_box_iou(
        box_cxcywh_to_xyxy(output['pred_boxes']).flatten(0, 1),
        box_xywh_to_xyxy(target).clamp(0, 1))
    return loss.detach(), iou.detach()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=False)
    started = time.time()
    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    seed(44101)
    assert sha256(CHECKPOINT) == CHECKPOINT_SHA
    update_config_from_file('experiments/cadtrack/cadtrack.yaml')
    model = load_pretrained(cfg, CHECKPOINT, repair=False).cuda().eval().requires_grad_(False)
    before = digest(model.state_dict())

    train_names = Path('lib/train/data_specs/lasher_train.txt').read_text().splitlines()
    assert len(train_names) == len(set(train_names)) == 882
    fitting = [name for index, name in enumerate(train_names) if index % 5]
    previous_reports = [
        'server28_fusion_adapter_pilot_20261102/report.json',
        'server28_head_supervision_20261102/report.json',
        'server28_modality_heads_20261102/report.json',
        'server28_response_geometry_20261102/report.json',
        'server28_absent_search_kv_20261102_v2/report.json',
    ]
    previous_ids = set()
    for filename in previous_reports:
        report = json.loads((REPORTS / filename).read_text())
        cache_manifests = report.get('cache_manifests')
        if cache_manifests:
            manifests = cache_manifests.values()
        elif 'manifest' in report:
            manifests = [report['manifest']]
        else:
            assert filename == 'server28_absent_search_kv_20261102_v2/report.json'
            for row in report['rows']:
                identity = row['identity']
                previous_ids.add((identity['sequence'], tuple(identity['template_frames']),
                                  tuple(identity['search_frames'])))
            continue
        for manifest in manifests:
            for item in manifest['input_manifest']:
                identity = item['identity']
                previous_ids.add((identity['sequence'], tuple(identity['template_frames']),
                                  tuple(identity['search_frames'])))

    loader = make_loader(cfg, '/data/wangwj/dataset/LasHeR/trainingset',
                         'train', 16, 128, 0)
    loader.dataset.datasets[0].sequence_list = list(fitting)
    identities = capture_identities(loader)
    aggregate = []
    base_rows = []
    intervention_rows = []
    count = 0
    states = [('11', [1., 1.]), ('01', [0., 1.]), ('10', [1., 0.])]
    for data in loader:
        templates = [image.cuda() for image in data['template_images']]
        search = data['search_images'][0].cuda()
        target = data['search_anno'][0].cuda()
        batch = search.shape[0]
        batch_ids = identities[count:count + batch]
        assert len(batch_ids) == batch
        for identity in batch_ids:
            key = (identity['sequence'], tuple(identity['template_frames']),
                   tuple(identity['search_frames']))
            assert key not in previous_ids
        count += batch
        for dynamic_name, dynamic_state in states:
            dynamic_flags = search.new_tensor(dynamic_state).expand(batch, -1)
            for search_name, search_state in states:
                search_flags = search.new_tensor(search_state).expand(batch, -1)
                availability = [search.new_ones(batch, 2), dynamic_flags, search_flags]
                native = model(templates, [search], availability)[0]
                native_loss, native_iou = metrics(native, target)
                complete = search_name == '11'
                missing_stream = search_state.index(0.) if not complete else 0
                routes = [('target', missing_stream), ('sham', 1 - missing_stream)]
                for direction, stream in routes:
                    if complete:
                        changed = native
                    else:
                        changed = without_search_position(
                            model, templates, search, availability, stream)
                    changed_loss, changed_iou = metrics(changed, target)
                    if complete:
                        assert all(torch.equal(native[key], changed[key]) for key in MAPS)
                    for index, identity in enumerate(batch_ids):
                        row = dict(identity=identity, cell=dynamic_name + '_' + search_name,
                                   direction=direction, loss_delta=float(native_loss[index] - changed_loss[index]),
                                   iou_delta=float(changed_iou[index] - native_iou[index]),
                                   complete_equal=complete)
                        aggregate.append(row)
                        if direction == 'target':
                            base_rows.append(dict(identity=identity, cell=row['cell'],
                                                  loss=float(native_loss[index]),
                                                  iou=float(native_iou[index])))
                            intervention_rows.append(row)
        print(json.dumps(dict(stage='batch', samples=count, expected=128)), flush=True)

    assert count == 128 and len(aggregate) == 2304
    effects = {}
    for state in ['01', '10']:
        selected = [row for row in aggregate
                    if row['cell'].endswith('_' + state)]
        target_rows = [row for row in selected if row['direction'] == 'target']
        sham_rows = [row for row in selected if row['direction'] == 'sham']
        effects[state] = dict(
            target_iou=clustered(target_rows, 'iou_delta', 44102),
            target_loss=clustered(target_rows, 'loss_delta', 44103),
            target_vs_sham_iou=clustered(
                [dict(identity=t['identity'], iou_delta=t['iou_delta'] - s['iou_delta'])
                 for t, s in zip(target_rows, sham_rows)], 'iou_delta', 44104))
    complete_rows = [row for row in aggregate
                     if row['direction'] == 'target' and row['cell'].endswith('_11')]
    assert len(complete_rows) == 384 and all(row['complete_equal'] for row in complete_rows)
    advance = all(effects[state]['target_iou']['mean'] >= 0.01 and
                  effects[state]['target_iou']['ci95_low'] > 0 and
                  effects[state]['target_vs_sham_iou']['ci95_low'] > 0
                  for state in ['01', '10'])
    assert digest(model.state_dict()) == before
    assert all(parameter.grad is None for parameter in model.parameters())
    sources = [Path(__file__), Path('miss_rgbt/model.py'), Path('miss_rgbt/data.py'),
               Path('miss_rgbt/objectives.py'), Path('lib/models/cadtrack/vit_care.py'),
               Path('lib/models/layers/head.py')]
    report = dict(status='complete', pid=os.getpid(), seed=44101,
        checkpoint=CHECKPOINT, checkpoint_sha256=CHECKPOINT_SHA,
        baseline_state_sha256=before, baseline_state_unchanged=True,
        baseline_gradients_absent=True, train_list_sha256=sha256(
            Path('lib/train/data_specs/lasher_train.txt')),
        sequence_names=fitting, exact_previous_tuple_overlap=False,
        source_sha256={str(path): sha256(path) for path in sources},
        rows=aggregate, baseline_rows=base_rows, intervention_rows=intervention_rows,
        effects=effects, complete_equal=True, advance=advance,
        intervention='subtract absolute search position embedding at first block input',
        sham='subtract the same positional embedding from the available search stream',
        seconds=time.time() - started,
        scope='Frozen train882 crop diagnostic; no training, val97, official test, or temporal claim.')
    save(root / 'report.json', report)
    save(root / 'completed.json', dict(status='complete',
        report_sha256=sha256(root / 'report.json'), advance=advance, seconds=report['seconds']))
    print(json.dumps(dict(stage='complete', effects=effects, advance=advance,
                          seconds=report['seconds'])), flush=True)


if __name__ == '__main__':
    main()
