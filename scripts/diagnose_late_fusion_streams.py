"""Train882 diagnostic of absent-stream contribution at the fusion head."""
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
REPORTS = Path('/home/gaob/Miss-RGBT/reports')
MAPS = ['score_map', 'size_map', 'offset_map', 'pred_boxes']


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


def seed(value):
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    torch.cuda.manual_seed_all(value)


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


def run_fusion_intervention(model, templates, search, availability, stream):
    def mask_branch(module, inputs):
        fused = inputs[0]
        channels = fused.shape[1] // 2
        branch = fused[:, stream * channels:(stream + 1) * channels]
        missing_search = availability[-1].sum(-1).eq(1)[:, None, None, None]
        masked = torch.where(missing_search, torch.zeros_like(branch), branch)
        assert torch.equal(masked[~missing_search.expand_as(branch)],
                           branch[~missing_search.expand_as(branch)])
        assert torch.count_nonzero(masked[missing_search.expand_as(branch)]) == 0
        return (torch.cat([fused[:, :stream * channels], masked,
                           fused[:, (stream + 1) * channels:]], dim=1),)

    handle = model.baseline.decode_fuse_search.register_forward_pre_hook(mask_branch)
    output = model(templates, [search], availability)[0]
    handle.remove()
    return output


def read_prior_tuple_ids():
    files = [
        'server28_fusion_adapter_pilot_20261102/report.json',
        'server28_head_supervision_20261102/report.json',
        'server28_modality_heads_20261102/report.json',
        'server28_response_geometry_20261102/report.json',
        'server28_absent_search_kv_20261102_v2/report.json',
        'server28_absent_position_20261102_v4/report.json',
    ]
    identities = set()
    for filename in files:
        report = json.loads((REPORTS / filename).read_text())
        cache_manifests = report.get('cache_manifests')
        if cache_manifests:
            manifests = cache_manifests.values()
        elif 'manifest' in report:
            manifests = [report['manifest']]
        else:
            assert filename in (
                'server28_absent_search_kv_20261102_v2/report.json',
                'server28_absent_position_20261102_v4/report.json')
            for row in report['rows']:
                item = row['identity']
                identities.add((item['sequence'], tuple(item['template_frames']),
                                tuple(item['search_frames'])))
            continue
        for manifest in manifests:
            for row in manifest['input_manifest']:
                item = row['identity']
                identities.add((item['sequence'], tuple(item['template_frames']),
                                tuple(item['search_frames'])))
    return identities


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
    seed(46101)
    assert sha256(CHECKPOINT) == CHECKPOINT_SHA
    update_config_from_file('experiments/cadtrack/cadtrack.yaml')
    model = load_pretrained(cfg, CHECKPOINT, repair=False).cuda().eval().requires_grad_(False)
    before = digest(model.state_dict())
    previous_ids = read_prior_tuple_ids()
    prior_reports = [
        'server28_fusion_adapter_pilot_20261102/report.json',
        'server28_head_supervision_20261102/report.json',
        'server28_modality_heads_20261102/report.json',
        'server28_response_geometry_20261102/report.json',
        'server28_absent_search_kv_20261102_v2/report.json',
        'server28_absent_position_20261102_v4/report.json',
    ]

    train_path = Path('lib/train/data_specs/lasher_train.txt')
    train_names = train_path.read_text().splitlines()
    assert len(train_names) == len(set(train_names)) == 882
    fitting = [name for index, name in enumerate(train_names) if index % 5]
    loader = make_loader(cfg, '/data/wangwj/dataset/LasHeR/trainingset',
                         'train', 16, 128, 0)
    loader.dataset.datasets[0].sequence_list = list(fitting)
    identities = capture_identities(loader)
    rows = []
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
                missing_stream = search_state.index(0.) if search_name != '11' else 0
                for direction, stream in [('target', missing_stream),
                                          ('sham', 1 - missing_stream)]:
                    changed = run_fusion_intervention(
                        model, templates, search, availability, stream)
                    changed_loss, changed_iou = metrics(changed, target)
                    if search_name == '11':
                        assert all(torch.equal(native[key], changed[key]) for key in MAPS)
                    for index, identity in enumerate(batch_ids):
                        rows.append(dict(
                            identity=identity,
                            cell=dynamic_name + '_' + search_name,
                            direction=direction,
                            loss_delta=float(native_loss[index] - changed_loss[index]),
                            iou_delta=float(changed_iou[index] - native_iou[index]),
                            complete_equal=search_name == '11'))
        print(json.dumps(dict(stage='batch', samples=count, expected=128)), flush=True)

    assert count == 128 and len(rows) == 2304
    effects = {}
    for state in ['01', '10']:
        selected = [row for row in rows if row['cell'].endswith('_' + state)]
        target_rows = [row for row in selected if row['direction'] == 'target']
        sham_rows = [row for row in selected if row['direction'] == 'sham']
        paired = []
        keyed = {(row['identity']['sequence'], tuple(row['identity']['template_frames']),
                  tuple(row['identity']['search_frames']), row['cell'], row['direction']): row
                 for row in rows}
        for row in target_rows:
            sham = keyed[(row['identity']['sequence'], tuple(row['identity']['template_frames']),
                          tuple(row['identity']['search_frames']), row['cell'], 'sham')]
            paired.append(dict(identity=row['identity'],
                               iou_delta=row['iou_delta'] - sham['iou_delta']))
        effects[state] = dict(
            target_iou=clustered(target_rows, 'iou_delta', 46102),
            target_loss=clustered(target_rows, 'loss_delta', 46103),
            target_vs_sham_iou=clustered(paired, 'iou_delta', 46104))
    complete_rows = [row for row in rows
                     if row['direction'] == 'target' and row['cell'].endswith('_11')]
    assert len(complete_rows) == 384 and all(row['complete_equal'] for row in complete_rows)
    advance = all(effects[state]['target_iou']['mean'] >= 0.01 and
                  effects[state]['target_iou']['ci95_low'] > 0 and
                  effects[state]['target_vs_sham_iou']['ci95_low'] > 0
                  for state in ['01', '10'])
    assert digest(model.state_dict()) == before
    assert all(parameter.grad is None for parameter in model.parameters())
    sources = [Path(__file__), train_path, Path('miss_rgbt/model.py'),
               Path('miss_rgbt/data.py'), Path('miss_rgbt/objectives.py'),
               Path('lib/models/cadtrack/cadtrack.py'),
               Path('lib/models/cadtrack/vit_care.py'),
               Path('lib/models/layers/head.py'),
               Path('/home/gaob/Miss-RGBT/scripts/launch_late_fusion_streams.py')]
    report = dict(status='complete', pid=os.getpid(), seed=46101,
        checkpoint=CHECKPOINT, checkpoint_sha256=CHECKPOINT_SHA,
        baseline_state_sha256=before, baseline_state_unchanged=True,
        baseline_gradients_absent=True, train_list_sha256=sha256(train_path),
        sequence_names=fitting, exact_previous_tuple_overlap=False,
        prior_report_sha256={name: sha256(REPORTS / name) for name in prior_reports},
        source_sha256={str(path): sha256(path) for path in sources},
        rows=rows, effects=effects, complete_equal=True, advance=advance,
        intervention='zero the absent stream feature channels immediately before decode_fuse_search',
        sham='zero the available stream feature channels at the same fusion input',
        seconds=time.time() - started,
        scope='Frozen train882 crop mechanism diagnostic; no training, val97, official test, or temporal claim.')
    with (root / 'report.json').open('x', newline='\n') as stream:
        json.dump(report, stream, indent=2)
        stream.write('\n')
    report_sha = sha256(root / 'report.json')
    with (root / 'completed.json').open('x', newline='\n') as stream:
        json.dump(dict(status='complete', report_sha256=report_sha,
                       advance=advance, seconds=report['seconds']), stream, indent=2)
        stream.write('\n')
    print(json.dumps(dict(stage='complete', effects=effects, advance=advance,
                          seconds=report['seconds'])), flush=True)


if __name__ == '__main__':
    main()
