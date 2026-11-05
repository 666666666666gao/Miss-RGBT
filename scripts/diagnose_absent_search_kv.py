"""Frozen E0 paired diagnostic of position-only missing search K/V tokens."""
import argparse
import json
import os
from pathlib import Path
import random
import time
import types

import cv2
import numpy as np
import torch

from lib.config.cadtrack.config import cfg, update_config_from_file
from lib.utils.box_ops import box_cxcywh_to_xyxy, box_xywh_to_xyxy, generalized_box_iou
from miss_rgbt.data import make_loader
from miss_rgbt.model import load_pretrained
from miss_rgbt.objectives import task_loss
from scripts.diagnose_joint_utility_sets import capture_identities, sha256
from scripts.probe_fusion_adapter import CHECKPOINT, CHECKPOINT_SHA, MAPS, digest
from scripts.analyze_response_geometry import clustered


def save(path, value):
    with path.open('x', newline='\n') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')


def seed(value):
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    torch.cuda.manual_seed_all(value)


def custom_attention(block, tokens, key_mask):
    attention = block.attn
    batch, length, channels = tokens.shape
    qkv = attention.qkv(tokens).reshape(
        batch, length, 3, attention.num_heads, channels // attention.num_heads
    ).permute(2, 0, 3, 1, 4)
    query, key, value = qkv[0], qkv[1], qkv[2]
    scores = (query @ key.transpose(-2, -1)) * attention.scale
    scores = scores.masked_fill(key_mask[:, None, None, :], float('-inf'))
    weights = attention.attn_drop(scores.softmax(dim=-1))
    result = (weights @ value).transpose(1, 2).reshape(batch, length, channels)
    return attention.proj_drop(attention.proj(result))


def run_with_key_mask(model, blocks, original_forwards, masks,
                      templates, search, availability):
    def forward(block, rgb, tir):
        rgb = rgb + block.drop_path(custom_attention(block, block.norm1(rgb), masks[0]))
        tir = tir + block.drop_path(custom_attention(block, block.norm1(tir), masks[1]))
        rgb = rgb + block.drop_path(block.mlp(block.norm2(rgb)))
        tir = tir + block.drop_path(block.mlp(block.norm2(tir)))
        return rgb, tir
    for block in blocks:
        block.forward = types.MethodType(forward, block)
    output = model(templates, [search], availability)[0]
    for block, original_forward in zip(blocks, original_forwards):
        block.forward = original_forward
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output')
    parser.add_argument('--witness-only', action='store_true')
    args = parser.parse_args()
    assert args.output or args.witness_only
    root = Path(args.output) if args.output else None
    if not args.witness_only:
        root.mkdir(parents=True, exist_ok=False)
    started = time.time()
    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    seed(43101)
    assert sha256(CHECKPOINT) == CHECKPOINT_SHA
    update_config_from_file('experiments/cadtrack/cadtrack.yaml')
    model = load_pretrained(cfg, CHECKPOINT, repair=False).cuda().eval().requires_grad_(False)
    before = digest(model.state_dict())
    train_names = Path('lib/train/data_specs/lasher_train.txt').read_text().splitlines()
    assert len(train_names) == len(set(train_names)) == 882
    fitting = [name for index, name in enumerate(train_names) if index % 5]
    old_reports = [
        ('server28_fusion_adapter_pilot_20261102/report.json', 'cache_manifests'),
        ('server28_head_supervision_20261102/report.json', 'cache_manifests'),
        ('server28_modality_heads_20261102/report.json', 'cache_manifests'),
        ('server28_response_geometry_20261102/report.json', 'manifest'),
    ]
    old_ids = set()
    for filename, key in old_reports:
        record = json.loads((Path('reports') / filename).read_text())
        manifests = record[key].values() if key == 'cache_manifests' else [record[key]]
        for manifest in manifests:
            for item in manifest['input_manifest']:
                identity = item['identity']
                old_ids.add((identity['sequence'], tuple(identity['template_frames']),
                             tuple(identity['search_frames'])))
    loader = make_loader(cfg, '/data/wangwj/dataset/LasHeR/trainingset',
                         'train', 16, 128, 0)
    loader.dataset.datasets[0].sequence_list = list(fitting)
    identities = capture_identities(loader)
    def capture_features(module, inputs, output):
        captured['features'] = output.detach()

    captured = {}
    hook = model.baseline.decode_fuse_search.register_forward_hook(capture_features)
    aggregate, base_rows, point_rows = [], [], []
    blocks = model.baseline.backbone.blocks
    original_forwards = [block.forward for block in blocks]
    count = 0
    states = [('11', [1., 1.]), ('01', [0., 1.]), ('10', [1., 0.])]
    for batch_index, data in enumerate(loader):
        templates = [image.cuda() for image in data['template_images']]
        search = data['search_images'][0].cuda()
        target = data['search_anno'][0].cuda()
        batch = search.shape[0]
        batch_ids = identities[count:count + batch]
        assert len(batch_ids) == batch
        for identity in batch_ids:
            key = (identity['sequence'], tuple(identity['template_frames']),
                   tuple(identity['search_frames']))
            assert key not in old_ids
        count += batch
        for dynamic_name, dynamic_state in states:
            dynamic_flags = search.new_tensor(dynamic_state).expand(batch, -1)
            for search_name, search_state in states:
                search_flags = search.new_tensor(search_state).expand(batch, -1)
                availability = [search.new_ones(batch, 2), dynamic_flags, search_flags]
                query_masks = []
                total = 1 + len(templates) * model.baseline.backbone.pos_embed_z.shape[1] + \
                    model.baseline.backbone.pos_embed_x.shape[1]
                start_search = total - model.baseline.backbone.pos_embed_x.shape[1]
                for modality in range(2):
                    mask = torch.zeros(batch, total, dtype=torch.bool, device=search.device)
                    unavailable = search_flags[:, modality].eq(0)
                    mask[:, start_search:] = unavailable[:, None]
                    query_masks.append(mask)

                baseline = model(templates, [search], availability)[0]
                base_feature = captured['features'].clone()
                sham = run_with_key_mask(model, blocks, original_forwards,
                    [torch.zeros_like(mask) for mask in query_masks],
                    templates, search, availability)
                sham_maps_equal = all(torch.equal(baseline[key], sham[key]) for key in MAPS)
                assert sham_maps_equal, (dynamic_name, search_name, 'custom attention sham')
                if args.witness_only:
                    hook.remove()
                    print(json.dumps(dict(stage='attention_witness_pass',
                        batch_samples=batch, maps=MAPS)), flush=True)
                    return
                masked = run_with_key_mask(model, blocks, original_forwards, query_masks,
                    templates, search, availability)
                masked_feature = captured['features'].clone()
                assert all(block.forward == original for block, original in
                           zip(blocks, original_forwards))
                base_loss = task_loss([baseline], target.unsqueeze(0), cfg)
                _, base_iou = generalized_box_iou(
                    box_cxcywh_to_xyxy(baseline['pred_boxes']).flatten(0, 1),
                    box_xywh_to_xyxy(target).clamp(0, 1))
                masked_loss = task_loss([masked], target.unsqueeze(0), cfg)
                _, masked_iou = generalized_box_iou(
                    box_cxcywh_to_xyxy(masked['pred_boxes']).flatten(0, 1),
                    box_xywh_to_xyxy(target).clamp(0, 1))
                complete = search_name == '11'
                if complete:
                    assert all(torch.equal(baseline[key], masked[key]) for key in MAPS)
                    assert torch.equal(base_feature, masked_feature)
                for index, identity in enumerate(batch_ids):
                    base_row = dict(identity=identity, cell=dynamic_name + '_' + search_name,
                        loss=float(base_loss[index]), iou=float(base_iou[index]))
                    masked_row = dict(identity=identity, cell=dynamic_name + '_' + search_name,
                        loss=float(masked_loss[index]), iou=float(masked_iou[index]))
                    base_rows.append(base_row)
                    point_rows.append(masked_row)
                    aggregate.append(dict(identity=identity, cell=base_row['cell'],
                        loss_delta=base_row['loss'] - masked_row['loss'],
                        iou_delta=masked_row['iou'] - base_row['iou'],
                        native_sham_exact=sham_maps_equal,
                        complete_equal=complete))
        print(json.dumps(dict(stage='batch', samples=count, expected=128)), flush=True)
    assert count == 128 and len(aggregate) == len(base_rows) == len(point_rows) == 1152
    hook.remove()
    cells = {}
    for cell in sorted(set(row['cell'] for row in aggregate)):
        selected = [row for row in aggregate if row['cell'] == cell]
        assert len(selected) == 128 and all(row['native_sham_exact'] for row in selected)
        cells[cell] = dict(loss_delta=clustered(selected, 'loss_delta', 43102),
                           iou_delta=clustered(selected, 'iou_delta', 43103))
    effects, gates = {}, []
    for state in ['01', '10']:
        selected = [row for row in aggregate if row['cell'].endswith('_' + state)]
        effects[state] = clustered(selected, 'iou_delta', 43104)
        gates.append(effects[state]['mean'] >= .01 and effects[state]['ci95_low'] > 0)
    complete_rows = [row for row in aggregate if row['cell'].endswith('_11')]
    assert len(complete_rows) == 384 and all(row['complete_equal'] for row in complete_rows)
    gates.append(all(row['complete_equal'] for row in complete_rows))
    assert digest(model.state_dict()) == before
    assert all(parameter.grad is None for parameter in model.parameters())
    sources = [Path(__file__), Path('scripts/diagnose_joint_utility_sets.py'),
        Path('scripts/probe_fusion_adapter.py'), Path('miss_rgbt/model.py'),
        Path('miss_rgbt/data.py'), Path('miss_rgbt/objectives.py'),
        Path('lib/models/cadtrack/vit_care.py'), Path('lib/models/layers/head.py')]
    report = dict(status='complete', pid=os.getpid(), seed=43101, checkpoint=CHECKPOINT,
        checkpoint_sha256=CHECKPOINT_SHA, baseline_state_sha256=before,
        baseline_state_unchanged=True, baseline_gradients_absent=True,
        train_list_sha256=sha256(Path('lib/train/data_specs/lasher_train.txt')),
        sequence_names=fitting, unique_sequences=len(set(row['identity']['sequence'] for row in aggregate)),
        exact_previous_tuple_overlap=False,
        source_sha256={str(path): sha256(path) for path in sources},
        rows=aggregate, baseline_rows=base_rows, masked_rows=point_rows, cells=cells,
        effects=effects, advance=all(gates), attention_control='identical custom qkv/softmax/MLP with all keys enabled',
        mask='unavailable modality current-search tokens as keys/values only; keep all queries and template keys',
        seconds=time.time() - started,
        scope='Train882 frozen single-crop mechanism diagnostic; no training, official metric or temporal claim.')
    save(root / 'report.json', report)
    save(root / 'completed.json', dict(status='complete',
        report_sha256=sha256(root / 'report.json'), advance=all(gates), seconds=report['seconds']))
    print(json.dumps(dict(stage='complete', effects=effects, cells=cells,
                          advance=all(gates), seconds=report['seconds'])), flush=True)


if __name__ == '__main__':
    main()
