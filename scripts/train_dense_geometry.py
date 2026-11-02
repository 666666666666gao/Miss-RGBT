"""Matched fixed-neighborhood regression versus native peak supervision."""
import argparse
import copy
import json
import os
from pathlib import Path
import time

import cv2
import numpy as np
import torch
from lib.config.cadtrack.config import cfg, update_config_from_file
from lib.test.utils.hann import hann2d
from lib.utils.box_ops import box_cxcywh_to_xyxy, box_xywh_to_xyxy, generalized_box_iou
from lib.utils.heapmap_utils import generate_heatmap
from miss_rgbt.data import make_loader
from miss_rgbt.model import load_pretrained
from miss_rgbt.objectives import task_loss
import diagnose_box_consensus as prior
import probe_fusion_adapter as fusion
import probe_window_training as measurement

common, PROJECT = prior.common, prior.PROJECT


def save(path, value):
    with Path(path).open('x', newline='\n') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')


def exclusions():
    used, hashes = prior.exclusions()
    path = PROJECT/'reports/server28_box_consensus_20261102/report.json'
    hashes[path.parent.name+'/report.json'] = common.sha256(path)
    for item in json.loads(path.read_text())['input_manifest']:
        identity = item['identity']
        used[identity['sequence']].update(identity['template_frames']+identity['search_frames'])
    return used, hashes


def cache_split(model, names, samples, sample_seed, root, used):
    root.mkdir()
    common.seed(sample_seed)
    loader = make_loader(cfg, prior.history.training.burst.DATASET, 'train', 16, samples, 0)
    dataset = loader.dataset.datasets[0]
    dataset.sequence_list = names
    original_info = dataset.get_sequence_info

    def excluded_info(index):
        info = dict(original_info(index))
        info['visible'] = info['visible'].clone()
        info['visible'][sorted(used[names[index]])] = False
        return info

    dataset.get_sequence_info = excluded_info
    identities = common.capture_identities(loader)
    captured = {}

    def capture(module, inputs, output):
        captured['features'] = output.detach()

    handle = model.baseline.decode_fuse_search.register_forward_hook(capture)
    features, boxes, labels, row_ids, manifests, count = [], [], [], [], [], 0
    with torch.no_grad():
        for data in loader:
            templates = [value.cuda() for value in data['template_images']]
            search = data['search_images'][0].cuda()
            target = data['search_anno'][0]
            batch_ids = identities[count:count+len(search)]
            for index, identity in enumerate(batch_ids):
                assert identity['sequence'] in names
                assert not used[identity['sequence']].intersection(
                    identity['template_frames']+identity['search_frames'])
                manifests.append(dict(identity=identity, tensors_sha256=common.digest(dict(
                    templates=data['template_images'][:, index],
                    search=data['search_images'][0, index], target=target[index]))))
            for dynamic_name, dynamic in prior.prior.STATES:
                for search_name, current in prior.prior.STATES:
                    out = model(templates, [search], [search.new_ones(len(search), 2),
                        search.new_tensor(dynamic).expand(len(search), -1),
                        search.new_tensor(current).expand(len(search), -1)])[0]
                    value = captured['features']
                    replay = fusion.head_output(model.baseline.box_head, value)
                    assert all(torch.equal(out[key], replay[key]) for key in common.MAPS)
                    features.append(value.cpu())
                    boxes.append(target)
                    labels.extend([dynamic_name+'_'+search_name]*len(search))
                    row_ids.extend(batch_ids)
            count += len(search)
            print(json.dumps(dict(stage='cache', split=root.name, tuples=count,
                                  expected=samples)), flush=True)
    handle.remove()
    assert count == samples
    cache = dict(features=torch.cat(features), boxes=torch.cat(boxes),
                 labels=labels, identities=row_ids)
    torch.save(cache, str(root/'features.pth'))
    record = dict(samples=samples, rows=len(labels), seed=sample_seed, sequence_names=names,
        input_manifest=manifests, native_head_replay_bitwise=True,
        cache_path=str(root/'features.pth'), cache_sha256=common.sha256(root/'features.pth'))
    save(root/'manifest.json', record)
    return cache, record


def geometry(output, target):
    boxes = prior.candidates.dense_boxes(output)
    predicted = box_cxcywh_to_xyxy(boxes).reshape(-1, 4)
    truth = box_xywh_to_xyxy(target).clamp(0, 1)
    repeated = truth[:, None].expand(-1, 256, -1).reshape(-1, 4)
    giou, _ = generalized_box_iou(predicted, repeated)
    regression = (2*(1-giou)+5*(predicted-repeated).abs().mean(-1)).reshape(-1, 256)
    heatmap = generate_heatmap(target[None], cfg.DATA.SEARCH.SIZE, cfg.MODEL.BACKBONE.STRIDE)[0]
    assert heatmap.eq(1).flatten(1).sum(-1).eq(1).all()
    peak = heatmap.flatten(1).argmax(-1)
    index = torch.arange(256, device=target.device)
    support = ((index[None] % 16-peak[:, None] % 16).abs() <= 1) & (
        (index[None]//16-peak[:, None]//16).abs() <= 1)
    count = support.sum(-1)
    assert (count >= 4).all() and (count <= 9).all()
    return regression, support


def objective(head, features, target, window, mode):
    output = measurement.decode(head, features, window, True)
    native = task_loss([output], target[None], cfg)
    if mode == 'native':
        return native, output
    regression, support = geometry(output, target)
    location = (output['score_map']*window).flatten(1).argmax(-1)
    native_regression = regression.gather(1, location[:, None]).squeeze(1)
    dense_regression = (regression*support).sum(-1)/support.sum(-1)
    return native-native_regression+dense_regression, output


def witness(head, cache, window, root):
    features, target = cache['features'][:16].cuda(), cache['boxes'][:16].cuda()
    head = copy.deepcopy(head).cuda().eval().requires_grad_(True)
    loss, output = objective(head, features, target, window, 'dense')
    regression, support = geometry(output, target)
    location = (output['score_map']*window).flatten(1).argmax(-1)
    native_box = box_cxcywh_to_xyxy(output['pred_boxes']).flatten(0, 1)
    truth = box_xywh_to_xyxy(target).clamp(0, 1)
    giou, _ = generalized_box_iou(native_box, truth)
    direct = 2*(1-giou)+5*(native_box-truth).abs().mean(-1)
    gathered = regression.gather(1, location[:, None]).squeeze(1)
    assert torch.equal(direct, gathered)
    dense_regression = (regression*support).sum(-1)/support.sum(-1)
    gradients = torch.autograd.grad(dense_regression.sum(), output['offset_map'])[0]
    assert torch.isfinite(gradients).all()
    outside = (~support).reshape(-1, 1, 16, 16).expand_as(gradients)
    assert gradients[outside].eq(0).all()
    assert (gradients.flatten(1).abs().sum(-1) > 0).all()
    stored = {key: output[key].detach().cpu() for key in ['score_map', 'size_map', 'offset_map']}
    stored.update(target=target.cpu(), support=support.cpu(), offset_gradient=gradients.cpu(),
                  dense_loss=loss.detach().cpu(), regression=regression.detach().cpu(),
                  native_loss=task_loss([output], target[None], cfg).detach().cpu())
    torch.save(stored, str(root/'objective_witness.pth'))
    return dict(rows=16, native_regression_onehot_exact=True, zero_outside_support=True,
        gradient_nonzero=True, path=str(root/'objective_witness.pth'),
        sha256=common.sha256(root/'objective_witness.pth'))


def profile(original, cache, window, root):
    head = copy.deepcopy(original).cuda().eval().requires_grad_(True)
    optimizer = torch.optim.AdamW(head.parameters(), lr=1e-5, weight_decay=1e-4)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    static = torch.cuda.memory_allocated()
    loss, output = objective(head, cache['features'][:64].cuda(),
                             cache['boxes'][:64].cuda(), window, 'dense')
    loss.mean().backward()
    optimizer.step()
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated()
    total = torch.cuda.get_device_properties(0).total_memory
    proposed = int(64*(.8*total-static)/(peak-static))
    batch = max(value for value in range(32, min(proposed, 2304)+1, 32) if 2304 % value == 0)
    result = dict(probe_batch=64, static_bytes=static, peak_bytes=peak, total_bytes=total,
        proposed=proposed, batch=batch, learning_rate=1e-5*batch/8,
        updates_per_epoch=2304//batch,
        policy='80percent linear incremental-peak estimate; divisor32 of actual2304 rows; no duplication/cap/retry.')
    save(root/'batch_sizing.json', result)
    del head, optimizer, loss, output
    torch.cuda.empty_cache()
    return result


def fit(mode, original, cache, development, window, sizing, root):
    root.mkdir()
    head = copy.deepcopy(original).cuda().eval().requires_grad_(True)
    initial, buffers = common.digest(head.state_dict()), common.digest(dict(head.named_buffers()))
    initial_metrics = measurement.measure(head, development, window)
    optimizer = torch.optim.AdamW(head.parameters(), lr=sizing['learning_rate'], weight_decay=1e-4)
    history, reference, stale = [], None, 0
    started = time.time()
    torch.cuda.reset_peak_memory_stats()
    for epoch in range(1, 21):
        lr = sizing['learning_rate']*(1 if epoch <= 6 else .1)
        for group in optimizer.param_groups:
            group['lr'] = lr
        order = torch.randperm(len(cache['labels']),
            generator=torch.Generator().manual_seed(62200+epoch))
        losses = []
        for start in range(0, len(order), sizing['batch']):
            selected = order[start:start+sizing['batch']]
            optimizer.zero_grad(set_to_none=True)
            values, output = objective(head, cache['features'][selected].cuda(),
                cache['boxes'][selected].cuda(), window, mode)
            loss = values.mean()
            assert torch.isfinite(loss)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(head.parameters(), .1)
            assert torch.isfinite(norm)
            optimizer.step()
            losses.append(float(loss.detach()))
        result = measurement.measure(head, development, window)
        if epoch >= 7:
            if reference is None or result['score'] < reference*.998:
                reference, stale = result['score'], 0
            else:
                stale += 1
        record = dict(epoch=epoch, updates=epoch*sizing['updates_per_epoch'], lr=lr,
            score=result['score'], cells=result['cells'], stale=stale,
            train_mean_loss=float(np.mean(losses)), seconds=time.time()-started)
        history.append(record)
        torch.save(dict(head=head.state_dict(), mode=mode, epoch=epoch),
                   str(root/('epoch%02d.pth' % epoch)))
        save(root/('epoch%02d.json' % epoch), record)
        print(json.dumps(dict(stage='epoch', mode=mode, **record)), flush=True)
        if epoch >= 10 and stale >= 3:
            break
    selected = min(history[-5:], key=lambda row: (row['score'], row['epoch']))
    path = root/('epoch%02d.pth' % selected['epoch'])
    head.load_state_dict(torch.load(str(path), map_location='cpu')['head'], strict=True)
    assert measurement.measure(head, development, window)['score'] == selected['score']
    assert common.digest(dict(head.named_buffers())) == buffers
    result = dict(mode=mode, initial_sha256=initial, initial_buffers_sha256=buffers,
        initial_metrics=initial_metrics, history=history, selected=selected,
        selected_path=str(path), selected_sha256=common.sha256(path),
        parameter_count=sum(value.numel() for value in head.parameters()),
        bn_buffers_unchanged=True, peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        final_five=[dict(epoch=row['epoch'], path=str(root/('epoch%02d.pth' % row['epoch'])),
            sha256=common.sha256(root/('epoch%02d.pth' % row['epoch']))) for row in history[-5:]])
    save(root/'selection.json', result)
    return result


def statistics(baseline, audits):
    paired = []
    for base, native, dense in zip(baseline['rows'], audits['native']['rows'], audits['dense']['rows']):
        assert all(base[key] == native[key] == dense[key] for key in ['identity', 'cell', 'target'])
        paired.append(dict(identity=base['identity'], cell=base['cell'],
            gain=dense['iou']-base['iou'], control=dense['iou']-native['iou']))
    effects = {state: common.clustered(
        [row for row in paired if row['cell'].endswith('_'+state)], 'gain', 62301)
        for state in ['01', '10']}
    effects['versus_native_training'] = common.clustered(
        [row for row in paired if not row['cell'].endswith('_11')], 'control', 62302)
    for cell in ['11_11', '01_11', '10_11']:
        effects[cell] = common.clustered([row for row in paired if row['cell'] == cell], 'gain', 62303)
    advance = (all(effects[state]['mean'] >= .01 and effects[state]['ci95_low'] > 0
                   for state in ['01', '10'])
        and effects['versus_native_training']['ci95_low'] > 0
        and all(effects[cell]['mean'] >= 0 and effects[cell]['ci95_low'] > -.005
                for cell in ['11_11', '01_11', '10_11'])
        and all(value['sequences'] >= 32 for value in effects.values()))
    return effects, bool(advance)


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
    assert common.sha256(common.CHECKPOINT) == common.CHECKPOINT_SHA
    model = load_pretrained(cfg, common.CHECKPOINT, repair=False).cuda().eval().requires_grad_(False)
    before = common.digest(model.state_dict())
    names_path = Path('lib/train/data_specs/lasher_train.txt')
    names = names_path.read_text().splitlines()
    assert len(names) == len(set(names)) == 882
    partitions = dict(train=[name for index, name in enumerate(names) if index % 5],
                      development=names[::5][:88], audit=names[::5][88:])
    used, previous_hashes = exclusions()
    caches, manifests = {}, {}
    for split, count, sample_seed in [('train', 256, 62101), ('development', 128, 62102)]:
        caches[split], manifests[split] = cache_split(
            model, partitions[split], count, sample_seed, root/split, used)
    window = hann2d(torch.tensor([16, 16]), centered=True).cuda()
    witnessed = witness(model.baseline.box_head, caches['train'], window, root)
    sizing = profile(model.baseline.box_head, caches['train'], window, root)
    print(json.dumps(dict(stage='sizing', **sizing)), flush=True)
    arms = {mode: fit(mode, model.baseline.box_head, caches['train'], caches['development'],
                     window, sizing, root/mode) for mode in ['native', 'dense']}
    assert arms['native']['initial_sha256'] == arms['dense']['initial_sha256']
    assert arms['native']['initial_metrics'] == arms['dense']['initial_metrics']
    caches['audit'], manifests['audit'] = cache_split(
        model, partitions['audit'], 128, 62103, root/'audit', used)
    baseline = measurement.measure(model.baseline.box_head, caches['audit'], window)
    audits = {}
    for mode in ['native', 'dense']:
        head = copy.deepcopy(model.baseline.box_head).cuda().eval()
        head.load_state_dict(torch.load(arms[mode]['selected_path'], map_location='cpu')['head'], strict=True)
        audits[mode] = measurement.measure(head, caches['audit'], window)
        del head
    effects, advance = statistics(baseline, audits)
    assert common.digest(model.state_dict()) == before
    assert all(value.grad is None for value in model.parameters())
    sources = [Path(__file__), Path(prior.__file__), Path(prior.prior.__file__),
        Path(common.__file__), Path(fusion.__file__), Path(measurement.__file__),
        Path(prior.candidates.__file__), Path(prior.history.__file__),
        Path(prior.history.training.__file__), names_path,
        Path('experiments/cadtrack/cadtrack.yaml'), Path('lib/test/utils/hann.py'),
        Path('lib/utils/heapmap_utils.py'), Path('lib/utils/box_ops.py'),
        Path('lib/train/data/processing_utils.py'), Path('lib/train/data/sampler.py'),
        Path('lib/train/dataset/lasher.py')]
    sources += sorted(Path('miss_rgbt').rglob('*.py'))+sorted(Path('lib/models').rglob('*.py'))
    report = dict(status='complete', pid=os.getpid(), partitions=partitions,
        cache_manifests=manifests, prior_reports_sha256=previous_hashes,
        checkpoint=common.CHECKPOINT, checkpoint_sha256=common.CHECKPOINT_SHA,
        baseline_state_sha256=before, baseline_unchanged=True, baseline_gradients_absent=True,
        objective_witness=witnessed, batch_sizing=sizing, arms=arms,
        baseline=baseline, audit=audits, effects=effects, advance=advance,
        source_sha256={str(path.resolve()): common.sha256(path) for path in sources},
        seconds=time.time()-started,
        scope='Matched cached-head dense regression control; not official tracking or a novel module.')
    save(root/'report.json', report)
    save(root/'completed.json', dict(status='complete', report_sha256=common.sha256(root/'report.json'),
        advance=advance, seconds=report['seconds']))
    print(json.dumps(dict(stage='complete', effects=effects, advance=advance,
                          seconds=report['seconds'])), flush=True)


if __name__ == '__main__':
    main()
