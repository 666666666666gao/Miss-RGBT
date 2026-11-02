"""Paired A replication with train882-only stopping and checkpoint selection."""
import argparse
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
from miss_rgbt.data import make_loader
from miss_rgbt.model import load_pretrained
from miss_rgbt.objectives import task_loss, training_loss, utility_supervision
from miss_rgbt.train import fix_bn, sample_masks
import train_dense_geometry as prior

PROJECT = Path('/home/gaob/Miss-RGBT')


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def seed(value):
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    torch.cuda.manual_seed_all(value)


def tensor_hash(value):
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def loader_for(config, names, samples, batch, workers, used):
    loader = make_loader(cfg, config['dataset'], 'train', batch, samples, workers)
    dataset = loader.dataset.datasets[0]
    dataset.sequence_list = list(names)
    original_info, original_frames = dataset.get_sequence_info, dataset.get_frames
    original_item = loader.dataset.getitem
    calls = []

    def info(index):
        value = dict(original_info(index))
        value['visible'] = value['visible'].clone()
        value['visible'][sorted(used[names[index]])] = False
        return value

    def frames(index, ids, anno=None):
        calls.append((index, [int(frame) for frame in ids]))
        return original_frames(index, ids, anno)

    def item():
        calls.clear()
        result = original_item()
        template, search = calls[-2:]
        assert template[0] == search[0]
        assert len(template[1]) == 2 and len(search[1]) == 3
        assert not used[names[template[0]]].intersection(template[1] + search[1])
        result['frame_identity'] = torch.tensor([template[0]] + template[1] + search[1])
        return result

    dataset.get_sequence_info, dataset.get_frames = info, frames
    loader.dataset.getitem = item
    return loader


def record_batch(data, names, phase, epoch, index, stream):
    identities = []
    for values in data['frame_identity'].t().tolist():
        identities.append(dict(sequence=names[values[0]],
            template_frames=values[1:3], search_frames=values[3:]))
    record = dict(phase=phase, epoch=epoch, microstep=index, identities=identities,
        tensors_sha256={key: tensor_hash(data[key]) for key in
                       ['template_images', 'search_images', 'search_anno']})
    stream.write(json.dumps(record) + '\n')
    stream.flush()
    return record


def validate(model, batches):
    selector = model.utility_repair.selection_rng_state.clone()
    model.utility_repair.reset_selection()
    model.eval()
    totals = dict.fromkeys(['11', '01', '10'], 0.)
    count = 0
    with torch.no_grad(), torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        for data in batches:
            templates = [value.cuda() for value in data['template_images']]
            searches = [value.cuda() for value in data['search_images']]
            boxes = data['search_anno'].cuda()
            size = searches[0].shape[0]
            for label, flags in [('11', [1., 1.]), ('01', [0., 1.]), ('10', [1., 0.])]:
                availability = [searches[0].new_ones(size, 2) for _ in templates]
                availability += [searches[0].new_tensor(flags).expand(size, -1)
                                 for _ in searches]
                loss = task_loss(model(templates, searches, availability), boxes, cfg)
                assert torch.isfinite(loss).all()
                totals[label] += float(loss.mean()) * size
            count += size
    model.utility_repair.selection_rng_state.copy_(selector)
    return dict(samples=count, task_loss={key: value/count for key, value in totals.items()},
                mean_task_loss=sum(totals.values())/(3*count))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    root = Path(config['output'])
    root.mkdir(exist_ok=False, parents=True)
    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    update_config_from_file('experiments/cadtrack/cadtrack.yaml')
    names_path = Path('lib/train/data_specs/lasher_train.txt')
    names = names_path.read_text().splitlines()
    assert len(names) == len(set(names)) == 882
    partitions = dict(train=[name for index, name in enumerate(names) if index % 5],
                      development=names[::5][:88], audit=names[::5][88:])
    assert [len(partitions[key]) for key in ['train', 'development', 'audit']] == [705, 88, 89]
    used, hashes = prior.exclusions()
    dense_path = PROJECT/'reports/server28_dense_geometry_20261102/report.json'
    hashes[str(dense_path)] = prior.common.sha256(dense_path)
    for manifest in json.loads(dense_path.read_text())['cache_manifests'].values():
        for entry in manifest['input_manifest']:
            identity = entry['identity']
            used[identity['sequence']].update(identity['template_frames'] + identity['search_frames'])
    for path in config['exclude_manifests']:
        hashes[path] = prior.common.sha256(path)
        for line in Path(path).read_text().splitlines():
            for identity in json.loads(line)['identities']:
                used[identity['sequence']].update(identity['template_frames'] + identity['search_frames'])
    assert prior.common.sha256(config['checkpoint']) == prior.common.CHECKPOINT_SHA
    save(root/'run_config.json', config)
    save(root/'data_boundary.json', dict(partitions=partitions,
        excluded_frames={key: sorted(value) for key, value in used.items()},
        prior_sha256=hashes, train_list_sha256=prior.common.sha256(names_path),
        scope='train882 only; audit names reserved; no benchmark or val split opened'))
    sources = sorted(Path('miss_rgbt').rglob('*.py')) + sorted(Path('lib').rglob('*.py'))
    sources += sorted((PROJECT/'scripts').glob('*.py'))
    save(root/'source_sha256.json', {str(path.resolve()): prior.common.sha256(path) for path in sources})
    started = time.time()
    with (root/'input_manifest.jsonl').open('x') as inputs, (root/'steps.jsonl').open('x') as steps:
        seed(config['development_seed'])
        dev_loader = loader_for(config, partitions['development'], config['development_samples'],
                                4, 0, used)
        dev_batches = []
        for index, data in enumerate(dev_loader):
            record_batch(data, partitions['development'], 'development', 0, index, inputs)
            dev_batches.append(data)
        assert sum(data['search_images'].shape[1] for data in dev_batches) == config['development_samples']
        seed(config['seed'])
        model = load_pretrained(cfg, config['checkpoint'], repair=False,
                               task_utility=config['task_utility']).cuda().train()
        fix_bn(model, True)
        save(root/'initial_state.json', dict(parameters_sha256=prior.common.digest(
            dict(model.named_parameters())), checkpoint_sha256=prior.common.CHECKPOINT_SHA))
        groups = [dict(params=[p for n, p in model.named_parameters() if p.requires_grad and 'backbone' in n],
                       lr=config['learning_rate']*config['backbone_multiplier']),
                  dict(params=[p for n, p in model.named_parameters() if p.requires_grad and 'backbone' not in n],
                       lr=config['learning_rate'])]
        optimizer = torch.optim.AdamW(groups, weight_decay=config['weight_decay'])
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, config['lr_drop_epoch'], gamma=.1)
        loader = loader_for(config, partitions['train'], config['samples_per_epoch'],
                            config['microbatch'], config['workers'], used)
        assert len(loader) % config['accumulation'] == 0
        generator = torch.Generator().manual_seed(config['seed'])
        history, reference, stale, updates = [], float('inf'), 0, 0
        for epoch in range(1, config['epochs']+1):
            seed(config['seed']+epoch)
            model.train()
            fix_bn(model, True)
            optimizer.zero_grad(set_to_none=True)
            for index, data in enumerate(loader):
                record = record_batch(data, partitions['train'], 'train', epoch, index, inputs)
                templates = [image.cuda() for image in data['template_images']]
                searches = [image.cuda() for image in data['search_images']]
                boxes = data['search_anno'].cuda()
                available, patch = sample_masks(templates, searches, config)
                microstep = (epoch-1)*len(loader)+index
                probe, view = None, None
                if microstep % config['utility_probe_interval'] == 0:
                    probe_id = microstep//config['utility_probe_interval']
                    view = 'C' if probe_id % 2 == 0 else 'M'
                    regions = torch.randint(model.utility_repair.region_tokens.shape[0],
                        (searches[0].shape[0], 1), generator=generator).cuda()
                    probe = dict(search_index=(probe_id//2) % len(searches), regions=regions)
                complete = model(templates, searches, utility_probe=probe if view == 'C' else None)
                missing = model(templates, searches, available, patch,
                                utility_probe=probe if view == 'M' else None)
                loss, status = training_loss(complete, missing, boxes, cfg,
                    config['forward_weight'], 0., config['reverse_margin'])
                if probe is not None:
                    auxiliary, utility = utility_supervision(complete if view == 'C' else missing, boxes, cfg)
                    loss = loss + config['utility_weight']*auxiliary
                    status.update(utility=utility, probe_view=view)
                assert torch.isfinite(loss), status
                (loss/config['accumulation']).backward()
                status.update(epoch=epoch, microstep=microstep, loss=float(loss.detach()),
                    input_sha256=record['tensors_sha256'],
                    masks_sha256=[tensor_hash(value) for value in available+patch])
                if (index+1) % config['accumulation'] == 0:
                    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config['clip_grad_norm'])
                    assert torch.isfinite(norm)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    updates += 1
                    status.update(update=updates, seconds=time.time()-started,
                                  peak_gpu_bytes=torch.cuda.max_memory_allocated())
                    if updates == 1 or updates % 25 == 0:
                        print(json.dumps(status), flush=True)
                steps.write(json.dumps(status)+'\n')
                steps.flush()
            scheduler.step()
            result = validate(model, dev_batches)
            result.update(epoch=epoch, updates=updates, seconds=time.time()-started)
            history.append(result)
            save(root/'development_history.json', history)
            if epoch == config['lr_drop_epoch']+1:
                reference, stale = float('inf'), 0
            if result['mean_task_loss'] < reference*(1-config['early_stop_relative_delta']):
                reference, stale = result['mean_task_loss'], 0
            elif epoch > config['lr_drop_epoch']:
                stale += 1
            path = root/('epoch%02d.pth.tar' % epoch)
            torch.save(dict(model=model.state_dict(), config=config, epoch=epoch, updates=updates,
                            optimizer=optimizer.state_dict()), str(path))
            stop = epoch >= config['early_stop_min_epochs'] and stale >= config['early_stop_patience']
            print('DEVELOPMENT '+json.dumps(dict(result, stale=stale, stop=stop)), flush=True)
            if stop:
                break
        final_five = history[-5:]
        assert len(final_five) == 5
        selected = min(final_five, key=lambda value: (value['mean_task_loss'], value['epoch']))
        selected_path = root/('epoch%02d.pth.tar' % selected['epoch'])
        save(root/'completed.json', dict(status='complete', epochs=epoch, updates=updates,
            early_stopped=stop, seconds=time.time()-started, pid=os.getpid(),
            final_five=[dict(epoch=value['epoch'], path=str(root/('epoch%02d.pth.tar' % value['epoch'])),
                sha256=prior.common.sha256(root/('epoch%02d.pth.tar' % value['epoch']))) for value in final_five],
            selected_epoch=selected['epoch'], selected_checkpoint=str(selected_path),
            selected_sha256=prior.common.sha256(selected_path),
            selection_rule='Lowest fixed train882 development task loss among final five; earlier tie',
            all_checkpoints_retained=True, official_test_selection_used=False,
            scope='Existing A replication; no new module or official result'))


if __name__ == '__main__':
    main()
