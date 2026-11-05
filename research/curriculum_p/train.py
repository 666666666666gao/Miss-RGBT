"""Full-network P control; existing partitioned A runtime stays unchanged."""
import argparse
import json
import os
from pathlib import Path
import time

import cv2
import torch

import train_partitioned_a as base
from schedule import epoch_config, sample_masks, schedule

PROJECT = Path('/home/gaob/Miss-RGBT')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    assert config['epochs'] == 20 and config['patch_mask_probability'] == .1
    assert config['task_utility']['selection'] == 'utility'
    for path, digest in config['parent_completion_sha256'].items():
        assert base.prior.common.sha256(path) == digest
        assert json.loads(Path(path).read_text())['status'] == 'complete'
    root = Path(config['output'])
    root.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    base.update_config_from_file('experiments/cadtrack/cadtrack.yaml')
    names_path = Path('lib/train/data_specs/lasher_train.txt')
    names = names_path.read_text().splitlines()
    assert len(names) == len(set(names)) == 882
    partitions = dict(train=[name for index, name in enumerate(names) if index % 5],
                      development=names[::5][:88], audit=names[::5][88:])
    assert [len(partitions[key]) for key in ['train', 'development', 'audit']] == [705, 88, 89]
    used, hashes = base.prior.exclusions()
    dense = PROJECT/'reports/server28_dense_geometry_20261102/report.json'
    hashes[str(dense)] = base.prior.common.sha256(dense)
    for manifest in json.loads(dense.read_text())['cache_manifests'].values():
        for entry in manifest['input_manifest']:
            identity = entry['identity']
            used[identity['sequence']].update(identity['template_frames'] + identity['search_frames'])
    for path, digest in config['exclude_manifest_sha256'].items():
        assert base.prior.common.sha256(path) == digest
        hashes[path] = digest
        with Path(path).open() as stream:
            for line in stream:
                for identity in json.loads(line)['identities']:
                    used[identity['sequence']].update(identity['template_frames'] + identity['search_frames'])
    assert base.prior.common.sha256(config['checkpoint']) == base.prior.common.CHECKPOINT_SHA
    base.save(root/'run_config.json', config)
    base.save(root/'data_boundary.json', dict(partitions=partitions,
        excluded_frames={key: sorted(value) for key, value in used.items()},
        prior_sha256=hashes, train_list_sha256=base.prior.common.sha256(names_path)))
    sources = sorted(Path('miss_rgbt').rglob('*.py')) + sorted(Path('lib').rglob('*.py'))
    sources += [Path(base.__file__), Path(base.prior.__file__),
                Path(base.prior.common.__file__), Path(__file__),
                Path(__file__).with_name('schedule.py')]
    base.save(root/'source_sha256.json',
              {str(path.resolve()): base.prior.common.sha256(path) for path in sources})
    started = time.time()
    with (root/'input_manifest.jsonl').open('x') as inputs, (root/'steps.jsonl').open('x') as steps:
        base.seed(config['development_seed'])
        dev_loader = base.loader_for(config, partitions['development'],
            config['development_samples'], 4, 0, used)
        dev_batches = []
        for index, data in enumerate(dev_loader):
            base.record_batch(data, partitions['development'], 'development', 0, index, inputs)
            dev_batches.append(data)
        assert sum(data['search_images'].shape[1] for data in dev_batches) == config['development_samples']
        base.seed(config['seed'])
        model = base.load_pretrained(base.cfg, config['checkpoint'], repair=False,
                                    task_utility=config['task_utility']).cuda().train()
        base.fix_bn(model, True)
        base.save(root/'initial_state.json',
            dict(parameters_sha256=base.prior.common.digest(dict(model.named_parameters())),
                 bn_sha256={name: base.tensor_hash(value) for name, value in model.named_buffers()
                            if any(key in name for key in
                                   ['running_mean', 'running_var', 'num_batches_tracked'])}))
        groups = [
            dict(params=[p for name, p in model.named_parameters() if p.requires_grad and 'backbone' in name],
                 lr=config['learning_rate']*config['backbone_multiplier']),
            dict(params=[p for name, p in model.named_parameters() if p.requires_grad and 'backbone' not in name],
                 lr=config['learning_rate'])]
        optimizer = torch.optim.AdamW(groups, weight_decay=config['weight_decay'])
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, config['lr_drop_epoch'], gamma=.1)
        loader = base.loader_for(config, partitions['train'], config['samples_per_epoch'],
                                config['microbatch'], config['workers'], used)
        assert len(loader) % config['accumulation'] == 0
        generator = torch.Generator().manual_seed(config['seed'])
        history, updates = [], 0
        for epoch in range(1, 21):
            base.seed(config['seed']+epoch)
            model.train()
            base.fix_bn(model, True)
            optimizer.zero_grad(set_to_none=True)
            local = epoch_config(config, epoch)
            for index, data in enumerate(loader):
                record = base.record_batch(data, partitions['train'], 'train', epoch, index, inputs)
                templates = [value.cuda() for value in data['template_images']]
                searches = [value.cuda() for value in data['search_images']]
                boxes = data['search_anno'].cuda()
                available, patch = sample_masks(templates, searches, config, epoch)
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
                loss, status = base.training_loss(complete, missing, boxes, base.cfg,
                    config['forward_weight'], 0., config['reverse_margin'])
                if probe is not None:
                    auxiliary, utility = base.utility_supervision(
                        complete if view == 'C' else missing, boxes, base.cfg)
                    loss = loss + config['utility_weight']*auxiliary
                    status.update(utility=utility, probe_view=view)
                assert torch.isfinite(loss), status
                (loss/config['accumulation']).backward()
                status.update(epoch=epoch, microstep=microstep, loss=float(loss.detach()),
                    missing_probability=schedule(config['curriculum_arm'])[epoch-1],
                    availability_probability=local['complete_rgb_missing_tir_missing_probability'],
                    input_sha256=record['tensors_sha256'],
                    masks_sha256=[base.tensor_hash(value) for value in available+patch],
                    patch_sha256=[base.tensor_hash(value) for value in patch])
                if (index+1) % config['accumulation'] == 0:
                    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config['clip_grad_norm'])
                    assert torch.isfinite(norm)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    updates += 1
                    status.update(update=updates, seconds=time.time()-started,
                        grad_norm=float(norm), learning_rates=[group['lr'] for group in optimizer.param_groups],
                        peak_gpu_bytes=torch.cuda.max_memory_allocated())
                    if updates == 1 or updates % 25 == 0:
                        print(json.dumps(status), flush=True)
                steps.write(json.dumps(status)+'\n')
                steps.flush()
            scheduler.step()
            result = base.validate(model, dev_batches)
            result.update(epoch=epoch, updates=updates, seconds=time.time()-started,
                          missing_probability=schedule(config['curriculum_arm'])[epoch-1])
            history.append(result)
            base.save(root/'development_history.json', history)
            torch.save(dict(model=model.state_dict(), config=config, epoch=epoch,
                            updates=updates, optimizer=optimizer.state_dict()),
                       str(root/('epoch%02d.pth.tar' % epoch)))
            print('DEVELOPMENT '+json.dumps(result), flush=True)
        selected = min(history[-5:], key=lambda row: (row['mean_task_loss'], row['epoch']))
        selected_path = root/('epoch%02d.pth.tar' % selected['epoch'])
        base.save(root/'completed.json', dict(status='complete', epochs=20, updates=updates,
            pid=os.getpid(), seconds=time.time()-started, early_stopped=False,
            curriculum_arm=config['curriculum_arm'], schedule=schedule(config['curriculum_arm']),
            final_five=[dict(epoch=row['epoch'], path=str(root/('epoch%02d.pth.tar' % row['epoch'])),
                sha256=base.prior.common.sha256(root/('epoch%02d.pth.tar' % row['epoch'])))
                for row in history[-5:]],
            selected_epoch=selected['epoch'], selected_checkpoint=str(selected_path),
            selected_sha256=base.prior.common.sha256(selected_path),
            selection_rule='Minimum fixed train882 development task loss within epochs16-20; earlier tie',
            all_checkpoints_retained=True, official_test_selection_used=False,
            scope='P training control; not a validated module or official benchmark result'))


if __name__ == '__main__':
    main()
