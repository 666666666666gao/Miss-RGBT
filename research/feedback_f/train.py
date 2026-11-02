"""Isolated F image-training loop; requires verified completed P controls."""
import argparse
import json
import os
from pathlib import Path
import time

import cv2
import torch

import train_partitioned_a as base
from research.curriculum_p.schedule import ARMS as P_ARMS, epoch_config
from training import ARMS, batch_loss, build_model, step_masks

PROJECT = Path('/home/gaob/Miss-RGBT')


def validate_parents(config):
    path = PROJECT/'reports/server28_curriculum_p_20261102_formal/verification.json'
    verified = json.loads(path.read_text())
    assert verified['status'] == 'PASS' and verified['phase'] == 'formal'
    assert set(verified['arms']) == set(P_ARMS)
    assert config['parent_verification_sha256'] == {str(path): base.prior.common.sha256(path)}
    for source, digest in verified['source_sha256'].items():
        assert base.prior.common.sha256(source) == digest
    prepared_path = path.with_name('prepared.json')
    assert base.prior.common.sha256(prepared_path) == verified['prepared_sha256']
    prepared = json.loads(prepared_path.read_text())
    for arm in P_ARMS:
        root = PROJECT/'runs/server28_curriculum_p_20261102_formal'/arm
        entry = next(row for row in prepared['arms'] if row['arm'] == arm)
        assert base.prior.common.sha256(entry['config']) == entry['config_sha256']
        parent_config = json.loads(Path(entry['config']).read_text())
        assert parent_config == json.loads((root/'run_config.json').read_text())
        for key in ['parent_completion_sha256', 'exclude_manifest_sha256']:
            for source, digest in parent_config[key].items():
                assert config[key][source] == digest
        completed = root/'completed.json'
        manifest = root/'input_manifest.jsonl'
        assert config['parent_completion_sha256'][str(completed)] == verified['arms'][arm]['completed_sha256']
        assert config['exclude_manifest_sha256'][str(manifest)] == verified['arms'][arm]['input_manifest_sha256']
    for key in ['parent_completion_sha256', 'exclude_manifest_sha256']:
        for source, digest in config[key].items():
            assert base.prior.common.sha256(source) == digest
            if key == 'parent_completion_sha256':
                assert json.loads(Path(source).read_text())['status'] == 'complete'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    assert config['feedback']['mode'] in ARMS
    assert config['epochs'] == 20 and config['patch_mask_probability'] == .1
    assert config['task_utility']['selection'] == 'utility'
    assert config['reverse_weight'] == 0. and config['freeze_all_bn']
    assert config['curriculum_arm'] in P_ARMS
    assert isinstance(config['mask_seed'], int)
    validate_parents(config)
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
    root = Path(config['output'])
    root.mkdir(parents=True, exist_ok=False)
    base.save(root/'run_config.json', config)
    base.save(root/'data_boundary.json', dict(partitions=partitions,
        excluded_frames={key: sorted(value) for key, value in used.items()},
        prior_sha256=hashes, train_list_sha256=base.prior.common.sha256(names_path)))
    sources = sorted(Path('miss_rgbt').rglob('*.py')) + sorted(Path('lib').rglob('*.py'))
    sources += [Path(base.__file__), Path(base.prior.__file__),
                Path(base.prior.common.__file__)]
    sources += sorted(Path(__file__).parent.glob('*.py'))
    sources += [PROJECT/'research/curriculum_p/schedule.py']
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
        model = build_model(base.cfg, config['checkpoint'], config['task_utility'],
            config['feedback']['mode'], config['feedback']['width']).cuda().train()
        base.fix_bn(model, True)
        shared = {name: value for name, value in model.named_parameters()
                  if not name.startswith('feedback.')}
        base.save(root/'initial_state.json', dict(
            shared_parameters_sha256=base.prior.common.digest(shared),
            parameters_sha256=base.prior.common.digest(dict(model.named_parameters())),
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
                microstep = (epoch-1)*len(loader)+index
                available, patch = step_masks(templates, searches, local, microstep)
                probe, view = None, None
                if microstep % config['utility_probe_interval'] == 0:
                    probe_id = microstep//config['utility_probe_interval']
                    view = 'C' if probe_id % 2 == 0 else 'M'
                    regions = torch.randint(model.utility_repair.region_tokens.shape[0],
                        (searches[0].shape[0], 1), generator=generator).cuda()
                    probe = dict(search_index=(probe_id//2) % len(searches), regions=regions)
                loss, status = batch_loss(model, templates, searches, boxes, available,
                                          patch, base.cfg, config, probe, view)
                assert torch.isfinite(loss), status
                (loss/config['accumulation']).backward()
                status.update(epoch=epoch, microstep=microstep, loss=float(loss.detach()),
                    mask_seed=config['mask_seed']+microstep,
                    availability_probability=local['complete_rgb_missing_tir_missing_probability'],
                    input_sha256=record['tensors_sha256'],
                    masks_sha256=[base.tensor_hash(value) for value in available+patch])
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
            result.update(epoch=epoch, updates=updates, seconds=time.time()-started)
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
            feedback=config['feedback'], curriculum_arm=config['curriculum_arm'],
            final_five=[dict(epoch=row['epoch'], path=str(root/('epoch%02d.pth.tar' % row['epoch'])),
                sha256=base.prior.common.sha256(root/('epoch%02d.pth.tar' % row['epoch'])))
                for row in history[-5:]],
            selected_epoch=selected['epoch'], selected_checkpoint=str(selected_path),
            selected_sha256=base.prior.common.sha256(selected_path),
            selection_rule='Lowest fixed train882 development loss in epochs16-20; earlier tie',
            all_checkpoints_retained=True, official_test_selection_used=False,
            scope='F training control; not an official result or validated module'))


if __name__ == '__main__':
    main()
