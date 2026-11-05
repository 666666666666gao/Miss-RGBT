"""Freeze R1/R2 sources, shared development tuples, and training plans."""
import shutil

import cv2
import torch

from research.protocol_r.common import (
    ROOT, REPORT, RUNS, SNAPSHOT, INITIALIZER, INITIALIZER_SHA,
    save, sha, seed, tensor_sha,
)


def main():
    REPORT.mkdir()
    SNAPSHOT.mkdir()
    frozen = ROOT / 'runtime/task_utility_20261018'
    for folder in ['lib', 'miss_rgbt', 'experiments/cadtrack']:
        shutil.copytree(str(frozen / folder), str(SNAPSHOT / folder),
                        ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    shutil.copytree(str(ROOT / 'research/protocol_r'),
                    str(SNAPSHOT / 'research/protocol_r'))
    assert sha(INITIALIZER) == INITIALIZER_SHA
    source = {str(p): sha(p) for p in sorted(SNAPSHOT.rglob('*'))
              if p.is_file() and p.suffix in ['.py', '.yaml', '.txt']}
    for name in ['evaluate_candidate_official28.py', 'verify_recovery_campaign28.py',
                 'evaluate_val97_recovery28.py', 'analyze_epoch05_health28.py']:
        source[str(ROOT / name)] = sha(ROOT / name)
    save(REPORT / 'source_sha256.json', source)
    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    from lib.config.cadtrack.config import cfg, update_config_from_file
    from miss_rgbt.data import make_loader
    update_config_from_file(str(SNAPSHOT / 'experiments/cadtrack/cadtrack.yaml'))
    dataset_root = '/data/wangwj/dataset/LasHeR/trainingset'
    train_names = (SNAPSHOT / 'lib/train/data_specs/lasher_train.txt').read_text().splitlines()
    val_names = (SNAPSHOT / 'lib/train/data_specs/lasher_val.txt').read_text().splitlines()
    assert len(train_names) == 882 and len(val_names) == 97
    assert not set(train_names).intersection(val_names)
    seed(731128)
    loader = make_loader(cfg, dataset_root, 'val', 4, 128, 0)
    dataset = loader.dataset.datasets[0]
    assert dataset.sequence_list == val_names
    calls = []
    original_frames = dataset.get_frames

    def frames(index, ids, anno=None):
        calls.append(dict(sequence=val_names[index], frames=[int(x) for x in ids]))
        return original_frames(index, ids, anno)

    dataset.get_frames = frames
    batches, manifest = [], []
    for index, data in enumerate(loader):
        batch = {key: data[key] for key in
                 ['template_images', 'search_images', 'search_anno']}
        batches.append(batch)
        manifest.append(dict(batch=index, frame_calls=list(calls),
            tensor_sha256={key: tensor_sha(value) for key, value in batch.items()}))
        calls.clear()
    assert len(batches) == 32
    cache = REPORT / 'development128.pth'
    torch.save(batches, str(cache))
    save(REPORT / 'development_manifest.json', dict(
        split='val97', samples=128, seed=731128, batches=manifest,
        cache_sha256=sha(cache), purpose='Fixed crop task loss only; no tracking scores',
        train_names=train_names, development_names=val_names))
    base = dict(stage='E3', init='ostrack', seed=2027, checkpoint=str(INITIALIZER),
        dataset=dataset_root, repair=False, repair_width=128, repair_layer=6,
        forward_weight=.1, reverse_weight=0., reverse_margin=.1,
        patch_mask_probability=.1,
        complete_rgb_missing_tir_missing_probability=[.4, .3, .3],
        accumulation=2, samples_per_epoch=20000, epochs=20, workers=2,
        learning_rate=1e-4, backbone_multiplier=.1, weight_decay=1e-4,
        lr_drop_epoch=10, warmup_updates=500, clip_grad_norm=.1,
        checkpoint_epochs=[10, 16, 17, 18, 19, 20], freeze_all_bn=False,
        development_cache=str(cache), development_cache_sha256=sha(cache),
        development_selection='Minimum mean 11/01/10 task loss among epochs16-20; earlier tie',
        official_test_selection_used=False)
    for arm, microbatch, gpu in [('r1', 4, 0), ('r2', 8, 1)]:
        config = dict(base, arm=arm, microbatch=microbatch, physical_gpu=gpu,
                      output=str(RUNS / arm))
        save(REPORT / (arm + '.json'), config)
    save(REPORT / 'registration.json', dict(status='prepared',
        source_manifest_sha256=sha(REPORT / 'source_sha256.json'),
        initializer_sha256=INITIALIZER_SHA, development_cache_sha256=sha(cache),
        epoch_budget=20, early_stop=False, official_epochs=[16, 17, 18, 19, 20],
        official_conditions=['complete', 'miss'], sequences=479, frames=337353,
        expected_missing_keys='CADTrack fusion and cls positional parameter; documented exact allowlist',
        old_p_scheduler_pid=2911513, old_p_scheduler_action='stopped while waiting; no children',
        successor_policy='P blocked pending R1/R2 full official protocol audit'))
    print('PREPARED R1/R2', flush=True)


if __name__ == '__main__':
    main()
