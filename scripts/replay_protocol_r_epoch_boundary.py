"""Compare an exhausted native image epoch with a seed-only epoch transition."""
import argparse
import json
import random
from pathlib import Path
import time

import cv2
import numpy as np
import torch

from lib.config.cadtrack.config import cfg, update_config_from_file
from miss_rgbt.data import make_loader
from replay_protocol_r_input_witness import FrameCapture
from research.protocol_r.common import (
    REPORT, RUNS, SNAPSHOT, read, save, sha, tensor_sha,
)


def seed_only_epoch(loader):
    """Use the installed native sampler, consuming no dataset items or images."""
    assert loader.generator is None and loader.sampler.generator is None
    assert loader.num_workers == 2 and not loader.persistent_workers
    assert loader._get_shared_seed() is None
    base_seed = int(torch.empty((), dtype=torch.int64).random_().item())
    count = sum(1 for _ in loader.batch_sampler)
    assert count == len(loader)
    return base_seed, torch.get_rng_state().clone()


def record(data, epoch, index):
    return dict(epoch=epoch, index=index,
        tensors_sha256={key: tensor_sha(data[key]) for key in
                        ('template_images', 'search_images', 'search_anno')},
        identities=[json.loads(text) for text in data['replayed_frame_identity']])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    output = Path(args.output)
    assert not output.exists()
    started = time.time()
    configs = {arm: read(RUNS / arm / 'run_config.json') for arm in ('r1', 'r2')}
    config = configs['r1']
    completed_epochs, first = {}, {}
    for arm, other in configs.items():
        for key in ('seed', 'dataset', 'microbatch', 'samples_per_epoch', 'workers'):
            assert other[key] == config[key], key
        history = read(RUNS / arm / 'development_history.json')
        assert [row['epoch'] for row in history] == list(range(1, len(history) + 1))
        assert len(history) >= 2
        completed_epochs[arm] = len(history)
        with (RUNS / arm / 'steps.jsonl').open() as stream:
            first[arm] = json.loads(stream.readline())['first_update_input_sha256']
        assert len(first[arm]) == other['accumulation']
    assert config['workers'] == 2 and config['microbatch'] == 4
    sources = read(REPORT / 'source_sha256.json')
    import lib.train.data.loader as native_loader
    import lib.train.data.processing as processing
    import lib.train.data.sampler as sampler
    import lib.train.data.transforms as transforms
    import lib.train.dataset.lasher as lasher
    import miss_rgbt.data as native_data
    paths = [Path(module.__file__).resolve() for module in
             (native_loader, processing, sampler, transforms, lasher, native_data)]
    paths += [SNAPSHOT / 'research/protocol_r/train.py']
    for path in paths:
        assert sha(path) == sources[str(path)], str(path)
    update_config_from_file(str(SNAPSHOT / 'experiments/cadtrack/cadtrack.yaml'))
    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    random.seed(config['seed'])
    np.random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    loader = make_loader(cfg, config['dataset'], 'train', config['microbatch'],
                         config['samples_per_epoch'], config['workers'])
    capture = FrameCapture(loader.dataset.datasets[0], loader.dataset.processing)
    loader.dataset.datasets[0].get_frames = capture.frames
    loader.dataset.processing = capture.process
    initial = torch.get_rng_state().clone()
    base1, after1 = seed_only_epoch(loader)
    base2, after2 = seed_only_epoch(loader)
    torch.set_rng_state(initial)
    output.mkdir()
    iterator = iter(loader)
    assert iterator._base_seed == base1
    count = 0
    with (output / 'epoch01_input_manifest.jsonl').open('x') as stream:
        for index, data in enumerate(iterator):
            row = record(data, 1, index)
            assert len(row['identities']) == config['microbatch']
            for expected in first.values():
                if index < len(expected):
                    assert row['tensors_sha256'] == expected[index], index
            assert torch.equal(torch.get_rng_state(), after1), index
            stream.write(json.dumps(row) + '\n')
            count += 1
            if count % 250 == 0:
                stream.flush()
                print(json.dumps(dict(epoch=1, microbatches=count,
                    total_microbatches=len(loader), seconds=time.time() - started)), flush=True)
    del iterator
    assert count == len(loader) and torch.equal(torch.get_rng_state(), after1)
    prefix = []
    iterator = iter(loader)
    assert iterator._base_seed == base2
    for index in range(4):
        prefix.append(record(next(iterator), 2, index))
        assert torch.equal(torch.get_rng_state(), after2)
    del iterator
    # Jump directly from the initial seed through one image-free native epoch.
    torch.set_rng_state(initial)
    jumped_seed, jumped_state = seed_only_epoch(loader)
    assert jumped_seed == base1 and torch.equal(jumped_state, after1)
    iterator = iter(loader)
    assert iterator._base_seed == base2
    for index, expected in enumerate(prefix):
        assert record(next(iterator), 2, index) == expected, index
        assert torch.equal(torch.get_rng_state(), after2)
    del iterator
    assert not torch.cuda.is_initialized()
    import inspect
    from timm.models.layers import DropPath
    paths += [Path(__file__).resolve(),
              Path(__file__).with_name('replay_protocol_r_input_witness.py').resolve(),
              Path(inspect.getfile(torch.utils.data.DataLoader)).resolve(),
              Path(inspect.getfile(torch.utils.data.RandomSampler)).resolve(),
              Path(inspect.getfile(DropPath)).resolve()]
    report = dict(status='passed', seconds=time.time() - started,
        torch_version=torch.__version__, completed_training_epochs_at_start=completed_epochs,
        exhausted_native_epoch_microbatches=count, returned_epoch1_tuples=count * config['microbatch'],
        first_update_hashes_matched=True, epoch1_main_cpu_rng_unchanged_after_prefetch=True,
        epoch2_prefix_bitwise_and_identity_equal=True, epoch2_prefix_microbatches=4,
        seed_only_epoch_worker_seeds=[base1, base2],
        initial_cpu_rng_sha256=tensor_sha(initial),
        after_epoch_cpu_rng_sha256=[tensor_sha(after1), tensor_sha(after2)],
        epoch1_manifest_sha256=sha(output / 'epoch01_input_manifest.jsonl'),
        epoch2_prefix=prefix, cuda_initialized=False,
        run_config_sha256={arm: sha(RUNS / arm / 'run_config.json') for arm in configs},
        source_sha256={str(path): sha(path) for path in paths},
        all_historical_frame_exclusions_verified=False, ohcas_data_gate_passed=False,
        scope='Real native CPU image-loader epoch/transition witness. Original trainers '
              'were not instrumented: later original inputs and training CPU-RNG '
              'noninterference are not independently certified. Not a full historical '
              'manifest, image cache, OHCAS decision, training or official metric.')
    save(output / 'report.json', report)
    save(output / 'completed.json', dict(status='complete',
        report_sha256=sha(output / 'report.json'), seconds=report['seconds']))
    print(json.dumps({key: value for key, value in report.items()
                     if key not in ('epoch2_prefix', 'source_sha256')}), flush=True)


if __name__ == '__main__':
    main()
