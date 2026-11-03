"""Reconstruct one completed R input epoch on CPU, without certifying provenance."""
import argparse
import json
from pathlib import Path
import random
import time

import cv2
import numpy as np
import torch

from lib.config.cadtrack.config import cfg, update_config_from_file
from miss_rgbt.data import make_loader
from replay_protocol_r_input_witness import FrameCapture
from research.protocol_r.common import REPORT, RUNS, SNAPSHOT, read, save, sha, tensor_sha


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()
    output = Path(args.output_dir)
    assert not output.exists()
    started = time.time()
    configs = {arm: read(RUNS / arm / 'run_config.json') for arm in ('r1', 'r2')}
    config = configs['r1']
    expected = {}
    observed_history = {}
    for arm, other in configs.items():
        for key in ('dataset', 'seed', 'workers', 'microbatch', 'samples_per_epoch'):
            assert config[key] == other[key], key
        assert other['workers'] == 2 and other['microbatch'] == 4
        history = read(RUNS / arm / 'development_history.json')
        assert history[0]['epoch'] == 1 and history[0]['updates'] == (
            config['samples_per_epoch'] // config['microbatch'] // other['accumulation'])
        observed_history[arm] = history[0]
        with (RUNS / arm / 'steps.jsonl').open() as stream:
            first = json.loads(stream.readline())
        expected[arm] = first['first_update_input_sha256']
        assert len(expected[arm]) == other['accumulation']

    import lib.train.data.processing as processing
    import lib.train.data.sampler as sampler
    import lib.train.data.transforms as transforms
    import lib.train.data.loader as native_loader
    import lib.train.dataset.lasher as lasher
    import miss_rgbt.data as native_data
    import torch.utils.data.dataloader as torch_loader
    import torch.utils.data.sampler as torch_sampler
    import torch.utils.data._utils.worker as torch_worker
    import replay_protocol_r_input_witness as witness
    modules = (processing, sampler, transforms, native_loader, lasher, native_data)
    sources = read(REPORT / 'source_sha256.json')
    for module in modules:
        path = Path(module.__file__).resolve()
        assert sha(path) == sources[str(path)], str(path)
    paths = [Path(module.__file__).resolve() for module in modules]
    paths += [Path(module.__file__).resolve() for module in
              (torch_loader, torch_sampler, torch_worker, witness)]
    paths += [Path(__file__).resolve(),
              SNAPSHOT / 'experiments/cadtrack/cadtrack.yaml',
              SNAPSHOT / 'research/protocol_r/train.py']
    assert sha(paths[-1]) == sources[str(paths[-1])]
    source_hashes = {str(path): sha(path) for path in paths}
    update_config_from_file(str(paths[-2]))
    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    random.seed(config['seed'])
    np.random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    initial_rng = torch.get_rng_state().clone()
    loader = make_loader(cfg, config['dataset'], 'train', config['microbatch'],
                         config['samples_per_epoch'], config['workers'])
    assert torch.equal(torch.get_rng_state(), initial_rng)
    assert len(loader) == 5000 and not loader.persistent_workers
    capture = FrameCapture(loader.dataset.datasets[0], loader.dataset.processing)
    loader.dataset.datasets[0].get_frames = capture.frames
    loader.dataset.processing = capture.process

    # Pinned Torch 1.12 draws a base seed and a RandomSampler seed per iterator.
    generator = torch.Generator()
    generator.set_state(initial_rng)
    predicted_base_seed = torch.empty((), dtype=torch.int64).random_(generator=generator).item()
    predicted_sampler_seed = torch.empty((), dtype=torch.int64).random_(generator=generator).item()
    predicted_next_rng = generator.get_state()
    output.mkdir(parents=True, exist_ok=False)
    save(output / 'registration.json', dict(
        status='registered', epoch=1, full_microbatches=len(loader),
        run_config_sha256={arm: sha(RUNS / arm / 'run_config.json') for arm in configs},
        observed_completed_epoch_one=observed_history,
        source_sha256=source_hashes, cuda_visible_devices='',
        freshness_gate_passed=False,
        scope='Candidate reconstruction only; first-update hashes are the only '
              'independently recorded native input witnesses.'))
    iterator = iter(loader)
    assert iterator._base_seed == predicted_base_seed
    assert torch.equal(torch.get_rng_state(), predicted_next_rng)
    rng_after_iterator = torch.get_rng_state().clone()
    seen = {}
    prefix_matches = {arm: 0 for arm in configs}
    count = 0
    with (output / 'candidate_input_manifest.jsonl').open('x') as stream:
        for index, data in enumerate(iterator):
            count += 1
            identities = [json.loads(text) for text in data['replayed_frame_identity']]
            assert len(identities) == config['microbatch']
            row = dict(epoch=1, microstep=count, identities=identities)
            if index < max(len(rows) for rows in expected.values()):
                actual = {key: tensor_sha(data[key]) for key in
                          ('template_images', 'search_images', 'search_anno')}
                row['tensor_sha256'] = actual
                row['matches_recorded_arms'] = []
                for arm, rows in expected.items():
                    if index < len(rows):
                        assert actual == rows[index], (arm, index)
                        prefix_matches[arm] += 1
                        row['matches_recorded_arms'].append(arm)
            for identity in identities:
                seen.setdefault(identity['sequence'], set()).update(
                    identity['template_frames'] + identity['search_frames'])
            assert torch.equal(torch.get_rng_state(), predicted_next_rng)
            stream.write(json.dumps(row, sort_keys=True) + '\n')
            if count % 250 == 0:
                stream.flush()
                elapsed = time.time() - started
                print(json.dumps(dict(microbatches=count, total=len(loader),
                    seconds=elapsed, estimated_remaining_seconds=elapsed/count*(
                        len(loader)-count))), flush=True)
    assert count == len(loader)
    assert prefix_matches == {arm: len(rows) for arm, rows in expected.items()}
    assert torch.equal(torch.get_rng_state(), predicted_next_rng)
    assert not torch.cuda.is_initialized()
    assert all(sha(path) == digest for path, digest in source_hashes.items())
    torch.save(dict(initial=initial_rng, after_iterator=rng_after_iterator,
                    after_epoch=torch.get_rng_state()), str(output / 'cpu_rng_states.pth'))
    save(output / 'candidate_frame_union.json',
         {name: sorted(frames) for name, frames in sorted(seen.items())})
    artifacts = ('registration.json', 'candidate_input_manifest.jsonl',
                 'candidate_frame_union.json', 'cpu_rng_states.pth')
    report = dict(status='candidate_replayed', epoch=1, microbatches=count,
        returned_tuples=count*config['microbatch'], seconds=time.time()-started,
        sequences=len(seen), unique_sequence_frames=sum(len(frames) for frames in seen.values()),
        independently_matched_prefix_microbatches=prefix_matches,
        base_seed=predicted_base_seed, sampler_seed=predicted_sampler_seed,
        replay_cpu_rng_draws=2, replay_cpu_rng_unchanged_after_iterator=True,
        cuda_initialized=False, all_historical_frame_exclusions_verified=False,
        freshness_gate_passed=False,
        artifact_sha256={name: sha(output / name) for name in artifacts},
        source_sha256=source_hashes,
        scope='Full native real-image epoch-one candidate reconstruction. Both live R '
              'runs had already completed this epoch. Loader replay RNG matches two '
              'main-CPU seed draws. Original trainer CPU RNG continuity and later-input '
              'identity are not independently observed. Not a verified historical '
              'manifest, OHCAS gate, training, or official metric.')
    save(output / 'report.json', report)
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
