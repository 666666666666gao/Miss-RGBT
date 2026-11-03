"""Replay remaining native image epochs on CPU; never authorize OHCAS by itself."""
import argparse
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
from replay_protocol_r_epoch_boundary import record, seed_only_epoch
from replay_protocol_r_input_witness import FrameCapture
from research.protocol_r.common import REPORT, RUNS, SNAPSHOT, read, save, sha, tensor_sha


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--witness', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    output, witness = Path(args.output), Path(args.witness)
    assert not output.exists() and not torch.cuda.is_initialized()
    prior = read(witness / 'report.json')
    done = read(witness / 'completed.json')
    assert prior['status'] == 'passed' and done['status'] == 'complete'
    assert sha(witness / 'report.json') == done['report_sha256']
    assert prior['epoch2_prefix_bitwise_and_identity_equal'] is True
    assert prior['first_update_hashes_matched'] is True
    assert prior['exhausted_native_epoch_microbatches'] == 5000
    assert prior['returned_epoch1_tuples'] == 20000
    assert sha(witness / 'epoch01_input_manifest.jsonl') == prior['epoch1_manifest_sha256']
    sources = dict(prior['source_sha256'])
    sources.update(read(REPORT / 'source_sha256.json'))
    sources[str(Path(__file__).resolve())] = sha(__file__)
    for name, digest in sources.items():
        assert sha(name) == digest, name
    configs = {arm: read(RUNS / arm / 'run_config.json') for arm in ('r1', 'r2')}
    config = configs['r1']
    for arm, other in configs.items():
        assert sha(RUNS / arm / 'run_config.json') == prior['run_config_sha256'][arm]
        for key in ('dataset', 'seed', 'workers', 'microbatch', 'samples_per_epoch', 'epochs'):
            assert other[key] == config[key], (arm, key)
    assert config['epochs'] == 20 and config['workers'] == 2
    assert config['samples_per_epoch'] == 20000 and config['microbatch'] == 4
    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    update_config_from_file(str(SNAPSHOT / 'experiments/cadtrack/cadtrack.yaml'))
    random.seed(config['seed'])
    np.random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    loader = make_loader(cfg, config['dataset'], 'train', config['microbatch'],
                         config['samples_per_epoch'], config['workers'])
    capture = FrameCapture(loader.dataset.datasets[0], loader.dataset.processing)
    loader.dataset.datasets[0].get_frames = capture.frames
    loader.dataset.processing = capture.process
    assert tensor_sha(torch.get_rng_state()) == prior['initial_cpu_rng_sha256']
    first_seed, state = seed_only_epoch(loader)
    assert first_seed == prior['seed_only_epoch_worker_seeds'][0]
    assert tensor_sha(state) == prior['after_epoch_cpu_rng_sha256'][0]
    output.mkdir()
    save(output / 'registration.json', dict(pid=os.getpid(), witness=str(witness),
        witness_report_sha256=sha(witness / 'report.json'),
        run_config_sha256=prior['run_config_sha256'], source_sha256=sources,
        epochs=list(range(2, 21)), microbatches_per_epoch=5000,
        original_epoch1_manifest=str(witness / 'epoch01_input_manifest.jsonl'),
        original_epoch1_manifest_sha256=prior['epoch1_manifest_sha256'],
        cpu_only=True, actual_original_inputs_independently_certified=False,
        ohcas_data_gate_passed=False, starts_training=False))
    started, epochs = time.time(), []
    for epoch in range(2, 21):
        for name, digest in sources.items():
            assert sha(name) == digest, name
        # Derive the expected native seed/RNG boundary without changing this replay.
        before = torch.get_rng_state().clone()
        expected_seed, expected_state = seed_only_epoch(loader)
        torch.set_rng_state(before)
        manifest = output / ('epoch%02d_input_manifest.jsonl' % epoch)
        iterator = iter(loader)
        assert iterator._base_seed == expected_seed
        count = 0
        with manifest.open('x') as stream:
            for index, data in enumerate(iterator):
                row = record(data, epoch, index)
                assert len(row['identities']) == 4
                assert torch.equal(torch.get_rng_state(), expected_state), (epoch, index)
                if epoch == 2 and index < len(prior['epoch2_prefix']):
                    assert row == prior['epoch2_prefix'][index], index
                stream.write(json.dumps(row) + '\n')
                count += 1
                if count % 250 == 0:
                    stream.flush()
                    status = dict(stage='REPLAYING_IMAGES', pid=os.getpid(), epoch=epoch,
                        microbatches=count, total_microbatches=5000,
                        completed_epochs=[row['epoch'] for row in epochs],
                        seconds=time.time() - started, cuda_initialized=False,
                        ohcas_data_gate_passed=False)
                    save(output / 'status.json', status)
                    print(json.dumps(status), flush=True)
        del iterator
        assert count == len(loader) == 5000
        assert torch.equal(torch.get_rng_state(), expected_state)
        entry = dict(epoch=epoch, microbatches=count, tuples=count * 4,
            worker_base_seed=expected_seed, cpu_rng_sha256=tensor_sha(expected_state),
            manifest=str(manifest), manifest_sha256=sha(manifest))
        epochs.append(entry)
        save(output / ('epoch%02d_completed.json' % epoch), entry)
    assert not torch.cuda.is_initialized()
    for name, digest in sources.items():
        assert sha(name) == digest, name
    original_state = {arm: dict(
        completed_exists=(RUNS / arm / 'completed.json').exists(),
        last_completed_epoch=read(RUNS / arm / 'development_history.json')[-1]['epoch'])
        for arm in configs}
    result = dict(status='complete', pid=os.getpid(), epochs=epochs,
        reconstructed_epochs=list(range(1, 21)), replayed_tuples_this_job=380000,
        witness=str(witness), witness_report_sha256=sha(witness / 'report.json'),
        epoch1_manifest=str(witness / 'epoch01_input_manifest.jsonl'),
        epoch1_manifest_sha256=prior['epoch1_manifest_sha256'],
        run_config_sha256=prior['run_config_sha256'], source_sha256=sources,
        original_training_state_at_completion=original_state,
        seconds=time.time() - started, cuda_initialized=False,
        actual_original_inputs_independently_certified=False,
        all_historical_frame_exclusions_verified=False, ohcas_data_gate_passed=False,
        scope='Twenty-epoch conditional native image-loader reconstruction, using the '
              'completed epoch-one witness plus epochs2-20. Original CUDA-training '
              'CPU-RNG noninterference and original terminal runs still need audit; '
              'future planned inputs may be included. No image cache, M1 decision, '
              'training, official tracking metrics or launch authorization.')
    save(output / 'report.json', result)
    save(output / 'completed.json', dict(status='complete',
        report_sha256=sha(output / 'report.json'), ohcas_data_gate_passed=False))
    save(output / 'status.json', dict(stage='COMPLETE', pid=os.getpid(),
        seconds=result['seconds'], ohcas_data_gate_passed=False))
    print(json.dumps(dict(status='complete', seconds=result['seconds'],
        reconstructed_epochs=20, ohcas_data_gate_passed=False)), flush=True)


if __name__ == '__main__':
    main()
