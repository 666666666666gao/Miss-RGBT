"""Replay recorded R input hashes on CPU and attach their source-frame identities."""
import argparse
import json
import random
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from lib.config.cadtrack.config import cfg, update_config_from_file
from miss_rgbt.data import make_loader
from research.protocol_r.common import REPORT, RUNS, SNAPSHOT, read, save, sha, tensor_sha


class FrameCapture:
    def __init__(self, dataset, processing):
        self.dataset = dataset
        self.get_frames = dataset.get_frames
        self.processing = processing
        self.calls = []

    def frames(self, sequence, frames, anno=None):
        self.calls.append(dict(sequence=self.dataset.sequence_list[sequence],
                               frames=list(frames)))
        return self.get_frames(sequence, frames, anno)

    def process(self, data):
        result = self.processing(data)
        assert len(self.calls) == 2
        templates, searches = self.calls
        assert templates['sequence'] == searches['sequence']
        if result['valid']:
            result['replayed_frame_identity'] = json.dumps(dict(
                sequence=templates['sequence'],
                template_frames=templates['frames'],
                search_frames=searches['frames']), sort_keys=True)
        self.calls = []
        return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    output = Path(args.output)
    assert not output.exists()
    started = time.time()
    configs = {arm: read(RUNS / arm / 'run_config.json') for arm in ('r1', 'r2')}
    config = configs['r1']
    for other in configs.values():
        for key in ('dataset', 'seed', 'workers', 'microbatch', 'samples_per_epoch'):
            assert config[key] == other[key], key
        assert other['workers'] == 2 and other['microbatch'] == 4
    expected = {}
    for arm in configs:
        with (RUNS / arm / 'steps.jsonl').open() as stream:
            first = json.loads(stream.readline())
        expected[arm] = first['first_update_input_sha256']
        assert len(expected[arm]) == configs[arm]['accumulation']
    sources = read(REPORT / 'source_sha256.json')
    import lib.train.data.processing as processing
    import lib.train.data.sampler as sampler
    import lib.train.data.transforms as transforms
    import lib.train.dataset.lasher as lasher
    import miss_rgbt.data as native_data
    selected_sources = [Path(module.__file__).resolve()
                        for module in (processing, sampler, transforms, lasher, native_data)]
    for path in selected_sources:
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
    records = []
    iterator = iter(loader)
    for index in range(max(len(value) for value in expected.values())):
        data = next(iterator)
        actual = {key: tensor_sha(data[key])
                  for key in ('template_images', 'search_images', 'search_anno')}
        checked = []
        for arm, rows in expected.items():
            if index < len(rows):
                assert actual == rows[index], (arm, index, actual, rows[index])
                checked.append(arm)
        identities = [json.loads(text) for text in data['replayed_frame_identity']]
        assert len(identities) == config['microbatch']
        records.append(dict(microbatch=index + 1, tensor_sha256=actual,
                            matches_recorded_arms=checked, identities=identities))
    del iterator
    assert not torch.cuda.is_initialized()
    report = dict(status='passed', seconds=time.time() - started,
        replayed_microbatches=len(records), returned_tuples=len(records)*config['microbatch'],
        cuda_initialized=False, records=records,
        run_config_sha256={arm: sha(RUNS / arm / 'run_config.json') for arm in configs},
        source_sha256={str(path): sha(path) for path in selected_sources + [Path(__file__).resolve()]},
        scope='Real native CPU image-loader prefix replay, matching every recorded R1/R2 '
              'first-update input hash. Frame capture adds metadata only. Worker prefetch '
              'may read additional images. Not a full-run frame manifest, epoch-boundary '
              'RNG verification, OHCAS data gate, training, or official metric.')
    save(output, report)
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
