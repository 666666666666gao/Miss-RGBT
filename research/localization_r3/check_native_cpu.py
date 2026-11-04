"""Actual native OSTrack initialization and strict tracker-load CPU contract."""
import argparse
import gc
import json
from pathlib import Path
import tempfile
from unittest.mock import patch

import torch

from lib.config.cadtrack.config import cfg, update_config_from_file
from miss_rgbt.tracker import MissingTracker
from research.localization_r3.prepare import ARMS, CAMPAIGN, SNAPSHOT, RUNS, make_config
from research.localization_r3.train import bn_state, build
from research.protocol_r.common import REPORT, read, sha, tensor_sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    update_config_from_file('experiments/cadtrack/cadtrack.yaml')
    parent = read(REPORT / 'r1.json')
    initials, counts = [], []
    with tempfile.TemporaryDirectory(prefix='r3_native_cpu_') as directory:
        for arm in ARMS:
            config = make_config(arm, parent, CAMPAIGN, SNAPSHOT, RUNS,
                                 'not-used-by-this-cpu-interface-check')
            assert sha(config['checkpoint']) == config['initializer_sha256']
            model, initial = build(config)
            initials.append(initial)
            counts.append(sum(value.numel() for value in model.parameters()))
            assert model.utility_repair is None
            checkpoint = Path(directory) / (arm + '.pth.tar')
            torch.save(dict(model=model.state_dict(), config=config), str(checkpoint))
            del model
            gc.collect()
            # Exercise the unchanged native tracker, explicitly mocking placement.
            with patch.object(torch.nn.Module, 'cuda', lambda self: self), \
                    patch.object(torch.Tensor, 'cuda', lambda self: self):
                tracker = MissingTracker(cfg, str(checkpoint))
            actual = {name: tensor_sha(value)
                      for name, value in tracker.network.named_parameters()}
            assert actual == initial['parameters_sha256']
            assert bn_state(tracker.network) == initial['bn_sha256']
            assert tracker.network.utility_repair is None
            del tracker
            gc.collect()
    assert initials[0] == initials[1] and counts[0] == counts[1]
    assert not torch.cuda.is_initialized()
    source = Path(__file__).resolve().parent
    result = dict(status='passed', parameters=counts[0],
        matched_parameter_tensors=len(initials[0]['parameters_sha256']),
        matched_bn_buffers=len(initials[0]['bn_sha256']),
        native_tracker_strict_loads=2, cuda_placement_mocked=True,
        cuda_initialized=False, initializer_sha256=initials[0]['initializer_sha256'],
        initial_state=initials[0], formal_training_started=False,
        source_sha256={str(source / name): sha(source / name) for name in
                       ('prepare.py', 'train.py', 'check_native_cpu.py')},
        scope='Actual native model initialization and strict tracker loading on CPU; '
              'not real-image training, GPU memory, tracking scores or efficacy')
    with Path(args.output).open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps({key: value for key, value in result.items()
                      if key not in ('initial_state', 'source_sha256')}, indent=2))


if __name__ == '__main__':
    main()
