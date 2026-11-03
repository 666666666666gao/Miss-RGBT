"""Isolated OSTrack-initialized protocol replication."""
import hashlib
import json
from pathlib import Path
import random

import numpy as np
import torch

ROOT = Path('/home/gaob/Miss-RGBT')
REPORT = ROOT / 'reports/server28_protocol_r_20261103'
RUNS = ROOT / 'runs/server28_protocol_r_20261103'
SNAPSHOT = ROOT / 'runtime/protocol_r_20261103'
INITIALIZER = Path('/home/gaob/rgbx-initializers/OSTrack_ep0300.pth.tar')
INITIALIZER_SHA = '8e01d6251569ac84dbb53fdd062cd938551be2d889582f789aa3070196f42f41'
MISSING = [
    'backbone.cls_pos_embed',
    'decode_fuse_search.0.weight', 'decode_fuse_search.0.bias',
    'decode_fuse_search.1.weight', 'decode_fuse_search.1.bias',
    'decode_fuse_search.1.running_mean', 'decode_fuse_search.1.running_var',
]


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + '\n')


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def seed(value):
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    torch.cuda.manual_seed_all(value)


def tensor_sha(value):
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def check_sources():
    for path, expected in read(REPORT / 'source_sha256.json').items():
        assert sha(path) == expected, path


def build(config):
    from lib.config.cadtrack.config import cfg, update_config_from_file
    from lib.models.cadtrack import build_cadtrack
    from miss_rgbt.model import MissingTrackerModel
    update_config_from_file(str(SNAPSHOT / 'experiments/cadtrack/cadtrack.yaml'))
    seed(config['seed'])
    core = build_cadtrack(cfg, training=False)
    saved = torch.load(str(INITIALIZER), map_location='cpu')
    result = core.load_state_dict(saved['net'], strict=False)
    assert result.missing_keys == MISSING and not result.unexpected_keys, result
    model = MissingTrackerModel(core, repair=False, width=128, repair_layer=6)
    return model, dict(missing_keys=result.missing_keys,
                      unexpected_keys=result.unexpected_keys,
                      initializer_sha256=sha(INITIALIZER),
                      parameters_sha256=hashlib.sha256(b''.join(
                          p.detach().cpu().numpy().tobytes()
                          for p in model.parameters())).hexdigest())
