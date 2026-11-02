"""Inspect the actual cached-input Complete parity failure before any update."""
import json
from pathlib import Path

import torch

from lib.config.cadtrack.config import cfg, update_config_from_file
from miss_rgbt.model import load_pretrained
from probe_cross_stream_context import CHECKPOINT, MAPS, ContextResidual, output_from_map, seed

torch.set_num_threads(1)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
update_config_from_file('experiments/cadtrack/cadtrack.yaml')
model = load_pretrained(cfg, CHECKPOINT, repair=False).cuda().eval().requires_grad_(False)
path = Path('/home/gaob/Miss-RGBT/reports/server28_cross_stream_context_20261102/development/features.pth')
cache = torch.load(str(path), map_location='cpu')
raw = cache['features'][:16].cuda()
flags = cache['flags'][:16].cuda()
assert flags[:, -2:].eq(1).all()
seed(2027)
adapter = ContextResidual(model.baseline.backbone.embed_dim,
                          model.baseline.feat_len_s, 'aligned').cuda()
with torch.no_grad():
    changed = adapter(raw, flags)
    native = output_from_map(model, raw)
    repeated = output_from_map(model, raw)
    altered = output_from_map(model, changed)
    contiguous = output_from_map(model, changed.contiguous())
    print(json.dumps(dict(input_equal=torch.equal(raw, changed),
        native_stride=raw.stride(), adapter_stride=changed.stride(),
        native_contiguous=raw.is_contiguous(), adapter_contiguous=changed.is_contiguous(),
        repeated_equal={key: torch.equal(native[key], repeated[key]) for key in MAPS},
        original_equal={key: torch.equal(native[key], altered[key]) for key in MAPS},
        contiguous_equal={key: torch.equal(native[key], contiguous[key]) for key in MAPS},
        max_original_difference={key: float((native[key] - altered[key]).abs().max())
                                 for key in MAPS})), flush=True)
