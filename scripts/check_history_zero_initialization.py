"""Check literal-zero replacement against native block-three missing tokens."""
import argparse
import json
from pathlib import Path
import time

import torch

from diagnose_absent_position import digest, sha256
from diagnose_history_stream import CHECKPOINT, CHECKPOINT_SHA
from lib.config.cadtrack.config import cfg, update_config_from_file
from lib.models.cadtrack import build_cadtrack
from miss_rgbt.model import MissingTrackerModel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    destination = Path(args.output)
    assert not destination.exists() and not torch.cuda.is_initialized()
    started = time.time()
    torch.set_num_threads(1)
    assert sha256(CHECKPOINT) == CHECKPOINT_SHA
    saved = torch.load(CHECKPOINT, map_location='cpu')
    update_config_from_file('experiments/cadtrack/cadtrack.yaml')
    model = MissingTrackerModel(build_cadtrack(cfg, training=False),
                                task_utility=saved['config']['task_utility'])
    model.load_state_dict(saved['model'], strict=True)
    del saved
    model.eval().requires_grad_(False)
    before = digest(model.state_dict())
    templates = [torch.zeros(1, 6, 128, 128) for _ in range(2)]
    search = torch.zeros(1, 6, 256, 256)
    count = model.baseline.backbone.pos_embed_x.shape[1]
    rows = []
    for missing, flags in enumerate(([0., 1.], [1., 0.])):
        captured = []

        def capture(block, inputs, outputs):
            captured.append(tuple(value[:, -count:].detach().clone() for value in outputs))

        handle = model.baseline.backbone.blocks[2].register_forward_hook(capture)
        available = [torch.ones(1, 2) for _ in templates] + [torch.tensor([flags])]
        with torch.no_grad():
            model(templates, [search], available)
        handle.remove()
        assert len(captured) == 1
        tokens = captured[0][missing]
        assert tokens.shape == (1, 256, 768) and torch.isfinite(tokens).all()
        nonzero = int(torch.count_nonzero(tokens))
        assert nonzero > 0
        rows.append(dict(missing_modality='rgb' if missing == 0 else 'tir',
            shape=list(tokens.shape), nonzero_elements=nonzero,
            max_absolute_value=float(tokens.abs().max()),
            root_mean_square=float(tokens.square().mean().sqrt()),
            native_equals_literal_zero=torch.equal(tokens, torch.zeros_like(tokens))))
    assert digest(model.state_dict()) == before
    assert all(parameter.grad is None for parameter in model.parameters())
    assert not model.baseline.backbone.blocks[2]._forward_hooks
    assert not torch.cuda.is_initialized()
    sources = [Path(__file__).resolve(),
               Path('miss_rgbt/model.py').resolve(),
               Path('lib/models/cadtrack/vit_care.py').resolve(),
               Path('experiments/cadtrack/cadtrack.yaml').resolve()]
    report = dict(status='passed', rows=rows, seconds=time.time() - started,
        checkpoint=CHECKPOINT, checkpoint_sha256=CHECKPOINT_SHA,
        strict_native_checkpoint_load=True, model_state_unchanged=True,
        gradients_absent=True, cuda_initialized=False, real_images_executed=False,
        source_sha256={str(path): sha256(path) for path in sources},
        ohcas_data_gate_passed=False, ohcas_training_authorized=False,
        finding='A zero-valued replacement at block3 is not the native missing path, '
                'even with synthetic all-zero normalized images. Zero initialization '
                'of a replacement map alone therefore cannot prove native parity.',
        scope='Synthetic structural counterexample on the frozen registered model. '
              'No image intervention cohort, data-gate decision, module architecture '
              'change, optimization, convergence result or official tracking metric.')
    with destination.open('x') as stream:
        json.dump(report, stream, indent=2)
        stream.write('\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
