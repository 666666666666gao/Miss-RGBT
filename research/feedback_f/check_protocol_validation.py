"""CPU diagnostic arithmetic and optional real OSTrack forward checks."""
import argparse
import gc
from pathlib import Path

import torch

from lib.config.cadtrack.config import cfg, update_config_from_file
from lib.models.layers.head import CenterPredictor
from miss_rgbt.objectives import task_loss
from research.feedback_f.protocol_training import ARMS, build
from research.feedback_f.protocol_validation import compare_rounds, crop_iou, validate
from research.protocol_r.common import INITIALIZER_SHA, save, sha


class Selector(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('selection_rng_state', torch.tensor([7]))

    def reset_selection(self):
        self.selection_rng_state.zero_()


class DiagnosticModel(torch.nn.Module):
    def __init__(self, arm):
        super().__init__()
        self.arm = arm
        self.weight = torch.nn.Parameter(torch.tensor(0.))
        self.utility_repair = Selector()
        self.baseline = torch.nn.Module()
        self.baseline.box_head = CenterPredictor(8, 8, 16, 16)
        self.calls = []

    def prediction(self, batch, centers):
        score = torch.full((batch, 1, 16, 16), .1)
        score[:, :, 8, 8] = .9
        # Raw argmax deliberately differs from the tracker's Hann argmax.
        score[:, :, 15, 15] = .95
        sizes = torch.full((batch, 2, 16, 16), .25)
        offsets = torch.zeros(batch, 2, 16, 16)
        offsets[:, 0, 8, 8] = (centers - .5) * 16
        return dict(score_map=score, size_map=sizes, offset_map=offsets,
            pred_boxes=self.baseline.box_head.cal_bbox(score, sizes, offsets).unsqueeze(1))

    def forward(self, templates, searches, available):
        assert not self.training and not torch.is_grad_enabled()
        assert len(templates) == 2 and len(searches) == 3
        self.calls.append([flags.clone() for flags in available])
        self.utility_repair.selection_rng_state.add_(1)
        torch.rand(5)
        batch = searches[0].shape[0]
        assert batch % 4 == 0
        centers1 = torch.tensor([.5, .05, .5, .05]).repeat(batch // 4)
        centers2 = torch.tensor([.05, .5, .5, .05]).repeat(batch // 4)
        outputs = []
        for _ in searches:
            first = self.prediction(batch, centers1)
            if self.arm == 'single':
                outputs.append(first)
            else:
                final = self.prediction(batch, centers2)
                final['round1'] = first
                outputs.append(final)
        return outputs


def check_diagnostics():
    boxes = torch.tensor([[[.375, .375, .25, .25]] * 64] * 3)
    data = dict(template_images=torch.ones(2, 64, 1),
                search_images=torch.ones(3, 64, 1), search_anno=boxes)
    results = {}
    for arm in ARMS:
        model = DiagnosticModel(arm).train()
        model.baseline.box_head.eval()
        modes = [module.training for module in model.modules()]
        state = {name: value.clone() for name, value in model.state_dict().items()}
        rng = torch.get_rng_state().clone()
        report = validate(model, [data, data], cfg, arm)
        assert torch.equal(rng, torch.get_rng_state())
        assert modes == [module.training for module in model.modules()]
        assert all(torch.equal(value, model.state_dict()[name])
                   for name, value in state.items())
        assert len(model.calls) == 6 and report['samples'] == 128
        for call, flags in zip(model.calls, ([1., 1.], [0., 1.], [1., 0.]) * 2):
            assert all(value.eq(1).all() for value in call[:2])
            assert all(value.eq(torch.tensor(flags)).all() for value in call[2:])
        final = model.prediction(64, torch.tensor([.05, .5, .5, .05]).repeat(16))
        raw = [final] * 3 if arm != 'single' else [
            model.prediction(64, torch.tensor([.5, .05, .5, .05]).repeat(16))] * 3
        expected = float(task_loss(raw, boxes, cfg).sum()) / 64
        assert all(abs(value - expected) < 1e-6 for value in report['task_loss'].values())
        assert abs(report['mean_task_loss'] - expected) < 1e-6
        if arm == 'single':
            assert 'round_comparison' not in report and 'round_samples' not in report
        else:
            for label in ('11', '01', '10'):
                stats = report['round_comparison'][label]
                assert stats['frames'] == 384
                assert stats['corrected'] == stats['destroyed'] == 96
                assert stats['higher_iou'] == stats['lower_iou'] == 96
                assert stats['equal_iou'] == 192
                assert stats['round1_mean_iou'] == stats['round2_mean_iou'] == .5
                assert stats['correction_fraction'] == stats['destruction_fraction'] == .25
                assert stats['correction_rate_given_failure'] == .5
                assert stats['destruction_rate_given_success'] == .5
                for row in report['round_samples'][label]:
                    assert row['round1_iou'] == [[1., 0., 1., 0.] * 16] * 3
                    assert row['round2_iou'] == [[0., 1., 1., 0.] * 16] * 3
        results[arm] = dict(native_hann_decode=True, final_task_loss_exact=True,
            rng_selector_parameters_bn_and_training_modes_preserved=True,
            all_three_states=True, cached_layout_128_tuples=True)
    boundary = compare_rounds([.49, .5, .75, .1], [.5, .49, .75, .2])
    assert boundary['corrected'] == boundary['destroyed'] == 1
    assert boundary['higher_iou'] == 2 and boundary['lower_iou'] == 1
    assert boundary['equal_iou'] == 1
    assert compare_rounds([0.], [1.])['destruction_rate_given_success'] is None
    assert compare_rounds([1.], [0.])['correction_rate_given_failure'] is None
    return results


def check_native(initializer):
    common = dict(init='ostrack', seed=65511, checkpoint=initializer,
        initializer_sha256=INITIALIZER_SHA,
        task_utility=dict(width=128, depth=1, region_grid=4, selected_regions=4,
                         refinement_limit=.1, selection='utility', selection_seed=65511))
    torch.manual_seed(65512)
    templates = [torch.randn(1, 6, 128, 128) for _ in range(2)]
    searches = [torch.randn(1, 6, 256, 256) for _ in range(3)]
    boxes = torch.tensor([[[.375, .375, .25, .25]]] * 3)
    reference, results = None, {}
    for arm in ARMS:
        model, _ = build(dict(common, feedback=dict(mode=arm, width=128)))
        model.eval()
        current = {}
        with torch.no_grad():
            for label, flags in [('11', [1., 1.]), ('01', [0., 1.]), ('10', [1., 0.])]:
                available = [torch.ones(1, 2) for _ in templates]
                available += [torch.tensor([flags]) for _ in searches]
                outputs = model(templates, searches, available)
                final_loss = task_loss(outputs, boxes, cfg)
                final_iou = crop_iou(outputs, boxes, model.baseline.box_head)
                assert torch.isfinite(final_loss).all()
                assert not final_loss.requires_grad and not final_iou.requires_grad
                if arm != 'single':
                    first = [out['round1'] for out in outputs]
                    assert torch.equal(task_loss(first, boxes, cfg), final_loss)
                    assert torch.equal(crop_iou(first, boxes, model.baseline.box_head), final_iou)
                current[label] = dict(
                    loss=final_loss.tolist(), iou=final_iou.tolist())
        if arm == 'single':
            reference = current
        assert current == reference
        results[arm] = current
        print('NATIVE_CPU_FORWARD ' + arm, flush=True)
        del model, outputs
        gc.collect()
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--initializer')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    assert not Path(args.output).exists() and not torch.cuda.is_initialized()
    torch.set_num_threads(1)
    update_config_from_file('experiments/cadtrack/cadtrack.yaml')
    checks = check_diagnostics()
    native = check_native(args.initializer) if args.initializer else None
    assert not torch.cuda.is_initialized()
    sources = [Path(__file__), Path(__file__).with_name('protocol_validation.py'),
        Path(__file__).with_name('protocol_training.py'), Path(__file__).with_name('model.py'),
        Path('lib/utils/box_ops.py'), Path('lib/test/utils/hann.py'),
        Path('lib/models/layers/head.py'), Path('miss_rgbt/objectives.py')]
    save(args.output, dict(status='PASS', checks=checks, native_forward=native,
        cuda_initialized=False, threshold_boundary_and_zero_denominators=True,
        scope='CPU synthetic cache diagnostics; native OSTrack forwards only if supplied',
        source_sha256={str(path.resolve()): sha(path) for path in sources},
        limitations=['No real images, learned feedback or CUDA smoke',
            'Not a full val97 model evaluation, training, latency or official tracking result',
            'F parent preparation, trainer and complete official dispatch remain unimplemented']))
    print('PASS: diagnostic arithmetic and state preservation', flush=True)


if __name__ == '__main__':
    main()
