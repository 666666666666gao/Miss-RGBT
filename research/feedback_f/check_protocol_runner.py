"""CPU synthetic optimizer checks; not an actual-image F smoke or run verifier."""
import argparse
from pathlib import Path

import torch

from lib.config.cadtrack.config import cfg, update_config_from_file
from miss_rgbt.objectives import utility_supervision
from research.curriculum_p.check_engine import compare_trees, inputs
from research.curriculum_p.engine import ARMS as CURRICULA, CurriculumState
from research.feedback_f.check_protocol_training import RoundModel, config
from research.feedback_f.protocol_runner import prediction_snapshot, set_learning_rates, step
from research.feedback_f.protocol_training import ARMS, forward_microbatch
from research.protocol_r.common import save, sha, tensor_sha


class ProbeModel(RoundModel):
    def __init__(self, arm):
        super().__init__(arm)
        self.utility_repair = torch.nn.Module()
        self.utility_repair.register_buffer('region_tokens', torch.arange(16)[:, None])
        self.utility_repair.logits = torch.nn.Parameter(torch.full((16,), .3))

    def prediction(self, image, mask, bias):
        out = super().prediction(image, mask, bias)
        out.update(size_map=out['score_map'].expand(-1, 2, -1, -1) * .25,
                   offset_map=out['score_map'].expand(-1, 2, -1, -1) * .1)
        return out

    def forward(self, templates, searches, availability=None, patch=None,
                utility_probe=None):
        outputs = super().forward(templates, searches, availability, patch, utility_probe)
        if utility_probe is not None:
            out = outputs[utility_probe['search_index']]
            # Equal detached counterfactuals give an exactly known zero Utility target.
            detached = {key: out[key].detach() for key in ('score_map', 'pred_boxes')}
            out['utility_probe'] = dict(off=detached, on=detached,
                                        regions=utility_probe['regions'])
            out['utility_logits'] = self.utility_repair.logits.expand(searches[0].shape[0], -1)
        return outputs


def optimizer(model):
    return torch.optim.AdamW([
        dict(params=[model.weight], lr=5e-6),
        dict(params=[model.utility_repair.logits], lr=5e-5)], weight_decay=.0001)


def reference(model, opt, local, index, data, state, generator, updates):
    """Compute the same contract directly, without the runner's step or LR helper."""
    factor = min((updates + 1) / 500., 1.)
    for group, rate in zip(opt.param_groups, (5e-6, 5e-5)):
        group['lr'] = rate * factor
    probe, view = None, None
    if index % 4 == 0:
        probe_id = index // 4
        view = 'C' if probe_id % 2 == 0 else 'M'
        probe = dict(search_index=(probe_id // 2) % 3,
                     regions=torch.randint(16, (2, 1), generator=generator))
    loss, _, complete, missing, available, patch = forward_microbatch(
        model, list(data['template_images']), list(data['search_images']),
        data['search_anno'], cfg, local, 1, state, probe, view)
    if probe is not None:
        auxiliary, _ = utility_supervision(
            complete if view == 'C' else missing, data['search_anno'], cfg)
        loss = loss + .1 * auxiliary
    (loss / 2).backward()
    norm = None
    if (index + 1) % 2 == 0:
        norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), .1))
        opt.step()
        opt.zero_grad(set_to_none=True)
    return float(loss.detach()), norm, [tensor_sha(value) for value in available + patch]


def check_steps():
    templates, searches, boxes = inputs()
    data = dict(template_images=torch.stack([x[:2] for x in templates]),
                search_images=torch.stack([x[:2] for x in searches]), search_anno=boxes[:, :2])
    rows = []
    for curriculum in CURRICULA:
        for arm in ARMS:
            local = dict(config(curriculum, arm), accumulation=2,
                         learning_rate=5e-5, backbone_multiplier=.1,
                         utility_probe_interval=4, utility_weight=.1,
                         warmup_updates=500, lr_drop_epoch=10, clip_grad_norm=.1)
            model, other = ProbeModel(arm), ProbeModel(arm)
            opt, other_opt = optimizer(model), optimizer(other)
            state, other_state = [CurriculumState(curriculum, 'cpu', 65502) for _ in range(2)]
            generator, other_generator = [torch.Generator().manual_seed(65503) for _ in range(2)]
            updates, probes = 0, []
            initial_weight, initial_logits = model.weight.detach().clone(), \
                model.utility_repair.logits.detach().clone()
            for index in range(24):
                torch.manual_seed(65504 + index)
                previous = {name: p.detach().clone() for name, p in model.named_parameters()}
                status, updates = step(
                    model, opt, local, 1, index, data, state, generator, updates)
                torch.manual_seed(65504 + index)
                loss, norm, masks = reference(
                    other, other_opt, local, index, data, other_state, other_generator, index // 2)
                assert status['loss'] == loss and status['masks_sha256'] == masks
                assert updates == (index + 1) // 2 and state.microsteps == index + 1
                assert ('update' in status) == ((index + 1) % 2 == 0)
                if norm is not None:
                    assert status['gradient_norm'] == norm
                else:
                    assert all(torch.equal(p, previous[name])
                               for name, p in model.named_parameters())
                assert ('utility' in status) == (index % 4 == 0)
                if index % 4 == 0:
                    assert status['utility']['target_mean'] == 0.
                    probes.append((status['probe_view'], status['utility']['search_index']))
                compare_trees(model.state_dict(), other.state_dict())
                compare_trees(opt.state_dict(), other_opt.state_dict())
                compare_trees(state.state_dict(), other_state.state_dict())
                assert torch.equal(generator.get_state(), other_generator.get_state())
                for a, b in zip(model.parameters(), other.parameters()):
                    assert (a.grad is None) == (b.grad is None)
                    if a.grad is not None:
                        assert torch.equal(a.grad, b.grad)
            assert probes == [('C', 0), ('M', 0), ('C', 1), ('M', 1), ('C', 2), ('M', 2)]
            assert model.weight != initial_weight
            assert not torch.equal(model.utility_repair.logits, initial_logits)
            rows.append(dict(curriculum=curriculum, feedback_mode=arm, microsteps=24,
                             updates=12, independently_matched=True,
                             all_searches_both_probe_views=True))
    return rows


def check_rates_and_snapshot():
    model = ProbeModel('deep')
    opt = optimizer(model)
    local = dict(learning_rate=5e-5, backbone_multiplier=.1,
                 warmup_updates=500, lr_drop_epoch=10)
    rows = []
    for epoch, update, factor in ((1, 1, .002), (1, 500, 1.),
                                  (10, 501, 1.), (11, 501, .1), (20, 100000, .1)):
        set_learning_rates(opt, local, epoch, update)
        expected = [5e-5 * .1 * factor, 5e-5 * factor]
        assert [g['lr'] for g in opt.param_groups] == expected
        rows.append(dict(epoch=epoch, next_update=update, learning_rates=expected))
    templates, searches, _ = inputs()
    for arm in ARMS:
        outputs = ProbeModel(arm)(templates, searches)
        copied = prediction_snapshot(outputs)
        assert len(copied) == 3 and all(len(x) == (1 if arm == 'single' else 2) for x in copied)
        original = copied[0][-1]['score_map'].clone()
        with torch.no_grad():
            outputs[0]['score_map'].zero_()
        assert torch.equal(copied[0][-1]['score_map'], original)
        assert all(not value.requires_grad and value.device.type == 'cpu'
                   for search in copied for row in search for value in row.values())
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    assert not Path(args.output).exists() and not torch.cuda.is_initialized()
    torch.set_num_threads(1)
    update_config_from_file('experiments/cadtrack/cadtrack.yaml')
    rows, rates = check_steps(), check_rates_and_snapshot()
    assert not torch.cuda.is_initialized()
    sources = [Path(__file__), Path(__file__).with_name('protocol_runner.py'),
               Path(__file__).with_name('protocol_training.py'),
               Path(__file__).with_name('check_protocol_training.py')]
    save(args.output, dict(status='PASS', cases=rows, learning_rates=rates,
        prediction_snapshots_all_modes=True, cuda_initialized=False,
        source_sha256={str(path.resolve()): sha(path) for path in sources},
        scope='Synthetic CPU step/AdamW/accumulation/Utility routing only; '
              'no actual-image smoke, checkpoint reload, full-run validation, '
              'GPU, learned feedback or official metrics'))
    print('PASS: 24 mode/curriculum cases, 576 microsteps, 288 updates, LR and snapshots')


if __name__ == '__main__':
    main()
