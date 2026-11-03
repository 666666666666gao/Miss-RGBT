"""CPU fixtures for the isolated intervention and paired decision rule."""
import argparse
import json
from pathlib import Path
import sys
import tempfile
from unittest import TestCase

import torch

from research.history_stream.diagnostic import (
    MODES, capture_search, decision, different_sequence_donors,
    forward_intervention, replace_search, summarize,
)


class PairBlock(torch.nn.Module):
    def forward(self, rgb, tir):
        return rgb + 1, tir + 2


class HookFixture(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.baseline = torch.nn.Module()
        self.baseline.backbone = torch.nn.Module()
        backbone = self.baseline.backbone
        backbone.register_buffer('pos_embed_x', torch.zeros(1, 4, 2))
        backbone.blocks = torch.nn.ModuleList([PairBlock() for _ in range(3)])

    def forward(self, templates, searches, availability=None):
        values = (searches[0], searches[0] + 10)
        for block in self.baseline.backbone.blocks:
            values = block(*values)
        return [dict(rgb=values[0], tir=values[1])]


def check_hooks():
    model = HookFixture()
    search = torch.zeros(1, 9, 2)
    missing = [torch.tensor([[0., 1.]])]
    complete = [torch.ones(1, 2)]
    history = (torch.full((1, 4, 2), 20.), torch.full((1, 4, 2), 30.))
    native = model([], [search])[0]
    for layer in (0, 3):
        captured = capture_search(model, [], search, layer)
        assert torch.equal(captured[0], torch.full_like(captured[0], float(layer)))
        assert torch.equal(captured[1], torch.full_like(captured[1], 10. + 2 * layer))
        for mode in MODES:
            result = forward_intervention(model, [], search, complete, history, mode, layer)
            assert all(torch.equal(result[key], native[key]) for key in native)
        result = forward_intervention(model, [], search, missing, history, 'true_hist', layer)
        assert torch.equal(result['rgb'][:, -4:],
                           torch.full((1, 4, 2), 20. + 3 - layer))
        assert torch.equal(result['rgb'][:, :-4], native['rgb'][:, :-4])
        assert torch.equal(result['tir'], native['tir'])
        zero = forward_intervention(model, [], search, missing, history, 'zero', layer)
        assert all(torch.equal(zero[key], native[key]) for key in native)
        assert all(not block._forward_hooks and not block._forward_pre_hooks
                   for block in model.baseline.backbone.blocks)
    assert decision(3, True) == 'accepted_block3'
    assert decision(3, False) == 'retry_embedding'
    assert decision(0, True) == 'accepted_embedding'
    assert decision(0, False) == 'closed_both_diagnostics_negative'


def check_retry_gate(rows, effects):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))
    from diagnose_history_stream import CHECKPOINT, CHECKPOINT_SHA, check_retry, sha256
    checks = TestCase()
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        config, receipt = root / 'config.json', root / 'cache_receipt.json'
        config.write_text('{}')
        receipt.write_text('{}')
        report_path, completed_path = root / 'report.json', root / 'completed.json'
        report = dict(status='complete', injection_layer=3, advance=False,
            m1_decision='retry_embedding', checkpoint=CHECKPOINT,
            checkpoint_sha256=CHECKPOINT_SHA, cache_receipt_sha256=sha256(receipt),
            source_sha256={str(config.resolve()): sha256(config)},
            complete_bitwise_parity=True, native_zero_bitwise_parity=True,
            state_unchanged=True, gradients_absent=True, rows=rows, effects=effects)

        def write_previous():
            report_path.write_text(json.dumps(report))
            completed_path.write_text(json.dumps(dict(status='complete',
                injection_layer=report['injection_layer'], advance=report['advance'],
                m1_decision=report['m1_decision'], report_sha256=sha256(report_path))))

        write_previous()
        assert check_retry(3, None, receipt, config) is None
        with checks.assertRaises(AssertionError):
            check_retry(0, None, receipt, config)
        result = check_retry(0, report_path, receipt, config)
        assert result['report_sha256'] == sha256(report_path)
        with checks.assertRaises(AssertionError):
            check_retry(3, report_path, receipt, config)
        config.write_text('{"changed":true}')
        with checks.assertRaises(AssertionError):
            check_retry(0, report_path, receipt, config)
        config.write_text('{}')
        receipt.write_text('{"changed":true}')
        with checks.assertRaises(AssertionError):
            check_retry(0, report_path, receipt, config)
        receipt.write_text('{}')
        report['advance'] = True
        write_previous()
        with checks.assertRaises(AssertionError):
            check_retry(0, report_path, receipt, config)
        report['advance'], report['injection_layer'] = False, 0
        write_previous()
        with checks.assertRaises(AssertionError):
            check_retry(0, report_path, receipt, config)
        report['injection_layer'] = 3
        report['rows'][0]['true_hist']['iou'] += .01
        write_previous()
        with checks.assertRaises(AssertionError):
            check_retry(0, report_path, receipt, config)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    check_hooks()
    torch.manual_seed(73101)
    output = (torch.randn(3, 9, 4), torch.randn(3, 9, 4))
    history = (torch.randn(3, 4, 4), torch.randn(3, 4, 4))
    flags = torch.tensor([[1., 1.], [0., 1.], [1., 0.]])
    for mode in MODES:
        assert replace_search(output, torch.ones_like(flags), history, mode, 4) is output
        result = replace_search(output, flags, history, mode, 4)
        if mode == 'zero':
            assert result is output
            continue
        for stream in range(2):
            assert torch.equal(result[stream][:, :-4], output[stream][:, :-4])
            assert torch.equal(result[stream][flags[:, stream] == 1],
                               output[stream][flags[:, stream] == 1])
            selected = flags[:, stream] == 0
            expected = output[1 - stream][:, -4:] if mode == 'copy_avail' else history[stream]
            assert torch.equal(result[stream][selected, -4:], expected[selected])
            assert torch.equal(output[stream], result[stream]) is False
    identities = [dict(sequence=value) for value in ('a', 'a', 'b', 'c')]
    donors = different_sequence_donors(identities)
    assert donors == [2, 2, 3, 0]
    assert all(identities[index]['sequence'] != identities[donor]['sequence']
               for index, donor in enumerate(donors))
    from scripts.diagnose_absent_position import clustered
    rows = []
    for state in ('01', '10'):
        for index in range(128):
            row = dict(identity=dict(sequence=str(index % 16)), state=state)
            for mode in MODES:
                row[mode] = dict(iou=.6 if mode == 'true_hist' else .5,
                                 task_loss=2. if mode == 'true_hist' else 3.)
            rows.append(row)
    effects, advance = summarize(rows, clustered)
    assert advance and effects['01']['zero']['iou_gain']['sequences'] == 16
    for row in rows:
        if row['state'] == '10':
            row['true_hist']['iou'] = .4
    effects, advance = summarize(rows, clustered)
    assert not advance
    check_retry_gate(rows, effects)
    for row in rows:
        row['true_hist']['iou'] = .6
        row['wrong_seq']['iou'] = .7
    assert not summarize(rows, clustered)[1]
    assert not torch.cuda.is_initialized()
    report = dict(status='PASS', device='cpu', cuda_initialized=False,
        checks=['complete_noop_all_modes', 'native_zero_noop', 'search_only',
                'observed_stream_unchanged', 'missing_stream_replacement',
                'different_sequence_donors', 'paired_sequence_bootstrap',
                'both_directions_required', 'wrong_sequence_control_required',
                'embedding_and_block3_hooks', 'hook_cleanup', 'layer_decisions',
                'negative_block3_required_for_retry', 'same_cache_and_config_required',
                'hash_bound_prior_report', 'prior_effects_recomputed'],
        real_images_executed=False, diagnostic_gate_passed=False)
    with Path(args.output).open('x') as stream:
        json.dump(report, stream, indent=2)
        stream.write('\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
