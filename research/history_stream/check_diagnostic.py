"""CPU fixtures for the isolated intervention and paired decision rule."""
import argparse
import json
from pathlib import Path

import torch

from research.history_stream.diagnostic import (
    MODES, different_sequence_donors, replace_search, summarize,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
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
    assert not summarize(rows, clustered)[1]
    for row in rows:
        row['true_hist']['iou'] = .6
        row['wrong_seq']['iou'] = .7
    assert not summarize(rows, clustered)[1]
    assert not torch.cuda.is_initialized()
    report = dict(status='PASS', device='cpu', cuda_initialized=False,
        checks=['complete_noop_all_modes', 'native_zero_noop', 'search_only',
                'observed_stream_unchanged', 'missing_stream_replacement',
                'different_sequence_donors', 'paired_sequence_bootstrap',
                'both_directions_required', 'wrong_sequence_control_required'],
        real_images_executed=False, diagnostic_gate_passed=False)
    with Path(args.output).open('x') as stream:
        json.dump(report, stream, indent=2)
        stream.write('\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
