"""Run the synthetic tracker checks on CPU with device placement mocked."""
import argparse
import json
from pathlib import Path
import sys
from unittest.mock import patch

import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    output = Path(args.output)
    assert not output.exists()
    assert not torch.cuda.is_initialized()
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import check_tracker

    tensor, ones = torch.tensor, torch.ones

    def cpu_tensor(*values, **kwargs):
        if kwargs.get('device') == 'cuda':
            kwargs['device'] = 'cpu'
        return tensor(*values, **kwargs)

    def cpu_ones(*values, **kwargs):
        if kwargs.get('device') == 'cuda':
            kwargs['device'] = 'cpu'
        return ones(*values, **kwargs)

    with patch.object(torch.nn.Module, 'cuda', lambda model: model), \
            patch.object(torch.Tensor, 'cuda', lambda value: value), \
            patch.object(torch, 'tensor', cpu_tensor), \
            patch.object(torch, 'ones', cpu_ones), \
            patch.object(sys, 'argv', ['check_tracker', '--checkpoint',
                                     args.checkpoint, '--output', str(output)]):
        check_tracker.main()
    assert not torch.cuda.is_initialized()
    report = json.loads(output.read_text())
    report.update(device='cpu', cuda_initialized=False,
        device_placement_mocked=['Module.cuda', 'Tensor.cuda',
                                 'torch.tensor(device=cuda)', 'torch.ones(device=cuda)'])
    report['source_sha256'][str(Path(__file__).resolve())] = check_tracker.sha(__file__)
    report['limitations'].append(
        'CUDA placement explicitly mocked; no GPU/device/parity/performance verification')
    output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(dict(status=report['status'], device='cpu',
        cuda_initialized=False, output=str(output))), flush=True)


if __name__ == '__main__':
    main()
