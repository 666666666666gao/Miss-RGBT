"""Wrap unchanged E0 tensors for explicit-availability inference; no training."""
import argparse
import hashlib
import json
from pathlib import Path

import torch


def sha(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    original, output = Path(args.checkpoint), Path(args.output)
    assert sha(original) == 'd20c8641e4263df3ceadf9e430d0ac96c181c13ad482bca28c2dddef592a8bb8'
    assert not output.exists()
    saved = torch.load(str(original), map_location='cpu')
    assert saved['epoch'] == 20 and saved['net_type'] == 'CADTrack'
    config = dict(repair=False, repair_width=128, repair_layer=6)
    state = {'baseline.' + name: value for name, value in saved['net'].items()}
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(model=state, config=config, epoch=20, updates=0), str(output))
    reloaded = torch.load(str(output), map_location='cpu')
    assert set(reloaded['model']) == set(state)
    assert all(torch.equal(value, reloaded['model']['baseline.' + name]) for name, value in saved['net'].items())
    record = dict(original=str(original), original_sha256=sha(original),
        packaged=str(output), packaged_sha256=sha(output), model_state_keys=len(state),
        tensor_equality=True, optimizer_updates=0, config=config,
        scope='Serialization envelope only; unchanged ablated-CADTrack E0 tensors. Evaluation still required.')
    output.with_suffix('.json').write_text(json.dumps(record, indent=2))
    print(json.dumps(record), flush=True)


if __name__ == '__main__':
    main()
