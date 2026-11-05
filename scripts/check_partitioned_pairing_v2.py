"""Exercise exact common-prefix pairing and real saved smoke verification."""
import copy
import json
from pathlib import Path
import subprocess
import sys

from verify_partitioned_a_v2 import compare_pair, sha


def main():
    dev = dict(phase='development', identities=['dev'], tensors_sha256='dev')
    train = [dict(phase='train', identities=[index], tensors_sha256=str(index))
             for index in range(4)]
    steps = [(str(index), ['mask'+str(index)]) for index in range(4)]
    equal = compare_pair([[dev]+train, [dev]+train], [steps, steps])
    unequal = compare_pair([[dev]+train[:2], [dev]+train], [steps[:2], steps])
    reverse = compare_pair([[dev]+train, [dev]+train[:2]], [steps, steps[:2]])
    assert equal['full_run_lengths_equal'] and equal['common_microsteps'] == 4
    assert not unequal['full_run_lengths_equal'] and unequal['common_microsteps'] == 2
    assert reverse['common_microsteps'] == 2
    rejected = []
    for kind in ['input', 'mask', 'development']:
        inputs = [[copy.deepcopy(dev)]+copy.deepcopy(train[:2]),
                  [copy.deepcopy(dev)]+copy.deepcopy(train)]
        masks = [copy.deepcopy(steps[:2]), copy.deepcopy(steps)]
        if kind == 'input':
            inputs[1][1]['identities'] = ['wrong']
        elif kind == 'development':
            inputs[1][0]['tensors_sha256'] = 'wrong'
        else:
            masks[1][0] = ('0', ['wrong'])
        failed = False
        try:
            compare_pair(inputs, masks)
        except AssertionError:
            failed = True
        assert failed, kind
        rejected.append(kind)
    root = Path('/home/gaob/Miss-RGBT')
    verifier = root/'scripts/verify_partitioned_a_v2.py'
    subprocess.run([sys.executable, str(verifier), '--phase', 'smoke'], check=True)
    receipt = root/'reports/server28_partitioned_a_20261102_smoke/verification_v2.json'
    smoke = json.loads(receipt.read_text())
    assert smoke['status'] == 'passed' and smoke['pairing']['common_microsteps'] == 24
    result = dict(status='passed', equal=equal, unequal=unequal, reverse=reverse,
        rejected_mismatches=rejected, real_smoke_verification_sha256=sha(receipt),
        source_sha256={str(path): sha(path) for path in [Path(__file__), verifier]},
        scope='Pairing contract fixtures plus complete saved smoke evidence; zero benchmark evaluations')
    with (root/'reports/server28_partitioned_pairing_v2_checks_20261102.json').open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
