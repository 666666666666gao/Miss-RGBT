"""Recompute full F metrics and verify every source/GT/trajectory receipt."""
import argparse
import json
from pathlib import Path

import numpy as np
import rgbt

import evaluate_candidate_official28 as native


def read(path):
    return json.loads(Path(path).read_text())


def verify(destination, checkpoint_sha, condition):
    run = read(destination/'run.json')
    done = read(destination/'completed.json')
    recorded = read(destination/'metrics.json')
    assert run['checkpoint_sha256'] == checkpoint_sha == native.sha(run['checkpoint'])
    assert run['condition'] == condition and run['limit'] is None
    assert run['sequences'] == done['sequences'] == 479
    assert run['frames'] == done['frames'] == 337353
    evaluator = Path(__file__).with_name('evaluate.py')
    assert run['standalone_evaluator_sha256'] == native.sha(evaluator)
    assert run['native_evaluator_sha256'] == native.sha(native.__file__)
    expected_sources = [evaluator, evaluator.with_name('tracker.py'),
                        evaluator.with_name('model.py'), Path(native.__file__)]
    assert run['feedback_source_sha256'] == {
        str(path.resolve()): native.sha(path) for path in expected_sources}
    expected_runtime = {
        str(path.relative_to(native.PROJECT)): native.sha(path)
        for folder in ['miss_rgbt', 'lib', 'scripts']
        for path in sorted((native.PROJECT/folder).rglob('*.py'))}
    assert run['source_sha256'] == expected_runtime
    inventory = read(native.PROJECT/'reports/e0/inventory.json')['datasets']
    gt_root = Path(rgbt.__file__).parent/'gt_file'
    all_receipts = read(destination/'all_receipts.json')
    receipts = {(row['dataset'], row['sequence']): row for row in all_receipts}
    assert len(receipts) == len(all_receipts) == 479
    assert set(receipts) == {(dataset, row['name']) for dataset, spec in inventory.items()
                             for row in spec['sequence_details']}
    measured = {}
    for dataset, cls, keys in [('LasHeR-Miss', rgbt.LasHeR, ['PR', 'NPR', 'SR']),
                               ('RGBT234-Miss', rgbt.RGBT234, ['MPR', 'MSR'])]:
        spec = inventory[dataset]
        assert native.sha(spec['mask']) == spec['mask_sha256']
        names = {row['name'] for row in spec['sequence_details']}
        for row in spec['sequence_details']:
            name, frames = row['name'], row['frames']
            gt_files = ([gt_root/'LasHeR/lasher_gt'/(name+'.txt')]
                        if dataset == 'LasHeR-Miss' else [
                            gt_root/'RGBT234/rgbt234_gt'/name/file
                            for file in ['visible.txt', 'infrared.txt']])
            assert len(gt_files) == len(row['gt_sha256'])
            for gt_file, digest in zip(gt_files, row['gt_sha256'].values()):
                assert native.sha(gt_file) == digest
                assert np.loadtxt(str(gt_file), delimiter=',', ndmin=2).shape == (frames, 4)
            path = destination/'bbox'/dataset/(name+'.txt')
            receipt = read(destination/'receipts'/dataset/(name+'.json'))
            assert receipt == receipts[(dataset, name)]
            assert receipt['frames'] == frames and native.sha(path) == receipt['bbox_sha256']
            boxes = np.loadtxt(str(path), ndmin=2)
            assert boxes.shape == (frames, 4) and np.isfinite(boxes).all()
        scorer = cls()
        assert set(scorer.seqs_name) == names
        scorer('verified', str(destination/'bbox'/dataset), bbox_type='ltwh')
        label = dataset if condition == 'miss' else dataset.replace('-Miss', '')
        actual = dict(sequences=len(names),
                      frames=sum(row['frames'] for row in spec['sequence_details']))
        curves = read(destination/(label+'_curves.json'))
        for key in keys:
            score, curve = getattr(scorer, key)('verified')
            actual[key] = float(score*100)
            assert np.isclose(actual[key], recorded[label][key], rtol=0, atol=1e-9)
            assert np.allclose(curve, curves[key], rtol=0, atol=1e-12)
        assert actual['sequences'] == recorded[label]['sequences']
        assert actual['frames'] == recorded[label]['frames']
        measured[label] = actual
    return measured


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--checkpoint-sha256', required=True)
    parser.add_argument('--condition', required=True, choices=['complete', 'miss'])
    args = parser.parse_args()
    destination = Path(args.output).resolve()
    result = dict(status='PASS', metrics=verify(destination, args.checkpoint_sha256, args.condition),
        verifier_sha256=native.sha(__file__),
        scope='Full source/GT/receipt/trajectory checks and independent metric recalculation; '
              'does not certify training or mask execution from saved boxes')
    with (destination/'independent_verification.json').open('x') as stream:
        stream.write(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
