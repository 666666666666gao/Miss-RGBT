"""Recalculate the saved diagnostic; this does not replay neural inference."""
import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def effect(rows, left, right, seed):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row['identity']['sequence']].append(row[right] - row[left])
    values = np.array([np.mean(grouped[name]) for name in sorted(grouped)])
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(values), size=(10000, len(values)))
    interval = np.quantile(values[draws].mean(axis=1), [.025, .975])
    return dict(mean_sequence_macro=float(values.mean()),
                ci95_low=float(interval[0]), ci95_high=float(interval[1]),
                sequences=len(values), tuples=len(rows),
                positive_sequences=int((values > 0).sum()),
                zero_sequences=int((values == 0).sum()),
                negative_sequences=int((values < 0).sum()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('directory')
    parser.add_argument('--train-list', default='lib/train/data_specs/lasher_train.txt')
    args = parser.parse_args()
    directory = Path(args.directory)
    path = directory / 'diagnostic.json'
    raw = path.read_bytes()
    report = json.loads(raw)
    completed = json.loads((directory / 'completed.json').read_text())
    digest = hashlib.sha256(raw).hexdigest()
    assert digest == completed['diagnostic_sha256']
    train_raw = Path(args.train_list).read_bytes()
    assert hashlib.sha256(train_raw).hexdigest() == report['train_sequence_list_sha256']
    names = set(train_raw.decode().splitlines())
    assert len(names) == report['train_sequences'] == 882
    assert report['samples'] == completed['tuples'] == 128
    assert len(report['rows']) == completed['condition_rows'] == 384
    seed = report['sample_seed']
    identities_by_condition = []
    summaries = {}
    for condition in ['complete', 'rgb_only', 'tir_only']:
        rows = [row for row in report['rows'] if row['condition'] == condition]
        assert len(rows) == 128
        identities = []
        for row in rows:
            assert row['identity']['sequence'] in names
            identities.append(json.dumps({key: row[key] for key in
                ['identity', 'templates_sha256', 'search_sha256', 'box_sha256']}, sort_keys=True))
            for key in ['utility_regions', 'random_regions', 'similarity_regions']:
                assert len(row[key]) == len(set(row[key])) == 4
                assert all(0 <= region < 16 for region in row[key])
            for key in ['off', 'utility_joint', 'random_joint', 'similarity_joint',
                        'utility_single_gain_sum', 'utility_joint_gain']:
                assert np.isfinite(row[key])
            assert np.isclose(row['off'] - row['utility_joint'], row['utility_joint_gain'],
                              atol=2e-6, rtol=1e-5)
        identities_by_condition.append(identities)
        summary = dict(means={key: float(np.mean([row[key] for row in rows])) for key in
            ['off', 'utility_joint', 'random_joint', 'similarity_joint']})
        contrasts = [
            ('utility_minus_random_loss_improvement', 'utility_joint', 'random_joint', 1),
            ('utility_minus_similarity_loss_improvement', 'utility_joint', 'similarity_joint', 2),
            ('utility_joint_gain', 'utility_joint', 'off', 3),
            ('utility_additivity_residual', 'utility_single_gain_sum', 'utility_joint_gain', 4),
        ]
        for name, left, right, offset in contrasts:
            summary[name] = effect(rows, left, right, seed + offset)
        assert summary == report['summaries'][condition], condition
        summaries[condition] = summary
    assert identities_by_condition[0] == identities_by_condition[1] == identities_by_condition[2]
    gate = all(summaries[state][contrast]['ci95_low'] > 0
        for state in ['rgb_only', 'tir_only']
        for contrast in ['utility_minus_random_loss_improvement',
                         'utility_minus_similarity_loss_improvement'])
    assert gate == report['advance_to_joint_utility_training']
    result = dict(status='saved_arithmetic_verified', diagnostic_sha256=digest,
                  condition_rows=384, tuples=128, summaries=summaries,
                  registered_gate=gate,
                  limitations='Saved-record recomputation only; no independent neural replay, frozen-state digest or training-seed replication.')
    with (directory / 'analysis.json').open('x', newline='\n') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
