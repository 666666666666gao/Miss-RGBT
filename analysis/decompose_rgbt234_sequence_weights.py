"""Compare sequence versus frame weights using unchanged recorded MSR curves."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from analyze_paired_baseline import sha


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project-root', required=True)
    parser.add_argument('--raw-root', required=True)
    parser.add_argument('--gt-root', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    root, raw, output = Path(args.project_root), Path(args.raw_root), Path(args.output)
    inventory_path = root / 'reports/e0/inventory.json'
    inventory = json.loads(inventory_path.read_text(encoding='utf-8'))['datasets']['RGBT234-Miss']
    list_path = Path(args.gt_root) / 'RGBT234/attr_txt/SequencesName.txt'
    names = list_path.read_text(encoding='utf-8').splitlines()
    sizes = {row['name']: row['frames'] for row in inventory['sequence_details']}
    assert len(names) == 234 and set(names) == set(sizes)
    frames = np.array([sizes[name] for name in names])
    assert frames.sum() == 116650
    inputs = {str(path): sha(path) for path in [Path(__file__), inventory_path, list_path]}
    scores = {}
    for model in ['e3', 'e4']:
        folder = raw / 'results' / (model + '_full20_miss_20261013')
        path = folder / 'RGBT234-Miss_curves.json'
        curve = np.array(json.loads(path.read_text())['MSR'])
        assert curve.shape == (234, 21) and np.isfinite(curve).all()
        score = float(json.loads((folder / 'metrics.json').read_text())['RGBT234-Miss']['MSR'])
        assert abs(float(curve.mean() * 100) - score) < 1e-12
        scores[model] = curve.mean(1) * 100
        inputs[str(path)] = sha(path)
        inputs[str(folder / 'metrics.json')] = sha(folder / 'metrics.json')
    delta = scores['e4'] - scores['e3']
    weight = frames / frames.sum()
    macro = float(delta.mean())
    weighted = float(np.dot(weight, delta))
    decomposition = float(np.dot(weight - 1 / len(names), delta))
    assert abs(weighted - macro - decomposition) < 1e-12
    rows = [dict(sequence=name, frames=int(frames[i]), e3_msr=float(scores['e3'][i]), e4_msr=float(scores['e4'][i]),
        delta_pp=float(delta[i]), macro_contribution_pp=float(delta[i] / 234),
        frame_weighted_contribution_pp=float(weight[i] * delta[i])) for i, name in enumerate(names)]
    quartiles = []
    for i, indices in enumerate(np.array_split(np.argsort(frames, kind='stable'), 4), 1):
        quartiles.append(dict(length_quartile=i, sequences=len(indices), frames=int(frames[indices].sum()),
            minimum_length=int(frames[indices].min()), maximum_length=int(frames[indices].max()),
            mean_delta_pp=float(delta[indices].mean()), macro_contribution_pp=float(delta[indices].sum() / 234)))
    report = dict(scope='Aggregation diagnosis of the same installed-toolkit MSR curves; benchmark MSR remains the equal-sequence mean. '
        'Frame-weighted numbers are diagnostics, never replacement primary scores. All 234 sequences and all 21 thresholds retained. '
        'Quartiles use sequence lengths only; no parameter selection from test results.',
        e3_msr_macro=float(scores['e3'].mean()), e4_msr_macro=float(scores['e4'].mean()),
        official_protocol_delta_pp=macro, diagnostic_frame_weighted_e3=float(np.dot(weight, scores['e3'])),
        diagnostic_frame_weighted_e4=float(np.dot(weight, scores['e4'])),
        diagnostic_frame_weighted_delta_pp=weighted, weighting_difference_pp=decomposition,
        improved_sequences=int((delta > 0).sum()), harmed_sequences=int((delta < 0).sum()),
        tied_sequences=int((delta == 0).sum()), length_quartiles=quartiles, inputs=inputs)
    (output / 'sequence_weighting.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    with (output / 'sequence_weighting.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
