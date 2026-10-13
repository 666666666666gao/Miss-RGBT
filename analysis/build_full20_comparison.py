"""Summarize completed E3/E4 epoch20 evaluations and their frozen E0 controls."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path


DATASETS = {
    'LasHeR': (245, 220703, ['PR', 'NPR', 'SR']),
    'RGBT234': (234, 116650, ['MPR', 'MSR']),
}
MISS_TARGETS = {'LasHeR': {'PR': 69.3, 'SR': 54.4}, 'RGBT234': {'MPR': 88.1, 'MSR': 64.3}}


def sha(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def read_json(path, inputs):
    inputs[str(path)] = sha(path)
    return json.loads(path.read_text(encoding='utf-8'))


def verify_metrics(metrics, condition):
    expected = {name + ('-Miss' if condition == 'miss' else '') for name in DATASETS}
    assert set(metrics) == expected, set(metrics)
    for dataset, (sequences, frames, keys) in DATASETS.items():
        row = metrics[dataset + ('-Miss' if condition == 'miss' else '')]
        assert row['sequences'] == sequences and row['frames'] == frames, row
        assert all(math.isfinite(row[key]) and 0 <= row[key] <= 100 for key in keys), row


def read_baselines(root, inputs):
    baselines = {}
    for condition, folder in [('complete', 'e0_complete'), ('miss', 'e0')]:
        completed = read_json(root / 'reports' / folder / 'completed.json', inputs)
        assert completed['sequences'] == 479, completed
        metrics = read_json(root / 'reports' / folder / 'metrics.json', inputs)
        verify_metrics(metrics, condition)
        baselines[condition] = metrics
    return baselines


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project-root', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    root = Path(args.project_root).resolve()
    inputs = {}
    baselines = read_baselines(root, inputs)
    inventory = read_json(root / 'reports/e0/inventory.json', inputs)
    models = {}
    rows = []
    for stage in ['e3', 'e4']:
        training = root / 'checkpoints' / (stage + '_full20_seed2027')
        completed = read_json(training / 'completed.json', inputs)
        config = read_json(training / 'run_config.json', inputs)
        assert completed['epochs'] == 20 and completed['updates'] == 50000, completed
        assert config == read_json(root / 'configs' / (stage + '_full20.json'), inputs), stage
        assert config['epochs'] == 20 and config['samples_per_epoch'] == 20000
        assert config['microbatch'] * config['accumulation'] == 8 and config['seed'] == 2027
        checkpoint = training / 'epoch20.pth.tar'
        checkpoint_sha256 = sha(checkpoint)
        inputs[str(checkpoint)] = checkpoint_sha256
        version = training / 'source_version.txt'
        inputs[str(version)] = sha(version)
        models[stage] = dict(training=completed, config=config, source_version=version.read_text().strip(),
                             checkpoint_sha256=checkpoint_sha256, evaluations={})
        for condition in ['complete', 'miss']:
            label = stage + '_full20_' + condition
            folder = root / 'results' / (label + '_20261013')
            run = read_json(folder / 'run.json', inputs)
            assert Path(run['checkpoint']).resolve() == checkpoint and run['checkpoint_sha256'] == checkpoint_sha256
            assert run['condition'] == condition and run['physical_gpus'] == [2, 3]
            finished = read_json(folder / 'completed.json', inputs)
            assert finished['sequences'] == 479 and finished['frames'] == 337353, finished
            metrics = read_json(folder / 'metrics.json', inputs)
            assert metrics == read_json(root / 'reports' / label / 'metrics.json', inputs), label
            verify_metrics(metrics, condition)
            receipts = read_json(folder / 'all_receipts.json', inputs)
            expected = {(dataset, item['name']): item['frames']
                        for dataset, spec in inventory['datasets'].items() for item in spec['sequence_details']}
            assert len(receipts) == 479 and {(v['dataset'], v['sequence']) for v in receipts} == set(expected)
            for receipt in receipts:
                key = (receipt['dataset'], receipt['sequence'])
                assert receipt['frames'] == expected[key] and receipt['physical_gpu'] in [2, 3], receipt
                bbox = folder / 'bbox' / key[0] / (key[1] + '.txt')
                inputs[str(bbox)] = sha(bbox)
                assert inputs[str(bbox)] == receipt['bbox_sha256'], bbox
                with bbox.open() as stream:
                    assert sum(1 for line in stream) == expected[key], bbox
            models[stage]['evaluations'][condition] = dict(completed=finished, metrics=metrics,
                checkpoint_sha256=run['checkpoint_sha256'], verified_receipts=len(receipts))
            for dataset, (sequences, frames, keys) in DATASETS.items():
                name = dataset + ('-Miss' if condition == 'miss' else '')
                for metric in keys:
                    value, baseline = metrics[name][metric], baselines[condition][name][metric]
                    row = dict(experiment=stage.upper(), dataset=name, metric=metric, value=value,
                               baseline=baseline, delta_pp=value - baseline, sequences=sequences, frames=frames)
                    if condition == 'miss' and metric in MISS_TARGETS[dataset]:
                        target = MISS_TARGETS[dataset][metric]
                        row.update(strict_target=target, target_exceeded=value > target)
                    rows.append(row)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    report = dict(scope='fixed final epoch20; 50000 updates per model; four completed full evaluations; '
                        'single seed; E5 and three effective modules remain unproven',
                  baselines=baselines, models=models, rows=rows, input_sha256=inputs)
    (output / 'comparison.json').write_bytes((json.dumps(report, indent=2) + '\n').encode('utf-8'))
    fields = ['experiment', 'dataset', 'metric', 'value', 'baseline', 'delta_pp', 'sequences', 'frames',
              'strict_target', 'target_exceeded']
    with (output / 'comparison.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    lines = ['# 20轮完整训练指标', '',
             '固定最终epoch20，各50000次更新；四组评估均验证479序列、337353帧、权重与bbox回执。增益单位为百分点。', '']
    for condition in ['complete', 'miss']:
        for dataset, (_, _, keys) in DATASETS.items():
            name = dataset + ('-Miss' if condition == 'miss' else '')
            lines += ['## ' + name, '', '| 实验 | ' + ' | '.join(keys) + ' |',
                      '|---|' + '|'.join(['---:'] * len(keys)) + '|']
            lines.append('| E0 | ' + ' | '.join('%.6f' % baselines[condition][name][key] for key in keys) + ' |')
            for stage in ['e3', 'e4']:
                selected = {v['metric']: v for v in rows if v['experiment'] == stage.upper() and v['dataset'] == name}
                lines.append('| ' + stage.upper() + ' | ' + ' | '.join('%.6f (%+.6f)' %
                             (selected[key]['value'], selected[key]['delta_pp']) for key in keys) + ' |')
            lines.append('')
    lines += ['## Miss严格目标', '', '| 实验 | 数据集 | 指标 | 目标 | 实际 | 是否超过 |', '|---|---|---|---:|---:|---|']
    for row in rows:
        if 'strict_target' in row:
            lines.append('| {experiment} | {dataset} | {metric} | {strict_target} | {value:.6f} | {passed} |'.format(
                passed='是' if row['target_exceeded'] else '否', **row))
    lines += ['', '单seed结果不能证明统计稳定性；E3−E0不能单独证明正向蒸馏有效，E4−E3需要同参数对照。E5尚未训练，不能据此宣称三个论文模块成立。', '']
    (output / 'RESULTS.md').write_bytes('\n'.join(lines).encode('utf-8'))
    print(json.dumps(dict(output=str(output), verified_evaluations=4, metric_rows=len(rows))))


if __name__ == '__main__':
    main()
