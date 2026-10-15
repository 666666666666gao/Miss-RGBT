"""Summarize closed training logs, source identity, BN buffers and budget changes."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import subprocess


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project-root', required=True)
    parser.add_argument('--raw-root', required=True)
    args = parser.parse_args()
    root, raw = Path(args.project_root), Path(args.raw_root)
    output = root / 'reports/full20_comparison_20261013_v1'
    training = {}
    keys = ['task_c', 'task_m', 'kd_forward', 'kd_reverse', 'reverse_fraction', 'loss']
    for stage in ['e3', 'e4']:
        folder = raw / 'training' / (stage + '_full20_seed2027')
        version = (folder / 'source_version.txt').read_text().strip()
        sources = json.loads((folder / 'source_sha256.json').read_text())
        for name, expected in sources.items():
            blob = subprocess.check_output(['git', 'show', version + ':' + name], cwd=str(root))
            assert hashlib.sha256(blob).hexdigest() == expected, name
            assert sha(raw / 'source' / name) == expected, name
        epochs = {epoch: {key: [] for key in keys} for epoch in range(1, 21)}
        with (folder / 'steps.jsonl').open() as stream:
            for update, line in enumerate(stream, 1):
                row = json.loads(line)
                assert row['update'] == update and row['epoch'] == (update - 1) // 2500 + 1
                assert all(math.isfinite(v) for v in row.values() if isinstance(v, (float, int)))
                for key in keys:
                    epochs[row['epoch']][key].append(row[key])
                if update == 1:
                    first_inputs = row['input_sha256_microbatches']
        assert update == 50000
        assert all(len(values['loss']) == 2500 for values in epochs.values())
        means = {epoch: {key: math.fsum(values[key]) / len(values[key]) for key in keys}
                 for epoch, values in epochs.items()}
        training[stage] = dict(updates=update, epochs=20, source_version=version, verified_source_files=len(sources),
            steps_sha256=sha(folder / 'steps.jsonl'), first_inputs=first_inputs, epoch_means=means,
            log_scope='Last accumulation microbatch at each update; not the full-train population mean.')
    training['first_update_inputs_match'] = training['e3']['first_inputs'] == training['e4']['first_inputs']
    deployed = json.loads((output / 'collection.json').read_text())['source_sha256']
    differences = {}
    for name, actual in deployed.items():
        blob = subprocess.check_output(['git', 'show', '4c473fa:' + name], cwd=str(root))
        expected = hashlib.sha256(blob).hexdigest()
        if expected != actual:
            differences[name] = dict(git_sha256=expected, deployed_sha256=actual)
    assert set(differences) == {'scripts/validate_focal_normalization.py'}
    (root / 'reports/full20_training/summary.json').write_bytes((json.dumps(training, indent=2)+'\n').encode('utf-8'))
    (output / 'source_identity.json').write_bytes((json.dumps(dict(recorded_files=len(deployed),
        python_files=sum(name.endswith('.py') for name in deployed), differences=differences,
        scope='Actual deployed source snapshot retained in the closed archive; one auxiliary exception to Git 4c473fa.'),
        indent=2)+'\n').encode('utf-8'))
    comparison = json.loads((output / 'comparison.json').read_text())
    first5 = json.loads((root / 'reports/first5_evaluation/comparison.json').read_text())
    old = {(v['experiment'], v['dataset'], v['metric']): v['value'] for v in first5['rows']}
    rows = []
    for value in comparison['rows']:
        row = dict(value)
        row['relative_change_vs_e0_percent'] = 100 * row['delta_pp'] / row['baseline']
        row['epoch05_value'] = old[row['experiment'], row['dataset'], row['metric']]
        row['delta_epoch20_minus_epoch05_pp'] = row['value'] - row['epoch05_value']
        rows.append(row)
    (output / 'budget_comparison.json').write_bytes((json.dumps(dict(rows=rows,
        scope='Descriptive fixed epoch05 versus fixed epoch20; different total budgets and LR drop epochs (5 versus 10). Not test checkpoint selection.'),
        indent=2)+'\n').encode('utf-8'))
    fields = ['experiment', 'dataset', 'metric', 'epoch05_value', 'value', 'delta_epoch20_minus_epoch05_pp',
              'baseline', 'delta_pp', 'relative_change_vs_e0_percent']
    with (output / 'budget_comparison.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    lines = ['# 融合BN检查点证据', '',
             '来自实际E0与epoch20检查点的CPU读取。完整元数据及缓冲SHA见checkpoint_metadata.json。', '',
             '| 模型 | 融合BN初始计数 | 最终计数 | 增量 | running_mean均值变化 | running_var均值变化 | 未变头部BN缓冲数 |',
             '|---|---:|---:|---:|---|---|---:|']
    metadata = json.loads((output / 'checkpoint_metadata.json').read_text())
    for stage, saved in metadata.items():
        buffers = saved['bn_buffer_differences_from_e0']
        count = buffers['decode_fuse_search.1.num_batches_tracked']
        mean = buffers['decode_fuse_search.1.running_mean']
        var = buffers['decode_fuse_search.1.running_var']
        unchanged = sum(v['max_abs_change'] == 0 for k,v in buffers.items() if not k.startswith('decode_fuse_search.'))
        lines.append('| %s | %d | %d | %d | %.6f→%.6f | %.6f→%.6f | %d |' %
            (stage.upper(), count['initial_mean'], count['final_mean'], count['max_abs_change'],
             mean['initial_mean'], mean['final_mean'], var['initial_mean'], var['final_mean'], unchanged))
    lines += ['', '600000 = 50000更新 × 2个累积microbatch × 3个search × C/M两次前向，与当前融合BN仍训练的调用次数一致。头部运行缓冲未变不代表affine参数被冻结。此观察确认统计更新，不单独证明退化的因果来源；后续从相同E0独立训练融合BN冻结对照，当前已结束的权重和结果保留。',
              '', '5轮与20轮均只有seed2027；训练量与学习率降阶轮次不同，不能把其差值当作纯长度效应，不能按Test将epoch05换成最终模型。', '']
    (output / 'BN_DIAGNOSTICS.md').write_bytes(('\n'.join(lines)).encode('utf-8'))
    print(json.dumps(dict(training_updates_each=50000, epoch_count_each=20, budget_metric_rows=len(rows),
                         source_files=len(deployed), source_exceptions=len(differences))))


if __name__ == '__main__':
    main()
