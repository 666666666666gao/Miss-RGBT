"""Aggregate the closed FT records; independent GT re-scoring is a separate audit."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path


DATASETS = {'LasHeR': (245, 220703, ['PR', 'NPR', 'SR']),
            'RGBT234': (234, 116650, ['MPR', 'MSR'])}
TARGETS = {'LasHeR-Miss': {'PR': 69.3, 'SR': 54.4}, 'RGBT234-Miss': {'MPR': 88.1, 'MSR': 64.3}}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', required=True)
    parser.add_argument('--raw', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    project, raw, output = Path(args.project), Path(args.raw), Path(args.output)
    collection = json.loads((raw / 'collection.json').read_text(encoding='utf-8'))
    for name, expected in collection['files'].items():
        assert sha(raw / name) == expected, name
    config = json.loads((raw / 'training/run_config.json').read_text(encoding='utf-8'))
    completed = json.loads((raw / 'training/completed.json').read_text(encoding='utf-8'))
    assert completed['epochs'] == 20 and completed['updates'] == 50000
    assert config['stage'] == 'E0-FT' and config['seed'] == 2027
    assert config['epochs'] * config['samples_per_epoch'] == 400000
    assert config['microbatch'] * config['accumulation'] == 8
    assert not config['repair'] and config['forward_weight'] == config['reverse_weight'] == 0
    logs = [json.loads(line) for line in (raw / 'training/steps.jsonl').read_text().splitlines()]
    assert len(logs) == 50000
    assert [row['update'] for row in logs] == list(range(1, 50001))
    assert all(row['epoch'] == 1 + (row['update'] - 1) // 2500 for row in logs)
    assert all(math.isfinite(float(value)) for row in logs for value in row.values() if isinstance(value, (int, float)))
    metadata = collection['checkpoint_metadata']
    assert metadata['strict_reload'] and metadata['all_state_values_finite']
    assert metadata['config'] == config and metadata['updates'] == 50000
    baseline_paths = [project / 'reports/e0_complete/metrics.json', project / 'reports/e0/metrics.json']
    baselines = {condition: json.loads(path.read_text(encoding='utf-8'))
                 for condition, path in zip(['complete', 'miss'], baseline_paths)}
    availability_path = project / 'reports/e0_availability_20261015_v1/comparison.json'
    availability = json.loads(availability_path.read_text(encoding='utf-8'))
    availability_metrics = {condition: availability['evaluations'][condition]['metrics']
                            for condition in ['complete', 'miss']}
    inventory = json.loads((raw / 'inventory.json').read_text(encoding='utf-8'))
    expected = {(dataset, row['name']): row['frames'] for dataset, spec in inventory['datasets'].items()
                for row in spec['sequence_details']}
    conditions, rows = {}, []
    for condition in ['complete', 'miss']:
        label = 'e0_ft_full20_' + condition
        folder = raw / 'results' / (label + '_20261015')
        run = json.loads((folder / 'run.json').read_text(encoding='utf-8'))
        done = json.loads((folder / 'completed.json').read_text(encoding='utf-8'))
        exit_record = json.loads((raw / 'records' / (label + '_exit.json')).read_text(encoding='utf-8'))
        receipts = json.loads((folder / 'all_receipts.json').read_text(encoding='utf-8'))
        metrics = json.loads((folder / 'metrics.json').read_text(encoding='utf-8'))
        assert run['checkpoint_sha256'] == metadata['checkpoint_sha256']
        assert run['physical_gpus'] == [2] and run['condition'] == condition
        assert exit_record['exit_code'] == 0
        assert done['sequences'] == 479 and done['frames'] == 337353
        assert len(receipts) == 479 and {(row['dataset'], row['sequence']) for row in receipts} == set(expected)
        for receipt in receipts:
            key = receipt['dataset'], receipt['sequence']
            path = folder / 'bbox' / key[0] / (key[1] + '.txt')
            assert sha(path) == receipt['bbox_sha256'] and receipt['frames'] == expected[key]
            assert receipt['physical_gpu'] == 2
            boxes = [line.split() for line in path.read_text().splitlines()]
            assert len(boxes) == expected[key]
            assert all(len(box) == 4 and all(math.isfinite(float(value)) for value in box) for box in boxes)
        conditions[condition] = dict(metrics=metrics, completed=done, exit=exit_record,
            receipts_verified=479, run=run)
        assert set(metrics) == {dataset + ('-Miss' if condition == 'miss' else '') for dataset in DATASETS}
        for dataset, (sequences, frames, keys) in DATASETS.items():
            name = dataset + ('-Miss' if condition == 'miss' else '')
            assert metrics[name]['sequences'] == sequences and metrics[name]['frames'] == frames
            for key in keys:
                value = metrics[name][key]
                assert math.isfinite(value) and 0 <= value <= 100
                row = dict(dataset=name, metric=key, value=value,
                    original_e0=baselines[condition][name][key],
                    delta_original_e0_pp=value - baselines[condition][name][key],
                    zero_update_availability=availability_metrics[condition][name][key],
                    delta_availability_pp=value - availability_metrics[condition][name][key])
                if key in TARGETS.get(name, {}):
                    row.update(strict_target=TARGETS[name][key], target_exceeded=value > TARGETS[name][key])
                rows.append(row)
    buffers = metadata['bn_buffer_differences_from_e0']
    assert len(buffers) == 39
    head = {name: value for name, value in buffers.items() if 'box_head.' in name}
    assert len(head) == 36 and all(value['equal'] for value in head.values())
    fusion = buffers['baseline.decode_fuse_search.1.num_batches_tracked']
    assert fusion['final_mean'] - fusion['initial_mean'] == 300000
    output.mkdir(parents=True, exist_ok=False)
    report = dict(scope='Recorded metrics aggregation, closed FT final20/50000, fixed seed2027. Bbox hashes/row counts/finite values verified; this producer does not independently reconstruct metrics from GT. Fresh audit pending.',
        collection_sha256=sha(raw / 'collection.json'), checkpoint_metadata=metadata,
        training=dict(completed=completed, config=config, log_updates_verified=50000,
            log_loss_scope='Per-update log stores the last accumulation microbatch, not the average over all microbatches.'),
        conditions=conditions, rows=rows,
        control_input_hashes={str(path.relative_to(project)): sha(path) for path in baseline_paths + [availability_path]},
        installed_package_files=len(collection['installed_package_sha256']),
        mask_sha256=collection['mask_sha256'], post_evaluation_seals=True,
        e2_final_metrics_available=False, efficacy_claim=False)
    (output / 'comparison.json').write_bytes((json.dumps(report, indent=2) + '\n').encode('utf-8'))
    (output / 'checkpoint_metadata.json').write_bytes((json.dumps(metadata, indent=2) + '\n').encode('utf-8'))
    with (output / 'comparison.csv').open('w', newline='', encoding='utf-8') as stream:
        fields = ['dataset', 'metric', 'value', 'original_e0', 'delta_original_e0_pp',
                  'zero_update_availability', 'delta_availability_pp', 'strict_target', 'target_exceeded']
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    lines = ['# E0-FT：完整20轮与四套全量指标', '',
        '固定最终epoch20/50000更新、seed2027，完整输入继续训练；两项评估各479序列337353帧、退出0，共958次序列运行674706帧。以下聚合真实记录，独立GT复算审计待完成。', '',
        '| 条件 | 指标 | E0-FT | 原始E0 | 差值pp | 零训练availability | 差值pp |',
        '|---|---|---:|---:|---:|---:|---:|']
    for row in rows:
        lines.append('| {dataset} | {metric} | {value:.6f} | {original_e0:.6f} | {delta_original_e0_pp:+.6f} | {zero_update_availability:.6f} | {delta_availability_pp:+.6f} |'.format(**row))
    lines += ['', '完整五项均低于E0：完整退化也出现在仅完整输入的继续训练，不能全部归因于mask/KD；这不是已证明某个优化或BN机制导致退化。Miss的额外训练效应应优先与同显式缺失/模板可用性路径的E0-availability比较，原始E0的复制处理不同。', '',
        'E2最终指标未出，尚不能分离双视图训练与KD。单seed/同样本元组和更新预算不等于训练前向成本相同；完整能力正迁移需用相同模板策略比较FT与缺失训练模型。四项严格Miss目标均未超过，完整约2pp与有效论文模块目标仍未达到。', '',
        '父进程已CPU严格重载固定末轮、核验权重SHA及39个BN缓冲；融合计数增加300000，头部36个运行缓冲保持。这是统计更新事实，不是性能因果证明。评分包/GT/mask为评估后核验，不追认为启动前封存，不认证作者工具箱/SCDT严格等价。审计未独立读取二进制权重前，不能扩大父进程重载证据。', '']
    (output / 'RESULTS.md').write_bytes('\n'.join(lines).encode('utf-8'))
    print(json.dumps(dict(output=str(output), metric_rows=len(rows), verified_sequence_runs=958, verified_frames=674706)))


if __name__ == '__main__':
    main()
