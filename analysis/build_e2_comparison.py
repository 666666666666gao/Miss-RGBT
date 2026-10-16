"""Aggregate closed E2 and pair it with existing FT/E3/E4; GT re-scoring is separate."""
import argparse
import csv
import json
import math
from pathlib import Path
import struct

from build_ft_comparison import DATASETS, TARGETS, sha


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def verify_evaluation(raw, condition, expected, checkpoint_sha):
    label = 'e2_full20_' + condition
    folder = raw / 'results' / (label + '_20261015')
    run, done, receipts = [read(folder / name) for name in ['run.json', 'completed.json', 'all_receipts.json']]
    metrics = read(folder / 'metrics.json')
    exit_record = read(raw / 'records' / (label + '_exit.json'))
    assert run['checkpoint_sha256'] == checkpoint_sha and run['physical_gpus'] == [2, 3]
    assert run['condition'] == condition and exit_record['exit_code'] == 0
    assert done['sequences'] == 479 and done['frames'] == 337353
    assert len(receipts) == 479 and {(row['dataset'], row['sequence']) for row in receipts} == set(expected)
    for receipt in receipts:
        key = receipt['dataset'], receipt['sequence']
        path = folder / 'bbox' / key[0] / (key[1] + '.txt')
        assert sha(path) == receipt['bbox_sha256'] and receipt['frames'] == expected[key]
        assert receipt['physical_gpu'] in [2, 3]
        boxes = [line.split() for line in path.read_text().splitlines()]
        assert len(boxes) == expected[key]
        assert all(len(box) == 4 and all(math.isfinite(float(value)) for value in box) for box in boxes)
    assert set(metrics) == {name + ('-Miss' if condition == 'miss' else '') for name in DATASETS}
    for dataset, (sequences, frames, keys) in DATASETS.items():
        name = dataset + ('-Miss' if condition == 'miss' else '')
        assert metrics[name]['sequences'] == sequences and metrics[name]['frames'] == frames
        assert all(math.isfinite(metrics[name][key]) and 0 <= metrics[name][key] <= 100 for key in keys)
    return dict(metrics=metrics, completed=done, exit=exit_record, run=run, verified_receipts=479)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', required=True)
    parser.add_argument('--raw', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    project, raw, output = Path(args.project), Path(args.raw), Path(args.output)
    collection = read(raw / 'collection.json')
    for name, expected_hash in collection['files'].items():
        assert sha(raw / name) == expected_hash, name
    assert read(raw / 'records/status.json')['stage'] == 'CAMPAIGN_COMPLETE'
    assert (raw / 'records/completed.json').is_file()
    config, completed = read(raw / 'training/run_config.json'), read(raw / 'training/completed.json')
    assert config['stage'] == 'E2' and config['seed'] == 2027 and not config['repair']
    assert config['forward_weight'] == config['reverse_weight'] == 0
    assert completed['epochs'] == 20 and completed['updates'] == 50000
    assert config['samples_per_epoch'] * config['epochs'] == 400000
    assert config['microbatch'] * config['accumulation'] == 8
    logs = [json.loads(line) for line in (raw / 'training/steps.jsonl').read_text().splitlines()]
    assert len(logs) == 50000 and [row['update'] for row in logs] == list(range(1, 50001))
    assert all(row['epoch'] == 1 + (row['update'] - 1) // 2500 for row in logs)
    assert all(math.isfinite(float(value)) for row in logs for value in row.values() if isinstance(value, (int, float)))
    task_sum_error = max(abs(row['loss'] - struct.unpack('f', struct.pack('f', row['task_c'] + row['task_m']))[0]) for row in logs)
    assert task_sum_error == 0
    metadata = collection['checkpoint_metadata']
    assert metadata['config'] == config and metadata['epoch'] == 20 and metadata['updates'] == 50000
    assert metadata['strict_reload'] and metadata['all_state_values_finite']
    buffers = metadata['bn_buffer_differences_from_e0']
    assert len(buffers) == 39
    head = {name: value for name, value in buffers.items() if 'box_head.' in name}
    assert len(head) == 36 and all(value['equal'] for value in head.values())
    fusion = buffers['baseline.decode_fuse_search.1.num_batches_tracked']
    assert fusion['final_mean'] - fusion['initial_mean'] == 600000
    inventory = read(raw / 'inventory.json')
    expected = {(name, item['name']): item['frames'] for name, spec in inventory['datasets'].items()
                for item in spec['sequence_details']}
    ft_path = project / 'reports/e0_ft_20261016_v1/comparison.json'
    prior_path = project / 'reports/full20_comparison_20261013_v1/comparison.json'
    availability_path = project / 'reports/e0_availability_20261015_v1/comparison.json'
    ft, prior, availability = read(ft_path), read(prior_path), read(availability_path)
    ft_config = ft['training']['config']
    assert {key for key in set(config) | set(ft_config) if config.get(key) != ft_config.get(key)} == {'stage', 'output'}
    conditions = {condition: verify_evaluation(raw, condition, expected, metadata['checkpoint_sha256'])
                  for condition in ['complete', 'miss']}
    for condition, record in conditions.items():
        assert record['run']['source_sha256'] == ft['conditions'][condition]['run']['source_sha256']
    rows = []
    for condition in ['complete', 'miss']:
        for dataset, (_, _, keys) in DATASETS.items():
            name = dataset + ('-Miss' if condition == 'miss' else '')
            for key in keys:
                e0 = prior['baselines'][condition][name][key]
                av = availability['evaluations'][condition]['metrics'][name][key]
                ft_value = ft['conditions'][condition]['metrics'][name][key]
                e2 = conditions[condition]['metrics'][name][key]
                e3 = prior['models']['e3']['evaluations'][condition]['metrics'][name][key]
                e4 = prior['models']['e4']['evaluations'][condition]['metrics'][name][key]
                row = dict(dataset=name, metric=key, original_e0=e0, zero_update_availability=av,
                    ft=ft_value, e2=e2, e3=e3, e4=e4, e2_minus_ft_pp=e2-ft_value,
                    e3_minus_e2_pp=e3-e2, e4_minus_e3_pp=e4-e3, e2_minus_original_e0_pp=e2-e0)
                if key in TARGETS.get(name, {}):
                    row.update(strict_target=TARGETS[name][key], e2_target_exceeded=e2 > TARGETS[name][key])
                rows.append(row)
    output.mkdir(parents=True, exist_ok=False)
    report = dict(scope='Recorded E2 final20/50000 aggregation, fixed seed2027, two full479 evaluations. Parent bbox/source/log checks; independent GT re-scoring audit pending.',
        collection_sha256=sha(raw / 'collection.json'), checkpoint_metadata=metadata,
        training=dict(config=config, completed=completed, verified_updates=50000,
            max_logged_loss_vs_float32_task_c_plus_task_m=task_sum_error,
            note='KD/reliability debug values may be nonzero, both KD weights are0. Log losses describe the last accumulation microbatch.'),
        conditions=conditions, rows=rows,
        control_input_hashes={str(path.relative_to(project)): sha(path) for path in [ft_path, prior_path, availability_path]},
        comparison_limits='E2-FT and E3-E2 are matched logical tuple/update-budget controls; one vs two views differ in forward cost. Entire sampled tuple histories were not recorded. E3/E4 fixed-final source/config context must also be assessed, not just arithmetic deltas.',
        post_evaluation_seals=True, source_match_ft_each_evaluation=True, bn_causality_proven=False,
        phase1_complete_record=read(raw / 'records/completed.json'))
    (output / 'comparison.json').write_bytes((json.dumps(report, indent=2) + '\n').encode('utf-8'))
    (output / 'checkpoint_metadata.json').write_bytes((json.dumps(metadata, indent=2) + '\n').encode('utf-8'))
    fields = ['dataset','metric','original_e0','zero_update_availability','ft','e2','e3','e4',
              'e2_minus_ft_pp','e3_minus_e2_pp','e4_minus_e3_pp','e2_minus_original_e0_pp',
              'strict_target','e2_target_exceeded']
    with (output / 'comparison.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    lines = ['# E2完整20轮：归因对照', '',
        '共享C/M两视图任务监督，无KD/R；固定seed2027、最终epoch20/50000更新。完整/Miss各479序列337353帧、退出0，共958次序列运行674706帧。这里聚合真实记录，独立审计待完成。', '',
        '| 条件 | 指标 | E0 | availability | FT | E2 | E3 | E4 | E2−FT | E3−E2 | E4−E3 |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for row in rows:
        lines.append('| {dataset} | {metric} | {original_e0:.6f} | {zero_update_availability:.6f} | {ft:.6f} | {e2:.6f} | {e3:.6f} | {e4:.6f} | {e2_minus_ft_pp:+.6f} | {e3_minus_e2_pp:+.6f} | {e4_minus_e3_pp:+.6f} |'.format(**row))
    complete_up = sum(row['e2_minus_original_e0_pp'] > 0 for row in rows if not row['dataset'].endswith('-Miss'))
    passed = sum(row.get('e2_target_exceeded', False) for row in rows)
    lines += ['', 'E2完整五项中%d项高于原E0，四项严格Miss阈值%d项超过。不能由单seed控制推断一般稳定性、BN因果或三个有效模块。' % (complete_up, passed), '',
        'FT与E2配置只差stage/输出，实际两项评估各106份源哈希相同；训练一视图与两视图的计算量不同。E3−E2、E4−E3给出后续KD/原修复的数值对照，仍需核对执行源码/配置及计算预算；不能仅以减法宣称方法机制普遍成立。', '',
        'E2日志forward/reverse KD与可靠比例只是诊断，权重均0，50000条loss严格等于float32(taskC+taskM)。融合BN计数增加600000、头部36个运行缓冲保持为父进程CPU元数据，不说明性能因果。评分器为已披露rgbt1.0.1口径；后封存不追认为启动封存，不认证作者/SCDT等价。', '']
    (output / 'RESULTS.md').write_bytes('\n'.join(lines).encode('utf-8'))
    print(json.dumps(dict(output=str(output), metric_rows=10, verified_runs=958, verified_frames=674706)))


if __name__ == '__main__':
    main()
