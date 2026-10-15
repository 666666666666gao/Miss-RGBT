"""Aggregate the closed zero-update availability control; no model or rescoring."""
import argparse
import csv
import json
from pathlib import Path

from build_full20_comparison import DATASETS, MISS_TARGETS, read_baselines, read_json, sha, verify_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project-root', required=True)
    parser.add_argument('--raw-root', required=True)
    parser.add_argument('--collection-record', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    root, raw = Path(args.project_root), Path(args.raw_root)
    inputs = {str(Path(__file__)):sha(Path(__file__))}
    baselines = read_baselines(root, inputs)
    inventory = read_json(root/'reports/e0/inventory.json', inputs)
    collection = read_json(Path(args.collection_record), inputs)
    pack = read_json(raw/'packaging.json', inputs)
    assert pack['optimizer_updates'] == 0 and pack['tensor_equality'] is True and pack['model_state_keys'] == 250
    expected = {(dataset, item['name']):item['frames'] for dataset, spec in inventory['datasets'].items()
        for item in spec['sequence_details']}
    evaluations, rows = {}, []
    for condition in ['complete','miss']:
        label = 'e0_availability_'+condition
        folder = raw/'results'/(label+'_20261015')
        meta = read_json(folder/'run.json', inputs)
        assert meta['condition'] == condition and meta['checkpoint_sha256'] == pack['packaged_sha256']
        assert meta['physical_gpus'] == [2,3] and meta['workers'] == 8
        completed = read_json(folder/'completed.json', inputs)
        assert completed['sequences'] == 479 and completed['frames'] == 337353
        exit_record = read_json(raw/'records'/(label+'_exit.json'), inputs)
        assert exit_record['exit_code'] == 0
        metrics = read_json(folder/'metrics.json', inputs)
        verify_metrics(metrics, condition)
        receipts = read_json(folder/'all_receipts.json', inputs)
        assert len(receipts) == 479 and {(row['dataset'],row['sequence']) for row in receipts} == set(expected)
        for row in receipts:
            dataset, name = row['dataset'], row['sequence']
            assert row['frames'] == expected[dataset,name] and row['physical_gpu'] in [2,3]
            assert row == read_json(folder/'receipts'/dataset/(name+'.json'), inputs)
            bbox = folder/'bbox'/dataset/(name+'.txt')
            inputs[str(bbox)] = sha(bbox)
            assert inputs[str(bbox)] == row['bbox_sha256']
            assert len(bbox.read_text().splitlines()) == row['frames']
        evaluations[condition] = dict(completed=completed,metrics=metrics,exit_record=exit_record,
            checkpoint_sha256=meta['checkpoint_sha256'],verified_receipts=479)
        for dataset, (sequences, frames, keys) in DATASETS.items():
            name = dataset+('-Miss' if condition == 'miss' else '')
            curves = read_json(folder/(name+'_curves.json'), inputs)
            for metric in keys:
                curve = curves[metric]
                success = metric in ['SR','MSR']
                assert len(curve) == sequences and all(len(value) == (21 if success else 51) for value in curve)
                mean = (sum(sum(value)/len(value) for value in curve) if success else
                    sum(value[20] for value in curve))/sequences*100
                assert abs(mean-metrics[name][metric]) < 1e-12
                value, baseline = metrics[name][metric], baselines[condition][name][metric]
                row = dict(dataset=name,metric=metric,value=value,baseline=baseline,delta_pp=value-baseline,
                    sequences=sequences,frames=frames)
                if condition == 'miss' and metric in MISS_TARGETS[dataset]:
                    row.update(strict_target=MISS_TARGETS[dataset][metric],target_exceeded=value>MISS_TARGETS[dataset][metric])
                rows.append(row)
    pair = read_json(raw/'records/e0_availability_complete_pair.json', inputs)
    assert len(pair['rows']) == 479 and sum(row['frames'] for row in pair['rows']) == 337353
    assert pair['identical_sequences'] == sum(row['identical'] for row in pair['rows'])
    complete_check = dict(exactly_equal_sequence_arrays=pair['identical_sequences'],sequences=479,
        max_abs_bbox_difference=max(row['max_abs_bbox_difference'] for row in pair['rows']),
        all_five_primary_metrics_exactly_equal=all(row['delta_pp'] == 0 for row in rows if '-Miss' not in row['dataset']))
    assert complete_check['all_five_primary_metrics_exactly_equal']
    output = Path(args.output)
    output.mkdir(parents=True,exist_ok=False)
    report = dict(scope='Zero-update E0 availability-aware inference-path control. Same original E0 tensors by packaging witness; '
        'two complete/Miss full evaluations, not new training. Aggregates recorded metrics/curves and verifies trajectory receipts; '
        'this producer does not reconstruct scores from GT. Independent audit separate. '
        'Input handling includes missing embeddings and historical-template availability, not an isolated one-line intervention.',
        source_version=(raw/'source_version.txt').read_text().strip(),packaging=pack,collection=collection,
        complete_wrapper_check=complete_check,baselines=baselines,evaluations=evaluations,rows=rows,input_sha256=inputs)
    (output/'comparison.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    fields=['dataset','metric','value','baseline','delta_pp','sequences','frames','strict_target','target_exceeded']
    with (output/'comparison.csv').open('w',newline='',encoding='utf-8') as handle:
        writer=csv.DictWriter(handle,fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    lines=['# E0-availability：零训练输入处理对照','',
        '同一原E0 epoch20张量，0次更新、无KD/R。两项完整/Miss评估均exit0，共958次序列运行、674706帧。不是新增训练结果。','',
        '| 条件 | 指标 | 原E0 | E0-availability | 差值pp |','|---|---|---:|---:|---:|']
    for row in rows:
        lines.append('| {dataset} | {metric} | {baseline:.6f} | {value:.6f} | {delta_pp:+.6f} |'.format(**row))
    lines += ['',
        '完整输入五项主指标与E0精确相等。479个完整序列中254个bbox数组逐值精确相等；其余存在极小浮点差，最大绝对差为%.16g，不能声称479条轨迹字节相同。'%complete_check['max_abs_bbox_difference'],
        '',
        '缺失输入五项均下降，严格Miss目标0/4。该对照说明无额外训练时，新可用性感知推理路径整体会改变缺失表现；不能把后续E3/E4相对原E0的变化全部归于KD或修复。它也不能单独区分embedding屏蔽与历史模板标识的效应，仍须等待同预算E0-FT、E2以及BN对照。',
        '',
        '原始闭合记录归档SHA256：`'+collection['archive_sha256']+'`。原/封装权重只由父执行器在服务器流式核验哈希，审计者未据此获得独立本地二进制验证。当前评分包范围3556文件中，先前本地快照1967文件全部逐字节哈希一致，含本次713份GT及评分代码；其余1589个GT树文件仅保留远端哈希元数据，未全部下载认证。封存时间在评估后，不能改写成运行前封存。',
        '',
        '完整独立GT/曲线复算与科学表述审计见后续EXPERIMENT_AUDIT.md/json。主指标仍沿用安装评分包；上游官方实现等价性未独立认证。',
        '']
    (output/'RESULTS.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps(dict(output=str(output),metrics=len(rows),complete_wrapper_check=complete_check)),flush=True)


if __name__ == '__main__':
    main()
