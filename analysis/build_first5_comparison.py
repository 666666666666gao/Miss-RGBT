"""Compare recorded, completed epoch05 results; does not rescore predictions."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess

from build_full20_comparison import DATASETS, MISS_TARGETS, read_baselines, read_json, verify_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project-root', required=True)
    args = parser.parse_args()
    root = Path(args.project_root).resolve()
    inputs = {}
    baselines = read_baselines(root, inputs)
    collection = read_json(root / 'reports/first5_evaluation/collection.json', inputs)
    models, rows, repair_deltas = {}, [], []
    for stage in ['e3', 'e4']:
        training = root / 'reports/first5_training' / stage
        completed = read_json(training / 'completed.json', inputs)
        config = read_json(training / 'run_config.json', inputs)
        assert completed['epochs'] == 5 and completed['updates'] == 12500
        executed_config_ref = '58fe5819c6e82770a5a36025924f5a05030ac7cc:configs/' + stage + '.json'
        executed_config = subprocess.check_output(['git', 'show', executed_config_ref], cwd=str(root))
        inputs['git:' + executed_config_ref] = hashlib.sha256(executed_config).hexdigest()
        assert config == json.loads(executed_config)
        assert config['seed'] == 2027 and config['reverse_weight'] == 0
        models[stage] = dict(training=completed, config=config, evaluations={})
        for condition in ['complete', 'miss']:
            label = stage + '_' + condition
            folder = root / 'reports' / label
            run = read_json(folder / 'run.json', inputs)
            finished = read_json(folder / 'completed.json', inputs)
            metrics = read_json(folder / 'metrics.json', inputs)
            verify_metrics(metrics, condition)
            assert finished['sequences'] == 479 and finished['frames'] == 337353
            assert Path(run['checkpoint']).name == 'epoch05.pth.tar'
            assert run['checkpoint_sha256'] == collection['checkpoint_sha256'][stage]
            assert run['condition'] == condition and run['physical_gpus'] == [2, 3]
            models[stage]['evaluations'][condition] = metrics
            for dataset, (sequences, frames, keys) in DATASETS.items():
                name = dataset + ('-Miss' if condition == 'miss' else '')
                for metric in keys:
                    value, baseline = metrics[name][metric], baselines[condition][name][metric]
                    row = dict(experiment=stage.upper(), dataset=name, metric=metric, value=value,
                               baseline=baseline, delta_pp=value-baseline,
                               relative_change_percent=100*(value-baseline)/baseline,
                               sequences=sequences, frames=frames)
                    if condition == 'miss' and metric in MISS_TARGETS[dataset]:
                        target = MISS_TARGETS[dataset][metric]
                        row.update(strict_target=target, target_exceeded=value > target)
                    rows.append(row)
    for e3, e4 in zip(rows[:10], rows[10:]):
        assert (e3['dataset'], e3['metric']) == (e4['dataset'], e4['metric'])
        repair_deltas.append(dict(dataset=e3['dataset'], metric=e3['metric'], delta_e4_minus_e3_pp=e4['value']-e3['value']))
    output = root / 'reports/first5_evaluation'
    report = dict(scope='Recorded final epoch05; 12500 updates each; four full evaluations; single seed. '
                        'This aggregation does not independently rescore bbox files or locally hash checkpoint bytes. '
                        'Not the running final epoch20 results; no E5 optimization or three-module efficacy claim.',
                  baselines=baselines, models=models, rows=rows, repair_deltas=repair_deltas, input_sha256=inputs)
    (output / 'comparison.json').write_bytes((json.dumps(report, indent=2)+'\n').encode('utf-8'))
    fields = ['experiment', 'dataset', 'metric', 'value', 'baseline', 'delta_pp', 'relative_change_percent',
              'sequences', 'frames', 'strict_target', 'target_exceeded']
    with (output / 'comparison.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    lines = ['# 首轮5轮全量指标', '',
             'E3/E4各5×20000采样样本、12500更新，seed2027，固定epoch05。每组479序列、337353帧。指标为百分数，括号为相对原始epoch20 E0的百分点差值。此表汇总记录值，独立bbox/曲线复算见审计。', '']
    for condition in ['complete', 'miss']:
        for dataset, (_, _, keys) in DATASETS.items():
            name = dataset + ('-Miss' if condition == 'miss' else '')
            lines += ['## '+name, '', '| 实验 | '+' | '.join(keys)+' |', '|---|'+'|'.join(['---:']*len(keys))+'|']
            lines.append('| E0 | '+' | '.join('%.6f' % baselines[condition][name][key] for key in keys)+' |')
            for stage in ['e3', 'e4']:
                selected = {v['metric']: v for v in rows if v['experiment']==stage.upper() and v['dataset']==name}
                lines.append('| '+stage.upper()+' | '+' | '.join('%.6f (%+.6f)' %
                             (selected[key]['value'], selected[key]['delta_pp']) for key in keys)+' |')
            lines.append('')
    lines += ['## E4相对E3：局部修复的配对差值', '', '| 数据集 | 指标 | 差值（pp） |', '|---|---|---:|']
    lines += ['| {dataset} | {metric} | {delta_e4_minus_e3_pp:+.6f} |'.format(**row) for row in repair_deltas]
    lines += ['', '## Miss严格目标', '', '| 实验 | 数据集 | 指标 | 目标 | 实际 | 超过目标 |', '|---|---|---|---:|---:|---|']
    for row in rows:
        if 'strict_target' in row:
            lines.append('| {experiment} | {dataset} | {metric} | >{strict_target} | {value:.6f} | {passed} |'.format(
                passed='是' if row['target_exceeded'] else '否', **row))
    lines += ['', '## 发现及证据限制', '',
              '1. E3仅LasHeR-Miss SR超过预设阈值；其余三个主要Miss阈值未超过。E4四个主要Miss阈值均未超过。完整条件主要指标均低于E0，尚未满足完整输入约+2pp目标。',
              '2. E4相对E3在LasHeR-Miss PR/SR下降、RGBT234-Miss MPR/MSR上升。局部修复的效果混合，不能称其在两个数据集均有效。',
              '3. 完整输入修复残差为零，训练共享参数仍会改变完整能力；此处完整成绩下降是实际观察，具体机制尚未通过因果实验确定。',
              '4. E3/E4 reverse_weight均为0；反向项日志及单批CPU梯度见证不能代替E5正式性能实验。单seed不报告标准差或显著性。',
              '5. 下一步先完成当前冻结20轮及固定epoch20四组评估。后续在train/val上验证目标相关补偿、完整任务保护，另补E2、参数容量对照、E5可靠性反馈对照及多seed；这些均为待验证假设。',
              '6. 所有新方法评估记录103个源码文件，其中102个匹配4c473fa；辅助验证脚本的实际部署差异和字节快照另存于source_difference.json及source_snapshots。不能将全部103文件称为逐字节匹配该提交。', '']
    (output / 'RESULTS.md').write_bytes(('\n'.join(lines)).encode('utf-8'))
    print(json.dumps(dict(metric_rows=len(rows), repair_delta_rows=len(repair_deltas), output=str(output))))


if __name__ == '__main__':
    main()
