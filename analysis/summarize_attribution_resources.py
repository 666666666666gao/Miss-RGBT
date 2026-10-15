"""Summarize the single planned GPU2/3+CPU sample; no live hardware query."""
import argparse
import json
from pathlib import Path
import statistics


GIB = 1024 ** 3


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--profile',required=True)
    parser.add_argument('--intake',required=True)
    parser.add_argument('--output',required=True)
    args = parser.parse_args()
    profile = json.loads(Path(args.profile).read_text(encoding='utf-8'))
    intake = json.loads(Path(args.intake).read_text(encoding='utf-8'))
    samples = profile['samples']
    assert len(samples) == profile['sample_count'] == 6 and profile['logical_cpu_count'] == 64
    gpus = []
    next_training_peak = intake['training']['e2']['last']['peak_gpu_bytes']/GIB
    for gpu in [2,3]:
        rows = [next(row for row in sample['gpus'] if row['physical_gpu'] == gpu) for sample in samples]
        free = min(row['free_mib'] for row in rows)/1024
        assert all(row['total_mib'] == 24576 for row in rows)
        gpus.append(dict(physical_gpu=gpu,total_gib=24,
            used_gib_mean=statistics.mean(row['used_mib']/1024 for row in rows),free_gib_min=free,
            utilization_percent_mean=statistics.mean(row['gpu_util_percent'] for row in rows),
            utilization_percent_min=min(row['gpu_util_percent'] for row in rows),
            utilization_percent_max=max(row['gpu_util_percent'] for row in rows),
            next_dual_view_allocated_peak_gib=next_training_peak,
            free_minus_allocated_peak_gib=free-next_training_peak,
            next_required_dual_view_training_fits_current_free_memory=free>=next_training_peak))
    busy = [row['cpu_busy_percent'] for row in samples if 'cpu_busy_percent' in row]
    waits = [row['cpu_iowait_percent'] for row in samples if 'cpu_iowait_percent' in row]
    assert len(busy) == len(waits) == 5
    host = dict(logical_cpu_count=64,cpu_busy_percent_interval_mean=statistics.mean(busy),
        cpu_iowait_percent_interval_mean=statistics.mean(waits),
        available_ram_gib_min=min(row['host_memory_bytes']['MemAvailable']/GIB for row in samples),
        available_ram_gib_mean=statistics.mean(row['host_memory_bytes']['MemAvailable']/GIB for row in samples),
        load_average_last=samples[-1]['load_average'])
    assert not any(row['next_required_dual_view_training_fits_current_free_memory'] for row in gpus)
    report = dict(scope='One60-second nominal sample,6 GPU/RAM point observations and5 CPU intervals; '
        'no temperature/power/other-GPU query, no training instrumentation, no same-GPU concurrency trial. '
        'nvidia-smi utilization is activity percentage, not a measured FLOP/s saturation claim. '
        'Allocated peak excludes extra CUDA/runtime overhead and is already above current free memory.',
        start_utc=samples[0]['observed_utc'],end_utc=samples[-1]['observed_utc'],
        source_profile=args.profile,source_training_intake=args.intake,gpus=gpus,host=host,
        decision='Keep one formal training per physical GPU2/3, two simultaneous jobs total. '
            'No additional required dual-view training fits observed free memory. CPU/RAM aggregate headroom does not resolve GPU memory admission.',
        same_gpu_parallel_speedup_or_slowdown_measured=False,
        training_rates={stage:row['recent_updates_per_second'] for stage,row in intake['training'].items()})
    output = Path(args.output)
    output.mkdir(parents=True,exist_ok=False)
    (output/'summary.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    lines=['# 2026-10-15 14:11 GPU2/3与主机资源采样','',
        '预定诊断节点的一次60秒采样：6个GPU/RAM时点、5个CPU区间。保持每卡一个正式训练，共两项；未叠加新任务。','',
        '| 物理卡 | 显存已用GiB均值 | 显存余量GiB最小值 | GPU利用率均值 | 范围 |','|---|---:|---:|---:|---:|']
    for row in gpus:
        lines.append('| {physical_gpu} | {used_gib_mean:.2f} | {free_gib_min:.2f} | {utilization_percent_mean:.2f}% | {utilization_percent_min}–{utilization_percent_max}% |'.format(**row))
    lines += ['',
        '64逻辑核CPU区间利用率均值%.2f%%，I/O等待%.3f%%；RAM可用均值%.2fGiB、最小%.2fGiB；最后1/5/15分钟load average为%s。'%(host['cpu_busy_percent_interval_mean'],host['cpu_iowait_percent_interval_mean'],host['available_ram_gib_mean'],host['available_ram_gib_min'],host['load_average_last']),
        '',
        '下一项双路训练实测CUDA分配峰值约%.2fGiB，已超过GPU2/3当前余量，且还未计额外运行时开销。基于显存准入与当前GPU活动率保持原安排。没有实施同卡并发速度对比，不能声称已经实测并发会变慢或给出减速比例。'%next_training_peak,
        '',
        '14:10训练日志尾部约200更新的速率：E0-FT %.6f updates/s，E2 %.6f updates/s；两者任务前向数量不同，不作为同负载的并发加速对比。'%(report['training_rates']['e0_ft'],report['training_rates']['e2']),
        '',
        '## 采样与汇总代码变更','',
        '| 文件 | 类型 | 内容 |','|---|---|---|',
        '| analysis/profile_gpu23_host.py | 新增 | 本次实际SSH执行的Linux探针正文；只读GPU2/3及聚合CPU/RAM，未修改训练代码 |',
        '| analysis/summarize_attribution_resources.py | 新增 | 对既有采样与训练快照离线汇总，不重复查询硬件 |',
        '| profile_output/resources_20261015_1410.json | 新增 | 原始6时点结果 |',
        '| profile_output/attribution_20261015_1410/ | 新增 | 本摘要与机器可读判断 |','']
    (output/'README.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps(report),flush=True)


if __name__ == '__main__':
    main()
