import json
from pathlib import Path


report = {}
for stage in ['e3', 'e4']:
    path = Path('checkpoints') / (stage + '_speed_smoke_seed2027')
    rows = [json.loads(line) for line in (path / 'steps.jsonl').read_text().splitlines()]
    assert len(rows) == 25 and rows[-1]['update'] == 25
    checkpoint = path / 'smoke.pth.tar'
    assert checkpoint.is_file()
    seconds_per_update = (rows[-1]['seconds'] - rows[0]['seconds']) / 24
    report[stage] = dict(optimizer_updates=25, seconds=rows[-1]['seconds'],
        seconds_per_update_after_first=seconds_per_update,
        predicted_5_epoch_training_hours=12500 * seconds_per_update / 3600,
        peak_gpu_bytes=rows[-1]['peak_gpu_bytes'], input_sha256=rows[0]['input_sha256'],
        effective_batch=8, final_status=rows[-1], checkpoint_exists=True)
assert report['e3']['input_sha256'] == report['e4']['input_sha256']
report['matched_first_update_last_microbatch_input'] = True
Path('reports/paired_speed_smoke.json').write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
