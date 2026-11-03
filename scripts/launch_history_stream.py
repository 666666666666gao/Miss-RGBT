"""Start the frozen diagnostic only on a free, capped physical GPU."""
import argparse
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', required=True, type=int, choices=range(4))
    parser.add_argument('--cache-receipt', required=True)
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--injection-layer', type=int, choices=(0, 3), default=3)
    parser.add_argument('--previous-diagnostic')
    args = parser.parse_args()
    root = Path('/home/gaob/Miss-RGBT')
    runtime = root / 'runtime/task_utility_20261018'
    output = Path(args.output).resolve()
    assert not output.exists()
    receipt = json.loads(Path(args.cache_receipt).read_text())
    assert receipt['status'] == 'complete' and receipt['all_historical_frame_exclusions_verified']
    assert Path(args.config).is_file()
    assert (args.previous_diagnostic is None) == (args.injection_layer == 3)
    if args.previous_diagnostic is not None:
        assert Path(args.previous_diagnostic).is_file()
    values = subprocess.check_output(['nvidia-smi', '--id=' + str(args.gpu),
        '--query-gpu=memory.used,temperature.gpu,power.limit',
        '--format=csv,noheader,nounits'], universal_newlines=True)
    memory, temperature, power = [float(value.strip()) for value in values.split(',')]
    assert memory < 500 and temperature < 85 and power == 250, values
    log_path = output.with_name(output.name + '_launch.log')
    launch_path = output.with_name(output.name + '_launch.json')
    assert not launch_path.exists()
    command = [sys.executable, '-u', str(root / 'scripts/diagnose_history_stream.py'),
        '--cache-receipt', str(Path(args.cache_receipt).resolve()),
        '--config', str(Path(args.config).resolve()), '--output', str(output),
        '--injection-layer', str(args.injection_layer)]
    if args.previous_diagnostic is not None:
        command += ['--previous-diagnostic', str(Path(args.previous_diagnostic).resolve())]
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES=str(args.gpu),
        OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
        WANDB_MODE='disabled', PYTHONPATH=str(runtime) + ':' + str(root))
    with log_path.open('xb') as stream:
        child = subprocess.Popen(command, cwd=str(runtime), env=environment,
            stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
    record = dict(pid=child.pid, gpu=args.gpu, command=command,
        started_utc=datetime.datetime.utcnow().isoformat() + 'Z',
        log=str(log_path), output=str(output),
        preflight=dict(memory_mib=memory, temperature_c=temperature, power_limit_w=power),
        scope='Diagnostic launch receipt; not a completion or a passed history gate.')
    with launch_path.open('x') as stream:
        json.dump(record, stream, indent=2)
        stream.write('\n')
    print(json.dumps(record), flush=True)


if __name__ == '__main__':
    main()
