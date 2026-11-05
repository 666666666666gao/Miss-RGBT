"""Register one detached controller for the existing training pair."""
import json
import os
from pathlib import Path
import subprocess
import sys

import partitioned_a_official_queue as queue


def main():
    checks_path = queue.ROOT/'reports/server28_partitioned_a_20261102_queue_checks.json'
    checks = queue.read(checks_path)
    assert checks['status'] == 'passed' and checks['benchmark_jobs_executed'] == 0
    assert checks['checkpoint_plan_jobs'] == 20
    for path, expected in checks['source_sha256'].items():
        assert queue.sha(path) == expected, path
    assert not queue.QUEUE.exists()
    log = queue.ROOT/'reports/server28_partitioned_a_20261102_official_queue.log'
    receipt = queue.ROOT/'reports/server28_partitioned_a_20261102_official_queue.launch.json'
    assert not receipt.exists()
    script = Path(queue.__file__).resolve()
    command = [sys.executable, '-u', str(script), '--register']
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1',
        MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', WANDB_MODE='disabled',
        PYTHONPATH=str(queue.RUNTIME)+':'+str(queue.ROOT))
    with log.open('xb') as stream:
        process = subprocess.Popen(command, cwd=str(queue.ROOT), env=environment,
            stdout=stream, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
            start_new_session=True)
    result = dict(pid=process.pid, command=command, log=str(log),
        registered_utc=queue.now(), poll_seconds=300, cuda_visible_devices='',
        script_sha256=queue.sha(script), checks_sha256=queue.sha(checks_path),
        queues_existing_training=True, restarts_training=False,
        benchmark_evaluation_starts_only_after_terminal_training_verification=True)
    with receipt.open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
