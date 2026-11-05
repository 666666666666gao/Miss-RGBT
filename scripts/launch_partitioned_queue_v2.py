"""Replace only the waiting queue after verified pairing-contract correction."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys

import partitioned_a_official_queue as old
import partitioned_a_official_queue_v2 as new


def main():
    checks_path = new.ROOT/'reports/server28_partitioned_pairing_v2_checks_20261102.json'
    checks = new.read(checks_path)
    assert checks['status'] == 'passed'
    for path, digest in checks['source_sha256'].items():
        assert new.sha(path) == digest, path
    assert new.read(old.QUEUE/'status.json')['stage'] == 'WAITING_FOR_TRAINING'
    old.check_sources()
    registration = old.read(old.QUEUE/'registration.json')
    assert registration['pid'] == 1318842
    for item in registration['training']:
        assert old.training_alive(item)
        assert not (Path(item['config']['output'])/'development_history.json').exists()
    assert not old.RESULTS.exists() and not new.QUEUE.exists()
    status = subprocess.check_output(['ps', '-p', str(registration['pid']), '-o', 'args='],
                                     universal_newlines=True)
    assert str(Path(old.__file__).resolve()) in status and '--register' in status
    assert not (old.QUEUE/'training_verification.log').exists()
    # Retire the sleeping controller, never either training worker.
    os.kill(registration['pid'], signal.SIGTERM)
    log = new.ROOT/'reports/server28_partitioned_a_20261102_official_queue_v2.log'
    command = [sys.executable, '-u', str(Path(new.__file__).resolve()), '--register']
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1',
        MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', WANDB_MODE='disabled',
        PYTHONPATH=str(new.RUNTIME)+':'+str(new.ROOT))
    with log.open('xb') as stream:
        process = subprocess.Popen(command, cwd=str(new.ROOT), env=environment,
            stdout=stream, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
            start_new_session=True)
    result = dict(pid=process.pid, command=command, log=str(log), registered_utc=new.now(),
        superseded_pid=registration['pid'], superseded_queue=str(old.QUEUE),
        reason='Correct verifier whole-log equality to common-prefix equality for independent early stopping',
        training_pids_unchanged=[item['pid'] for item in registration['training']],
        checks_sha256=new.sha(checks_path),
        script_sha256=new.sha(Path(new.__file__)),
        original_queue_artifacts_preserved=True, training_restarted=False)
    with (new.ROOT/'reports/server28_partitioned_a_20261102_official_queue_v2.launch.json').open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    old.save(old.QUEUE/'superseded.json', result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
