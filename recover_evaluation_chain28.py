"""Restore interrupted deterministic development evaluation and its queue."""
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from evaluate_val97_recovery28 import sha


ROOT = Path("/home/gaob/Miss-RGBT")
OLD = ROOT / "reports/server28-state-offset-replication16-20261031"
OUTPUT = ROOT / "reports/server28-state-offset-replication16-recovered-20261031"


def save(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def check_memory():
    rows = subprocess.check_output([
        "nvidia-smi", "--id=0,1", "--query-gpu=index,memory.free",
        "--format=csv,noheader,nounits"], universal_newlines=True).splitlines()
    assert len(rows) == 2 and all(int(row.split(",")[1]) >= 3000 for row in rows)


def main():
    check_memory()
    OUTPUT.mkdir(exist_ok=False)
    launch = json.loads((OLD / "launch.json").read_text())
    assert sha(OLD / "registration.json") == launch["registration_sha256"]
    for path, expected in launch["source_sha256"].items():
        assert sha(path) == expected
    (OUTPUT / "registration.json").write_bytes((OLD / "registration.json").read_bytes())
    launch["recovery"] = dict(
        old_output=str(OLD), old_launch_sha256=sha(OLD / "launch.json"),
        boot_id=Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
        observed_utc=datetime.datetime.utcnow().isoformat() + "Z",
        original_outputs_preserved=True, fresh_full_video_for_unfinished_tracks=True)
    for name in ["recover_state_offset28.py", "recover_evaluation_chain28.py"]:
        launch["source_sha256"][str(ROOT / name)] = sha(ROOT / name)
    processes = {}
    for gpu in ["0", "1"]:
        spec = OUTPUT / ("gpu%s_spec.json" % gpu)
        spec.write_bytes((OLD / spec.name).read_bytes())
        with (OUTPUT / ("gpu%s.log" % gpu)).open("xb") as log:
            process = subprocess.Popen(
                [sys.executable, "-u", str(ROOT / "recover_state_offset28.py"),
                 "--gpu", gpu, "--spec", str(spec), "--output", str(OUTPUT / ("gpu" + gpu)),
                 "--previous-output", str(OLD / ("gpu" + gpu))],
                cwd=str(ROOT), stdout=log, stderr=subprocess.STDOUT,
                env=dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, OMP_NUM_THREADS="1",
                         MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1"))
        processes[gpu] = process
    launch["children"] = {gpu: process.pid for gpu, process in processes.items()}
    launch["pid"] = os.getpid()
    save(OUTPUT / "launch.json", launch)
    print("RESTORED " + json.dumps(launch["children"]), flush=True)
    while any(process.poll() is None for process in processes.values()):
        time.sleep(300)
    assert all(process.returncode == 0 for process in processes.values())
    subprocess.check_call([sys.executable, str(ROOT / "analyze_state_offset28.py"),
                           "--output", str(OUTPUT)], cwd=str(ROOT))
    previous = json.loads((OUTPUT / "report.json").read_text())
    assert sha(OUTPUT / "report.json") == json.loads(
        (OUTPUT / "completed.json").read_text())["report_sha256"]
    assert previous["past_only_offsets_verified"] and previous["sham_and_complete_bitwise_equal"]
    previous_report = OUTPUT / "report.json"
    for name in ["run_spatial_calibration28", "run_calibration_gap28"]:
        check_memory()
        module = __import__(name)
        old_output = module.OUTPUT
        registration = json.loads((old_output / "registration.json").read_text())
        for path, expected in registration["source_sha256"].items():
            assert sha(path) == expected
        module.OUTPUT = ROOT / "reports" / (old_output.name + "-recovered")
        module.OUTPUT.mkdir(exist_ok=False)
        (module.OUTPUT / "registration.json").write_bytes(
            (old_output / "registration.json").read_bytes())
        current_launch = dict(registration, registration_sha256=sha(
            module.OUTPUT / "registration.json"), recovery_previous_report_sha256=sha(
                previous_report), recovered_controller_pid=os.getpid())
        current_launch["source_sha256"][str(Path(__file__))] = sha(__file__)
        save(module.OUTPUT / "launch.json", current_launch)
        for gpu in ["0", "1"]:
            save(module.OUTPUT / ("gpu%s_spec.json" % gpu), dict(
                cohort=registration["parts"][gpu],
                registration_sha256=current_launch["registration_sha256"]))
        workers = {}
        for gpu in ["0", "1"]:
            with (module.OUTPUT / ("gpu%s.log" % gpu)).open("xb") as log:
                command = [sys.executable, "-u", str(ROOT / "recover_evaluation_chain28.py"),
                           "--worker-module", name, "--gpu", gpu,
                           "--output", str(module.OUTPUT)]
                workers[gpu] = subprocess.Popen(command, cwd=str(ROOT), stdout=log,
                                                stderr=subprocess.STDOUT)
        save(module.OUTPUT / "children.json", {gpu: p.pid for gpu, p in workers.items()})
        while any(process.poll() is None for process in workers.values()):
            time.sleep(300)
        assert all(process.returncode == 0 for process in workers.values())
        module.analyze()
        previous_report = module.OUTPUT / "report.json"
        assert sha(previous_report) == json.loads(
            (module.OUTPUT / "completed.json").read_text())["report_sha256"]
    print("RECOVERED_CHAIN_COMPLETE", flush=True)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker-module", choices=["run_spatial_calibration28", "run_calibration_gap28"])
    parser.add_argument("--gpu", choices=["0", "1"])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.worker_module:
        module = __import__(args.worker_module)
        module.OUTPUT = args.output
        module.worker(args)
    else:
        main()
