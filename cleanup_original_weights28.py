"""Remove original campaign weights only after all four final verifications."""
import argparse
import datetime
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time

from evaluate_val97_recovery28 import sha


ROOT = Path("/home/gaob/Miss-RGBT")
LABELS = ["a_utility_full20_seed2027", "a_random_full20_seed2027",
          "batch10_utility_seed2027", "batch10_random_seed2027"]
OUTPUT = ROOT / "reports/server28-original-weight-cleanup-20261031"
WITNESS = ROOT / "reports/server28-original-weight-cleanup-witness-20261031.json"


def read(path):
    return json.loads(path.read_text())


def save(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")


def inventory(base):
    records = []
    for label in LABELS:
        folder = base / "runs/server28" / label
        assert folder.resolve() == folder and folder.is_dir()
        for path in sorted(folder.glob("epoch*.pth.tar")):
            assert not path.is_symlink() and path.resolve().parent == folder
            assert re.fullmatch(r"epoch[0-9]{2}\.pth\.tar", path.name)
            assert 1 <= int(path.name[5:7]) <= 20
            records.append(dict(label=label, path=str(path), bytes=path.stat().st_size,
                                sha256=sha(path)))
    return records


def remove_registered(base, records, selected, journal):
    assert set(selected) == set(LABELS)
    assert inventory(base) == records
    selected_paths = []
    for label, item in selected.items():
        path = Path(item["path"])
        assert path.parent == base / "runs/server28-recovery-20261031" / label
        assert path.resolve() == path and sha(path) == item["sha256"]
        selected_paths.append(path)
    removed = []
    with journal.open("x") as stream:
        for item in records:
            path = Path(item["path"])
            assert path not in selected_paths
            links = path.stat().st_nlink
            path.unlink()
            receipt = dict(item, links_before=links, removed=True)
            stream.write(json.dumps(receipt) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
            removed.append(receipt)
    assert inventory(base) == []
    for item in selected.values():
        assert sha(item["path"]) == item["sha256"]
    return removed


def witness():
    with tempfile.TemporaryDirectory(prefix="weight-cleanup-witness-",
                                     dir=str(ROOT / "reports")) as directory:
        base = Path(directory).resolve()
        selected = {}
        for label in LABELS:
            original = base / "runs/server28" / label
            retained = base / "runs/server28-recovery-20261031" / label
            original.mkdir(parents=True)
            retained.mkdir(parents=True)
            old = original / "epoch10.pth.tar"
            old.write_bytes(("old-" + label).encode())
            target = retained / "epoch14.pth.tar"
            target.write_bytes(("selected-" + label).encode())
            selected[label] = dict(path=str(target), sha256=sha(target))
        extra = base / "unrelated.pth.tar"
        extra.write_bytes(b"unrelated")
        extra_sha = sha(extra)
        records = inventory(base)
        assert len(records) == 4
        removed = remove_registered(base, records, selected, base / "deleted.jsonl")
        assert len(removed) == 4 and sha(extra) == extra_sha
        assert len((base / "deleted.jsonl").read_text().splitlines()) == 4
    save(WITNESS, dict(source_sha256=sha(__file__), files_removed=4,
                       selected_weights_preserved=True, unrelated_weight_preserved=True,
                       scope="Actual deletion of disposable fixture files through the shared "
                             "inventory/removal body, not final campaign verification."))


def launch():
    assert read(WITNESS)["source_sha256"] == sha(__file__)
    verification_queue = ROOT / "reports/server28-recovery-20261031/independent_verification_queue.json"
    jobs = read(verification_queue)["jobs"]
    for label, item in jobs.items():
        command = (Path("/proc") / str(item["pid"]) / "cmdline").read_bytes()
        assert b"/home/gaob/Miss-RGBT/verify_recovery_campaign28.py" in command
        assert label.encode() in command
    records = inventory(ROOT)
    assert all(any(item["label"] == label for item in records) for label in LABELS)
    OUTPUT.mkdir(exist_ok=False)
    save(OUTPUT / "registration.json", dict(
        registered_utc=datetime.datetime.utcnow().isoformat() + "Z",
        source_sha256=sha(__file__), witness_sha256=sha(WITNESS), labels=LABELS,
        original_files=records, verification_queue_sha256=sha(verification_queue),
        poll_seconds=300,
        scope="Only registered original epoch weights from these four campaigns. "
              "Wait for all independent final-five and official evaluations; "
              "keep all four validation-selected recovery weights. "
              "No threshold or three-module certification."))
    with (OUTPUT / "worker.log").open("xb") as log:
        command = [sys.executable, "-u", str(Path(__file__)), "--mode", "wait"]
        process = subprocess.Popen(
            command, cwd=str(ROOT), stdout=log, stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL, start_new_session=True,
            env=dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1",
                     MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1"))
    result = dict(pid=process.pid, command=command,
                  registration_sha256=sha(OUTPUT / "registration.json"))
    save(OUTPUT / "launch.json", result)
    print(json.dumps(result), flush=True)


def wait():
    registration = read(OUTPUT / "registration.json")
    assert registration["source_sha256"] == sha(__file__)
    campaign = ROOT / "reports/server28-recovery-20261031"
    queue_path = campaign / "independent_verification_queue.json"
    assert sha(queue_path) == registration["verification_queue_sha256"]
    queue = read(queue_path)
    while True:
        pending = []
        for label in LABELS:
            if not (campaign / label / "independent_verification.json").exists():
                pending.append(label)
                process = Path("/proc") / str(queue["jobs"][label]["pid"])
                command = (process / "cmdline").read_bytes()
                assert b"/home/gaob/Miss-RGBT/verify_recovery_campaign28.py" in command
                assert label.encode() in command
        if not pending:
            break
        print("WAITING " + json.dumps(pending), flush=True)
        time.sleep(300)
    for path, expected in queue["source_sha256"].items():
        assert sha(path) == expected
    selected, verifications = {}, {}
    for label in LABELS:
        folder = campaign / label
        path = folder / "independent_verification.json"
        verified, official = read(path), read(folder / "official_completed.json")
        assert verified["label"] == label
        assert verified["verifier_sha256"] == sha(ROOT / "verify_recovery_campaign28.py")
        assert verified["selected_weight_and_recovery_cleanup_verified"]
        assert verified["development_trajectories_verified"] == 970
        assert verified["official_trajectories_verified"] == 958
        best = official["best"]
        assert verified["selected_epoch"] == best["epoch"]
        assert verified["checkpoint_sha256"] == best["checkpoint_sha256"]
        assert official["official_test_selection_used"] is False
        for condition in ["complete", "miss"]:
            assert verified["official_metrics"][condition] == official["official_" + condition]
        selected[label] = dict(path=best["checkpoint"], sha256=best["checkpoint_sha256"])
        verifications[label] = dict(path=str(path), sha256=sha(path))
    cpu = ROOT / "reports/server28-candidate-repair-audit-20261031"
    assert sha(cpu / "analysis_corrected.json") == read(
        cpu / "analysis_corrected_completed.json")["analysis_sha256"]
    assert sha(cpu / "report.json") == read(cpu / "completed.json")["report_sha256"]
    save(OUTPUT / "approved_plan.json", dict(
        registration_sha256=sha(OUTPUT / "registration.json"),
        selected=selected, verification_reports=verifications,
        cpu_analysis_sha256=sha(cpu / "analysis_corrected.json"),
        original_files=registration["original_files"]))
    removed = remove_registered(ROOT, registration["original_files"], selected,
                                OUTPUT / "deleted.jsonl")
    save(OUTPUT / "completed.json", dict(
        selected=selected, removed=removed, original_weight_cleanup_verified=True,
        removed_path_bytes=sum(item["bytes"] for item in removed),
        last_link_removed_bytes=sum(item["bytes"] for item in removed if item["links_before"] == 1),
        approved_plan_sha256=sha(OUTPUT / "approved_plan.json"),
        journal_sha256=sha(OUTPUT / "deleted.jsonl"),
        scope="Only these four original training directories were cleaned. "
              "Independent metric reports remain unchanged; no benchmark/module attainment claim."))
    print("ORIGINAL_WEIGHT_CLEANUP_COMPLETE", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["witness", "launch", "wait"], required=True)
    args = parser.parse_args()
    {"witness": witness, "launch": launch, "wait": wait}[args.mode]()
