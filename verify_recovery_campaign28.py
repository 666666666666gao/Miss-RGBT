"""Read-only independent verification of final-five selection and official scores."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import rgbt

from analyze_epoch05_health28 import metrics
from evaluate_val97_recovery28 import sha


ROOT = Path("/home/gaob/Miss-RGBT")
LABELS = ["a_utility_full20_seed2027", "a_random_full20_seed2027",
          "batch10_utility_seed2027", "batch10_random_seed2027"]


def read(path):
    return json.loads(path.read_text())


def verify_development(destination, protocol_path, expected_checkpoint):
    protocol = read(protocol_path)
    launch, completed = read(destination / "launch.json"), read(destination / "completed.json")
    assert launch["protocol_sha256"] == sha(protocol_path)
    assert launch["checkpoint_sha256"] == completed["checkpoint_sha256"] == expected_checkpoint
    condition = launch["condition"]
    mask = protocol["conditions"][condition]
    assert sha(mask["path"]) == mask["sha256"]
    cohort = {item["name"]: item for item in protocol["sequences"]}
    receipts = read(destination / "receipts.json")
    by_name = {item["sequence"]: item for item in receipts}
    assert len(receipts) == len(by_name) == len(cohort) == 97
    assert set(by_name) == set(cohort)
    verified = []
    for name, item in cohort.items():
        assert sha(item["gt_path"]) == item["gt_sha256"]
        gt = np.loadtxt(item["gt_path"], delimiter=",", ndmin=2)
        path = destination / (name + ".txt")
        assert sha(path) == by_name[name]["bbox_sha256"]
        boxes = np.loadtxt(str(path), ndmin=2)
        assert boxes.shape == gt.shape == (item["frames"], 4)
        assert np.isfinite(boxes).all()
        measured = metrics(boxes, gt)
        for key, value in measured.items():
            assert np.isclose(value, by_name[name][key], rtol=0, atol=1e-9)
        assert by_name[name]["video_frames"] == item["frames"]
        verified.append(measured)
    assert completed["frames"] == sum(item["frames"] for item in cohort.values()) == 54646
    assert completed["sequences"] == 97
    for key in ["PR20", "SR"]:
        assert np.isclose(np.mean([item[key] for item in verified]), completed[key],
                          rtol=0, atol=1e-9)
    return completed


def verify_official(destination, checkpoint, condition, expected_metrics):
    run = read(destination / "run.json")
    done, saved_metrics = read(destination / "completed.json"), read(destination / "metrics.json")
    assert run["checkpoint_sha256"] == checkpoint
    assert run["condition"] == condition and run["limit"] is None
    assert run["sequences"] == done["sequences"] == 479
    assert run["frames"] == done["frames"] == 337353
    assert saved_metrics == expected_metrics
    evaluator = ROOT / "evaluate_candidate_official28.py"
    assert run["standalone_evaluator_sha256"] == sha(evaluator)
    runtime = ROOT / "runtime/task_utility_20261018"
    for name, expected in run["source_sha256"].items():
        assert sha(runtime / name) == expected, name
    inventory = read(runtime / "reports/e0/inventory.json")["datasets"]
    gt_root = Path(rgbt.__file__).parent / "gt_file"
    all_receipts = read(destination / "all_receipts.json")
    receipts = {(item["dataset"], item["sequence"]): item for item in all_receipts}
    assert len(receipts) == len(all_receipts) == 479
    assert set(receipts) == {
        (dataset, item["name"]) for dataset, spec in inventory.items()
        for item in spec["sequence_details"]}
    measured = {}
    for dataset, cls, keys in [
        ("LasHeR-Miss", rgbt.LasHeR, ["PR", "NPR", "SR"]),
        ("RGBT234-Miss", rgbt.RGBT234, ["MPR", "MSR"]),
    ]:
        spec = inventory[dataset]
        assert sha(spec["mask"]) == spec["mask_sha256"]
        names = {item["name"] for item in spec["sequence_details"]}
        for item in spec["sequence_details"]:
            name, frames = item["name"], item["frames"]
            gt_files = ([gt_root / "LasHeR/lasher_gt" / (name + ".txt")]
                        if dataset == "LasHeR-Miss" else [
                            gt_root / "RGBT234/rgbt234_gt" / name / filename
                            for filename in ["visible.txt", "infrared.txt"]])
            expected_gt = list(item["gt_sha256"].values())
            assert len(gt_files) == len(expected_gt)
            for gt_file, expected in zip(gt_files, expected_gt):
                assert sha(gt_file) == expected
            path = destination / "bbox" / dataset / (name + ".txt")
            receipt = read(destination / "receipts" / dataset / (name + ".json"))
            assert receipt == receipts[(dataset, name)]
            assert receipt["frames"] == frames and sha(path) == receipt["bbox_sha256"]
            boxes = np.loadtxt(str(path), ndmin=2)
            assert boxes.shape == (frames, 4) and np.isfinite(boxes).all()
        scorer = cls()
        assert set(scorer.seqs_name) == names
        scorer("verified", str(destination / "bbox" / dataset), bbox_type="ltwh")
        label = dataset if condition == "miss" else dataset.replace("-Miss", "")
        actual = dict(sequences=len(names),
                      frames=sum(item["frames"] for item in spec["sequence_details"]))
        curves = read(destination / (label + "_curves.json"))
        for key in keys:
            score, curve = getattr(scorer, key)("verified")
            actual[key] = float(score * 100)
            assert np.isclose(actual[key], saved_metrics[label][key], rtol=0, atol=1e-9)
            assert np.allclose(curve, curves[key], rtol=0, atol=1e-12)
        assert actual["sequences"] == saved_metrics[label]["sequences"]
        assert actual["frames"] == saved_metrics[label]["frames"]
        measured[label] = actual
    return measured


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", choices=LABELS, required=True)
    parser.add_argument("--wait", action="store_true")
    args = parser.parse_args()
    folder = ROOT / "reports/server28-recovery-20261031" / args.label
    launch = read(folder / "launch.json")
    if args.wait:
        while not (folder / "official_completed.json").exists():
            process = Path("/proc") / str(launch["controller_pid"])
            assert process.exists(), "Controller exited without official completion"
            command = (process / "cmdline").read_bytes()
            assert b"/home/gaob/Miss-RGBT/recovery_campaign28.py" in command
            assert args.label.encode() in command
            time.sleep(300)
    official = read(folder / "official_completed.json")
    queue = read(folder.parent / "independent_verification_queue.json")
    for path, expected in queue["source_sha256"].items():
        assert sha(path) == expected, path
    for path, expected in launch["source_sha256"].items():
        assert sha(path) == expected, path
    run = ROOT / "runs/server28-recovery-20261031" / args.label
    training = read(run / "completed.json")
    epochs = training["final_five_epochs"]
    assert epochs == list(range(training["epochs"] - 4, training["epochs"] + 1))
    rows = read(folder / "last_five_metrics.json")
    assert [row["epoch"] for row in rows] == epochs and len(rows) == 5
    protocol = ROOT / "development/recovery_val97_20261030/protocol.json"
    for row in rows:
        assert set(row["conditions"]) == {"complete", "alternating_outage"}
        for condition, recorded in row["conditions"].items():
            destination = folder / ("epoch%02d_%s" % (row["epoch"], condition))
            actual = verify_development(destination, protocol, row["checkpoint_sha256"])
            assert read(destination / "launch.json")["condition"] == condition
            assert sha(ROOT / "evaluate_recovery_val97_28.py") == read(
                destination / "launch.json")["source_sha256"]
            assert actual == recorded
        score = sum(value["SR"] for value in row["conditions"].values()) / 2
        assert score == row["selection_score"]
    best = max(rows, key=lambda row: (row["selection_score"], -row["epoch"]))
    selected = read(folder / "selected.json")
    cleanup = read(folder / "completed.json")
    assert selected["best"] == cleanup["best"] == official["best"] == best
    assert selected["official_test_selection_used"] is False
    assert cleanup["official_test_selection_used"] is official["official_test_selection_used"] is False
    assert cleanup["cleanup_completed"] and official["cleanup_completed"]
    retained = run / ("epoch%02d.pth.tar" % best["epoch"])
    assert Path(best["checkpoint"]) == retained
    assert sha(retained) == best["checkpoint_sha256"]
    assert sorted(run.glob("epoch*.pth.tar")) == [retained]
    assert cleanup["cleanup"] == selected["cleanup"] == official["cleanup"]
    assert len(cleanup["cleanup"]) == 4
    expected_removed = {run / ("epoch%02d.pth.tar" % epoch)
                        for epoch in epochs if epoch != best["epoch"]}
    assert {Path(item["path"]) for item in cleanup["cleanup"]} == expected_removed
    hashes = {row["epoch"]: row["checkpoint_sha256"] for row in rows}
    assert all(item["sha256"] == hashes[int(Path(item["path"]).name[5:7])]
               for item in cleanup["cleanup"])
    assert all(not path.exists() for path in expected_removed)
    official_metrics = {}
    for condition in ["complete", "miss"]:
        destination = ROOT / "results/server28_recovery_20261031" / (
            args.label + "_" + condition)
        official_metrics[condition] = verify_official(
            destination, best["checkpoint_sha256"], condition, official["official_" + condition])
    baseline_path = ROOT / "runtime/task_utility_20261018/reports/e0_complete/metrics.json"
    baseline = read(baseline_path)
    complete_deltas = {
        dataset: {key: value - baseline[dataset][key]
                  for key, value in scores.items() if key not in ["sequences", "frames"]}
        for dataset, scores in official_metrics["complete"].items()}
    thresholds = {"LasHeR-Miss": {"PR": 69.3, "SR": 54.4},
                  "RGBT234-Miss": {"MPR": 88.1, "MSR": 64.3}}
    attainment = {
        dataset: {key: dict(score=official_metrics["miss"][dataset][key], threshold=limit,
                            passed=official_metrics["miss"][dataset][key] > limit)
                  for key, limit in goals.items()} for dataset, goals in thresholds.items()}
    result = dict(
        label=args.label, verifier_sha256=sha(__file__), selected_epoch=best["epoch"],
        checkpoint_sha256=best["checkpoint_sha256"], development_trajectories_verified=970,
        official_trajectories_verified=958, official_metrics=official_metrics,
        baseline_sha256=sha(baseline_path), complete_delta_pp=complete_deltas,
        complete_at_least2pp={dataset: {key: delta >= 2 for key, delta in values.items()}
                             for dataset, values in complete_deltas.items()},
        strict_miss_targets=attainment,
        all_miss_targets_met=all(item["passed"] for goals in attainment.values()
                                 for item in goals.values()),
        selected_weight_and_recovery_cleanup_verified=True,
        original_weight_cleanup_verified=False,
        scope="Read-only receipt/source/trajectory checks and independent metric recalculation. "
              "Baseline+2pp shown as a reference for the requested approximate gain; "
              "no invented tolerance. Does not certify inference mask execution from boxes, "
              "training convergence, three paper modules or historical data exposure.")
    with (folder / "independent_verification.json").open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
