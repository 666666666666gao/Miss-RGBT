"""Independently verify the matched template-coverage pilot receipts."""
import json
from pathlib import Path

import numpy as np

from evaluate_val97_recovery28 import sha


ROOT = Path("/home/gaob/Miss-RGBT")
OUTPUT = ROOT / "reports/server28-protected-mixed-pair-20261031"
PREVIOUS = ROOT / "reports/server28-protected-sixepoch-20261030"
CELLS = [dynamic + "_" + search
         for dynamic in ["11", "01", "10"] for search in ["11", "01", "10"]]


def read(path):
    return json.loads(path.read_text())


def verify_measurement(measured):
    assert set(measured["cells"]) == set(CELLS)
    for row in measured["cells"].values():
        values = np.asarray(row["losses"])
        assert values.ndim == 1 and len(values) == 16 and np.isfinite(values).all()
        assert np.isclose(values.mean(), row["mean_task_loss"], rtol=0, atol=1e-12)
    score = np.mean([measured["cells"][key]["mean_task_loss"]
                     for key in CELLS if key.split("_")[1] != "11"])
    assert np.isclose(score, measured["selection_score"], rtol=0, atol=1e-12)


def main():
    completion = read(OUTPUT / "completed.json")
    comparison = read(OUTPUT / "comparison.json")
    assert completion["comparison_sha256"] == sha(OUTPUT / "comparison.json")
    assert completion["updates"] == 384
    launch = comparison["launch"]
    assert launch["input_manifest_sha256"] == sha(PREVIOUS / "inputs.json")
    for path, expected in launch["source_sha256"].items():
        assert sha(path) == expected, path
    assert comparison["same_initial_state_and_inputs"]
    arms, receipts = {}, {}
    for arm in ["complete_templates", "mixed_templates"]:
        folder = OUTPUT / arm
        report = read(folder / "report.json")
        initial = read(folder / "initial.json")
        history = read(folder / "development_history.json")
        finished = read(folder / "completed.json")
        assert report == comparison["arms"][arm]
        assert report["initial"] == initial["development"]
        assert report["initial_repair_sha256"] == initial["initial_repair_sha256"]
        assert report["baseline_state_unchanged"] and report["complete_history_outputs_bitwise_equal"]
        assert len(history) == 6 and [row["epoch"] for row in history] == list(range(1, 7))
        verify_measurement(report["initial"])
        for row in history:
            epoch = row["epoch"]
            assert row["updates"] == epoch * 32
            assert read(folder / ("epoch%02d_inputs.json" % epoch)) == read(
                PREVIOUS / ("epoch%02d_inputs.json" % epoch))
            verify_measurement(row["development"])
            assert row["development"]["cells"]["11_11"] == report["initial"]["cells"]["11_11"]
            counts = row["dynamic_template_counts"]
            assert sum(counts.values()) == 64
            if arm == "complete_templates":
                assert counts == {"11": 64, "01": 0, "10": 0}
        best = min(history[-5:], key=lambda row: (row["development"]["selection_score"], row["epoch"]))
        assert best == report["selected"]
        retained = Path(report["retained"])
        assert retained == folder / ("repair_epoch%02d.pth" % best["epoch"])
        assert sha(retained) == report["retained_sha256"] == finished["retained_sha256"]
        assert finished["updates"] == report["updates"] == 192
        assert finished["selected_epoch"] == best["epoch"] and finished["cleanup_completed"]
        removed = {Path(item["path"]) for item in report["cleanup"]}
        assert removed == {folder / ("repair_epoch%02d.pth" % epoch)
                           for epoch in range(2, 7) if epoch != best["epoch"]}
        assert all(not path.exists() for path in removed)
        assert set(folder.glob("repair_epoch*.pth")) == {retained}
        arms[arm] = report
        receipts[arm] = dict(selected_epoch=best["epoch"], retained_sha256=sha(retained),
                             report_sha256=sha(folder / "report.json"))
    first, second = arms["complete_templates"], arms["mixed_templates"]
    assert first["initial"] == second["initial"]
    assert first["initial_repair_sha256"] == second["initial_repair_sha256"]
    cells = {}
    for key in CELLS:
        values = {}
        for arm, report in arms.items():
            row = report["selected"]["development"]["cells"][key]
            values[arm] = row["mean_task_loss"]
        delta = values["mixed_templates"] - values["complete_templates"]
        assert np.isclose(delta, comparison["selected_loss_delta_mixed_minus_complete"][key],
                          rtol=0, atol=1e-12)
        cells[key] = dict(selected_task_loss=values, mixed_minus_complete=delta)
    result = dict(
        comparison_sha256=sha(OUTPUT / "comparison.json"), analyzer_sha256=sha(Path(__file__)),
        verified_updates=384, arm_receipts=receipts, cells=cells,
        selection_score_delta=second["selected"]["development"]["selection_score"]
                              - first["selected"]["development"]["selection_score"],
        scope="One-seed reused development task losses, selected on these same tuples; "
              "descriptive only, no tracking/generalization/official/novelty claim",
        baseline_and_complete_outputs_checks="Training receipts verified; tensors are not re-executed here")
    with (OUTPUT / "analysis.json").open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
