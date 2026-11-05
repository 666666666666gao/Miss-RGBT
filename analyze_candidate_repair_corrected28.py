"""Independently check saved repair-intervention receipts and arithmetic."""
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from evaluate_val97_recovery28 import sha


ROOT = Path("/home/gaob/Miss-RGBT")
OUTPUT = ROOT / "reports/server28-candidate-repair-audit-20261031"


def main():
    report = json.loads((OUTPUT / "report.json").read_text())
    completed = json.loads((OUTPUT / "completed.json").read_text())
    assert sha(OUTPUT / "report.json") == completed["report_sha256"]
    launch = json.loads((OUTPUT / "launch.json").read_text())
    assert launch == report["launch"]
    assert launch["checkpoint_epoch"] == 6 and launch["checkpoint_updates"] == 12000
    assert launch["samples"] == 128 and launch["batch_size"] == 4
    for path, expected in launch["source_sha256"].items():
        assert sha(path) == expected, path
    configuration = ROOT / "runtime/task_utility_20261018/experiments/cadtrack/cadtrack.yaml"
    settings = yaml.safe_load(configuration.read_text())
    searches = settings["DATA"]["SEARCH"]["NUMBER"]
    templates = settings["DATA"]["TEMPLATE"]["NUMBER"]
    assert searches == 3 and templates == 2
    arrays, summaries = {}, {}
    for label in ["utility", "random"]:
        model = report["models"][label]
        assert model["network_state_unchanged"] and model["native_tail_bitwise_equal"]
        assert sha(model["checkpoint"]) == model["checkpoint_sha256"]
        assert model["checkpoint_sha256"] == launch["checkpoint_sha256"][label]
        records = model["records"]
        assert len(records) == 32
        assert [row["batch"] for row in records] == list(range(32))
        generator = torch.Generator().manual_seed(launch["random_control_seed"])
        for row in records:
            assert row["samples"] == 4 and set(row["states"]) == {"11", "01", "10"}
            for state in ["11", "01", "10"]:
                item = row["states"][state]
                assert item["native_tail_bitwise_equal"]
                assert set(item["task_loss"]) == set(launch["arms"])
                assert len(item["native_regions"]) == len(item["random_regions"]) == searches
                for search in range(searches):
                    assert len(item["native_regions"][search]) == 4
                    expected = [torch.randperm(16, generator=generator)[:4].tolist()
                                for _ in range(4)]
                    assert item["random_regions"][search] == expected
                    for regions in item["native_regions"][search]:
                        assert len(set(regions)) == 4 and all(0 <= x < 16 for x in regions)
                for values in item["task_loss"].values():
                    assert len(values) == 4 and np.isfinite(values).all()
        arrays[label], summaries[label] = {}, {}
        for state in ["11", "01", "10"]:
            values = {arm: np.asarray([
                value for row in records for value in row["states"][state]["task_loss"][arm]])
                for arm in launch["arms"]}
            arrays[label][state] = values
            summary = model["summary"][state]
            means = {arm: float(value.mean()) for arm, value in values.items()}
            assert means == summary["mean_task_loss"]
            contrasts = {}
            for first, second in [
                ("native4", "off"), ("native4", "random4"), ("all16", "native4")
            ]:
                delta = values[first] - values[second]
                computed = dict(
                    mean_loss_delta=float(delta.mean()), wins=int((delta < 0).sum()),
                    ties=int((delta == 0).sum()), losses=int((delta > 0).sum()))
                assert computed == summary["contrasts"][first + "_minus_" + second]
                contrasts[first + "_minus_" + second] = computed
            summaries[label][state] = dict(mean_task_loss=means, contrasts=contrasts)
    comparisons = {
        state: {
            arm: float((arrays["utility"][state][arm] -
                        arrays["random"][state][arm]).mean())
            for arm in launch["arms"]
        } for state in ["11", "01", "10"]
    }
    result = dict(
        report_sha256=sha(OUTPUT / "report.json"), analyzer_sha256=sha(__file__),
        model_summaries=summaries, utility_minus_random_loss=comparisons,
        random_regions_reproduced=True, samples_per_model_state=128,
        searches_per_tuple=searches, templates_per_tuple=templates,
        configuration_sha256=sha(configuration),
        original_analyzer_sha256=sha(ROOT / "analyze_candidate_repair28.py"),
        original_failure_log_sha256=sha(Path(str(OUTPUT) + "-analysis.log")),
        correction="Original analyzer assumed two search frames; frozen YAML and "
                   "producer data have three. Original producer/report/log preserved. "
                   "This analyzer checks every recorded search and all random draws. "
                   "The producer protocol string saying both search frames is incorrect; "
                   "the producer processes all three configured search frames.",
        scope="Independent file/hash/shape/random-subset and aggregate-arithmetic "
              "verification. Native-tail and unchanged-model execution witnesses "
              "come from the producer, not independently re-executed here. "
              "Reused cached tuples, full templates and batch-coupled focal loss; "
              "no video-independent confidence interval or PR/SR claim. "
              "Between-model differences also include trained backbone/correction; "
              "they cannot identify the effect of selection alone.")
    with (OUTPUT / "analysis_corrected.json").open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    with (OUTPUT / "analysis_corrected_completed.json").open("x") as stream:
        json.dump(dict(analysis_sha256=sha(OUTPUT / "analysis_corrected.json"),
                       report_sha256=sha(OUTPUT / "report.json"),
                       analyzer_sha256=sha(__file__)), stream, indent=2)
        stream.write("\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
