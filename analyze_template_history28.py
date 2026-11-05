"""Describe actual missing-template exposure in completed paired trajectories."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--traces", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(args.traces)
    report = json.loads((root / "report.json").read_text())
    assert report["status"] == "CPU_COHORT8_COMPLETE_NOT_BENCHMARK"
    sequences, pooled = {}, Counter()
    for name, sequence in report["sequences"].items():
        conditions = {}
        for condition, receipt in sequence["conditions"].items():
            path = root / (name + "_" + condition + ".json")
            assert sha(path) == receipt["trace_sha256"]
            rows = json.loads(path.read_text())
            assert len(rows) == receipt["frames"] - 1
            availability = {0: [1, 1]}
            source = 0
            counts, groups = Counter(), {}
            for row in rows:
                assert row["frame"] == len(availability)
                availability[row["frame"]] = row["available"]
                # Read the previous source before this frame's optional template write.
                dynamic = availability[source]
                search = row["available"]
                assert any(dynamic)
                counts["video_frames_after_initialization"] += 1
                if row["observed"]:
                    counts["observed_frames"] += 1
                    missing_template = not all(dynamic)
                    counts["observed_with_missing_dynamic_template"] += missing_template
                    counts["complete_search_frames"] += all(search)
                    counts["complete_search_with_missing_dynamic_template"] += all(search) and missing_template
                    counts["single_search_with_missing_dynamic_template"] += sum(search) == 1 and missing_template
                    counts["single_search_opposite_dynamic_modality"] += sum(search) == sum(dynamic) == 1 and search != dynamic
                    key = "search%s_dynamic%s" % (
                        "".join(map(str, search)), "".join(map(str, dynamic)))
                    group = groups.setdefault(key, dict(frames=0, valid_geometry_frames=0,
                                                       iou_sum=0., zero_crop_overlap=0))
                    group["frames"] += 1
                    if row["diagnostic"] is not None:
                        group["valid_geometry_frames"] += 1
                        group["iou_sum"] += row["diagnostic"]["iou"]
                        group["zero_crop_overlap"] += not row["diagnostic"]["target_overlaps_previous_crop"]
                if row["template_written"]:
                    assert row["observed"]
                    counts["template_writes"] += 1
                    counts["single_modality_template_writes"] += sum(search) == 1
                    source = row["frame"]
                assert row["dynamic_template_source_frame"] == source
            for group in groups.values():
                total = group["valid_geometry_frames"]
                group["mean_iou"] = group.pop("iou_sum") / total if total else None
            conditions[condition] = dict(counts=dict(counts), groups=groups,
                                         trace_sha256=sha(path))
            if condition == "alternating_outage":
                pooled.update(counts)
        sequences[name] = conditions
    observed = pooled["observed_frames"]
    complete = pooled["complete_search_frames"]
    result = dict(
        scope="Descriptive existing fixed8 CPU trajectories of historical A. No intervention, "
              "causal effect, selected LocalRepair outcome or benchmark claim. "
              "Groups differ in frames, history and tracking state.",
        report_sha256=sha(root / "report.json"), source_sha256=sha(Path(__file__)),
        sequences=sequences, alternating_counts=dict(pooled),
        observed_missing_dynamic_percent=100 * pooled["observed_with_missing_dynamic_template"] / observed,
        complete_search_missing_dynamic_percent=100 * pooled[
            "complete_search_with_missing_dynamic_template"] / complete,
        interpretation="Complete current search is not equivalent to all-complete history. "
                       "The frozen six-epoch pilot uses complete templates only. Its sampled "
                       "complete-output hash therefore does not certify mixed-history tracking. "
                       "Historical A formal training already samples dynamic-template outages.")
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: result[key] for key in [
        "alternating_counts", "observed_missing_dynamic_percent",
        "complete_search_missing_dynamic_percent", "interpretation"]}, indent=2))


if __name__ == "__main__":
    main()
