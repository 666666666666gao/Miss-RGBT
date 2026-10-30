"""Separate fixed-grid coverage, localization ceiling and actual selection."""
import argparse
import hashlib
import json
import math
from pathlib import Path

ROOT = Path("/home/gaob/Miss-RGBT/reports/server28-native-grid-20261030")


def summarize(rows):
    result = dict(observations=len(rows))
    if not rows:
        return result
    for label, group in [
        ("all", rows),
        ("native_unsupported", [
            row for row in rows
            if not row["candidates"][0]["diagnostic"]["target_overlaps_previous_crop"]]),
    ]:
        values = []
        for row in group:
            candidates = row["candidates"]
            native = candidates[0]
            selected = max(candidates, key=lambda candidate: candidate["confidence"])
            oracle = max(candidates, key=lambda candidate: candidate["diagnostic"]["iou"])
            overlapping = [
                candidate for candidate in candidates
                if candidate["diagnostic"]["target_overlaps_previous_crop"]]
            centered = [
                candidate for candidate in candidates
                if candidate["diagnostic"]["target_center_in_previous_crop"]]
            values.append(dict(
                native_iou=native["diagnostic"]["iou"],
                selected_iou=selected["diagnostic"]["iou"],
                oracle_iou=oracle["diagnostic"]["iou"],
                coverage=bool(overlapping), center_coverage=bool(centered),
                selected_support=selected["diagnostic"]["target_overlaps_previous_crop"],
                selected_scale=selected["scale"],
                oracle_success=oracle["diagnostic"]["iou"] >= .5,
                actual_success=selected["diagnostic"]["iou"] >= .5,
                selected_native=selected is native))
        totals = dict(observations=len(values))
        for key in ["coverage", "center_coverage", "selected_support",
                    "oracle_success", "actual_success", "selected_native"]:
            totals[key] = sum(value[key] for value in values)
        totals["localization_failure_despite_center_coverage"] = sum(
            value["center_coverage"] and not value["oracle_success"] for value in values)
        totals["selection_failure_despite_oracle_success"] = sum(
            value["oracle_success"] and not value["actual_success"] for value in values)
        totals["harms_over_point_one"] = sum(
            value["selected_iou"] < value["native_iou"] - .1 for value in values)
        totals["selected_scales"] = {
            scale: sum(value["selected_scale"] == scale for value in values)
            for scale in ["native", "previous", "initial"]}
        if values:
            for key in ["native_iou", "selected_iou", "oracle_iou"]:
                totals["mean_" + key] = sum(value[key] for value in values) / len(values)
        result[label] = totals
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--report-dir", default=str(ROOT))
    parser.add_argument("--raw", action="store_true")
    args = parser.parse_args()
    root = Path(args.report_dir)
    completion = json.loads((root / "completed.json").read_text())
    source_path = root / "report.json"
    raw = source_path.read_bytes()
    report = json.loads(raw)
    assert completion["observations"] == len(report["observations"]) == 46
    assert all(len(row["candidates"]) == 51 for row in report["observations"])
    rows = [row for row in report["observations"]
            if row["candidates"][0]["diagnostic"] is not None]
    assert len(rows) == 45
    result = dict(
        report_sha256=hashlib.sha256(raw).hexdigest(),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        scope="Post-hoc diagnostic of the preregistered51 proposals at46 fixed points. "
              "GT labels coverage and IoU only; no new selector or threshold fitted. "
              "Not an independent validation, closed-loop benefit or benchmark score.",
        aggregate=summarize(rows),
        per_state={"".join(str(int(flag)) for flag in state): summarize([
            row for row in rows if row["available"] == state])
                   for state in [[1, 1], [0, 1], [1, 0]]})
    if args.raw:
        prior = json.loads((ROOT / "report.json").read_text())
        prior_rows = [row for row in prior["observations"]
                      if row["candidates"][0]["diagnostic"] is not None]
        assert len(prior_rows) == len(rows)
        differences = []
        for current, previous in zip(rows, prior_rows):
            assert (current["sequence"], current["frame"], current["available"]) == (
                previous["sequence"], previous["frame"], previous["available"])
            for candidate, old in zip(current["candidates"], previous["candidates"]):
                assert candidate["crop_box"] == old["crop_box"]
                assert all(math.isclose(value, reference, rel_tol=1e-4, abs_tol=.05)
                           for value, reference in zip(candidate["bbox"], old["bbox"]))
                differences.extend(abs(value - reference)
                                   for value, reference in zip(candidate["bbox"], old["bbox"]))
        result["hann_vs_prior_max_bbox_difference"] = max(differences)
        raw_rows = [
            dict(row, candidates=[
                dict(candidate, diagnostic=candidate["raw_diagnostic"],
                     confidence=candidate["raw_confidence"])
                for candidate in row["candidates"]])
            for row in rows]
        result["raw_aggregate"] = summarize(raw_rows)
        result["raw_per_state"] = {
            "".join(str(int(flag)) for flag in state): summarize([
                row for row in raw_rows if row["available"] == state])
            for state in [[1, 1], [0, 1], [1, 0]]}
        result["raw_selected_vs_native_hann"] = dict(
            mean_iou_delta=sum(
                max(raw_row["candidates"], key=lambda candidate: candidate["confidence"])
                ["diagnostic"]["iou"] - row["candidates"][0]["diagnostic"]["iou"]
                for row, raw_row in zip(rows, raw_rows)) / len(rows),
            harms_over_point_one=sum(
                max(raw_row["candidates"], key=lambda candidate: candidate["confidence"])
                ["diagnostic"]["iou"] < row["candidates"][0]["diagnostic"]["iou"] - .1
                for row, raw_row in zip(rows, raw_rows)))
    (root / "selection_analysis.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
