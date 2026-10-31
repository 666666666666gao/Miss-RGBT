"""Measure saved calibration effects on one shared geometry-arm history."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

from analyze_epoch05_health28 import metrics
from audit_offset_mechanism28 import scores, summarize
from evaluate_val97_recovery28 import sha
from run_calibration_gap28 import ARMS, CONDITIONS, ROOT, RUNTIME, flags_for


def read(path):
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(RUNTIME))
    from lib.utils.box_ops import clip_box

    final = read(args.source / "report.json")
    completed = read(args.source / "completed.json")
    assert sha(args.source / "report.json") == completed["report_sha256"]
    assert final["trajectories_verified"] == completed["trajectories_verified"] == 64
    assert final["past_only_center_and_scale_verified"] and final["controls_bitwise_equal"]
    launch = final["launch"]
    assert sha(args.source / "registration.json") == launch["registration_sha256"]
    for path, expected in launch["source_sha256"].items():
        assert sha(path) == expected
    names = sorted(item["name"] for item in launch["cohort"])
    immediate = {
        condition: {state: {contrast: [] for contrast in ["center_minus_raw", "geometry_minus_center"]}
                    for state in ["01", "10"]}
        for condition in CONDITIONS if condition != "complete"}
    closed_loop = {condition: {state: [] for state in ["11", "01", "10", "00"]}
                   for condition in CONDITIONS if condition != "complete"}
    diagnostics, source_hashes = {}, {}
    trajectories = 0
    for gpu, cohort in launch["parts"].items():
        folder = args.source / ("gpu" + gpu)
        worker = read(folder / "report.json")
        assert sha(folder / "report.json") == final["worker_receipts"][gpu]["report_sha256"]
        assert sha(folder / "report.json") == read(folder / "completed.json")["report_sha256"]
        for item in cohort:
            name = item["name"]
            gt_path = Path(item["folder"]) / "init.txt"
            assert sha(gt_path) == item["gt_sha256"]
            gt = np.loadtxt(str(gt_path), delimiter=",", ndmin=2)
            for condition in CONDITIONS:
                traces, boxes = {}, {}
                for arm in ARMS:
                    prefix = folder / (arm + "_" + name + "_" + condition)
                    receipt = worker["records"][arm][name][condition]
                    for suffix, key in [(".txt", "bbox_sha256"), (".json", "trace_sha256")]:
                        path = Path(str(prefix) + suffix)
                        assert sha(path) == receipt[key]
                        source_hashes[str(path)] = receipt[key]
                    boxes[arm] = np.loadtxt(str(prefix) + ".txt", ndmin=2)
                    traces[arm] = read(Path(str(prefix) + ".json"))
                    assert boxes[arm].shape == gt.shape
                    assert len(traces[arm]) == len(gt) - 1
                    assert np.isfinite(boxes[arm]).all()
                    measured = metrics(boxes[arm], gt)
                    assert all(np.isclose(value, receipt[key], rtol=0, atol=1e-10)
                               for key, value in measured.items())
                    assert all(np.isclose(value, final["per_video"][name][arm][condition][key],
                                          rtol=0, atol=1e-10)
                               for key, value in measured.items())
                    trajectories += 1
                assert np.array_equal(boxes["native"], boxes["sham"])
                if condition == "complete":
                    assert all(np.array_equal(boxes[arm], boxes["native"]) for arm in ARMS)
                    continue
                diag = dict(template_write_changes=[], scale_by_state={"01": [], "10": []},
                            age_by_state={"01": [], "10": []})
                offsets = {"01": np.zeros(2), "10": np.zeros(2)}
                scales = {"01": np.ones(2), "10": np.ones(2)}
                last_calibration = None
                for frame, row in enumerate(traces["geometry"], start=1):
                    flags = flags_for(frame, condition)
                    state = "".join(map(str, flags))
                    native = traces["native"][frame - 1]
                    assert row["frame"] == native["frame"] == frame
                    assert row["available"] == native["available"] == flags
                    assert np.array_equal(row["bbox"], boxes["geometry"][frame])
                    if row["template_written"] != native["template_written"]:
                        diag["template_write_changes"].append(frame)
                    if state == "11":
                        result = row["prediction"]
                        offsets = {key: np.asarray(value)
                                   for key, value in result["calibration_offsets"].items()}
                        sizes = result["calibration_sizes"]
                        scales = {key: np.asarray(sizes["11"]) / np.asarray(sizes[key])
                                  for key in ["01", "10"]}
                        last_calibration = frame
                    if (gt[frame, 2:] > 0).all():
                        closed_loop[condition][state].append(dict(
                            sequence=name, frame=frame,
                            before=scores(native["bbox"], gt[frame]),
                            after=scores(row["bbox"], gt[frame])))
                    if state not in ["01", "10"]:
                        continue
                    result = row["prediction"]
                    offset, scale = offsets[state], scales[state]
                    assert np.array_equal(result["applied_offset"], offset)
                    assert np.array_equal(result["applied_scale"], scale)
                    raw = np.asarray(result["before_scale_bbox"])
                    center = raw.copy()
                    center[:2] -= offset
                    geometry = np.concatenate([
                        center[:2] + raw[2:] / 2 - raw[2:] * scale / 2, raw[2:] * scale])
                    clipped = {key: np.asarray(clip_box(value.tolist(), *row["image_shape"], margin=10))
                               for key, value in [("raw", raw), ("center", center), ("geometry", geometry)]}
                    assert np.allclose(clipped["geometry"], row["bbox"], rtol=0, atol=1e-9)
                    diag["scale_by_state"][state].append(scale.tolist())
                    if last_calibration is not None:
                        assert last_calibration < frame
                        diag["age_by_state"][state].append(frame - last_calibration)
                    if not (gt[frame, 2:] > 0).all():
                        continue
                    values = {key: scores(box, gt[frame]) for key, box in clipped.items()}
                    for key, box in clipped.items():
                        values[key]["size_log_error"] = float(np.linalg.norm(
                            np.log(box[2:] / gt[frame, 2:])))
                    for first, second in [("center", "raw"), ("geometry", "center")]:
                        immediate[condition][state][first + "_minus_" + second].append(dict(
                            sequence=name, frame=frame, before=values[second], after=values[first]))
                diagnostics.setdefault(name, {})[condition] = diag
    assert trajectories == 64

    def summarize_present(rows):
        present = [name for name in names if any(row["sequence"] == name for row in rows)]
        if not present:
            return dict(exposed_videos=[], unexposed_videos=names, scored_frames=0)
        result = summarize(rows, present)
        result.update(exposed_videos=present, unexposed_videos=[name for name in names if name not in present],
                      scored_frames=len(rows))
        return result

    immediate_results = {}
    for condition, states in immediate.items():
        immediate_results[condition] = {}
        for state, contrasts in states.items():
            immediate_results[condition][state] = {}
            for contrast, rows in contrasts.items():
                result = summarize_present(rows)
                result["size_log_error_by_video"] = {
                    name: {side: float(np.mean([row[side]["size_log_error"] for row in rows
                                               if row["sequence"] == name]))
                           for side in ["before", "after"]}
                    for name in result["exposed_videos"]}
                immediate_results[condition][state][contrast] = result
    result = dict(
        source_report_sha256=sha(args.source / "report.json"),
        audit_sha256=sha(Path(__file__)), trajectories_verified=trajectories,
        source_trace_and_bbox_hashes=source_hashes, new_model_calls=0,
        immediate_same_geometry_history=immediate_results,
        closed_loop_geometry_minus_native={
            condition: {state: summarize_present(rows) for state, rows in states.items()}
            for condition, states in closed_loop.items()},
        diagnostics=diagnostics,
        scope="Post-hoc arithmetic on all64 saved trajectories. Immediate raw/center/geometry "
              "boxes share the geometry arm's prediction and prior history, with native clipping. "
              "These do not reproduce trajectories of separately rerun arms. Closed-loop state "
              "slices include earlier crop/template effects. Unexposed video/state pairs are "
              "listed, not imputed or removed from overall benchmark scores. GT scores only; "
              "no parameter, expiry or gate selected. Not isolated age causality, official "
              "performance, an independent replication or a validated module.")
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps(dict(output=str(args.output), report_sha256=sha(args.output),
                         trajectories_verified=trajectories, new_model_calls=0)))


if __name__ == "__main__":
    main()
