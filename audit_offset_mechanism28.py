"""Separate saved same-history offset effects from closed-loop trajectory effects."""
import argparse
import json
from pathlib import Path

import numpy as np

from evaluate_val97_recovery28 import sha


ROOT = Path("/home/gaob/Miss-RGBT")


def read(path):
    return json.loads(path.read_text())


def scores(box, target):
    box, target = np.asarray(box), np.asarray(target)
    error = np.linalg.norm(box[:2] + box[2:] / 2 - target[:2] - target[2:] / 2)
    extent = np.maximum(0, np.minimum(box[:2] + box[2:], target[:2] + target[2:]) -
                        np.maximum(box[:2], target[:2]))
    intersection = float(extent.prod())
    overlap = intersection / (box[2:].prod() + target[2:].prod() - intersection)
    return dict(center_error_px=float(error), PR20=float(error <= 20) * 100,
                SR=float((overlap > np.linspace(0, 1, 21)).mean()) * 100)


def summarize(rows, names):
    per_video = {}
    for name in names:
        selected = [row for row in rows if row["sequence"] == name]
        assert selected, name
        fields = ["center_error_px", "PR20", "SR"]
        per_video[name] = dict(
            frames=len(selected),
            before={key: float(np.mean([row["before"][key] for row in selected]))
                    for key in fields},
            after={key: float(np.mean([row["after"][key] for row in selected]))
                   for key in fields})
    indices = np.random.RandomState(2027).randint(0, len(names), (10000, len(names)))
    contrasts = {}
    for key in ["center_error_px", "PR20", "SR"]:
        delta = np.asarray([per_video[name]["after"][key] -
                            per_video[name]["before"][key] for name in names])
        contrasts[key] = dict(
            video_macro_before=float(np.mean([per_video[name]["before"][key] for name in names])),
            video_macro_after=float(np.mean([per_video[name]["after"][key] for name in names])),
            mean_delta=float(delta.mean()),
            paired_video_bootstrap95=np.percentile(delta[indices].mean(1), [2.5, 97.5]).tolist(),
            negative_videos=int((delta < 0).sum()), zero_videos=int((delta == 0).sum()),
            positive_videos=int((delta > 0).sum()))
    return dict(per_video=per_video, contrasts=contrasts)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    final = read(args.source / "report.json")
    assert sha(args.source / "report.json") == read(args.source / "completed.json")["report_sha256"]
    assert final["sham_and_complete_bitwise_equal"] and final["past_only_offsets_verified"]
    launch = final["launch"]
    names = sorted(item["name"] for item in launch["cohort"])
    immediate = {"01": [], "10": []}
    trajectory = {state: [] for state in ["11", "01", "10", "00"]}
    diagnostics = {}
    for gpu, cohort in launch["parts"].items():
        folder = args.source / ("gpu" + gpu)
        worker = read(folder / "report.json")
        assert sha(folder / "report.json") == final["worker_receipts"][gpu]["report_sha256"]
        for item in cohort:
            name = item["name"]
            gt_path = Path(item["folder"]) / "init.txt"
            assert sha(gt_path) == item["gt_sha256"]
            gt = np.loadtxt(str(gt_path), delimiter=",", ndmin=2)
            traces = {}
            for arm in ["native", "offset_corrected"]:
                prefix = folder / (arm + "_" + name + "_alternating_outage")
                receipt = worker["records"][arm][name]["alternating_outage"]
                assert sha(str(prefix) + ".json") == receipt["trace_sha256"]
                assert sha(str(prefix) + ".txt") == receipt["bbox_sha256"]
                traces[arm] = read(Path(str(prefix) + ".json"))
            assert len(traces["native"]) == len(traces["offset_corrected"]) == len(gt) - 1
            changed_frames, write_changes = [], []
            writes = {"native": 0, "offset_corrected": 0}
            nonzero_offsets = 0
            for frame, (native, corrected) in enumerate(
                    zip(traces["native"], traces["offset_corrected"]), start=1):
                assert native["frame"] == corrected["frame"] == frame
                assert native["available"] == corrected["available"]
                state = "".join(map(str, corrected["available"]))
                for arm, row in [("native", native), ("offset_corrected", corrected)]:
                    writes[arm] += int(row["template_written"])
                if native["template_written"] != corrected["template_written"]:
                    write_changes.append(frame)
                if not np.array_equal(native["bbox"], corrected["bbox"]):
                    changed_frames.append(frame)
                if state in immediate:
                    nonzero_offsets += int(np.any(np.asarray(corrected["applied_offset"]) != 0))
                if not (gt[frame, 2:] > 0).all():
                    continue
                trajectory[state].append(dict(
                    sequence=name, frame=frame,
                    before=scores(native["bbox"], gt[frame]),
                    after=scores(corrected["bbox"], gt[frame])))
                if state in immediate:
                    immediate[state].append(dict(
                        sequence=name, frame=frame,
                        before=scores(corrected["raw_bbox"], gt[frame]),
                        after=scores(corrected["bbox"], gt[frame])))
            selected = [row for rows in trajectory.values() for row in rows
                        if row["sequence"] == name]
            for arm, field in [("native", "before"), ("offset_corrected", "after")]:
                receipt = worker["records"][arm][name]["alternating_outage"]
                assert len(selected) == receipt["scored_frames"]
                for key in ["PR20", "SR"]:
                    assert np.isclose(np.mean([row[field][key] for row in selected]),
                                      receipt[key], rtol=0, atol=1e-10)
            diagnostics[name] = dict(
                trajectory_changed_frames=len(changed_frames),
                first_trajectory_change=changed_frames[:1],
                template_write_counts=writes, template_write_decision_changes=write_changes,
                nonzero_offset_frames=nonzero_offsets)
    result = dict(
        source_report_sha256=sha(args.source / "report.json"),
        audit_sha256=sha(Path(__file__)), videos=len(names), new_model_calls=0,
        immediate_same_history={state: summarize(rows, names) for state, rows in immediate.items()},
        closed_loop_state_slices={state: summarize(rows, names) for state, rows in trajectory.items()},
        diagnostics=diagnostics,
        scope="Post-hoc saved-trajectory mechanism audit. Immediate comparisons use the same "
              "corrected tracker history, comparing the clipped no-offset and clipped offset "
              "boxes from one raw prediction. "
              "Closed-loop state slices include all prior crop/template effects. Template write "
              "counts do not isolate a causal template mechanism. GT is evaluation-only. "
              "Correlated frame strata and video bootstrap do not establish new-seed, official "
              "or independent prospective efficacy; no algorithm parameter is selected.")
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
