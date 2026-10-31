"""Offline same-history control for sensor-state changes in velocity estimates."""
import json
from pathlib import Path

import numpy as np

from analyze_e0_availability28 import frame_metrics
from analyze_epoch05_health28 import compare
from evaluate_val97_recovery28 import sha


ROOT = Path("/home/gaob/Miss-RGBT")
FOLDER = ROOT / "reports/server28-blind-motion-shortgap16-20261031"


def read(path):
    return json.loads(path.read_text())


def center(boxes):
    return boxes[..., :2] + boxes[..., 2:] / 2


def main():
    parent = read(FOLDER / "report.json")
    assert sha(FOLDER / "report.json") == read(FOLDER / "completed.json")["report_sha256"]
    videos, records = {}, []
    for gpu in [0, 1]:
        folder = FOLDER / ("gpu%d" % gpu)
        worker = read(folder / "report.json")
        assert sha(folder / "report.json") == parent["worker_receipts"][str(gpu)]["report_sha256"]
        assert worker["launch"]["frame_block"] == 1
        for item in worker["launch"]["cohort"]:
            name = item["name"]
            gt_path = Path(item["folder"]) / "init.txt"
            prefix = folder / ("velocity_" + name + "_alternating_outage")
            receipt = worker["records"]["velocity"][name]["alternating_outage"]
            assert sha(gt_path) == item["gt_sha256"]
            assert sha(str(prefix) + ".txt") == receipt["bbox_sha256"]
            assert sha(str(prefix) + ".json") == receipt["trace_sha256"]
            gt = np.loadtxt(str(gt_path), delimiter=",", ndmin=2)
            boxes = np.loadtxt(str(prefix) + ".txt", ndmin=2)
            trace = read(Path(str(prefix) + ".json"))
            available = np.asarray([[1, 1]] + [row["available"] for row in trace])
            blind_frames = np.flatnonzero(~available.any(1))
            predictions = {arm: [] for arm in ["hold", "adjacent", "matched"]}
            selected = []
            for frame in blind_frames:
                before, last = np.flatnonzero(available[:frame].any(1))[-2:]
                assert last == frame - 1
                assert available[before].tolist() == [1, 0]
                assert available[last].tolist() == [1, 1]
                same_state = np.flatnonzero((available[:last] == available[last]).all(1))
                assert len(same_state) >= 1
                matched_before = same_state[-1]
                adjacent_velocity = (center(boxes[last]) - center(boxes[before])) / (last - before)
                matched_velocity = (center(boxes[last]) - center(boxes[matched_before])) / (last - matched_before)
                measured = boxes[last].copy()
                measured[:2] += (frame - last) * adjacent_velocity
                assert np.allclose(measured, boxes[frame], rtol=0, atol=1e-9)
                matched = boxes[last].copy()
                matched[:2] += (frame - last) * matched_velocity
                if not (gt[frame, 2:] > 0).all():
                    continue
                selected.append(frame)
                predictions["hold"].append(boxes[last])
                predictions["adjacent"].append(measured)
                predictions["matched"].append(matched)
                row = dict(
                    sequence=name, frame=int(frame), adjacent_frames=[int(before), int(last)],
                    matched_frames=[int(matched_before), int(last)],
                    adjacent_velocity=adjacent_velocity.tolist(),
                    matched_velocity=matched_velocity.tolist())
                if (gt[[before, matched_before, last], 2:] > 0).all():
                    row["retrospective_gt"] = dict(
                        adjacent_velocity_error=float(np.linalg.norm(
                            adjacent_velocity - (center(gt[last]) - center(gt[before])) / (last - before))),
                        matched_velocity_error=float(np.linalg.norm(
                            matched_velocity - (center(gt[last]) - center(gt[matched_before])) /
                            (last - matched_before))))
                records.append(row)
            assert selected
            videos[name] = {
                arm: {key: float(value.mean()) for key, value in frame_metrics(
                    np.asarray(predicted), gt[selected]).items()}
                for arm, predicted in predictions.items()}
    names = sorted(videos)
    assert len(names) == 16
    indices = np.random.RandomState(2027).randint(0, len(names), (10000, len(names)))
    arms = {arm: {name: videos[name][arm] for name in names}
            for arm in ["hold", "adjacent", "matched"]}
    result = dict(
        parent_report_sha256=sha(FOLDER / "report.json"), source_sha256=sha(Path(__file__)),
        videos=16, valid_blind_frames=len(records), per_video=videos, records=records,
        video_macro={arm: {key: float(np.mean([arms[arm][name][key] for name in names]))
                           for key in ["PR20", "SR"]} for arm in arms},
        matched_minus_adjacent=compare(arms["matched"], arms["adjacent"], names, indices),
        matched_minus_hold=compare(arms["matched"], arms["hold"], names, indices),
        scope="Offline blind-only control on the same saved adjacent-CV trajectory. "
              "No new closed-loop crop/template history. Matched availability also changes "
              "the observation time interval, so differences do not isolate modality bias. "
              "No inference GT, fitted parameter, official score or validated contribution.")
    with (FOLDER / "availability_velocity_audit.json").open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps({key: result[key] for key in [
        "videos", "valid_blind_frames", "video_macro", "matched_minus_adjacent"]}, indent=2))


if __name__ == "__main__":
    main()
