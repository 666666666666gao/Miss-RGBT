"""Geometry of the fixed51-crop rule over all observed valid cohort frames."""
import json
import math
from pathlib import Path

import cv2
import numpy as np

ROOT = Path("/home/gaob/Miss-RGBT")
BASE = ROOT / "reports/server28-val97-gpu-consistency-20261030/alternating_outage"


def summarize(rows):
    failed = [row for row in rows if not row["native_support"]]
    result = dict(observed_valid=len(rows), native_unsupported=len(failed))
    for label in ["previous", "initial", "combined"]:
        result[label] = dict(
            support_restored=sum(row[label + "_support"] for row in failed),
            center_restored=sum(row[label + "_center"] for row in failed))
    return result


def main():
    protocol = json.loads((ROOT / "development/recovery_val97_20261030/protocol.json").read_text())
    cohort = json.loads((BASE / "report.json").read_text())["sequences"]
    specs = {item["name"]: item for item in protocol["sequences"]}
    by_sequence, all_rows = {}, []
    for name in cohort:
        spec = specs[name]
        gt = np.loadtxt(spec["gt_path"], delimiter=",", ndmin=2)
        image_path = sorted((Path(protocol["data_root"]) / name / "visible").glob("*.jpg"))[0]
        height, width = cv2.imread(str(image_path)).shape[:2]
        trace = json.loads((BASE / (name + ".json")).read_text())
        rows = []
        for row in trace:
            if not row["observed"] or row["diagnostic"] is None:
                continue
            target = gt[row["frame"]]
            target_center = target[:2] + target[2:] / 2.
            record = dict(native_support=row["diagnostic"]["target_overlaps_previous_crop"])
            for label, size in [("previous", row["previous_box"][2:]), ("initial", gt[0, 2:])]:
                side = math.ceil(math.sqrt(size[0] * size[1]) * 4)
                supports, centers = [], []
                for y in range(5):
                    for x in range(5):
                        center = [width * (x + .5) / 5., height * (y + .5) / 5.]
                        lower = [round(value - side / 2.) for value in center]
                        supports.append(all(min(lower[axis] + side, target[axis] + target[axis + 2]) >
                                            max(lower[axis], target[axis]) for axis in [0, 1]))
                        centers.append(all(lower[axis] <= target_center[axis] < lower[axis] + side
                                           for axis in [0, 1]))
                record[label + "_support"] = any(supports)
                record[label + "_center"] = any(centers)
            record["combined_support"] = record["previous_support"] or record["initial_support"]
            record["combined_center"] = record["previous_center"] or record["initial_center"]
            rows.append(record)
        by_sequence[name] = summarize(rows)
        all_rows.extend(rows)
    result = dict(
        summary=summarize(all_rows), per_sequence=by_sequence,
        scope="Pure crop geometry, all observed valid first8-cohort GPU trajectory frames. "
              "Same fixed5x5 centers and previous/initial sizes as native-grid neural audit. "
              "No proposal quality, identity, acceptance, closed-loop or benchmark evidence. "
              "Image dimensions assumed constant within each sequence; source frames and "
              "ground truth are existing fixed audit assets.",
    )
    destination = ROOT / "reports/server28-native-grid-coverage-20261030.json"
    destination.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
