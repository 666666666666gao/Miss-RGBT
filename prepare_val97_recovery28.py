import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path("/home/gaob/Miss-RGBT")
RUNTIME = ROOT / "runtime/task_utility_20261018"
OUTPUT = ROOT / "development/recovery_val97_20261030"
DATA = Path("/data/wangwj/dataset/LasHeR/trainingset")
PATTERNS = {
    "complete": [[1, 1]],
    "rgb_outage": [[1, 1], [0, 1]],
    "tir_outage": [[1, 1], [1, 0]],
    "alternating_outage": [[1, 1], [0, 1], [1, 1], [1, 0], [1, 1], [0, 0]],
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    split_dir = RUNTIME / "lib/train/data_specs"
    names = (split_dir / "lasher_val.txt").read_text().splitlines()
    train = set((split_dir / "lasher_train.txt").read_text().splitlines())
    assert len(names) == len(set(names)) == 97
    assert not (set(names) & train)
    sequences = []
    masks = {condition: {} for condition in PATTERNS}
    counts = {condition: {state: 0 for state in ("00", "01", "10", "11")}
              for condition in PATTERNS}
    for name in names:
        folder = DATA / name
        gt_file = folder / "init.txt"
        gt = np.loadtxt(str(gt_file), delimiter=",", ndmin=2)
        assert gt.shape[1] == 4 and np.isfinite(gt).all()
        assert (gt[0, 2:] > 0).all()
        frames = len(gt)
        assert len(list((folder / "visible").glob("*.jpg"))) == frames
        assert len(list((folder / "infrared").glob("*.jpg"))) == frames
        sequences.append(dict(name=name, frames=frames, gt_path=str(gt_file),
                              gt_sha256=sha(gt_file),
                              invalid_size_frames=int((gt[:, 2:] <= 0).any(1).sum())))
        for condition, pattern in PATTERNS.items():
            states = [pattern[(index // 32) % len(pattern)][:] for index in range(frames)]
            assert states[0] == [1, 1]
            masks[condition][name] = dict(frames=frames, data=states)
            for rgb, tir in states:
                counts[condition][str(rgb) + str(tir)] += 1
    assert sum(item["frames"] for item in sequences) == 54646
    OUTPUT.mkdir(parents=True, exist_ok=False)
    files = {}
    for condition, mask in masks.items():
        path = OUTPUT / (condition + ".json")
        path.write_text(json.dumps(mask, separators=(",", ":")) + "\n")
        files[condition] = dict(path=str(path), sha256=sha(path), states=counts[condition])
    protocol = dict(
        dataset="LasHeR project val97", data_root=str(DATA),
        sequences=sequences, conditions=files,
        split_sha256={name: sha(split_dir / name)
                      for name in ("lasher_train.txt", "lasher_val.txt")},
        frame_block=32, patterns=PATTERNS,
        rationale="Fixed 32-video-frame blocks for controlled outage/recovery comparisons; "
                  "not fitted to published test masks or model scores.",
        initialization="First frame complete for every condition",
        missing_protocol="Duplicate the available image for one missing sensor; "
                         "both missing holds the previous box without tracker call",
        annotation_protocol="Track every frame; exclude nonpositive-width/height GT only "
                            "from development geometry/IoU aggregates; retain frame records",
        limitations="Development diagnostics only, not official Miss benchmark. "
                    "Original E0 exposure to val97 is unverified. "
                    "Current train882 and val97 names do not overlap.",
        prepared_by_sha256=sha(Path(__file__)),
        model_execution=False,
    )
    (OUTPUT / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    summary = dict(output=str(OUTPUT), sequences=len(sequences), frames=54646,
                   conditions=files, invalid_size_frames=sum(s["invalid_size_frames"] for s in sequences),
                   model_execution=False)
    (ROOT / "reports/server28-val97-recovery-preparation.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
