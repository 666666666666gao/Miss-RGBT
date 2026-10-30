"""Summarize held-out joint interventions from the frozen selector pilot."""
import argparse
import json
from pathlib import Path
import statistics


def summarize(rows):
    count = len(rows)
    result = {"slates": count}
    for rule in ["native", "selected", "random", "oracle"]:
        result[rule + "_mean_loss"] = sum(row[rule + "_loss"] for row in rows) / count
        result[rule + "_mean_relative_gain_percent"] = sum(
            100 * (row["off_loss"] - row[rule + "_loss"]) / row["off_loss"]
            for row in rows) / count
    for baseline in ["native", "random", "oracle"]:
        differences = [row["selected_loss"] - row[baseline + "_loss"] for row in rows]
        result["selected_minus_" + baseline] = {
            "mean_loss_delta": sum(differences) / count,
            "wins": sum(value < 0 for value in differences),
            "ties": sum(value == 0 for value in differences),
            "losses": sum(value > 0 for value in differences),
        }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("report")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = json.loads(Path(args.report).read_text())
    result = {
        "scope": "Frozen offline pilot, four held-out sequences and one tuple per sequence/state. "
                 "Correlated states; no tracker, benchmark or novelty evidence. "
                 "Oracle ranks single-region benefits; it is not the best joint subset.",
        "seconds": report["seconds"],
        "controls": {},
    }
    labels = json.loads(Path(args.report).with_name("labels.json").read_text())
    result["label_scale_by_split"] = {}
    for split in ["train", "development", "heldout"]:
        slates = [row["targets"] for row in labels if row["split"] == split]
        values = [value for slate in slates for value in slate]
        result["label_scale_by_split"][split] = {
            "slates": len(slates),
            "median_absolute_benefit": statistics.median(abs(value) for value in values),
            "median_within_slate_span": statistics.median(max(row) - min(row) for row in slates),
            "positive_region_benefits": sum(value > 0 for value in values),
            "negative_region_benefits": sum(value < 0 for value in values),
            "exact_zero_region_benefits": sum(value == 0 for value in values),
        }
    for kind, control in report["controls"].items():
        rows = control["joint_interventions"]
        result["controls"][kind] = {
            "epochs": control["epochs"],
            "selected_epoch": control["selected_epoch"],
            "selected_sha256": control["selected_sha256"],
            "heldout_ranking": control["heldout"],
            "joint_all": summarize(rows),
            "joint_by_state": {
                state: summarize([row for row in rows if row["state"] == state])
                for state in ["11", "01", "10"]
            },
        }
    regression = report["controls"]["regression"]["joint_interventions"]
    pairwise = report["controls"]["pairwise"]["joint_interventions"]
    comparisons = []
    for first, second in zip(regression, pairwise):
        assert (first["sequence"], first["state"]) == (second["sequence"], second["state"])
        for key in ["off_loss", "native_loss", "random_loss", "oracle_loss"]:
            assert first[key] == second[key], key
        comparisons.append({
            "sequence": first["sequence"], "state": first["state"],
            "pairwise_minus_regression_loss": second["selected_loss"] - first["selected_loss"],
        })
    assert len(regression) == len(pairwise) == 12
    result["paired_comparisons"] = comparisons
    result["pairwise_minus_regression_mean_loss"] = sum(
        row["pairwise_minus_regression_loss"] for row in comparisons) / len(comparisons)
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
