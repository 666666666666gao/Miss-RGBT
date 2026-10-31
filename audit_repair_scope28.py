"""Privileged feature interventions on existing train882 development tuples."""
import datetime
import json
import os
from pathlib import Path
import time

os.environ["CUDA_VISIBLE_DEVICES"] = ""
from probe_protected_sixepoch28 import (
    ROOT, RUNTIME, cfg, cv2, digest, load_pretrained, make_loader,
    seed, sha, task_loss, torch, update_config_from_file,
)


OUTPUT = ROOT / "reports/server28-repair-scope-20261031"
PILOT = ROOT / "reports/server28-protected-sixepoch-20261030"
STATES = [("11", [1, 1]), ("01", [0, 1]), ("10", [1, 0])]
ARMS = ["native", "missing_tokens_oracle", "affected_stream_oracle"]


def output_hash(outputs):
    return digest({str(index) + "_" + field: output[field]
                   for index, output in enumerate(outputs)
                   for field in ["score_map", "size_map", "offset_map", "pred_boxes"]})


def main():
    OUTPUT.mkdir(exist_ok=False)
    torch.set_num_threads(4)
    cv2.setNumThreads(1)
    seed(2027)
    started = time.time()
    pilot = json.loads((PILOT / "report.json").read_text())
    for path, expected in pilot["launch"]["source_sha256"].items():
        assert sha(Path(path)) == expected
    checkpoint = pilot["launch"]["checkpoint"]
    assert sha(checkpoint) == pilot["launch"]["checkpoint_sha256"]
    update_config_from_file(str(RUNTIME / "experiments/cadtrack/cadtrack.yaml"))
    model = load_pretrained(cfg, checkpoint, repair=True, width=128,
                            repair_layer=6).cpu().eval().requires_grad_(False)
    model_sha = digest(model.state_dict())
    data_root = "/data/wangwj/dataset/LasHeR/trainingset"
    train = make_loader(cfg, data_root, "train", 2, 64, 0)
    development_names = train.dataset.datasets[0].sequence_list[::5]
    loader = make_loader(cfg, data_root, "val", 2, 16, 0)
    loader.dataset.datasets[0].sequence_list = development_names
    seed(27182)
    batches = list(loader)[:2]
    manifest = json.loads((PILOT / "inputs.json").read_text())
    hashes = [digest({key: batch[key] for key in [
        "template_images", "search_images", "search_anno"]}) for batch in batches]
    assert hashes == manifest["development"][:2]
    assert not set(development_names) & set(manifest["heldout_sequences"])
    launch = dict(
        pid=os.getpid(), started_utc=datetime.datetime.utcnow().isoformat() + "Z",
        device="cpu", cpu_threads=4, samples=4, input_sha256=hashes,
        checkpoint_sha256=sha(checkpoint), layer=6,
        source_sha256={str(Path(__file__)): sha(Path(__file__)),
                       **pilot["launch"]["source_sha256"]},
        arms=ARMS, scope="Privileged complete-feature replacement on reused train882 dev tuples; "
                        "not a deployable method, training, tracking or official evaluation")
    (OUTPUT / "launch.json").write_text(json.dumps(launch, indent=2) + "\n")
    print("LAUNCH " + json.dumps(launch), flush=True)
    context = dict(mode="capture", references=[], index=0)

    def intervene(module, inputs, outputs):
        if context["mode"] == "capture":
            context["references"].append(tuple(value.detach().clone() for value in outputs))
            return outputs
        reference = context["references"][context["index"]]
        context["index"] += 1
        mask = inputs[2]
        if context["mode"] == "native":
            return outputs
        if context["mode"] == "missing_tokens_oracle":
            return tuple(torch.where(mask[..., view:view + 1] == 0, reference[view], outputs[view])
                         for view in range(2))
        assert context["mode"] == "affected_stream_oracle"
        affected = (mask == 0).any(1)
        return tuple(torch.where(affected[:, view, None, None], reference[view], outputs[view])
                     for view in range(2))

    hook = model.repair.register_forward_hook(intervene)
    records = {dynamic + "_" + search: {arm: [] for arm in ARMS}
               for dynamic, _ in STATES for search, _ in STATES}
    teacher_losses = []
    complete_equal = True
    affected_equal = True
    with torch.no_grad():
        for batch_index, batch in enumerate(batches):
            templates, searches = list(batch["template_images"]), list(batch["search_images"])
            assert len(templates) == 2
            size = searches[0].shape[0]
            context.update(mode="capture", references=[], index=0)
            teacher = model(templates, searches)
            assert len(context["references"]) == len(searches)
            teacher_hash = output_hash(teacher)
            teacher_losses.extend(task_loss(teacher, batch["search_anno"], cfg).tolist())
            for dynamic_label, dynamic_flags in STATES:
                for search_label, search_flags in STATES:
                    key = dynamic_label + "_" + search_label
                    availability = [torch.ones(size, 2)]
                    availability += [torch.tensor([dynamic_flags], dtype=torch.float32).expand(size, -1)]
                    availability += [torch.tensor([search_flags], dtype=torch.float32).expand(size, -1)
                                     for _ in searches]
                    for arm in ARMS:
                        context.update(mode=arm, index=0)
                        actual = model(templates, searches, availability)
                        assert context["index"] == len(searches)
                        losses = task_loss(actual, batch["search_anno"], cfg)
                        assert torch.isfinite(losses).all()
                        actual_hash = output_hash(actual)
                        if key == "11_11":
                            complete_equal &= actual_hash == teacher_hash
                        if arm == "affected_stream_oracle":
                            affected_equal &= actual_hash == teacher_hash
                        records[key][arm].extend(losses.tolist())
            print("BATCH " + json.dumps(dict(batch=batch_index, seconds=time.time() - started)),
                  flush=True)
    hook.remove()
    assert digest(model.state_dict()) == model_sha
    assert complete_equal and affected_equal
    means = {key: {arm: sum(losses) / len(losses) for arm, losses in arms.items()}
             for key, arms in records.items()}
    deltas = {key: {arm: values[arm] - values["native"] for arm in ARMS[1:]}
              for key, values in means.items()}
    report = dict(
        launch=launch, teacher_losses=teacher_losses,
        teacher_mean_task_loss=sum(teacher_losses) / len(teacher_losses),
        losses=records, means=means, oracle_minus_native=deltas,
        model_state_unchanged=True, complete_history_outputs_bitwise_equal=complete_equal,
        affected_stream_outputs_match_complete_teacher=affected_equal,
        seconds=time.time() - started,
        interpretation="These interventions use unavailable complete-input features. Differences "
                       "measure a repair-scope diagnostic only, not achievable inference gains. "
                       "Mixed-history sample size is four reused development tuples, one seed. "
                       "A teacher can itself have higher task loss on a sample; replacing exact "
                       "features is not assumed to improve supervised loss.")
    (OUTPUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    (OUTPUT / "completed.json").write_text(json.dumps(dict(
        cells=27, samples_per_cell=4, report_sha256=sha(OUTPUT / "report.json"),
        seconds=time.time() - started), indent=2) + "\n")
    print("COMPLETE " + json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
