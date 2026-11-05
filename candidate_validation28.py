"""Fixed causal validation tuples for a preregistered convergence proxy."""
import hashlib
import random

import numpy as np
import torch

from miss_rgbt.data import make_loader
from miss_rgbt.objectives import task_loss


def fixed_validation(cfg, config):
    python_state, numpy_state = random.getstate(), np.random.get_state()
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        seed = config["validation_seed"]
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        loader = make_loader(cfg, config["dataset"], "val", 4,
                             config["validation_samples"], 0)
        batches = list(loader)
    random.setstate(python_state)
    np.random.set_state(numpy_state)
    digest = hashlib.sha256()
    for batch in batches:
        for key in ["template_images", "search_images", "search_anno"]:
            digest.update(batch[key].numpy().tobytes())
    return batches, digest.hexdigest()


def validate(model, batches, cfg):
    state = model.utility_repair.selection_rng_state.clone()
    model.utility_repair.reset_selection()
    model.eval()
    totals = {key: 0. for key in ["11", "01", "10"]}
    count = 0
    with torch.no_grad(), torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        for batch in batches:
            templates = [image.cuda() for image in batch["template_images"]]
            searches = [image.cuda() for image in batch["search_images"]]
            boxes = batch["search_anno"].cuda()
            size = searches[0].shape[0]
            for label, available in [("11", [1., 1.]), ("01", [0., 1.]), ("10", [1., 0.])]:
                availability = [searches[0].new_ones(size, 2) for _ in templates]
                availability += [
                    searches[0].new_tensor(available).expand(size, -1) for _ in searches]
                loss = task_loss(model(templates, searches, availability), boxes, cfg)
                assert torch.isfinite(loss).all()
                totals[label] += float(loss.mean()) * size
            count += size
    model.utility_repair.selection_rng_state.copy_(state)
    return dict(samples=count, task_loss={key: value / count for key, value in totals.items()},
                mean_task_loss=sum(totals.values()) / (3 * count))
