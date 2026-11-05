"""Per-search availability normalization; native objectives stay immutable."""
import torch

from miss_rgbt.objectives import response_kd, task_loss
from miss_rgbt.objectives import training_loss as native_training_loss

STATES = ((1., 1.), (0., 1.), (1., 0.))


class StateLossEMA:
    def __init__(self, device, momentum=.99):
        self.momentum = momentum
        self.values = torch.ones(3, device=device)

    def normalize(self, losses, availability):
        """Update once per microbatch using all search/sample observations."""
        assert losses.ndim == 2 and availability.shape == (*losses.shape, 2)
        states = availability.new_tensor(STATES)
        matches = availability.unsqueeze(-2).eq(states).all(-1)
        assert matches.sum(-1).eq(1).all()
        indices = matches.to(torch.long).argmax(-1)
        for index in range(3):
            selected = indices.eq(index)
            if selected.any():
                mean = losses.detach()[selected].mean()
                self.values[index].mul_(self.momentum).add_(
                    mean, alpha=1. - self.momentum)
        normalized = losses / self.values[indices]
        return normalized.sum(0), {
            'ema_m_' + str(int(state[0])) + str(int(state[1])): float(value)
            for state, value in zip(STATES, self.values)}


def training_loss(complete, missing, boxes, cfg, forward_weight=0.,
                  reverse_weight=0., margin=0., state_norm=False,
                  normalizer=None, search_availability=None):
    if not state_norm:
        return native_training_loss(complete, missing, boxes, cfg,
                                    forward_weight, reverse_weight, margin)
    assert normalizer is not None and search_availability is not None
    assert len(missing) == len(boxes) == len(search_availability)
    loss_c = task_loss(complete, boxes, cfg)
    # Preserve the native focal reduction over the whole batch for each search.
    per_search = torch.stack([task_loss([output], boxes[i:i + 1], cfg)
                              for i, output in enumerate(missing)])
    raw_m = per_search.sum(0)
    loss_m, status = normalizer.normalize(
        per_search, torch.stack(search_availability))
    reliable = (raw_m.detach() + margin < loss_c.detach()).to(loss_c.dtype)
    kd_forward = response_kd(missing, complete)
    kd_reverse = response_kd(complete, missing)
    total = loss_c.mean() + loss_m.mean() + forward_weight * kd_forward.mean()
    total = total + reverse_weight * (reliable * kd_reverse).mean()
    status.update(task_c=float(loss_c.mean().detach()),
                  task_m=float(raw_m.mean().detach()),
                  task_m_normalized=float(loss_m.mean().detach()),
                  kd_forward=float(kd_forward.mean().detach()),
                  kd_reverse=float((reliable * kd_reverse).mean().detach()),
                  reverse_fraction=float(reliable.mean()))
    return total, status
