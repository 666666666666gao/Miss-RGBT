"""Train-only task-loss diagnostics; not tracking accuracy or official scores."""
import torch

from miss_rgbt.objectives import task_loss
from model import FeedbackTrackerModel


def validate(model, batches, cfg):
    selector = model.utility_repair.selection_rng_state.clone()
    model.utility_repair.reset_selection()
    model.eval()
    labels = [('11', [1., 1.]), ('01', [0., 1.]), ('10', [1., 0.])]
    totals = dict.fromkeys([label for label, _ in labels], 0.)
    paired = isinstance(model, FeedbackTrackerModel)
    records = {label: [] for label, _ in labels}
    count = 0
    with torch.no_grad(), torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        for batch_index, data in enumerate(batches):
            templates = [value.cuda() for value in data['template_images']]
            searches = [value.cuda() for value in data['search_images']]
            boxes = data['search_anno'].cuda()
            size = searches[0].shape[0]
            for label, flags in labels:
                availability = [searches[0].new_ones(size, 2) for _ in templates]
                availability += [searches[0].new_tensor(flags).expand(size, -1)
                                 for _ in searches]
                outputs = model(templates, searches, availability)
                final = task_loss(outputs, boxes, cfg)
                assert torch.isfinite(final).all()
                totals[label] += float(final.mean()) * size
                if paired:
                    first = task_loss([out['round1'] for out in outputs], boxes, cfg)
                    assert torch.isfinite(first).all()
                    records[label].append(dict(batch_index=batch_index,
                        round1=first.cpu().tolist(), round2=final.cpu().tolist()))
            count += size
    model.utility_repair.selection_rng_state.copy_(selector)
    result = dict(samples=count, task_loss={key: value/count for key, value in totals.items()},
                  mean_task_loss=sum(totals.values())/(3*count))
    if paired:
        comparison = {}
        for label, _ in labels:
            first = [loss for row in records[label] for loss in row['round1']]
            final = [loss for row in records[label] for loss in row['round2']]
            assert len(first) == len(final) == count
            improved = sum(after < before for before, after in zip(first, final))
            worsened = sum(after > before for before, after in zip(first, final))
            equal = count-improved-worsened
            comparison[label] = dict(samples=count, round1_mean=sum(first)/count,
                round2_mean=sum(final)/count, mean_delta=(sum(final)-sum(first))/count,
                lower_loss=improved, higher_loss=worsened, equal_loss=equal,
                lower_loss_fraction=improved/count, higher_loss_fraction=worsened/count,
                equal_loss_fraction=equal/count)
        result.update(round_comparison=comparison, round_samples=records,
            round_comparison_scope='Per-tuple sum of task losses over search frames; '
                'strict comparison, not correction/destruction of tracked frames or official accuracy')
    return result
