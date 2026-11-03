"""Frozen block-three interventions, without changing the native model."""
import torch


MODES = ('zero', 'true_hist', 'wrong_seq', 'copy_avail')


def replace_search(output, availability, history, mode, search_tokens):
    assert mode in MODES
    assert availability.ndim == 2 and availability.shape[1] == 2
    assert bool(((availability == 0) | (availability == 1)).all())
    assert bool(availability.sum(1).ge(1).all())
    if mode == 'zero' or bool(availability.eq(1).all()):
        return output
    rgb, tir = output
    assert rgb.shape == tir.shape
    assert 0 < search_tokens < rgb.shape[1]
    source = (rgb[:, -search_tokens:], tir[:, -search_tokens:]) if mode == 'copy_avail' else history
    assert len(source) == 2
    values = []
    for stream, value in enumerate(output):
        missing = availability[:, stream].eq(0)
        replacement = source[1 - stream] if mode == 'copy_avail' else source[stream]
        assert replacement.shape == value[:, -search_tokens:].shape
        changed = value.clone()
        changed[missing, -search_tokens:] = replacement[missing]
        values.append(changed)
    return tuple(values)


def capture_search(model, templates, search):
    captured = []
    count = model.baseline.backbone.pos_embed_x.shape[1]

    def capture(block, inputs, output):
        captured.append(tuple(value[:, -count:].detach().clone() for value in output))

    handle = model.baseline.backbone.blocks[2].register_forward_hook(capture)
    with torch.no_grad():
        model(templates, [search])
    handle.remove()
    assert len(captured) == 1
    return captured[0]


def forward_intervention(model, templates, search, availability, history, mode):
    count = model.baseline.backbone.pos_embed_x.shape[1]

    def intervene(block, inputs, output):
        return replace_search(output, availability[-1], history, mode, count)

    handle = model.baseline.backbone.blocks[2].register_forward_hook(intervene)
    with torch.no_grad():
        result = model(templates, [search], availability)[0]
    handle.remove()
    return result


def different_sequence_donors(identities):
    """Deterministic cyclic offset; never use the same sequence as a control."""
    names = [entry['sequence'] for entry in identities]
    assert len(set(names)) >= 2
    result = []
    for index, name in enumerate(names):
        result.append(next((index + offset) % len(names)
                           for offset in range(1, len(names))
                           if names[(index + offset) % len(names)] != name))
    return result


def summarize(rows, clustered):
    effects = {}
    for state in ('01', '10'):
        chosen = [row for row in rows if row['state'] == state]
        assert len(chosen) == 128
        contrasts = {}
        for control in ('zero', 'wrong_seq', 'copy_avail'):
            paired = [dict(identity=row['identity'],
                iou_gain=row['true_hist']['iou'] - row[control]['iou'],
                loss_gain=row[control]['task_loss'] - row['true_hist']['task_loss'])
                for row in chosen]
            contrasts[control] = {
                key: clustered(paired, key, 73103) for key in ('iou_gain', 'loss_gain')}
        effects[state] = contrasts
    advance = all(effects[state]['zero']['iou_gain']['ci95_low'] > 0
                  and effects[state]['wrong_seq']['iou_gain']['mean'] > 0
                  for state in ('01', '10'))
    return effects, advance
