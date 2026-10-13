"""Real GPU/data/optimizer validation, not a benchmark result."""
import json
from pathlib import Path
import random
import time

import numpy as np
import torch

from lib.config.cadtrack.config import cfg, update_config_from_file
from lib.utils.box_ops import giou_loss
from lib.train.actors.cadtrack import CADTRACKActor
from lib.utils.focal_loss import FocalLoss
from miss_rgbt.data import make_loader
from miss_rgbt.model import load_pretrained
from miss_rgbt.objectives import task_loss, training_loss
from miss_rgbt.train import sample_masks, fix_bn


def main():
    config = json.loads(Path('configs/e4.json').read_text())
    random.seed(2027)
    np.random.seed(2027)
    torch.manual_seed(2027)
    torch.cuda.manual_seed_all(2027)
    torch.set_num_threads(1)
    update_config_from_file('experiments/cadtrack/cadtrack.yaml')
    loader = make_loader(cfg, config['dataset'], 'train', 2, 8, 0)
    data = next(iter(loader))
    templates = [x.cuda() for x in data['template_images']]
    searches = [x.cuda() for x in data['search_images']]
    boxes = data['search_anno'].cuda()
    model = load_pretrained(cfg, config['checkpoint'], repair=True).cuda().eval()
    report = dict(scope='real GPU training data validation only; no full benchmark', seed=2027,
                  templates=list(data['template_images'].shape), searches=list(data['search_images'].shape))
    with torch.no_grad():
        reference = model.baseline(templates, searches)
        current = model(templates, searches)
        errors = {key: max(float((a[key] - b[key]).abs().max()) for a, b in zip(reference, current))
                  for key in ['pred_boxes', 'score_map', 'size_map', 'offset_map']}
        assert max(errors.values()) == 0., errors
        report['complete_baseline_max_difference'] = errors
        actor = CADTRACKActor(model.baseline, dict(giou=giou_loss, l1=torch.nn.functional.l1_loss, focal=FocalLoss()),
            dict(giou=2., l1=5., focal=1.), type('Settings', (), dict(batchsize=2))(), cfg)
        native_loss, _ = actor.compute_losses(reference, dict(search_anno=boxes))
        own_loss = task_loss(current, boxes, cfg).mean()
        assert torch.allclose(native_loss, own_loss, atol=1e-5, rtol=1e-6), (native_loss, own_loss)
        report['native_loss'] = float(native_loss)
        report['per_sample_loss_mean'] = float(own_loss)
        available = [torch.ones(2, 2, device='cuda')] + [torch.tensor([[0., 1.], [0., 1.]], device='cuda')
            for _ in range(len(templates) - 1 + len(searches))]
        output = model(templates, searches, available)
        changed = [x.clone() for x in templates + searches]
        for image in changed[1:]:
            image[:, :3] = torch.randn_like(image[:, :3]) * 100
        again = model(changed[:len(templates)], changed[len(templates):], available)
        invisible_error = max(float((a['score_map'] - b['score_map']).abs().max()) for a, b in zip(output, again))
        assert invisible_error == 0., invisible_error
        report['unobserved_rgb_perturbation_max_difference'] = invisible_error
    del reference, current, output, again, native_loss, own_loss
    torch.cuda.empty_cache()
    model.train()
    fix_bn(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    before = model.repair.delta.weight.detach().clone()
    start = time.time()
    available, patch = sample_masks(templates, searches, config)
    complete = model(templates, searches)
    missing = model(templates, searches, available, patch)
    loss, status = training_loss(complete, missing, boxes, cfg, .1, .02, .1)
    assert torch.isfinite(loss)
    loss.backward()
    grads = {name: float(p.grad.abs().sum()) for name, p in model.named_parameters() if p.grad is not None}
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    assert grads['repair.delta.weight'] > 0
    optimizer.step()
    assert not torch.equal(before, model.repair.delta.weight)
    torch.cuda.synchronize()
    report.update(real_optimizer_updates=1, update_seconds=time.time() - start,
                  peak_gpu_bytes=torch.cuda.max_memory_allocated(), status=status,
                  repair_output_gradient_l1=grads['repair.delta.weight'],
                  all_present_gradients_finite=True,
                  trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                  repair_parameters=sum(p.numel() for p in model.repair.parameters()))
    checkpoint = Path('checkpoints/method_native_validation.pth.tar')
    checkpoint.parent.mkdir(exist_ok=True)
    torch.save(dict(model=model.state_dict(), config=config), str(checkpoint))
    with torch.no_grad():
        model.eval()
        expected = model(templates, searches, available)
        model.load_state_dict(torch.load(str(checkpoint), map_location='cpu')['model'], strict=True)
        restored = model(templates, searches, available)
        reload_error = max(float((a['score_map'] - b['score_map']).abs().max()) for a, b in zip(expected, restored))
        assert reload_error == 0.
        report['strict_reload_max_difference'] = reload_error
    Path('reports/method_native_validation.json').write_text(json.dumps(report, indent=2))
    print('METHOD_NATIVE_PASS ' + json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
