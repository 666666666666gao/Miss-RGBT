"""Execute the preregistered diagnostic on an independently prepared fresh cache."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import cv2
import torch

from diagnose_absent_position import MAPS, clustered, digest, metrics, seed, sha256
from lib.config.cadtrack.config import cfg, update_config_from_file
from miss_rgbt.model import load_pretrained
from research.history_stream.diagnostic import (
    MODES, capture_search, decision, different_sequence_donors, forward_intervention,
    summarize,
)

CHECKPOINT = '/home/gaob/Miss-RGBT/runs/server28-recovery-20261031/batch10_utility_seed2027/epoch17.pth.tar'
CHECKPOINT_SHA = 'fbeaf71a9b4bbfb46e926251b34c7fb1960c4ba41d3da75745342638fd4dc062'


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')


def load_cache(receipt_path):
    receipt = read(receipt_path)
    assert receipt['status'] == 'complete'
    assert receipt['split'] == 'train882_fitting' and receipt['samples'] == 128
    assert receipt['history_max_age'] == 64
    assert receipt['crop_policy'] == 'previous_gt_shared_current_history_factor4'
    assert receipt['all_historical_frame_exclusions_verified']
    # Receipts must bind terminal jobs and immutable manifests, not live snapshots.
    assert receipt['excluded_runs']
    for run in receipt['excluded_runs']:
        assert read(run['completed'])['status'] == 'complete'
        assert sha256(run['completed']) == run['completed_sha256']
        assert sha256(run['manifest']) == run['manifest_sha256']
    for path, expected in receipt['source_sha256'].items():
        assert sha256(path) == expected, path
    assert sha256(receipt['frame_exclusion_audit']) == receipt['frame_exclusion_audit_sha256']
    audit = read(receipt['frame_exclusion_audit'])
    assert audit['status'] == 'passed' and audit['overlapping_frames'] == 0
    assert sha256(receipt['cache']) == receipt['cache_sha256']
    cache = torch.load(receipt['cache'], map_location='cpu')
    assert len(cache) == 128
    names = Path('lib/train/data_specs/lasher_train.txt').read_text().splitlines()
    assert len(names) == len(set(names)) == 882
    fitting = {name for index, name in enumerate(names) if index % 5}
    assert audit['identities_sha256'] == hashlib.sha256(
        json.dumps([item['identity'] for item in cache], sort_keys=True).encode()).hexdigest()
    for item in cache:
        identity = item['identity']
        assert identity['sequence'] in fitting
        assert len(identity['template_frames']) == len(identity['history_template_frames']) == 2
        assert max(identity['template_frames']) < identity['search_frame']
        assert max(identity['history_template_frames']) < identity['history_frame']
        assert 1 <= identity['search_frame'] - identity['history_frame'] <= 64
        assert identity['history_visible'] is True
        assert len(item['templates']) == len(item['history_templates']) == 2
        for image in item['templates'] + item['history_templates']:
            assert image.shape == (1, 6, 128, 128) and torch.isfinite(image).all()
        for image in (item['search'], item['history']):
            assert image.shape == (1, 6, 256, 256) and torch.isfinite(image).all()
        assert item['target'].shape == (1, 4) and torch.isfinite(item['target']).all()
        assert item['crop']['current'] == item['crop']['history']
        assert item['crop']['factor'] == 4
    return receipt, cache


def check_retry(injection_layer, previous_path, receipt_path, config_path):
    """Only a completed negative block-three experiment authorizes layer zero."""
    assert injection_layer in (0, 3)
    if injection_layer == 3:
        assert previous_path is None
        return None
    assert previous_path is not None
    path = Path(previous_path).resolve()
    previous = read(path)
    completed_path = path.parent / 'completed.json'
    completed = read(completed_path)
    assert completed['status'] == previous['status'] == 'complete'
    assert completed['report_sha256'] == sha256(path)
    assert previous['injection_layer'] == completed['injection_layer'] == 3
    assert previous['advance'] is completed['advance'] is False
    assert previous['m1_decision'] == completed['m1_decision'] == 'retry_embedding'
    assert previous['checkpoint'] == CHECKPOINT
    assert previous['checkpoint_sha256'] == CHECKPOINT_SHA
    assert previous['cache_receipt_sha256'] == sha256(receipt_path)
    assert previous['source_sha256'][str(Path(config_path).resolve())] == sha256(config_path)
    for source, expected in previous['source_sha256'].items():
        assert sha256(source) == expected, source
    assert previous['complete_bitwise_parity'] and previous['native_zero_bitwise_parity']
    assert previous['state_unchanged'] and previous['gradients_absent']
    effects, advance = summarize(previous['rows'], clustered)
    assert effects == previous['effects'] and advance is False
    return dict(report=str(path), report_sha256=sha256(path),
                completed=str(completed_path), completed_sha256=sha256(completed_path),
                injection_layer=3, advance=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache-receipt', required=True)
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--injection-layer', type=int, choices=(0, 3), default=3)
    parser.add_argument('--previous-diagnostic')
    args = parser.parse_args()
    receipt, cache = load_cache(args.cache_receipt)
    previous = check_retry(args.injection_layer, args.previous_diagnostic,
                           args.cache_receipt, args.config)
    assert sha256(CHECKPOINT) == CHECKPOINT_SHA
    config = read(args.config)
    assert config['task_utility']['selection'] == 'utility'
    root = Path(args.output)
    root.mkdir(exist_ok=False)
    started = time.time()
    seed(73101)
    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    update_config_from_file('experiments/cadtrack/cadtrack.yaml')
    model = load_pretrained(cfg, config['checkpoint'], repair=False,
                           task_utility=config['task_utility']).cuda().eval().requires_grad_(False)
    checkpoint = torch.load(CHECKPOINT, map_location='cpu')
    assert checkpoint['epoch'] == 17 and checkpoint['config']['task_utility'] == config['task_utility']
    model.load_state_dict(checkpoint['model'], strict=True)
    before = digest(model.state_dict())
    del checkpoint
    donors = different_sequence_donors([item['identity'] for item in cache])
    histories = []
    for item in cache:
        histories.append(tuple(value.cpu() for value in capture_search(model,
            [image.cuda() for image in item['history_templates']], item['history'].cuda(),
            args.injection_layer)))
    rows = []
    for index, item in enumerate(cache):
        templates = [image.cuda() for image in item['templates']]
        search, target = item['search'].cuda(), item['target'].cuda()
        for state, flags in (('11', [1., 1.]), ('01', [0., 1.]), ('10', [1., 0.])):
            availability = [search.new_ones(1, 2) for _ in templates]
            availability.append(search.new_tensor([flags]))
            row = dict(identity=item['identity'], state=state,
                       wrong_sequence=cache[donors[index]]['identity']['sequence'])
            with torch.no_grad():
                native = model(templates, [search], availability)[0]
            for mode in MODES:
                source = histories[donors[index] if mode == 'wrong_seq' else index]
                output = forward_intervention(model, templates, search, availability,
                    tuple(value.cuda() for value in source), mode, args.injection_layer)
                if mode == 'zero' or state == '11':
                    assert all(torch.equal(native[key], output[key]) for key in MAPS)
                with torch.no_grad():
                    loss, iou = metrics(output, target)
                assert torch.isfinite(loss).all() and torch.isfinite(iou).all()
                row[mode] = dict(task_loss=float(loss.item()), iou=float(iou.item()))
            rows.append(row)
        print(json.dumps(dict(samples=index + 1, seconds=time.time() - started)), flush=True)
    assert before == digest(model.state_dict())
    assert all(parameter.grad is None for parameter in model.parameters())
    effects, advance = summarize(rows, clustered)
    m1_decision = decision(args.injection_layer, advance)
    project = Path(__file__).resolve().parents[1]
    source_paths = [Path(__file__), project / 'research/history_stream/diagnostic.py',
        project / 'scripts/diagnose_absent_position.py', Path(args.config),
        Path('experiments/cadtrack/cadtrack.yaml'),
        Path('lib/train/data_specs/lasher_train.txt'), Path('lib/utils/box_ops.py'),
        Path('lib/utils/heapmap_utils.py')]
    source_paths += sorted(Path('miss_rgbt').rglob('*.py'))
    source_paths += sorted(Path('lib/models').rglob('*.py'))
    report = dict(status='complete', checkpoint=CHECKPOINT,
        checkpoint_sha256=CHECKPOINT_SHA, cache_receipt=receipt,
        cache_receipt_sha256=sha256(args.cache_receipt),
        source_sha256={str(path.resolve()): sha256(path) for path in source_paths},
        input_manifest=[item['identity'] for item in cache], rows=rows,
        injection_layer=args.injection_layer, previous_diagnostic=previous,
        m1_decision=m1_decision,
        effects=effects, advance=advance, complete_bitwise_parity=True,
        native_zero_bitwise_parity=True, state_unchanged=True, gradients_absent=True,
        seconds=time.time() - started, peak_gpu_bytes=torch.cuda.max_memory_allocated(),
        scope='Train882 frozen crop diagnostic; not tracking or official benchmark metrics.')
    save(root / 'report.json', report)
    save(root / 'completed.json', dict(status='complete', advance=advance,
        injection_layer=args.injection_layer, m1_decision=m1_decision,
        report_sha256=sha256(root / 'report.json')))
    print(json.dumps(dict(status='complete', advance=advance, effects=effects,
        injection_layer=args.injection_layer, m1_decision=m1_decision)), flush=True)


if __name__ == '__main__':
    main()
