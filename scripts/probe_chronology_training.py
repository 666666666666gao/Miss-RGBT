"""Matched temporal-mask training using identical small native-tail adapters."""
import argparse
import json
import os
from pathlib import Path
import time

import cv2
import torch

from lib.config.cadtrack.config import cfg, update_config_from_file
from miss_rgbt.model import load_pretrained
import probe_tail_lowrank as tail
from diagnose_template_chronology import old_ids
from diagnose_absent_position import CHECKPOINT, CHECKPOINT_SHA, clustered, digest, sha256


class PairedLoader:
    def __init__(self, loader, chronological, inputs):
        self.loader = loader
        self.dataset = loader.dataset
        self.chronological = chronological
        self.inputs = inputs

    def __iter__(self):
        for data in self.loader:
            for index in range(data['template_images'].shape[1]):
                self.inputs.append(digest(dict(
                    templates=data['template_images'][:, index],
                    search=data['search_images'][0, index],
                    boxes=data['search_anno'][0, index])))
            # Keep identity ids in original sampler order; explicitly record the transform.
            if self.chronological:
                data['template_images'] = data['template_images'].flip(0)
            yield data


def cache(model, names, samples, sample_seed, root, excluded, chronological):
    original_loader = tail.make_loader
    inputs = []
    def make_loader(*args, **kwargs):
        return PairedLoader(original_loader(*args, **kwargs), chronological, inputs)
    tail.make_loader = make_loader
    result, manifest = tail.cache_split(model, names, samples, sample_seed, root, excluded)
    tail.make_loader = original_loader
    manifest['template_images_reversed'] = chronological
    manifest['original_order_input_sha256'] = inputs
    tail.save(root / 'temporal_binding.json', dict(
        original_identity_order='newer,older',
        actual_template_order='older,newer' if chronological else 'newer,older',
        complete_template='older' if chronological else 'newer'))
    return result, manifest


def effects_from(audits):
    effects = {}
    for name, suffix in [('complete', '11_11'), ('rgb_missing', '_01'), ('tir_missing', '_10')]:
        values = [dict(identity=row['identity'], gain=row['iou'] - row['native_iou'])
            for row in audits['chronological']['rows'] if row['cell'].endswith(suffix)]
        effects[name] = clustered(values, 'gain', 53301)
    paired = []
    for first, second in zip(audits['chronological']['rows'], audits['native']['rows']):
        assert all(first[key] == second[key] for key in
                   ['identity', 'cell', 'target', 'native_box', 'native_iou', 'native_loss'])
        paired.append(dict(identity=first['identity'], gain=first['iou'] - second['iou']))
    effects['chronological_minus_native_training'] = clustered(paired, 'gain', 53302)
    advance = all(effects[name]['mean'] >= .01 and effects[name]['ci95_low'] > 0
                  for name in ['complete', 'rgb_missing', 'tir_missing'])
    advance = advance and effects['chronological_minus_native_training']['ci95_low'] > 0
    return effects, advance


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    root = Path(args.output)
    root.mkdir(exist_ok=False)
    started = time.time()
    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    update_config_from_file('experiments/cadtrack/cadtrack.yaml')
    assert sha256(CHECKPOINT) == CHECKPOINT_SHA
    model = load_pretrained(cfg, CHECKPOINT, repair=False).cuda().eval().requires_grad_(False)
    before = digest(model.state_dict())
    names = Path('lib/train/data_specs/lasher_train.txt').read_text().splitlines()
    assert len(names) == len(set(names)) == 882
    partitions = dict(train=[name for index, name in enumerate(names) if index % 5],
                      development=names[::5][:88], audit=names[::5][88:])
    excluded = old_ids()
    prior = json.loads((tail.PROJECT / 'reports/server28_template_chronology_20261102/report.json').read_text())
    assert prior['diagnostic_supported']
    for item in prior['input_manifest']:
        value = item['identity']
        excluded.add((value['sequence'], tuple(value['template_frames']), tuple(value['search_frames'])))
    caches, manifests = {}, {}
    for mode in ['native', 'chronological']:
        key = mode + '_train'
        caches[key], manifests[key] = cache(model, partitions['train'], 256, 53101,
            root / key, excluded, mode == 'chronological')
    first = manifests['native_train']['input_manifest']
    second = manifests['chronological_train']['input_manifest']
    assert [item['identity'] for item in first] == [item['identity'] for item in second]
    assert (manifests['native_train']['original_order_input_sha256'] ==
            manifests['chronological_train']['original_order_input_sha256'])
    caches['development'], manifests['development'] = cache(
        model, partitions['development'], 128, 53102, root / 'development', excluded, True)
    sizing = tail.size_batch(model, caches['native_train'], root)
    arms = {}
    for mode in ['native', 'chronological']:
        arms[mode] = tail.fit(model, 'attention', caches[mode + '_train'],
                             caches['development'], sizing, root / mode)
        arms[mode]['training_binding'] = mode
    caches['audit'], manifests['audit'] = cache(
        model, partitions['audit'], 128, 53103, root / 'audit', excluded, True)
    audits = {}
    for mode in ['native', 'chronological']:
        adapters, handles = tail.attach(model, 'attention')
        adapters.load_state_dict(torch.load(arms[mode]['selected_path'],
                                           map_location='cpu')['adapter'], strict=True)
        audits[mode] = tail.evaluate(model, caches['audit'])
        for handle in handles:
            handle.remove()
    assert before == digest(model.state_dict())
    assert all(parameter.grad is None for parameter in model.parameters())
    effects, advance = effects_from(audits)
    sources = [Path(__file__), Path(tail.__file__),
               Path(__file__).with_name('diagnose_template_chronology.py'),
               Path('lib/train/data_specs/lasher_train.txt'),
               Path('experiments/cadtrack/cadtrack.yaml')]
    sources += sorted(Path('miss_rgbt').rglob('*.py'))
    sources += sorted(Path('lib/models').rglob('*.py'))
    report = dict(status='complete', pid=os.getpid(), checkpoint=CHECKPOINT,
        checkpoint_sha256=CHECKPOINT_SHA, partitions=partitions, cache_manifests=manifests,
        batch_sizing=sizing, arms=arms, audit=audits, effects=effects, advance=advance,
        baseline_unchanged=True, baseline_gradients_absent=True, baseline_state_sha256=before,
        source_sha256={str(path): sha256(path) for path in sources}, seconds=time.time() - started,
        scope='Train882 matched cached-tail temporal mask training; not official or closed-loop metrics.')
    tail.save(root / 'report.json', report)
    tail.save(root / 'completed.json', dict(status='complete',
        report_sha256=sha256(root / 'report.json'), advance=advance, seconds=report['seconds']))
    print(json.dumps(dict(stage='complete', effects=effects, advance=advance,
                          seconds=report['seconds'])), flush=True)


if __name__ == '__main__':
    main()
