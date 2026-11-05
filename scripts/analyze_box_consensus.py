"""Reconstruct fixed box averaging and saved native maps independently."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np
from analyze_template_attention import exclusions as prior_exclusions, iou
from analyze_cross_stream_context import sha256

CELLS = [a+'_'+b for a in ['11', '01', '10'] for b in ['11', '01', '10']]


def clustered(rows, key):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row['identity']['sequence']].append(row[key])
    values = np.asarray([np.mean(grouped[name]) for name in sorted(grouped)])
    rng = np.random.default_rng(61301)
    draws = rng.integers(0, len(values), (10000, len(values)))
    low, high = np.quantile(values[draws].mean(1), [.025, .975])
    return dict(mean=float(values.mean()), ci95_low=float(low), ci95_high=float(high),
                sequences=len(values), rows=len(rows))


def server_checks(report, root, replay):
    import cv2
    import torch
    from lib.config.cadtrack.config import cfg, update_config_from_file
    from lib.train.dataset import LasHeR
    from miss_rgbt.model import load_pretrained
    import diagnose_box_consensus as worker

    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    update_config_from_file('experiments/cadtrack/cadtrack.yaml')
    for path, expected in report['source_sha256'].items():
        assert sha256(path) == expected, path
    assert sha256(report['checkpoint']) == report['checkpoint_sha256']
    for record in report['caches'].values():
        assert sha256(record['path']) == record['sha256']
    maps = torch.load(report['caches']['maps']['path'], map_location='cpu')
    images = torch.load(report['caches']['first_images']['path'], map_location='cpu')
    assert maps['target'].shape == (1152, 4) and len(images['search']) == 16
    names = Path('lib/train/data_specs/lasher_train.txt').read_text().splitlines()
    assert report['fitting_names'] == [name for index, name in enumerate(names) if index % 5]
    dataset = LasHeR(root=worker.history.training.burst.DATASET, split='train')
    for position, item in enumerate(report['input_manifest']):
        identity = item['identity']
        info = dataset.get_sequence_info(dataset.sequence_list.index(identity['sequence']))
        assert info['visible'][identity['template_frames']+identity['search_frames']].all()
        if position < 16:
            assert worker.common.digest({key: value[position] for key, value in images.items()}
                                        ) == item['tensors_sha256']
    side = 16
    for index, row in enumerate(report['rows']):
        raw = maps['score_map'][index].numpy().reshape(-1)
        weighted_score = (maps['score_map'][index]*maps['window']).numpy().reshape(-1)
        selected = np.argsort(-weighted_score, kind='mergesort')[:8]
        assert selected.tolist() == [candidate['index'] for candidate in row['candidates']]
        size = maps['size_map'][index].numpy().reshape(2, -1)
        offset = maps['offset_map'][index].numpy().reshape(2, -1)
        target = maps['target'][index].numpy()
        assert target.tolist() == row['target']
        for location, candidate in zip(selected, row['candidates']):
            decoded = [(location % side+float(offset[0, location]))/side,
                       (location//side+float(offset[1, location]))/side,
                       float(size[0, location]), float(size[1, location])]
            np.testing.assert_allclose(decoded, candidate['box'], rtol=0, atol=6e-8)
            assert float(raw[location]) == candidate['raw_score']
            assert float(weighted_score[location]) == candidate['hann_score']
    replay_rows = 0
    if replay:
        model = load_pretrained(cfg, report['checkpoint'], repair=False).cuda().eval().requires_grad_(False)
        before = worker.common.digest(model.state_dict())
        templates = [images['templates'][:, index].cuda() for index in range(2)]
        search = images['search'].cuda()
        with torch.no_grad():
            for cell_index, cell in enumerate(CELLS):
                dynamic, current = cell.split('_')
                flags = {'11': [1., 1.], '01': [0., 1.], '10': [1., 0.]}
                output = model(templates, [search], [search.new_ones(16, 2),
                    search.new_tensor(flags[dynamic]).expand(16, -1),
                    search.new_tensor(flags[current]).expand(16, -1)])[0]
                for key in ['score_map', 'size_map', 'offset_map']:
                    assert torch.equal(output[key].cpu(), maps[key][cell_index*16:(cell_index+1)*16])
                replay_rows += 16
        assert before == worker.common.digest(model.state_dict()) == report['baseline_state_sha256']
        assert all(parameter.grad is None for parameter in model.parameters())
    return dict(all_saved_maps_checked=True, physical_visibility_checked=True,
        source_checkpoint_cache_hashes_checked=True, normalized_image_replay_rows=replay_rows,
        limitation='Shared native network; no independent raw-image decode/full-input replay.')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--server-files', action='store_true')
    parser.add_argument('--replay-first-batch', action='store_true')
    args = parser.parse_args()
    root = Path(args.root)
    report = json.loads((root/'report.json').read_text())
    completed = json.loads((root/'completed.json').read_text())
    launch = json.loads(Path(str(root)+'.launch.json').read_text())
    assert report['status'] == completed['status'] == 'complete'
    assert completed['report_sha256'] == sha256(root/'report.json')
    assert launch['protocol_sha256'] == sha256(Path(str(root)+'.preregistered_protocol.md'))
    assert launch['script_sha256'] == sha256(root.parent.parent/'scripts/diagnose_box_consensus.py')
    assert launch['pid'] == report['pid']
    used, hashes = prior_exclusions(root.parent)
    previous = root.parent/'server28_template_attention_20261102/report.json'
    hashes[previous.parent.name+'/report.json'] = sha256(previous)
    for item in json.loads(previous.read_text())['input_manifest']:
        identity = item['identity']
        used[identity['sequence']].update(identity['template_frames']+identity['search_frames'])
    for key, value in report['prior_reports_sha256'].items():
        assert hashes[key] == value
    assert len(report['input_manifest']) == 128 and len(report['rows']) == 1152
    identities = []
    for item in report['input_manifest']:
        identity = item['identity']
        assert identity['sequence'] in report['fitting_names']
        assert not used[identity['sequence']].intersection(
            identity['template_frames']+identity['search_frames'])
        identities.append(identity)
    for position, row in enumerate(report['rows']):
        batch, within = divmod(position, 144)
        cell_index, item_index = divmod(within, 16)
        assert row['cell'] == CELLS[cell_index]
        assert row['identity'] == identities[batch*16+item_index]
        choices = row['candidates']
        assert len(choices) == 8 and len({choice['index'] for choice in choices}) == 8
        boxes = np.asarray([choice['box'] for choice in choices], dtype=np.float64)
        prior = np.asarray([choice['hann_score'] for choice in choices])
        assert np.isfinite(boxes).all() and (boxes[:, 2:] > 0).all() and (prior > 0).all()
        assert (np.diff(prior) <= 0).all()
        for choice in choices:
            assert abs(iou(choice['box'], row['target'])-choice['iou']) < 2e-6
        expected = dict(weighted_box=np.sum(boxes*(prior/prior.sum())[:, None], axis=0),
                        uniform_box=boxes.mean(0))
        for key, value in expected.items():
            np.testing.assert_allclose(row[key], value, rtol=0, atol=2e-7)
        for label in ['weighted', 'uniform']:
            actual_iou = iou(row[label+'_box'], row['target'])
            assert abs(actual_iou-row[label+'_iou']) < 2e-6
            assert row[label+'_gain'] == row[label+'_iou']-row['native_iou']
        assert row['native_iou'] == choices[0]['iou']
        assert row['weighted_minus_uniform'] == row['weighted_iou']-row['uniform_iou']
    groups = dict(complete=[r for r in report['rows'] if r['cell'] == '11_11'],
        recovery01=[r for r in report['rows'] if r['cell'] == '01_11'],
        recovery10=[r for r in report['rows'] if r['cell'] == '10_11'],
        rgb_missing=[r for r in report['rows'] if r['cell'].endswith('_01')],
        tir_missing=[r for r in report['rows'] if r['cell'].endswith('_10')],
        missing=[r for r in report['rows'] if not r['cell'].endswith('_11')])
    effects = {name: {key: clustered(values, key)
        for key in ['weighted_gain', 'uniform_gain', 'weighted_minus_uniform']}
        for name, values in groups.items()}
    assert effects == report['effects']
    advance = (all(effects[name]['weighted_gain']['mean'] >= .01
        and effects[name]['weighted_gain']['ci95_low'] > 0
        and effects[name]['weighted_gain']['sequences'] >= 64
        for name in ['rgb_missing', 'tir_missing'])
        and effects['missing']['weighted_minus_uniform']['ci95_low'] > 0
        and effects['missing']['weighted_minus_uniform']['sequences'] >= 64
        and all(effects[name]['weighted_gain']['mean'] >= 0
            and effects[name]['weighted_gain']['ci95_low'] > -.005
            and effects[name]['weighted_gain']['sequences'] >= 64
            for name in ['complete', 'recovery01', 'recovery10']))
    assert bool(advance) == report['advance'] == completed['advance']
    result = dict(status='verified', report_sha256=completed['report_sha256'],
        analyzer_sha256=sha256(__file__), effects=effects, advance=bool(advance),
        independently_hashed_prior_reports=hashes,
        server_checks=server_checks(report, root, args.replay_first_batch) if args.server_files else None,
        limitation='Train882 crop diagnostic; not official metrics, closed-loop gains or a paper module.')
    with Path(args.output).open('x', newline='\n') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
