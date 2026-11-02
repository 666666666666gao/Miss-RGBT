"""Independent saved evidence and NumPy attention/candidate reconstruction."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np
from analyze_cross_stream_context import sha256
from analyze_last_observed_template import exclusions as previous_exclusions

CELLS = [dynamic + '_' + search for dynamic in ['11', '01', '10'] for search in ['11', '01', '10']]


def exclusions(folder):
    used, hashes = previous_exclusions(folder)
    path = folder / 'server28_last_observed_template_20261102/selected_clips.json'
    hashes[path.parent.name + '/selected_clips.json'] = sha256(path)
    for clip in json.loads(path.read_text()):
        used[clip['sequence']].update(clip['frame_ids'])
    return used, hashes


def bootstrap(values):
    values = np.asarray(values)
    rng = np.random.default_rng(60301)
    draws = rng.integers(0, len(values), (10000, len(values)))
    low, high = np.quantile(values[draws].mean(1), [.025, .975])
    return dict(mean=float(values.mean()), ci95_low=float(low), ci95_high=float(high))


def iou(box, target):
    box = np.asarray(box, dtype=np.float64)
    first = np.r_[box[:2] - box[2:] / 2, box[:2] + box[2:] / 2]
    target = np.asarray(target, dtype=np.float64)
    second = np.clip(np.r_[target[:2], target[:2] + target[2:]], 0, 1)
    area = np.maximum(0, np.minimum(first[2:], second[2:]) - np.maximum(first[:2], second[:2])).prod()
    return float(area / (box[2:].prod() + (second[2:] - second[:2]).prod() - area))


def reconstruct_attention(qkv, weights):
    values = qkv.double().numpy()[0].reshape(385, 3, 12, 64).transpose(1, 2, 0, 3)
    score = (values[0, :, 1:65] @ values[1].transpose(0, 2, 1)) / 8
    probability = np.exp(score - score.max(-1, keepdims=True))
    probability /= probability.sum(-1, keepdims=True)
    weighted = np.einsum('hqk,wq->whk', probability[..., 129:], weights)
    weighted /= weights.sum(-1)[:, None, None]
    return weighted.mean(1)


def server_checks(report, root, replay):
    import cv2
    import torch
    from lib.config.cadtrack.config import cfg, update_config_from_file
    from lib.train.dataset import LasHeR
    from miss_rgbt.model import load_pretrained
    import diagnose_template_attention as worker

    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    update_config_from_file('experiments/cadtrack/cadtrack.yaml')
    for path, value in report['source_sha256'].items():
        assert sha256(path) == value, path
    assert sha256(report['checkpoint']) == report['checkpoint_sha256']
    for cache in report['caches'].values():
        assert sha256(cache['path']) == cache['sha256']
    images = torch.load(report['caches']['images']['path'], map_location='cpu')
    maps = torch.load(report['caches']['maps']['path'], map_location='cpu')
    assert len(images['search']) == 128 and len(maps['attention']) == 1152
    names = Path('lib/train/data_specs/lasher_train.txt').read_text().splitlines()
    assert report['fitting_names'] == [name for index, name in enumerate(names) if index % 5]
    dataset = LasHeR(root=worker.history.training.burst.DATASET, split='train')
    for index, item in enumerate(report['input_manifest']):
        tensors = {key: value[index] for key, value in images.items()}
        assert worker.common.digest(tensors) == item['tensors_sha256']
        assert all(torch.isfinite(value).all() for value in tensors.values())
        identity = item['identity']
        info = dataset.get_sequence_info(dataset.sequence_list.index(identity['sequence']))
        assert info['visible'][identity['template_frames'] + identity['search_frames']].all()
    max_error = 0.
    for cell_index, cell in enumerate(CELLS):
        flags = {'11': [1, 1], '01': [0, 1], '10': [1, 0]}[cell[-2:]]
        weights = maps['weights'][cell_index * 16].double().numpy()
        pieces = [reconstruct_attention(value, weights) for value in maps['raw_qkv'][cell]]
        attention = sum(piece * flag for piece, flag in zip(pieces, flags)) / sum(flags)
        error = float(np.abs(attention - maps['attention'][cell_index * 16].numpy()).max())
        max_error = max(max_error, error)
        assert error < 2e-6
    for index, row in enumerate(report['rows']):
        raw = maps['score_map'][index].numpy().reshape(-1)
        prior = (maps['score_map'][index].numpy().squeeze() * maps['window'].numpy().squeeze()).reshape(-1)
        chosen = np.argsort(-prior, kind='mergesort')[:8]
        assert chosen.tolist() == [candidate['index'] for candidate in row['candidates']]
        assert chosen.tolist() == maps['indices'][index].tolist()
        offset = maps['offset_map'][index].numpy().reshape(2, 256)
        size = maps['size_map'][index].numpy().reshape(2, 256)
        locations = np.arange(256)
        geometry = np.stack([(locations % 16 + offset[0]) / 16,
                             (locations // 16 + offset[1]) / 16, size[0], size[1]], -1)
        for slot, candidate in enumerate(row['candidates']):
            position = chosen[slot]
            assert np.max(np.abs(geometry[position] - candidate['box'])) < 2e-6
            assert abs(float(raw[position]) - candidate['raw_score']) < 1e-7
            assert abs(float(prior[position]) - candidate['hann_score']) < 1e-7
        assert np.array_equal(maps['target'][index].numpy(), np.asarray(row['target'], dtype=np.float32))
        assert maps['attention'][index, 0, chosen].tolist() == row['foreground']
        assert maps['attention'][index, 1, chosen].tolist() == row['background']
        template_index = (index // 144) * 16 + index % 16
        label = images['template_box'][template_index].numpy()
        axes = np.arange(8) / 8
        yy, xx = np.meshgrid(axes, axes, indexing='ij')
        low = np.stack([xx, yy], -1).reshape(64, 2)
        coverage = np.maximum(0, np.minimum(low + .125, label[:2] + label[2:])
                              - np.maximum(low, label[:2])).prod(-1) * 64
        assert np.max(np.abs(maps['weights'][index].numpy() - np.stack([coverage, 1 - coverage]))) < 2e-6
    replayed = 0
    if replay:
        model = load_pretrained(cfg, report['checkpoint'], repair=False).cuda().eval().requires_grad_(False)
        before = worker.common.digest(model.state_dict())
        assert before == report['baseline_state_sha256']
        templates = [images['templates'][:16, index].cuda() for index in range(2)]
        search = images['search'][:16].cuda()
        with torch.no_grad():
            for cell_index, cell in enumerate(CELLS):
                states = {'11': [1., 1.], '01': [0., 1.], '10': [1., 0.]}
                availability = [search.new_ones(16, 2),
                    search.new_tensor(states[cell[:2]]).expand(16, -1),
                    search.new_tensor(states[cell[-2:]]).expand(16, -1)]
                output = model(templates, [search], availability)[0]
                for key in ['score_map', 'size_map', 'offset_map']:
                    assert torch.equal(output[key].cpu(), maps[key][cell_index * 16:(cell_index + 1) * 16])
                replayed += 16
        assert worker.common.digest(model.state_dict()) == before
        assert all(value.grad is None for value in model.parameters())
    return dict(native_candidate_maps_verified=1152, numpy_qkv_cells=9,
        numpy_attention_max_error=max_error, normalized_image_rows_replayed=replayed,
        actual_visibility_verified=True, raw_image_decode_replayed=False,
        shared_native_network=True, full_image_replay=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('directory')
    parser.add_argument('--server-files', action='store_true')
    parser.add_argument('--replay-first-batch', action='store_true')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    root = Path(args.directory)
    report = json.loads((root / 'report.json').read_text())
    completed = json.loads((root / 'completed.json').read_text())
    launch = json.loads((root.parent / (root.name + '.launch.json')).read_text())
    assert report['status'] == completed['status'] == 'complete'
    assert report['pid'] == launch['pid'] and report['seed'] == 60101
    assert sha256(root / 'report.json') == completed['report_sha256']
    assert sha256(root.parent / (root.name + '.preregistered_protocol.md')) == launch['protocol_sha256']
    assert report['source_sha256'][launch['command'][2]] == launch['script_sha256']
    assert report['controls'] == dict(hook_parity_calls=9, frozen_state=True, gradients_absent=True)
    used, hashes = exclusions(root.parent)
    assert all(hashes[name] == value for name, value in report['prior_reports_sha256'].items())
    manifests = report['input_manifest']
    assert len(manifests) == 128 and len(report['rows']) == 1152
    assert len(report['fitting_names']) == len(set(report['fitting_names'])) == 705
    keys = set()
    for item in manifests:
        identity = item['identity']
        key = identity['sequence'], tuple(identity['template_frames']), tuple(identity['search_frames'])
        assert key not in keys and identity['sequence'] in report['fitting_names']
        keys.add(key)
        assert not used[identity['sequence']].intersection(identity['template_frames'] + identity['search_frames'])
    for index, row in enumerate(report['rows']):
        expected_identity = manifests[(index // 144) * 16 + index % 16]['identity']
        assert row['identity'] == expected_identity and row['cell'] == CELLS[(index % 144) // 16]
        choices = row['candidates']
        assert len(choices) == len({choice['index'] for choice in choices}) == 8
        assert np.isfinite(row['foreground'] + row['background'] + row['target']).all()
        assert all(value > 0 for value in row['foreground'] + row['background'])
        assert choices == sorted(choices, key=lambda choice: (-choice['hann_score'], choice['index']))
        for choice in choices:
            assert abs(iou(choice['box'], row['target']) - choice['iou']) < 2e-6
        scores = np.asarray([choice['iou'] for choice in choices])
        counts, correct_fg, correct_bg = 0, 0., 0.
        for first in range(8):
            for second in range(first + 1, 8):
                if abs(scores[first] - scores[second]) >= .05:
                    counts += 1
                    for label in ['foreground', 'background']:
                        sign = (row[label][first] - row[label][second]) * (scores[first] - scores[second])
                        value = 1. if sign > 0 else .5 if sign == 0 else 0.
                        if label == 'foreground':
                            correct_fg += value
                        else:
                            correct_bg += value
        assert [counts, correct_fg, correct_bg] == [
            row['pair_count'], row['foreground_correct'], row['background_correct']]
        prior = np.asarray([choice['hann_score'] for choice in choices])
        for label in ['foreground', 'background']:
            choice = int((prior * row[label]).argmax())
            assert row[label + '_choice'] == choice
            assert row[label + '_gain'] == float(scores[choice] - scores[0])
        assert row['native_iou'] == float(scores[0])
    effects = {}
    for category, suffix in [('complete', '11_11'), ('rgb_missing', '_01'), ('tir_missing', '_10')]:
        rows = [row for row in report['rows'] if (
            row['cell'] == suffix if category == 'complete' else row['cell'].endswith(suffix))]
        totals = defaultdict(lambda: np.zeros(3))
        gains = defaultdict(lambda: [[], []])
        for row in rows:
            name = row['identity']['sequence']
            totals[name] += [row['pair_count'], row['foreground_correct'], row['background_correct']]
            gains[name][0].append(row['foreground_gain'])
            gains[name][1].append(row['background_gain'])
        values = np.asarray([totals[name] for name in sorted(totals) if totals[name][0] > 0])
        foreground, background = values[:, 1] / values[:, 0], values[:, 2] / values[:, 0]
        result = dict(foreground=bootstrap(foreground), background=bootstrap(background),
            difference=bootstrap(foreground - background),
            sequences=len(values), pairs=int(values[:, 0].sum()), rows=len(rows))
        for label, position in [('foreground_gain', 0), ('background_gain', 1)]:
            result[label] = dict(bootstrap([np.mean(gains[name][position]) for name in sorted(gains)]),
                sequences=len(gains), rows=len(rows))
        effects[category] = result
    assert effects == report['effects']
    advance = all(value['sequences'] >= 64 and value['pairs'] >= 100
        and value['foreground']['ci95_low'] > .5 and value['difference']['mean'] >= .05
        and value['difference']['ci95_low'] > 0 for value in effects.values())
    assert advance == report['advance'] == completed['advance']
    result = dict(status='verified', report_sha256=completed['report_sha256'],
        analyzer_sha256=sha256(__file__), effects=effects, advance=bool(advance),
        independently_hashed_prior_reports=hashes,
        server_checks=server_checks(report, root, args.replay_first_batch) if args.server_files else None,
        limitation='Contextual attention information diagnostic, not independent identity evidence, '
                   'closed-loop gains, official metrics or a validated module.')
    with Path(args.output).open('x', newline='\n') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
