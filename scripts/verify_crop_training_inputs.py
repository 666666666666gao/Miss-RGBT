"""Reconstruct the first fitting video's paired image inputs from raw frames."""
import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from lib.config.cadtrack.config import cfg, update_config_from_file
from lib.train.dataset import LasHeR
from lib.train.data.processing_utils import sample_target
from miss_rgbt.tracker import MissingTracker
from analyze_cross_stream_context import sha256
from analyze_causal_crop_training import digest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('directory')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    root = Path(args.directory)
    manifest = json.loads((root / 'image_manifest.json').read_text())
    assert sha256(manifest['cache_path']) == manifest['cache_sha256']
    cache = torch.load(manifest['cache_path'], map_location='cpu')
    clip = manifest['clips'][0]
    accepted = {item['identity']['frame']: (index, item) for index, item in
                enumerate(manifest['accepted']) if item['identity']['sequence'] == clip['sequence']}
    rejected = {item['identity']['frame']: item for item in manifest['rejected']
                if item['identity']['sequence'] == clip['sequence']}
    assert set(accepted).isdisjoint(rejected)
    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    update_config_from_file('experiments/cadtrack/cadtrack.yaml')
    tracker = MissingTracker(cfg, '/home/gaob/Miss-RGBT/checkpoints/e0_availability/epoch20.pth.tar')
    tracker.network.eval().requires_grad_(False)
    baseline_digest = digest(tracker.network.state_dict())
    torch.manual_seed(58111)
    torch.cuda.manual_seed_all(58111)
    dataset = LasHeR(root='/data/wangwj/dataset/LasHeR/trainingset', split='train')
    assert dataset.sequence_list[clip['dataset_index']] == clip['sequence']
    info = dataset.get_sequence_info(clip['dataset_index'])
    frames, annotations, _ = dataset.get_frames(clip['dataset_index'], clip['frame_ids'], info)
    boxes = [value.tolist() for value in annotations['bbox']]
    tracker.initialize(frames[0], dict(init_bbox=boxes[0]))
    states = {'11': [1., 1.], '01': [0., 1.], '10': [1., 0.]}
    checked, invalid, label_max_error = 0, 0, 0.
    for position in range(1, len(frames)):
        frame = clip['frame_ids'][position]
        flags = states[clip['mixed_schedule'][position]]
        if position % 8 == 0:
            gt = torch.tensor(boxes[position])
            wh = gt[2:] * torch.exp(torch.randn(2) * .25)
            center = gt[:2] + .5 * gt[2:] + wh.prod().sqrt() * 3. * (torch.rand(2) - .5)
            jitter = torch.cat([center - .5 * wh, wh]).tolist()
            predicted = list(tracker.state)
            images, labels, valid = {}, {}, {}
            for mode, extract in [('native', jitter), ('causal', predicted)]:
                crop, factor, _ = sample_target(frames[position], extract, 4., output_sz=256)
                images[mode] = tracker.preprocessor.process(crop)[0].cpu()
                # Independent expression for normalized search-target coordinates.
                gt_array, crop_array = np.asarray(boxes[position]), np.asarray(extract)
                center_crop = np.array([127.5, 127.5]) + (
                    gt_array[:2] + .5 * gt_array[2:] - crop_array[:2] - .5 * crop_array[2:]) * factor
                wh_crop = gt_array[2:] * factor
                labels[mode] = np.concatenate([center_crop - .5 * wh_crop, wh_crop]) / 256.
                label = labels[mode]
                valid[mode] = bool((label[:2] >= 0).all()
                    and (label[:2] + label[2:] <= 1).all() and (label[2:] > 0).all())
            if all(valid.values()):
                assert frame in accepted and frame not in rejected
                index, item = accepted[frame]
                assert item['gt'] == boxes[position] and item['state'] == clip['mixed_schedule'][position]
                assert item['jittered_crop_box'] == jitter and item['predicted_crop_box'] == predicted
                tensors = {key: value[index] for key, value in cache.items()}
                assert torch.equal(torch.cat(tracker.templates).cpu(), tensors['templates'])
                available = torch.cat(tracker.availability + [
                    torch.tensor([flags], device='cuda')]).cpu()
                assert torch.equal(available, tensors['availability'])
                for mode in ['native', 'causal']:
                    assert torch.equal(images[mode], tensors[mode + '_search'])
                    error = float(np.abs(labels[mode] - tensors[mode + '_target'].numpy()).max())
                    label_max_error = max(label_max_error, error)
                    assert error < 1e-6
                assert digest(tensors) == item['tensors_sha256']
                checked += 1
            else:
                assert frame in rejected and frame not in accepted
                assert rejected[frame]['native_valid'] == valid['native']
                assert rejected[frame]['causal_valid'] == valid['causal']
                invalid += 1
        tracker.track(frames[position], flags)
    assert checked == len(accepted) and invalid == len(rejected) and checked + invalid == 11
    assert digest(tracker.network.state_dict()) == baseline_digest
    assert all(value.grad is None for value in tracker.network.parameters())
    result = dict(status='verified', sequence=clip['sequence'], frame_ids=clip['frame_ids'],
        accepted_image_pairs_bitwise=checked, rejection_decisions_reproduced=invalid,
        prediction_crop_boxes_exact=True, jitter_boxes_exact=True, template_history_exact=True,
        independent_numpy_target_max_error=label_max_error,
        image_manifest_sha256=sha256(root / 'image_manifest.json'),
        image_cache_sha256=manifest['cache_sha256'], verifier_sha256=sha256(__file__),
        raw_frames_sha256=hashlib.sha256(b''.join(
            np.ascontiguousarray(image).tobytes() for image in frames)).hexdigest(),
        model_unchanged=True, gradients_absent=True,
        limitation='First fitting clip only; shared native tracker/crop/preprocessor. '
                   'No optimizer replay, full-cache image reconstruction or official scores.')
    with Path(args.output).open('x', newline='\n') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
