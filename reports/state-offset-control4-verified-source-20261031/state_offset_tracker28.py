"""Isolated E0 tracker with past fully-observed-frame offset calibration."""
import numpy as np
import torch

from lib.train.data.processing_utils import sample_target
from lib.utils.box_ops import clip_box
from miss_rgbt.tracker import MissingTracker


class StateOffsetTracker(MissingTracker):
    def __init__(self, cfg, checkpoint, apply_offset):
        super().__init__(cfg, checkpoint)
        assert self.network.repair is None and self.network.utility_repair is None
        self.apply_offset = apply_offset

    def initialize(self, image, info):
        super().initialize(image, info)
        self.offsets = {(0, 1): np.zeros(2), (1, 0): np.zeros(2)}
        self.last_calibration_frame = None

    def predict(self, image, available):
        crop, factor, _ = sample_target(image, self.state, self.cfg.TEST.SEARCH_FACTOR,
                                       output_sz=self.cfg.TEST.SEARCH_SIZE)
        search = self.preprocessor.process(crop)
        observed = torch.tensor([available], device=search.device, dtype=torch.float32)
        out = self.network(self.templates, [search], self.availability + [observed])[0]
        response = self.window * out["score_map"]
        predicted = self.network.baseline.box_head.cal_bbox(
            response, out["size_map"], out["offset_map"])
        cx, cy, w, h = (predicted.view(-1, 4).mean(0) *
                        self.cfg.TEST.SEARCH_SIZE / factor).tolist()
        cx_prev = self.state[0] + .5 * self.state[2]
        cy_prev = self.state[1] + .5 * self.state[3]
        half = .5 * self.cfg.TEST.SEARCH_SIZE / factor
        box = [cx + cx_prev - half - .5 * w, cy + cy_prev - half - .5 * h, w, h]
        return box, float(response.max()), observed

    def track(self, image, available, calibration_images):
        self.frame_id += 1
        state = tuple(available)
        assert state in [(1, 1), (0, 1), (1, 0)]
        used_offset = np.zeros(2)
        with torch.no_grad():
            box, confidence, observed = self.predict(image, available)
            raw_unclipped = list(box)
            raw_clipped = clip_box(box, image.shape[0], image.shape[1], margin=10)
            if state == (1, 1):
                assert set(calibration_images) == {(0, 1), (1, 0)}
                complete_center = np.asarray(raw_clipped[:2]) + np.asarray(raw_clipped[2:]) / 2
                for flags, counterfactual_image in calibration_images.items():
                    alternative, _, _ = self.predict(counterfactual_image, list(flags))
                    alternative = clip_box(alternative, counterfactual_image.shape[0],
                                           counterfactual_image.shape[1], margin=10)
                    self.offsets[flags] = (np.asarray(alternative[:2]) +
                                           np.asarray(alternative[2:]) / 2 - complete_center)
                self.last_calibration_frame = self.frame_id
            else:
                assert not calibration_images
                if self.apply_offset:
                    used_offset = self.offsets[state]
                    box[0] -= float(used_offset[0])
                    box[1] -= float(used_offset[1])
            self.state = clip_box(box, image.shape[0], image.shape[1], margin=10)
        if (self.num_template > 1 and self.frame_id % self.cfg.TEST.UPDATE_INTERVALS == 0
                and confidence > self.cfg.TEST.UPDATE_THRESHOLD):
            crop, _, _ = sample_target(image, self.state, self.cfg.TEST.TEMPLATE_FACTOR,
                                       output_sz=self.cfg.TEST.TEMPLATE_SIZE)
            self.templates.append(self.preprocessor.process(crop))
            self.availability.append(observed)
            self.templates.pop(1)
            self.availability.pop(1)
        return dict(target_bbox=self.state, best_score=confidence,
                    raw_target_bbox=raw_clipped, applied_offset=used_offset.tolist(),
                    unclipped_target_bbox=raw_unclipped,
                    calibration_offsets={
                        "".join(map(str, flags)): value.tolist()
                        for flags, value in self.offsets.items()} if state == (1, 1) else None,
                    last_calibration_call=self.last_calibration_frame,
                    model_calls=3 if state == (1, 1) else 1)
