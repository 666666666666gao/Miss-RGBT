"""Extend the isolated center-offset control with past-frame size calibration."""
import numpy as np

from lib.utils.box_ops import clip_box
from state_offset_tracker28 import StateOffsetTracker


class SpatialCalibrationTracker(StateOffsetTracker):
    def __init__(self, cfg, checkpoint, apply_offset, apply_scale):
        super().__init__(cfg, checkpoint, apply_offset)
        self.apply_scale = apply_scale

    def initialize(self, image, info):
        super().initialize(image, info)
        self.scales = {(0, 1): np.ones(2), (1, 0): np.ones(2)}
        self.calibration_sizes = {}
        self.calibrating = False

    def predict(self, image, available):
        box, confidence, observed = super().predict(image, available)
        flags = tuple(available)
        if flags == (1, 1):
            self.calibration_sizes = {
                "11": clip_box(box, image.shape[0], image.shape[1], margin=10)[2:]}
            self.primary_raw_box = list(box)
        elif self.calibrating:
            size = clip_box(box, image.shape[0], image.shape[1], margin=10)[2:]
            self.calibration_sizes["".join(map(str, flags))] = list(size)
            self.scales[flags] = np.asarray(self.calibration_sizes["11"]) / np.asarray(size)
            assert np.isfinite(self.scales[flags]).all() and (self.scales[flags] > 0).all()
        else:
            self.primary_raw_box = list(box)
            if self.apply_scale:
                sx, sy = self.scales[flags]
                w, h = box[2] * float(sx), box[3] * float(sy)
                box = [box[0] + .5 * box[2] - .5 * w,
                       box[1] + .5 * box[3] - .5 * h, w, h]
        return box, confidence, observed

    def track(self, image, available, calibration_images):
        self.calibrating = available == [1, 1]
        result = super().track(image, available, calibration_images)
        applied = (self.scales[tuple(available)] if self.apply_scale and not self.calibrating
                   else np.ones(2))
        result.update(
            before_scale_bbox=list(self.primary_raw_box),
            applied_scale=applied.tolist(),
            calibration_scales={"".join(map(str, flags)): value.tolist()
                                for flags, value in self.scales.items()} if self.calibrating else None,
            calibration_sizes=self.calibration_sizes if self.calibrating else None)
        return result
