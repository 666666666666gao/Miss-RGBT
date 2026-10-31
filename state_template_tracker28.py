"""Ordinary control: read the most recent template for the current availability."""
import torch

from lib.train.data.processing_utils import sample_target
from lib.utils.box_ops import clip_box
from miss_rgbt.tracker import MissingTracker


class StateTemplateTracker(MissingTracker):
    def initialize(self, image, info):
        super().initialize(image, info)
        self.reset_bank()

    def reset_bank(self):
        assert self.num_template == 2
        self.bank = {key: (self.templates[-1], self.availability[-1], 0)
                     for key in [(1, 1), (0, 1), (1, 0)]}
        self.latest = self.bank[(1, 1)]

    def track(self, image, available):
        self.frame_id += 1
        key = tuple(available)
        selected = self.bank[key] if self.match_state else self.latest
        self.templates[-1], self.availability[-1] = selected[:2]
        crop, factor, _ = sample_target(image, self.state, self.cfg.TEST.SEARCH_FACTOR,
                                       output_sz=self.cfg.TEST.SEARCH_SIZE)
        search = self.preprocessor.process(crop)
        observed = torch.tensor([available], device=search.device, dtype=torch.float32)
        with torch.no_grad():
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
            self.state = clip_box(box, image.shape[0], image.shape[1], margin=10)
            confidence = float(response.max())
        written = (self.frame_id % self.cfg.TEST.UPDATE_INTERVALS == 0 and
                   confidence > self.cfg.TEST.UPDATE_THRESHOLD)
        if written:
            crop, _, _ = sample_target(image, self.state, self.cfg.TEST.TEMPLATE_FACTOR,
                                       output_sz=self.cfg.TEST.TEMPLATE_SIZE)
            template = self.preprocessor.process(crop)
            self.latest = (template, observed, self.frame_id)
            self.bank[key] = self.latest
            self.templates[-1], self.availability[-1] = template, observed
        return dict(target_bbox=self.state, best_score=confidence,
                    selected_source_call=selected[2],
                    selected_availability=selected[1].reshape(-1).tolist(),
                    template_written=bool(written),
                    bank_source_calls={"".join(map(str, state)): entry[2]
                                       for state, entry in self.bank.items()})
