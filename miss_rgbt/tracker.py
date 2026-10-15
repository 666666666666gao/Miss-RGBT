import torch

from lib.test.tracker.data_utils import PreprocessorMM
from lib.test.utils.hann import hann2d
from lib.train.data.processing_utils import sample_target
from lib.utils.box_ops import clip_box
from lib.models.cadtrack import build_cadtrack
from miss_rgbt.model import MissingTrackerModel


class MissingTracker:
    """Native crop/window/update logic with explicit template availability."""

    def __init__(self, cfg, checkpoint):
        saved = torch.load(checkpoint, map_location='cpu')
        config = saved['config']
        self.cfg = cfg
        self.network = MissingTrackerModel(build_cadtrack(cfg, training=False), repair=config['repair'],
            width=config['repair_width'], repair_layer=config['repair_layer'],
            task_utility=config.get('task_utility'))
        self.network.load_state_dict(saved['model'], strict=True)
        self.network.cuda().eval()
        self.preprocessor = PreprocessorMM(mean=cfg.DATA.MEAN, std=cfg.DATA.STD)
        self.num_template = cfg.DATA.TEMPLATE.NUMBER
        size = cfg.TEST.SEARCH_SIZE // cfg.MODEL.BACKBONE.STRIDE
        self.window = hann2d(torch.tensor([size, size]).long(), centered=True).cuda()

    def initialize(self, image, info):
        if self.network.utility_repair is not None:
            self.network.utility_repair.reset_selection()
        crop, _, _ = sample_target(image, info['init_bbox'], self.cfg.TEST.TEMPLATE_FACTOR,
                                   output_sz=self.cfg.TEST.TEMPLATE_SIZE)
        template = self.preprocessor.process(crop)
        self.templates = [template] * self.num_template
        self.availability = [torch.ones(1, 2, device='cuda') for _ in self.templates]
        self.state = info['init_bbox']
        self.frame_id = 0

    def track(self, image, available):
        self.frame_id += 1
        crop, factor, _ = sample_target(image, self.state, self.cfg.TEST.SEARCH_FACTOR,
                                       output_sz=self.cfg.TEST.SEARCH_SIZE)
        search = self.preprocessor.process(crop)
        observed = torch.tensor([available], device='cuda', dtype=torch.float32)
        with torch.no_grad():
            out = self.network(self.templates, [search], self.availability + [observed])[0]
            response = self.window * out['score_map']
            predicted = self.network.baseline.box_head.cal_bbox(response, out['size_map'], out['offset_map'])
            cx, cy, w, h = (predicted.view(-1, 4).mean(0) * self.cfg.TEST.SEARCH_SIZE / factor).tolist()
            cx_prev = self.state[0] + .5 * self.state[2]
            cy_prev = self.state[1] + .5 * self.state[3]
            half = .5 * self.cfg.TEST.SEARCH_SIZE / factor
            box = [cx + cx_prev - half - .5 * w, cy + cy_prev - half - .5 * h, w, h]
            self.state = clip_box(box, image.shape[0], image.shape[1], margin=10)
            confidence = float(response.max())
        if self.num_template > 1 and self.frame_id % self.cfg.TEST.UPDATE_INTERVALS == 0 and confidence > self.cfg.TEST.UPDATE_THRESHOLD:
            crop, _, _ = sample_target(image, self.state, self.cfg.TEST.TEMPLATE_FACTOR,
                                       output_sz=self.cfg.TEST.TEMPLATE_SIZE)
            self.templates.append(self.preprocessor.process(crop))
            self.availability.append(observed)
            self.templates.pop(1)
            self.availability.pop(1)
        return dict(target_bbox=self.state, best_score=confidence)
