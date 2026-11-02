"""F checkpoint loading with inherited native per-frame tracking behavior."""
import torch

from lib.models.cadtrack import build_cadtrack
from lib.test.tracker.data_utils import PreprocessorMM
from lib.test.utils.hann import hann2d
from miss_rgbt.tracker import MissingTracker
from model import FeedbackTrackerModel


class FeedbackTracker(MissingTracker):
    def __init__(self, cfg, checkpoint):
        saved = torch.load(checkpoint, map_location='cpu')
        config = saved['config']
        feedback = config['feedback']
        self.cfg = cfg
        self.network = FeedbackTrackerModel(build_cadtrack(cfg, training=False),
            task_utility=config['task_utility'], feedback_mode=feedback['mode'],
            feedback_width=feedback['width'])
        self.network.load_state_dict(saved['model'], strict=True)
        self.network.cuda().eval()
        self.preprocessor = PreprocessorMM(mean=cfg.DATA.MEAN, std=cfg.DATA.STD)
        self.num_template = cfg.DATA.TEMPLATE.NUMBER
        size = cfg.TEST.SEARCH_SIZE // cfg.MODEL.BACKBONE.STRIDE
        self.window = hann2d(torch.tensor([size, size]).long(), centered=True).cuda()
