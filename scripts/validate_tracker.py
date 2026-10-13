import json
from pathlib import Path

import numpy as np
import torch

from lib.config.cadtrack.config import cfg, update_config_from_file
from lib.test.tracker.cadtrack import CADTrack
from lib.test.utils import TrackerParams
from lib.train.dataset.depth_utils import get_x_frame
from miss_rgbt.tracker import MissingTracker


torch.set_num_threads(1)
update_config_from_file('experiments/cadtrack/cadtrack.yaml')
checkpoint = 'checkpoints/e4_speed_smoke_seed2027/smoke.pth.tar'
own = MissingTracker(cfg, checkpoint)
params = TrackerParams()
params.cfg = cfg
params.checkpoint = 'checkpoints/pretrained/CADTrack_ep0020.pth.tar'
params.template_factor = cfg.TEST.TEMPLATE_FACTOR
params.template_size = cfg.TEST.TEMPLATE_SIZE
params.search_factor = cfg.TEST.SEARCH_FACTOR
params.search_size = cfg.TEST.SEARCH_SIZE
params.save_all_boxes = False
params.debug = 0
native = CADTrack(params)
saved = torch.load(checkpoint, map_location='cpu')['model']
native.network.load_state_dict({key[len('baseline.'):]: value for key, value in saved.items()
                               if key.startswith('baseline.')}, strict=True)
sequence = Path('lib/train/data_specs/lasher_train.txt').read_text().splitlines()[0]
folder = Path('/data2/wangwj/dataset/LasHeR-0428/trainingset') / sequence
rgb = sorted((folder / 'visible').glob('*.jpg'))
tir = sorted((folder / 'infrared').glob('*.jpg'))
gt = np.loadtxt(str(folder / 'init.txt'), delimiter=',')
first = get_x_frame(str(rgb[0]), str(tir[0]), dtype='rgbrgb')
own.initialize(first, dict(init_bbox=gt[0].tolist()))
native.initialize(first, dict(init_bbox=gt[0].tolist()))
max_difference = 0.
for i in range(1, 11):
    frame = get_x_frame(str(rgb[i]), str(tir[i]), dtype='rgbrgb')
    a = own.track(frame, [1, 1])['target_bbox']
    b = native.track(frame)['target_bbox']
    max_difference = max(max_difference, float(np.abs(np.asarray(a) - b).max()))
assert max_difference == 0., max_difference
for state in [[0, 1], [1, 0]]:
    r = rgb[11] if state[0] else tir[11]
    t = tir[11] if state[1] else rgb[11]
    out = own.track(get_x_frame(str(r), str(t), dtype='rgbrgb'), state)
    assert np.isfinite(out['target_bbox']).all()
assert len(own.templates) == len(own.availability) == cfg.DATA.TEMPLATE.NUMBER
report = dict(scope='native tracker equivalence on 10 complete tracking frames + 2 missing inference checks; not benchmark',
              training_sequence=sequence, checkpoint=checkpoint, complete_bbox_max_difference=max_difference,
              missing_states_finite=True, single_view_inference=True, template_availability_length=len(own.availability))
Path('reports/tracker_native_validation.json').write_text(json.dumps(report, indent=2))
print('TRACKER_NATIVE_PASS ' + json.dumps(report), flush=True)
