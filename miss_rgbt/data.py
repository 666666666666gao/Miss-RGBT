import random
from types import SimpleNamespace

from lib.train.dataset import LasHeR
from lib.train.data import sampler, processing, LTRLoader
from lib.train.data import transforms as tfm
from lib.utils import TensorDict


class VisibleErrorSampler(sampler.TrackingSampler):
    """Native causal sampling and crop rejection, without swallowing decode errors."""

    def getitem(self):
        while True:
            dataset = self.datasets[0]
            seq_id, visible, info = self.sample_seq_from_dataset(dataset, True)
            search_ids = None
            gap = 0
            while search_ids is None:
                anchor = self._sample_visible_ids(visible, num_ids=1,
                    min_id=self.num_template_frames - 1, max_id=len(visible) - self.num_search_frames)
                previous = self._sample_visible_ids(visible, num_ids=self.num_template_frames - 1,
                    min_id=anchor[0] - self.max_gap - gap, max_id=anchor[0])
                if previous is None:
                    gap += 5
                    continue
                template_ids = anchor + previous
                search_ids = self._sample_visible_ids(visible, min_id=anchor[0] + 1,
                    max_id=anchor[0] + self.max_gap + gap, num_ids=self.num_search_frames)
                gap += 5
            templates, anno_z, _ = dataset.get_frames(seq_id, template_ids, info)
            searches, anno_x, meta = dataset.get_frames(seq_id, search_ids, info)
            data = self.processing(TensorDict(template_images=templates, template_anno=anno_z['bbox'],
                search_images=searches, search_anno=anno_x['bbox'], dataset=dataset.get_name(),
                test_class=meta.get('object_class_name')))
            if data['valid']:
                return data


def make_loader(cfg, root, split, batch, samples, workers):
    training = split == 'train'
    joint = tfm.Transform(tfm.ToGrayscale(probability=0.), tfm.RandomHorizontalFlip(probability=.5 if training else 0.))
    transforms = [tfm.ToTensorAndJitter(.2), tfm.RandomHorizontalFlip_Norm(.5)] if training else [tfm.ToTensor()]
    transforms.append(tfm.Normalize(mean=cfg.DATA.MEAN, std=cfg.DATA.STD))
    processor = processing.ViPTProcessing(
        search_area_factor=dict(template=cfg.DATA.TEMPLATE.FACTOR, search=cfg.DATA.SEARCH.FACTOR),
        output_sz=dict(template=cfg.DATA.TEMPLATE.SIZE, search=cfg.DATA.SEARCH.SIZE),
        center_jitter_factor=dict(template=cfg.DATA.TEMPLATE.CENTER_JITTER, search=cfg.DATA.SEARCH.CENTER_JITTER),
        scale_jitter_factor=dict(template=cfg.DATA.TEMPLATE.SCALE_JITTER, search=cfg.DATA.SEARCH.SCALE_JITTER),
        mode='sequence', transform=tfm.Transform(*transforms), joint_transform=joint,
        settings=SimpleNamespace())
    dataset = LasHeR(root=root, split=split, dtype='rgbrgb')
    sampled = VisibleErrorSampler([dataset], [1], samples_per_epoch=samples,
        max_gap=cfg.DATA.MAX_SAMPLE_INTERVAL, num_search_frames=cfg.DATA.SEARCH.NUMBER,
        num_template_frames=cfg.DATA.TEMPLATE.NUMBER, processing=processor, frame_sample_mode='causal')
    return LTRLoader(split, sampled, training=training, batch_size=batch, shuffle=training,
                     num_workers=workers, drop_last=True, stack_dim=1, worker_init_fn=seed_worker)


def seed_worker(worker_id):
    import numpy as np
    import torch
    seed = torch.initial_seed() % (2 ** 32)
    random.seed(seed)
    np.random.seed(seed)
