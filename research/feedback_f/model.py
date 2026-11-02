"""Isolated same-view second pass; the active A runtime is unchanged."""
import torch
from torch import nn

from miss_rgbt.model import MissingTrackerModel
from miss_rgbt.objectives import training_loss


class SearchFeedback(nn.Module):
    def __init__(self, dim, width, mode):
        super().__init__()
        assert mode in ['deep', 'shallow']
        self.mode = mode
        self.shallow_norm = nn.LayerNorm(2 * dim)
        self.deep_norm = nn.LayerNorm(2 * dim)
        self.hidden = nn.Sequential(nn.Linear(4 * dim + 7, width), nn.GELU())
        self.delta = nn.Linear(width, 2 * dim)
        nn.init.zeros_(self.delta.weight)
        nn.init.zeros_(self.delta.bias)

    def forward(self, shallow, deep, prediction, available):
        first = torch.cat(shallow, -1)
        last = torch.cat(deep, -1)
        response = torch.cat([prediction[key].flatten(2).transpose(1, 2)
                              for key in ['score_map', 'size_map', 'offset_map']], -1)
        if self.mode == 'shallow':
            last = torch.zeros_like(last)
            response = torch.zeros_like(response)
        hidden = self.hidden(torch.cat(
            [self.shallow_norm(first), self.deep_norm(last), response, available], -1))
        delta_r, delta_t = self.delta(hidden).chunk(2, -1)
        return shallow[0] + delta_r, shallow[1] + delta_t


class FeedbackTrackerModel(MissingTrackerModel):
    def __init__(self, baseline, task_utility=None, feedback_mode='deep', feedback_width=128):
        assert feedback_mode in ['none', 'shallow', 'deep']
        # Random selection needs a separately specified two-round RNG protocol.
        assert task_utility is None or task_utility['selection'] != 'random'
        super().__init__(baseline, repair=False, task_utility=task_utility)
        self.feedback_mode = feedback_mode
        self.feedback_width = feedback_width
        self.feedback = (None if feedback_mode == 'none' else
                         SearchFeedback(baseline.backbone.embed_dim, feedback_width, feedback_mode))
        assert len(baseline.backbone.blocks) == 12

    def embed_tokens(self, template, search, availability, patch_keep):
        backbone = self.baseline.backbone
        rgb, tir, mask = [], [], []
        for index, image in enumerate(template + [search]):
            pos = backbone.pos_embed_z if index < len(template) else backbone.pos_embed_x
            visible = availability[index].unsqueeze(1).expand(-1, pos.shape[1], -1)
            if index == len(template) and patch_keep is not None:
                visible = visible * patch_keep
            rgb.append(backbone.patch_embed(image[:, :3]) * visible[..., :1] + pos)
            tir.append(backbone.patch_embed(image[:, 3:]) * visible[..., 1:] + pos)
            mask.append(visible)
        query = backbone.cls_token.expand(search.shape[0], 1, -1)
        rgb = torch.cat([query, backbone.pos_drop(torch.cat(rgb, 1))], 1)
        tir = torch.cat([query, backbone.pos_drop(torch.cat(tir, 1))], 1)
        mask = torch.cat([search.new_ones(search.shape[0], 1, 2), torch.cat(mask, 1)], 1)
        return rgb, tir, mask

    def run_tail(self, rgb, tir, mask, probe_regions, lens_z, lens_x):
        auxiliary = {}
        for index in range(3, 12):
            rgb, tir = self.baseline.backbone.blocks[index](rgb, tir)
            if self.utility_repair is not None and index + 1 == self.repair_layer:
                before_r, before_t = rgb, tir
                rgb, tir, auxiliary = self.utility_repair(rgb, tir, mask)
                context = auxiliary.pop('context')
                if probe_regions is not None:
                    auxiliary['utility_probe'] = self.counterfactual_outputs(
                        before_r, before_t, mask, context, probe_regions, lens_z, lens_x)
        return rgb, tir, auxiliary

    def forward_tokens(self, template, search, availability, patch_keep, probe_regions=None):
        lens_z = len(template) * self.baseline.backbone.pos_embed_z.shape[1]
        lens_x = self.baseline.backbone.pos_embed_x.shape[1]
        rgb, tir, mask = self.embed_tokens(template, search, availability, patch_keep)
        for block in self.baseline.backbone.blocks[:3]:
            rgb, tir = block(rgb, tir)
        shallow_r, shallow_t = rgb, tir
        deep_r, deep_t, auxiliary1 = self.run_tail(
            rgb, tir, mask, None, lens_z, lens_x)
        round1 = self.baseline.forward_head(self.finish_features(deep_r, deep_t, lens_z, lens_x))
        round1.update(auxiliary1)
        if self.feedback is not None:
            search_r, search_t = self.feedback(
                (shallow_r[:, -lens_x:], shallow_t[:, -lens_x:]),
                (deep_r[:, -lens_x:], deep_t[:, -lens_x:]), round1, mask[:, -lens_x:])
            rgb = torch.cat([shallow_r[:, :-lens_x], search_r], 1)
            tir = torch.cat([shallow_t[:, :-lens_x], search_t], 1)
        deep_r, deep_t, auxiliary2 = self.run_tail(
            rgb, tir, mask, probe_regions, lens_z, lens_x)
        auxiliary2['round1'] = round1
        return self.finish_features(deep_r, deep_t, lens_z, lens_x), auxiliary2


def two_round_training_loss(complete, missing, boxes, cfg, **kwargs):
    """Mean of the existing C/M objective at both rounds; no extra loss scale."""
    first, status1 = training_loss([out['round1'] for out in complete],
                                  [out['round1'] for out in missing], boxes, cfg, **kwargs)
    second, status2 = training_loss(complete, missing, boxes, cfg, **kwargs)
    return .5 * (first + second), dict(round1=status1, round2=status2)
