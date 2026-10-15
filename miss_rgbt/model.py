import torch
from torch import nn

from lib.models.cadtrack import build_cadtrack
from lib.models.cadtrack.vit_care import recover_tokens
from miss_rgbt.task_utility import TaskUtilityRepair


class LocalRepair(nn.Module):
    """Pointwise residual compensation from the missing view alone."""

    def __init__(self, dim, width):
        super().__init__()
        self.norm = nn.LayerNorm(2 * dim)
        self.hidden = nn.Sequential(nn.Linear(2 * dim + 2, width), nn.GELU())
        self.delta = nn.Linear(width, 2 * dim)
        self.gate = nn.Linear(width, 2)
        nn.init.zeros_(self.delta.weight)
        nn.init.zeros_(self.delta.bias)

    def forward(self, rgb, tir, available):
        h = self.hidden(torch.cat([self.norm(torch.cat([rgb, tir], -1)), available], -1))
        delta_r, delta_t = self.delta(h).chunk(2, -1)
        strength = self.gate(h).sigmoid() * (1 - available)
        return rgb + strength[..., :1] * delta_r, tir + strength[..., 1:] * delta_t


class MissingTrackerModel(nn.Module):
    def __init__(self, baseline, repair=False, width=128, repair_layer=6, task_utility=None):
        super().__init__()
        self.baseline = baseline
        self.repair_layer = repair_layer
        self.repair = LocalRepair(baseline.backbone.embed_dim, width) if repair else None
        self.utility_repair = None
        if task_utility is not None:
            assert not repair and repair_layer == 6
            self.utility_repair = TaskUtilityRepair(baseline.backbone.embed_dim,
                int(baseline.backbone.pos_embed_x.shape[1] ** .5),
                int(baseline.backbone.pos_embed_z.shape[1] ** .5), **task_utility)

    def forward(self, template, search, availability=None, patch_keep=None, utility_probe=None):
        """availability: templates then search, each [B,2], 1 means observed."""
        batch = search[0].shape[0]
        if availability is None:
            availability = [search[0].new_ones(batch, 2) for _ in template + search]
        if utility_probe is not None:
            assert self.utility_repair is not None
            assert 0 <= utility_probe['search_index'] < len(search)
        output = []
        for index, image in enumerate(search):
            keep = None if patch_keep is None else patch_keep[index]
            regions = (utility_probe['regions'] if utility_probe is not None and
                       index == utility_probe['search_index'] else None)
            features, auxiliary = self.forward_tokens(template, image,
                availability[:len(template)] + [availability[len(template) + index]], keep, regions)
            out = self.baseline.forward_head(features)
            out.update(attn=None, track_query_before=None, backbone_feat=None)
            out.update(auxiliary)
            output.append(out)
        return output

    def forward_tokens(self, template, search, availability, patch_keep, probe_regions=None):
        backbone = self.baseline.backbone
        lens_z = len(template) * backbone.pos_embed_z.shape[1]
        lens_x = backbone.pos_embed_x.shape[1]
        rgb, tir, mask = [], [], []
        for index, image in enumerate(template + [search]):
            pos = backbone.pos_embed_z if index < len(template) else backbone.pos_embed_x
            visible = availability[index].unsqueeze(1).expand(-1, pos.shape[1], -1)
            if index == len(template) and patch_keep is not None:
                visible = visible * patch_keep
            # Remove unobserved sensor/patch values before any Transformer block.
            rgb.append(backbone.patch_embed(image[:, :3]) * visible[..., :1] + pos)
            tir.append(backbone.patch_embed(image[:, 3:]) * visible[..., 1:] + pos)
            mask.append(visible)
        rgb = backbone.pos_drop(torch.cat(rgb, 1))
        tir = backbone.pos_drop(torch.cat(tir, 1))
        query = backbone.cls_token.expand(search.shape[0], 1, -1)
        rgb = torch.cat([query, rgb], 1)
        tir = torch.cat([query, tir], 1)
        mask = torch.cat([search.new_ones(search.shape[0], 1, 2), torch.cat(mask, 1)], 1)
        auxiliary = {}
        for index, block in enumerate(backbone.blocks):
            rgb, tir = block(rgb, tir)
            if self.repair is not None and index + 1 == self.repair_layer:
                rgb, tir = self.repair(rgb, tir, mask)
            if self.utility_repair is not None and index + 1 == self.repair_layer:
                before_r, before_t = rgb, tir
                rgb, tir, auxiliary = self.utility_repair(rgb, tir, mask)
                context = auxiliary.pop('context')
                if probe_regions is not None:
                    auxiliary['utility_probe'] = self.counterfactual_outputs(
                        before_r, before_t, mask, context, probe_regions, lens_z, lens_x)
        return self.finish_features(rgb, tir, lens_z, lens_x), auxiliary

    def finish_features(self, rgb, tir, lens_z, lens_x):
        backbone = self.baseline.backbone
        rgb = recover_tokens(rgb, lens_z, lens_x, mode=backbone.cat_mode)
        tir = recover_tokens(tir, lens_z, lens_x, mode=backbone.cat_mode)
        features = backbone.norm(torch.cat([rgb, tir], 1))
        rgb, tir = features.chunk(2, 1)
        return torch.cat([rgb[:, -lens_x:], tir[:, -lens_x:]], -1)

    def predict_tail(self, rgb, tir, lens_z, lens_x):
        for block in self.baseline.backbone.blocks[self.repair_layer:]:
            rgb, tir = block(rgb, tir)
        return self.baseline.forward_head(self.finish_features(rgb, tir, lens_z, lens_x))

    def counterfactual_outputs(self, rgb, tir, mask, context, regions, lens_z, lens_x):
        assert all(not module.training for module in self.baseline.modules()
                   if isinstance(module, nn.BatchNorm2d)), 'Utility probes require frozen BN statistics'
        devices = [rgb.device.index] if rgb.is_cuda else []
        with torch.no_grad(), torch.random.fork_rng(devices=devices):
            cpu_state = torch.get_rng_state()
            cuda_state = torch.cuda.get_rng_state(rgb.device) if rgb.is_cuda else None
            off = self.predict_tail(rgb.detach(), tir.detach(), lens_z, lens_x)
            torch.set_rng_state(cpu_state)
            if rgb.is_cuda:
                torch.cuda.set_rng_state(cuda_state, rgb.device)
            on_r, on_t, _ = self.utility_repair.apply_regions(
                rgb.detach(), tir.detach(), mask, context.detach(), regions)
            on = self.predict_tail(on_r, on_t, lens_z, lens_x)
        return dict(off=off, on=on, regions=regions)


def load_pretrained(cfg, checkpoint, **kwargs):
    core = build_cadtrack(cfg, training=False)
    saved = torch.load(checkpoint, map_location='cpu')
    assert saved['epoch'] == 20 and saved['net_type'] == 'CADTrack'
    core.load_state_dict(saved['net'], strict=True)
    return MissingTrackerModel(core, **kwargs)
