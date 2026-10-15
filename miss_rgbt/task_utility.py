"""Region utility selection and compensation from the current view alone."""
import torch
from torch import nn
from torch.nn import functional as F


class TaskUtilityRepair(nn.Module):
    def __init__(self, dim, search_side, template_side, width=128, depth=1,
                 region_grid=4, selected_regions=4, refinement_limit=.1,
                 selection='utility', selection_seed=2027):
        super().__init__()
        assert search_side % region_grid == 0
        assert 1 <= selected_regions <= region_grid ** 2 and depth >= 1
        assert 0 <= refinement_limit <= 1
        assert selection in ['utility', 'similarity', 'random', 'all']
        self.search_tokens = search_side ** 2
        self.template_tokens = template_side ** 2
        self.selected_regions = selected_regions
        self.refinement_limit = refinement_limit
        self.selection = selection
        self.selection_seed = selection_seed
        size = search_side // region_grid
        regions = torch.arange(self.search_tokens).reshape(region_grid, size, region_grid, size)
        self.register_buffer('region_tokens', regions.permute(0, 2, 1, 3).reshape(region_grid ** 2, -1))
        grid = torch.arange(self.template_tokens).reshape(template_side, template_side)
        foreground = grid[template_side // 4:3 * template_side // 4,
                          template_side // 4:3 * template_side // 4].flatten()
        background = torch.ones(self.template_tokens, dtype=torch.bool)
        background[foreground] = False
        self.register_buffer('reference_foreground', foreground)
        self.register_buffer('reference_background', torch.arange(self.template_tokens)[background])
        generator = torch.Generator().manual_seed(selection_seed)
        self.register_buffer('selection_rng_state', generator.get_state())
        self.norm = nn.LayerNorm(2 * dim)
        inputs = 2 * dim + 4
        self.selector = nn.Sequential(nn.Linear(inputs, width), nn.GELU(), nn.Linear(width, 1))
        layers = [nn.Linear(inputs, width), nn.GELU()]
        for _ in range(depth - 1):
            layers.extend([nn.Linear(width, width), nn.GELU()])
        self.correction = nn.Sequential(*layers)
        self.delta = nn.Linear(width, 2 * dim)
        self.gate = nn.Linear(width, 2)
        self.deficit = nn.Linear(width, 2)
        nn.init.zeros_(self.delta.weight)
        nn.init.zeros_(self.delta.bias)

    def reset_selection(self):
        state = torch.Generator().manual_seed(self.selection_seed).get_state()
        self.selection_rng_state.copy_(state.to(self.selection_rng_state.device))

    def context(self, rgb, tir, available):
        references = []
        for tokens in [rgb, tir]:
            initial = tokens[:, 1:1 + self.template_tokens]
            reference = (initial[:, self.reference_foreground].mean(1) -
                         initial[:, self.reference_background].mean(1))
            similarity = (F.normalize(tokens[:, -self.search_tokens:], dim=-1) *
                          F.normalize(reference, dim=-1).unsqueeze(1)).sum(-1)
            references.append(similarity)
        similarity = torch.stack(references, -1)
        observed = available[:, -self.search_tokens:]
        context = torch.cat([self.norm(torch.cat([rgb[:, -self.search_tokens:],
                                                  tir[:, -self.search_tokens:]], -1)),
                             observed, similarity], -1)
        return context, (similarity * observed).sum(-1) / observed.sum(-1).clamp_min(1)

    def select(self, logits, similarity):
        if self.selection == 'all':
            return torch.arange(logits.shape[1], device=logits.device).expand(logits.shape[0], -1)
        if self.selection == 'random':
            generator = torch.Generator()
            generator.set_state(self.selection_rng_state.cpu())
            scores = torch.rand(logits.shape, generator=generator).to(logits.device)
            self.selection_rng_state.copy_(generator.get_state().to(self.selection_rng_state.device))
        elif self.selection == 'similarity':
            scores = similarity
        else:
            scores = logits
        return scores.topk(self.selected_regions, dim=1).indices

    def apply_regions(self, rgb, tir, available, context, regions):
        tokens = self.region_tokens[regions].flatten(1)
        inputs = context.gather(1, tokens.unsqueeze(-1).expand(-1, -1, context.shape[-1]))
        hidden = self.correction(inputs)
        delta_r, delta_t = self.delta(hidden).chunk(2, -1)
        observed = available[:, -self.search_tokens:].gather(1, tokens.unsqueeze(-1).expand(-1, -1, 2))
        deficit = self.refinement_limit * self.deficit(hidden).sigmoid()
        strength = self.gate(hidden).sigmoid() * ((1 - observed) + observed * deficit)
        index = tokens.unsqueeze(-1).expand(-1, -1, rgb.shape[-1])
        search_r = rgb[:, -self.search_tokens:].scatter_add(1, index, strength[..., :1] * delta_r)
        search_t = tir[:, -self.search_tokens:].scatter_add(1, index, strength[..., 1:] * delta_t)
        return (torch.cat([rgb[:, :-self.search_tokens], search_r], 1),
                torch.cat([tir[:, :-self.search_tokens], search_t], 1), tokens)

    def forward(self, rgb, tir, available):
        context, similarity = self.context(rgb, tir, available)
        region_context = context[:, self.region_tokens].mean(2)
        # Utility targets train the selector; task/KD train the representation and correction.
        logits = self.selector(region_context.detach()).squeeze(-1)
        regions = self.select(logits, similarity[:, self.region_tokens].mean(2))
        rgb, tir, tokens = self.apply_regions(rgb, tir, available, context, regions)
        return rgb, tir, dict(utility_logits=logits, utility_regions=regions,
                             utility_tokens=tokens, context=context)


def utility_targets(loss_off, loss_on):
    """Signed relative single-region benefit; never a feature-recovery label."""
    return ((loss_off.detach() - loss_on.detach()) /
            loss_off.detach().clamp_min(1e-6)).clamp(-1, 1)
