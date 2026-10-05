"""Isolated location / lower-gap / upper-gap fusion prototype.

Seven fixed levels: .05, .10, .25, .50, .75, .90, .95; three experts.
No model imports, learned router, training loss, checkpoint or data access.
Only float32/float64 are supported in this CPU-stage prototype, not AMP.
"""

from dataclasses import dataclass

import torch
from torch import Tensor, nn


POWER_UPPER_BOUND = 1.05


@dataclass(frozen=True)
class FusionOutput:
    raw_quantiles: Tensor  # After night masking, before final physical clamp.
    quantiles: Tensor
    weights: Tensor  # [B,H,role,expert]; roles: center, lower gaps, upper gaps.

    @property
    def clipped_mask(self) -> Tensor:
        return self.raw_quantiles != self.quantiles

    @property
    def tie_mask(self) -> Tensor:
        return self.quantiles.diff(dim=-1) == 0

    @property
    def gap_change(self) -> Tensor:
        """Post-clamp minus pre-clamp adjacent gaps, in ascending level order."""
        return self.quantiles.diff(dim=-1) - self.raw_quantiles.diff(dim=-1)


def _validate_experts_and_night(q: Tensor, night: Tensor) -> None:
    if not isinstance(q, Tensor) or q.dtype not in (torch.float32, torch.float64):
        raise TypeError('expert_quantiles must be a float32 or float64 tensor')
    if q.ndim != 4 or q.shape[-2:] != (3, 7) or min(q.shape[:2]) < 1:
        raise ValueError('expert_quantiles must have nonempty shape [B,H,3,7]')
    if not torch.isfinite(q).all():
        raise ValueError('expert_quantiles must be finite')
    if ((q < 0) | (q > POWER_UPPER_BOUND)).any():
        raise ValueError('expert_quantiles must already be projected to [0,1.05]')
    if (q.diff(dim=-1) < 0).any():
        raise ValueError('expert_quantiles must be nondecreasing (ties allowed)')
    if not isinstance(night, Tensor) or night.dtype != torch.bool:
        raise TypeError('night must be a boolean tensor')
    if night.shape != q.shape[:2] or night.device != q.device:
        raise ValueError('night must have shape [B,H] and match the expert device')


def _validate_matching(name: str, tensor: Tensor, q: Tensor, shape: tuple) -> None:
    if not isinstance(tensor, Tensor) or tensor.dtype != q.dtype:
        raise TypeError(f'{name} must have the same floating dtype as experts')
    if tensor.shape != shape or tensor.device != q.device:
        raise ValueError(f'{name} must have shape {shape} and match the expert device')
    if not torch.isfinite(tensor).all():
        raise ValueError(f'{name} must be finite')


def _fuse_validated(q: Tensor, weights: Tensor, night: Tensor) -> FusionOutput:
    # Same night semantics as the original C fusion: zero all experts first.
    q = q.masked_fill(night[..., None, None], 0.0)
    median = (weights[..., 0, :] * q[..., 3]).sum(dim=-1)
    adjacent = q.diff(dim=-1)
    lower = (weights[..., 1, :, None] * adjacent[..., :3].flip(-1)).sum(dim=-2)
    upper = (weights[..., 2, :, None] * adjacent[..., 3:]).sum(dim=-2)
    raw = torch.cat((
        median[..., None] - lower.cumsum(-1).flip(-1),
        median[..., None],
        median[..., None] + upper.cumsum(-1),
    ), dim=-1)
    # Separate center/shape weights may exceed the expert physical bounds.
    projected = raw.clamp(0.0, POWER_UPPER_BOUND)
    return FusionOutput(raw_quantiles=raw, quantiles=projected, weights=weights)


def fuse_location_gaps(expert_quantiles: Tensor, weights: Tensor, *, night: Tensor) -> FusionOutput:
    """Fuse already ordered/projected experts with explicit simplex weights.

    Shapes: experts [B,H,3,7], weights [B,H,3,3], night bool [B,H].
    Each role's weights must be nonnegative and sum to one, allowing only
    floating rounding tolerance (8 * dtype epsilon). They are not renormalized.
    With all three role weights tied to w this equals sum_e(w_e * q_e)
    in real arithmetic, not necessarily bitwise floating-point arithmetic.
    Identical experts are preserved; missing-history fallback is upstream.
    """
    q = expert_quantiles
    _validate_experts_and_night(q, night)
    _validate_matching('weights', weights, q, (*q.shape[:2], 3, 3))
    if (weights < 0).any():
        raise ValueError('weights must be nonnegative')
    if not torch.allclose(weights.sum(-1), torch.ones_like(weights[..., 0]),
                          rtol=0, atol=8 * torch.finfo(q.dtype).eps):
        raise ValueError('weights must sum to one along the expert axis')
    return _fuse_validated(q, weights, night)


class LocationGapFusion(nn.Module):
    """Parameter-free fusion of supplied nine logits, NOT a complete new model.

    logits [B,H,9] are contiguous role groups: center's three expert logits,
    lower's three, upper's three. Each group receives its own softmax. The
    future learned router is deliberately not implemented or connected here.
    Gradients are retained for experts and logits; tail losses still update
    the center. This is not gradient isolation or a guarantee of accuracy.
    """

    def forward(self, expert_quantiles: Tensor, logits: Tensor, *, night: Tensor) -> FusionOutput:
        q = expert_quantiles
        _validate_experts_and_night(q, night)
        _validate_matching('logits', logits, q, (*q.shape[:2], 9))
        weights = logits.reshape(*q.shape[:2], 3, 3).softmax(dim=-1)
        return _fuse_validated(q, weights, night)
