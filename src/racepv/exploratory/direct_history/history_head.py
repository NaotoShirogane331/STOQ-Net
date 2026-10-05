"""Fixed recent-history features; no future observations or data loading."""
import torch
from torch import Tensor, nn
from torch.nn import functional as F


def extract_recent_features(history_values: Tensor, history_mask: Tensor) -> tuple[Tensor, Tensor]:
    if not isinstance(history_values, Tensor) or not isinstance(history_mask, Tensor):
        raise TypeError('history and mask must be tensors')
    if not history_values.is_floating_point():
        raise TypeError('history must be real floating point')
    if history_values.ndim != 3 or history_values.shape[0] == 0 or history_values.shape[1:] != (192, 14):
        raise ValueError('history must have nonempty shape [B,192,14]')
    if history_mask.shape != history_values.shape or history_mask.device != history_values.device:
        raise ValueError('mask shape/device must match history')
    if history_mask.is_complex() or not bool(((history_mask == 0) | (history_mask == 1)).all()):
        raise ValueError('history mask must be finite binary')
    with torch.autocast(device_type=history_values.device.type, enabled=False):
        p, s = history_values[:, -24:, 0], history_values[:, -24:, 1]
        mp = history_mask[:, -24:, 0].bool() & torch.isfinite(p)
        ms = history_mask[:, -24:, 1].bool() & torch.isfinite(s)
        mr = mp & ms
        pc = torch.where(mp, p, 0.).float()
        sc = torch.where(ms, s, 0.).float()
        residual = torch.where(mr, pc - sc, 0.)
        features = torch.cat((pc, residual, mp.float(), mr.float()), dim=1)
        if not bool(torch.isfinite(features).all()):
            raise ValueError('history features overflow FP32')
    return features, mp.any(dim=1)


class BoundedHistoryHead(nn.Module):
    """Zero-init 96→8 affine location head, local FP32; no RNG-consuming layers."""
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(8, 96, dtype=torch.float32))
        self.bias = nn.Parameter(torch.zeros(8, dtype=torch.float32))

    def forward(self, features: Tensor, valid: Tensor) -> tuple[Tensor, dict[str, Tensor]]:
        if not isinstance(features, Tensor) or not features.is_floating_point():
            raise TypeError('features must be real floating tensor')
        if features.ndim != 2 or features.shape[0] == 0 or features.shape[1] != 96:
            raise ValueError('features must have nonempty shape [B,96]')
        if not isinstance(valid, Tensor) or valid.dtype != torch.bool:
            raise TypeError('valid must be a boolean tensor')
        if valid.shape != features.shape[:1] or valid.device != features.device:
            raise ValueError('valid shape/device mismatch')
        if self.weight.dtype != torch.float32 or self.bias.dtype != torch.float32:
            raise ValueError('head parameters must remain FP32; do not cast model to half/double')
        if features.device != self.weight.device:
            raise ValueError('head and features must share device')
        if not bool(torch.isfinite(features).all()):
            raise ValueError('features must be finite')
        with torch.autocast(device_type=features.device.type, enabled=False):
            x = features.float()
            value_term = F.linear(x[:, :48], self.weight[:, :48])
            mask_term = F.linear(x[:, 48:], self.weight[:, 48:])
            bias_term = self.bias.expand(features.shape[0], -1)
            z = value_term + mask_term + bias_term
            if not bool(torch.isfinite(z).all()):
                raise FloatingPointError('non-finite direct-history affine result')
            delta = torch.where(valid[:, None], .08 * torch.tanh(z), 0.)
        return delta, {name: value.detach() for name, value in
                       (('value_term', value_term), ('mask_term', mask_term), ('bias_term', bias_term))}

