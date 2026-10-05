"""Isolated fixed direct-history candidate; explicit backend qualification."""
import torch
from torch import Tensor
from racepv.models import RaCEPVConfig
from racepv.reliability_experiment import ReliabilityControlRaCEPV
from .history_head import BoundedHistoryHead, extract_recent_features

REVISION = 'racepv.direct-history-linear.v1'
CANDIDATE_ID = 'C-DH-linear'


class DirectHistoryC(ReliabilityControlRaCEPV):
    def __init__(self, config: RaCEPVConfig):
        expected = RaCEPVConfig(history_features=14, future_features=9, site_features=6,
            future_hour_sin_index=4, future_hour_cos_index=5, backbone=config.backbone,
            clear_sky_history_feature_indices=(2,7))
        if type(config) is not RaCEPVConfig or config != expected or config.backbone not in ('gru','mamba'):
            raise ValueError('fixed Mamba production or GRU synthetic contract required')
        # Preserve the exact C constructor, including A wrapper construction/unwrapping.
        super().__init__(config)
        self.direct_history_head = BoundedHistoryHead()

    @property
    def model_metadata(self):
        metadata = dict(super().model_metadata)
        metadata.update(parent_model_revision=metadata['model_revision'],
            model_revision=REVISION, architecture_version=REVISION, candidate_id=CANDIDATE_ID,
            cpu_contract_only=self.config.backbone == 'gru', scientific_training_ready=self.config.backbone == 'mamba',
            direct_history_steps=24, direct_history_corrected_steps=8,
            direct_history_cap=.08, direct_history_extra_parameters=776,
            direct_history_initialization='zero_without_rng',
            direct_history_feature_order='power24_residual24_power_mask24_residual_mask24')
        return metadata

    def forward(self, history_values, history_mask, future_known, site_metadata,
                future_weather=None, future_weather_mask=None) -> dict[str, Tensor]:
        if future_weather is not None or future_weather_mask is not None:
            raise ValueError('future weather is not admitted')
        device = 'cuda' if self.config.backbone == 'mamba' else 'cpu'
        if any(p.device.type != device or p.dtype != torch.float32 for p in self.parameters()):
            raise ValueError('FP32 master parameters on explicit backend required')
        for value in (history_values, future_known, site_metadata):
            if not isinstance(value, Tensor) or value.device.type != device or value.dtype != torch.float32:
                raise ValueError('FP32 inputs on explicit backend required')
        if torch.is_autocast_enabled('cpu'):
            raise ValueError('whole-model CPU autocast is not qualified; head island is tested separately')
        features, valid = extract_recent_features(history_values, history_mask)
        outputs = super().forward(history_values, history_mask, future_known, site_metadata)
        q = outputs['quantiles']
        delta, terms = self.direct_history_head(features, valid)
        clean_solar = torch.nan_to_num(future_known[:, :8, 0], nan=0., posinf=0., neginf=0.)
        delta = delta.masked_fill(clean_solar < self.config.daylight_threshold, 0.).to(dtype=q.dtype)
        corrected = (q[:, :8] + delta.unsqueeze(-1)).clamp(0., self.config.power_upper_bound)
        result = dict(outputs)
        result['quantiles'] = torch.cat((corrected, q[:, 8:]), dim=1)
        result['direct_history_delta'] = torch.cat((delta, q.new_zeros(q.shape[0], 16)), dim=1).detach()
        result['direct_history_valid'] = valid.detach()
        result.update({f'direct_history_{key}': value for key, value in terms.items()})
        return result

