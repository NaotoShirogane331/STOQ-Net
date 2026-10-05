"""Separate engineering qualification adapter; never registered as scientific C.

The old CPU adapter is unchanged. CUDA fusion is explicitly evaluated in FP32,
which is a numerical qualification choice, not a pure-architecture experiment.
"""
import copy
import threading
import torch
from torch import nn
from racepv.models.config import DEFAULT_QUANTILES, EXPERT_NAMES
from racepv.reliability_experiment import ReliabilityControlRaCEPV
from .fusion import LocationGapFusion


def fuse_amp_roles(experts, logits, night):
    # float() retains the same forward graph. Never re-clamp expert inputs.
    with torch.autocast(device_type=experts.device.type, enabled=False):
        return LocationGapFusion()(experts.float(), logits.float(), night=night)


class LocationGapQualificationAdapter(nn.Module):
    def __init__(self, control, *, backend='cuda-mamba'):
        super().__init__()
        if type(control) is not ReliabilityControlRaCEPV:
            raise TypeError('unwrapped original C required')
        cfg = control.config
        expected = {'cpu-gru': ('gru', 'causal_conv_gru'), 'cuda-mamba': ('mamba', 'mamba_ssm')}
        if backend not in expected or (cfg.backbone, control.backbone_name) != expected[backend]:
            raise ValueError('explicit qualified backend required; no fallback')
        if (cfg.history_length != 192 or cfg.horizon != 24 or cfg.sampling_minutes != 15
                or cfg.d_model != 64 or cfg.router_hidden_size != 128
                or cfg.quantiles != DEFAULT_QUANTILES or cfg.power_upper_bound != 1.05
                or cfg.daylight_threshold != .01 or cfg.router_temperature != 1.
                or cfg.prediction_state_coupling != 'decoupled'
                or cfg.state_representation_mode != 'expert_detached' or cfg.state_head_mode != 'flat'
                or cfg.router_mode != 'contextual' or cfg.fusion_mode != 'learned'
                or cfg.active_experts != EXPERT_NAMES or cfg.enable_auxiliary_gate_path
                or cfg.tail_residual_mode != 'off' or cfg.future_weather_features != 0
                or cfg.forecast_weather_mode != 'off' or not cfg.use_clear_sky_prior):
            raise ValueError('fixed original C configuration required')
        # Construct on CPU first so projection initialization cannot consume CUDA RNG.
        if any(p.device.type != 'cpu' or p.dtype != torch.float32 for p in control.parameters()):
            raise ValueError('construct from CPU FP32 parameters, then move to CUDA')
        self.base = copy.deepcopy(control)
        final = self.base.fusion_router.contextual_network[-1]
        if type(final) is not nn.Linear or (final.in_features, final.out_features) != (64, 3):
            raise ValueError('original projection required')
        with torch.random.fork_rng(devices=[]):
            projection = nn.Linear(64, 9)
        with torch.no_grad():
            projection.weight.copy_(final.weight.repeat(3, 1))
            projection.bias.copy_(final.bias.repeat(3))
        projection.weight.requires_grad_(final.weight.requires_grad)
        projection.bias.requires_grad_(final.bias.requires_grad)
        projection.train(final.training)
        self.base.fusion_router.contextual_network[-1] = projection
        self.backend = backend
        self.training = control.training
        self._forward_lock = threading.Lock()

    @property
    def config(self): return self.base.config

    @property
    def state_head(self): return self.base.state_head

    @property
    def backbone_name(self): return self.base.backbone_name

    def make_objective(self): return self.base.make_objective()

    @property
    def model_metadata(self):
        md = dict(self.base.model_metadata)
        md.update(parent_model_revision=md['model_revision'],
                  model_revision='racepv.location-gap-fusion.cuda-qualification.v1',
                  architecture_version='racepv.location-gap-fusion.cuda-qualification.v1',
                  qualification_backend=self.backend, scientific_training_ready=False,
                  initialization='copied_synthetic_control_rows_for_equivalence_only',
                  fusion_precision='FP32 island after original AMP expert computation',
                  role_names=['center','lower_gaps','upper_gaps'],
                  fusion_weight_semantics='location_lower_gap_upper_gap_role_simplexes',
                  fusion_gate_semantics='location_lower_gap_upper_gap_role_simplexes',
                  output_schema='location_gap_roles_not_legacy_gate.v1')
        return md

    def forward(self, history_values, history_mask, future_known, site_metadata,
                future_weather=None, future_weather_mask=None):
        device = 'cuda' if self.backend == 'cuda-mamba' else 'cpu'
        if any(p.dtype != torch.float32 or p.device.type != device for p in self.parameters()):
            raise ValueError('FP32 master parameters on the declared backend required')
        if any(v.dtype != torch.float32 or v.device.type != device
               for v in (history_values, future_known, site_metadata)):
            raise ValueError('FP32 inputs on the declared backend required')
        if history_mask.device != history_values.device:
            raise ValueError('mask device mismatch')
        if future_weather is not None or future_weather_mask is not None:
            raise ValueError('future weather forbidden')
        if not self._forward_lock.acquire(blocking=False):
            raise RuntimeError('concurrent/reentrant qualification forward forbidden')
        captured = []
        handle = None
        def capture(module, args, logits):
            if captured or logits.shape[-1] != 9:
                raise RuntimeError('exactly one nine-logit router call required')
            captured.append(logits)
            return logits[..., :3]
        try:
            handle = self.base.fusion_router.register_forward_hook(capture)
            out = dict(self.base(history_values, history_mask, future_known, site_metadata))
            if len(captured) != 1: raise RuntimeError('role logits absent')
            clean = torch.nan_to_num(future_known, nan=0., posinf=0., neginf=0.)
            night = clean[..., self.config.future_clear_sky_index] < self.config.daylight_threshold
            fused = fuse_amp_roles(out['expert_quantiles'], captured[0], night)
            out.pop('gate_weights'); out.pop('fusion_logits')
            out.update(quantiles=fused.quantiles, base_quantiles=fused.quantiles,
                       role_fusion_logits=captured[0].reshape(*captured[0].shape[:2],3,3),
                       role_gate_weights=fused.weights, role_raw_quantiles=fused.raw_quantiles,
                       role_clipped_mask=fused.clipped_mask, role_tie_mask=fused.tie_mask,
                       role_gap_change=fused.gap_change)
            return out
        finally:
            if handle is not None: handle.remove()
            captured.clear()
            self._forward_lock.release()
