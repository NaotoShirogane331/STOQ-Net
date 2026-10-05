"""Fresh C-LG screen adapter; original C/source/trainer remain untouched.

Caller binds original racepv source and seeds once before build. CPU-GRU is
synthetic test evidence only. CUDA-Mamba uses the qualified FP32 fusion island.
Serialize state_dict/checkpoint tensors, never the whole module. No resume or
checkpoint migration is supplied by this screen adapter.
"""
import copy
import hashlib
import json
import time

import torch
from torch import nn
from .cuda_adapter import fuse_amp_roles
from racepv.models.config import DEFAULT_QUANTILES, EXPERT_NAMES
from racepv.reliability_experiment import ReliabilityControlRaCEPV
from racepv.training.engine import _model_inputs
from racepv.training.research_runtime import ResearchTrainer
from racepv.training.research_baseline_bridge import masked_research_loss

REVISION = 'racepv.location-gap-fusion.scientific-screen.v1'
ROLE_NAMES = ('center', 'lower_gaps', 'upper_gaps')


class LocationGapScientificAdapter(nn.Module):
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
        if any(p.device.type != 'cpu' or p.dtype != torch.float32 for p in control.parameters()):
            raise ValueError('construct from fresh CPU FP32 C before device transfer')
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
        self._forward_active = False

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
        md.update(parent_model_revision=md['model_revision'], model_id='C-LG',
            model_revision=REVISION, architecture_version=REVISION,
            screen_backend=self.backend, scientific_training_ready=self.backend == 'cuda-mamba',
            cpu_contract_only=self.backend == 'cpu-gru',
            initialization='fresh_control_shared_state_repeated_role_projection',
            fusion_precision='FP32 island after original AMP expert computation',
            role_names=list(ROLE_NAMES), extra_parameters=390,
            fusion_weight_semantics='location_lower_gap_upper_gap_role_simplexes',
            fusion_gate_semantics='location_lower_gap_upper_gap_role_simplexes',
            output_schema='location_gap_roles_not_legacy_gate.v1')
        return md

    def forward(self, history_values, history_mask, future_known, site_metadata,
                future_weather=None, future_weather_mask=None):
        device = 'cuda' if self.backend == 'cuda-mamba' else 'cpu'
        if any(p.dtype != torch.float32 or p.device.type != device for p in self.parameters()):
            raise ValueError('FP32 master parameters on declared backend required')
        if any(v.dtype != torch.float32 or v.device.type != device
                for v in (history_values, future_known, site_metadata)):
            raise ValueError('FP32 inputs on declared backend required')
        if history_mask.device != history_values.device:
            raise ValueError('mask device mismatch')
        if future_weather is not None or future_weather_mask is not None:
            raise ValueError('future weather forbidden')
        if self._forward_active:
            raise RuntimeError('concurrent/reentrant screen forward forbidden')
        self._forward_active = True
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
                role_fusion_logits=captured[0].reshape(*captured[0].shape[:2], 3, 3),
                role_gate_weights=fused.weights, role_raw_quantiles=fused.raw_quantiles,
                role_clipped_mask=fused.clipped_mask, role_tie_mask=fused.tie_mask,
                role_gap_change=fused.gap_change)
            return out
        finally:
            if handle is not None: handle.remove()
            self._forward_active = False


def _canonical_state_sha256(state):
    """Hash sorted named tensors, dtype/shape and raw bytes, not torch.save ZIPs."""
    h = hashlib.sha256()
    for name in sorted(state):
        value = state[name].detach().cpu().contiguous()
        header = json.dumps([name, str(value.dtype), list(value.shape)], separators=(',', ':')).encode()
        data = value.reshape(-1).view(torch.uint8).numpy().tobytes()
        h.update(len(header).to_bytes(8, 'big')); h.update(header)
        h.update(len(data).to_bytes(8, 'big')); h.update(data)
    return h.hexdigest()


def shared_initialization_proof(control, candidate):
    """Exact shared parameters/buffers and independent three-row-copy proof."""
    if type(control) is not ReliabilityControlRaCEPV:
        raise TypeError('original C required for initialization proof')
    base = candidate.base if isinstance(candidate, LocationGapScientificAdapter) else candidate
    cs, ads = control.state_dict(), base.state_dict()
    if set(cs) != set(ads): raise ValueError('shared state keys differ')
    prefix = 'fusion_router.contextual_network.' + str(len(control.fusion_router.contextual_network) - 1)
    projection_keys = (prefix + '.weight', prefix + '.bias')
    is_candidate = isinstance(candidate, LocationGapScientificAdapter)
    for key in cs:
        expected = cs[key]
        if is_candidate and key in projection_keys:
            expected = expected.repeat(3, 1) if key.endswith('.weight') else expected.repeat(3)
        if not torch.equal(expected, ads[key]):
            raise ValueError('projection repetition differs' if key in projection_keys else 'shared tensor differs: ' + key)
    shared_c = {k: v for k, v in cs.items() if k not in projection_keys}
    shared_a = {k: v for k, v in ads.items() if k not in projection_keys}
    ch, ah = _canonical_state_sha256(shared_c), _canonical_state_sha256(shared_a)
    return dict(canonical_shared_sha256=ch, control_shared_sha256=ch, candidate_shared_sha256=ah,
        repeated_projection_verified=True, repeated_three_role_rows_exact=True,
        projection_proof='three_role_groups_repeat_original_rows' if is_candidate else 'original_C_projection_self_verified',
        projection_keys=list(projection_keys), shared_tensor_count=len(shared_c),
        shared_buffer_count=sum(n not in projection_keys for n, _ in control.named_buffers()),
        control_full_sha256=_canonical_state_sha256(cs), candidate_full_sha256=_canonical_state_sha256(ads),
        hash_schema='sorted_named_tensor_dtype_shape_raw_bytes.v1')


class LocationGapResearchTrainer(ResearchTrainer):
    """Only C-LG dispatch differs; original fit/AdamW/AMP/step/scoring reused."""
    def __init__(self, *args, model_id='C-LG', **kwargs):
        if model_id != 'C-LG' or not kwargs.get('separate_state_optimizer', False):
            raise ValueError('C-LG requires original separated forecast/state optimizer')
        if not args or not isinstance(args[0], LocationGapScientificAdapter):
            raise TypeError('C-LG scientific adapter required')
        super().__init__(*args, model_id=model_id, **kwargs)

    def predict_batch(self, batch):
        # Same candidate input and final projection semantics as ResearchTrainer.
        mask = batch['history_mask']
        if not bool(((mask == 0) | (mask == 1)).all()):
            raise ValueError('history mask is not binary')
        batch = dict(batch, history_mask=mask.bool())
        inputs = _model_inputs(batch)
        inputs['history_values'] = torch.where(inputs['history_mask'], inputs['history_values'], 0.)
        output = self.model(**inputs)
        q = output['quantiles']
        if q.shape != (*batch['targets'].shape, 7) or not bool(torch.isfinite(q).all()):
            raise ValueError('non-finite or malformed candidate quantiles')
        if bool((q[..., 1:] < q[..., :-1]).any()): raise ValueError('candidate quantile order differs')
        output['quantiles'] = torch.where(batch['daylight_mask'].bool()[..., None], q.clamp(0., 1.05), 0.)
        return output

    def _run_epoch(self, loader, training):
        # Original ResearchTrainer loop specialized only to its candidate branch.
        # Inherited _criterion consumes projected qCRPS + detached state CE.
        started = time.perf_counter()
        self.model.train(training)
        total, score, count = 0., 0., 0
        context = torch.enable_grad() if training else torch.inference_mode()
        with context:
            for raw_batch in loader:
                batch = self._move(raw_batch)
                with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.use_amp):
                    output = self.predict_batch(batch)
                    result = masked_research_loss(output['quantiles'], batch, model_id='tcn_qr')
                    if result['skip']: continue
                    loss, terms = self._criterion(output, batch)
                if not bool(torch.isfinite(loss)): raise FloatingPointError('non-finite research loss')
                if training:
                    self._step_separated_losses(forecast_loss=terms['forecast_loss'], state_loss=terms['state_loss'])
                n = result['valid_count']
                total += float(loss.detach()) * n
                score += float(result['daylight_qcrps'].detach()) * n
                count += n
        if count == 0: raise ValueError('no legal daylight target population')
        return dict(loss=total / count, daylight_qcrps=score / count,
            valid_count=float(count), elapsed_seconds=time.perf_counter() - started)


def build(signature, row, *, backend='cuda-mamba'):
    """Build one fresh approved row without reseeding the caller's RNG stream."""
    from .lg_policy import validate_candidate
    from racepv.training.research_campaign import candidate_config
    validate_candidate(row)
    if backend not in ('cuda-mamba', 'cpu-gru'): raise ValueError('explicit backend required')
    cfg = candidate_config(signature, synthetic_test=backend == 'cpu-gru')
    control = ReliabilityControlRaCEPV(cfg)
    model = control if row['architecture'] == 'C' else LocationGapScientificAdapter(control, backend=backend)
    proof = shared_initialization_proof(control, model)
    if backend == 'cuda-mamba':
        expected = 207014 if row['architecture'] == 'C' else 207404
        if sum(p.numel() for p in model.parameters()) != expected:
            raise ValueError('original C/C-LG parameter count differs')
    return model, model.make_objective(), proof


def trainer(model, objective, row, *, device, checkpoint_path=None, metadata=None):
    """Original C trainer; C-LG trainer differs only in candidate dispatch."""
    from .lg_policy import validate_candidate
    validate_candidate(row)
    kind = ResearchTrainer if row['architecture'] == 'C' else LocationGapResearchTrainer
    return kind(model, objective, model_id=row['architecture'], device=device,
        learning_rate=row['learning_rate'], weight_decay=row['weight_decay'],
        max_epochs=1, patience=8, min_epochs=1, gradient_clip_norm=1.,
        mixed_precision=device == 'cuda', separate_state_optimizer=True,
        selection_metric='daylight_qcrps', checkpoint_path=checkpoint_path,
        checkpoint_metadata=metadata)
