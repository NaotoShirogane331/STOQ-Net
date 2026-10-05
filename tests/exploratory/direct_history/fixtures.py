"""Small synthetic fixtures. No experiment data/checkpoints or CUDA calls."""
import random
import numpy as np
import torch
from racepv.models import RaCEPVConfig
from racepv.reliability_experiment import ReliabilityControlRaCEPV


def make_config():
    return RaCEPVConfig(history_features=14, future_features=9, site_features=6,
                       future_hour_sin_index=4, future_hour_cos_index=5, backbone='gru', clear_sky_history_feature_indices=(2,7))


def reset_rng(seed=29):
    random.seed(seed)
    np.random.seed(seed)
    # CPU generator only; no torch.manual_seed's lazy CUDA hooks.
    torch.random.default_generator.manual_seed(seed)


def rng_state():
    return random.getstate(), np.random.get_state(), torch.get_rng_state().clone()


def restore_rng(state):
    random.setstate(state[0])
    np.random.set_state(state[1])
    torch.set_rng_state(state[2])


def make_inputs(seed=13, batch_size=2):
    gen = torch.Generator(device='cpu').manual_seed(seed)
    history = .15 + .65 * torch.rand(batch_size, 192, 14, generator=gen)
    future = .2 + .4 * torch.rand(batch_size, 24, 9, generator=gen)
    future[:, ::5, 0] = 0
    phase = torch.linspace(0, 1, 24)
    future[..., 4], future[..., 5] = phase.sin(), phase.cos()
    return dict(history_values=history, history_mask=torch.ones_like(history, dtype=torch.bool),
                future_known=future, site_metadata=torch.rand(batch_size, 6, generator=gen))


def make_batch(seed=13):
    batch = make_inputs(seed)
    batch.update(targets=torch.full((2, 24), .4),
                 state_labels=torch.arange(48).reshape(2, 24) % 3,
                 daylight_mask=batch['future_known'][..., 0] >= .01,
                 target_mask=torch.ones(2, 24, dtype=torch.bool))
    return batch


def paired_models(candidate_type, seed=29):
    reset_rng(seed)
    control = ReliabilityControlRaCEPV(make_config())
    reset_rng(seed)
    candidate = candidate_type(make_config())
    return control, candidate

