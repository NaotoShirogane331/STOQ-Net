"""Research-only identities. Registration is not execution or scientific readiness."""

from copy import deepcopy

QUANTILES = (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95)
FULL_INFORMATION = "origin_history_values_and_masks+deterministic_future_known+site_metadata"


def _entry(name, kind, *, role="legacy_registered_execution_unverified", prediction="quantile",
           training="supervised", config=None, dependencies=(), status="ADAPTER_AVAILABLE",
           information=FULL_INFORMATION):
    return dict(model_id=name, historical_role=role, implementation_kind=kind,
                input_information=information, prediction_kind=prediction,
                training_kind=training, default_config=config or {},
                dependency_requirements=list(dependencies), bridge_status=status)


_REGISTRY = {
    name: _entry(name, "deterministic_reference", prediction="point", training="none",
                 information="historical_power_and_clear_sky+deterministic_future_clear_sky"
                 if name == "smart_persistence" else "deterministic_future_clear_sky")
    for name in ("smart_persistence", "clear_sky_persistence")
}
_REGISTRY.update({
    "quantile_lightgbm": _entry("quantile_lightgbm", "per_lead_quantile_tree", dependencies=("lightgbm",),
        config=dict(n_estimators=1000, learning_rate=.04, num_leaves=31, min_child_samples=100,
                    bagging_fraction=.8, bagging_freq=1, feature_fraction=1, verbosity=-1, random_state=13)),
    "random_forest_mondrian": _entry("random_forest_mondrian", "random_forest_point_external_calibration_pending",
        prediction="point", dependencies=("sklearn",),
        config=dict(n_estimators=500, min_samples_leaf=2, n_jobs=-1, random_state=13)),
    "lstm_qr": _entry("lstm_qr", "local_quantile_baseline", config=dict(hidden_dim=64, num_layers=2), dependencies=("torch",)),
    "tcn_qr": _entry(
        "tcn_qr",
        "research_protocol_tcn_two_conv_full_history_pool_v1",
        config=dict(hidden_dim=64),
        dependencies=("torch",),
    ),
    "patchtst_qr": _entry("patchtst_qr", "local_patchtst_style_baseline",
        config=dict(patch_length=16, patch_stride=8, d_model=64, n_heads=4, n_layers=2), dependencies=("torch",)),
    "s_mamba_qr": _entry("s_mamba_qr", "local_official_mamba_operator_adaptation",
        config=dict(d_model=64, d_state=16, d_conv=4), dependencies=("torch", "mamba_ssm", "CUDA runtime")),
    "timemixerpp_qr": _entry("timemixerpp_qr", "paper_derived_protocol_reproduction",
        config=dict(d_model=64, layers=2, downsampling_scales=[1, 2, 4], periods=[4, 8, 24], dropout=.1), dependencies=("torch",)),
    "timer_xl_qr": _entry("timer_xl_qr", "official_backbone_protocol_adaptation",
        config=dict(patch_len=16, d_model=256, n_heads=8, e_layers=4, dropout=.1),
        dependencies=("torch", "audited OpenLTM checkout and source_identity")),
    "rqr_tft": _entry("rqr_tft", "paper_derived_protocol_reproduction",
        config=dict(hidden_size=64, attention_heads=4, lstm_layers=1, dropout=.1,
                    ramp_loss_weight=1., ramp_weighted_quantile_multiplier=6., ramp_decision_threshold=.35), dependencies=("torch",)),
})
for _name in ("nhits", "tft", "xlstm"):
    _REGISTRY[_name] = _entry(_name, "official_NeuralForecast_backbone_origin_window_adaptation",
        dependencies=("torch", "neuralforecast", "xlstm") if _name == "xlstm" else ("torch", "neuralforecast"))
for _name in ("fact_qr", "itransformer_qr"):
    _REGISTRY[_name] = _entry(_name, "paper_derived_protocol_adaptation", dependencies=("torch", "fixed_audited_upstream_checkout"),
        config={"upstream_defaults": dict(d_model=512, d_ff=1024, dropout=.1, use_norm=True, freq="n", num_kernels=4,
            dilation=[1, 2, 3, 2, 1], core=.5) if _name == "fact_qr" else dict(d_model=512, n_heads=8, e_layers=2,
            d_ff=2048, factor=1, dropout=.1, embed="timeF", freq="h", activation="gelu", output_attention=False,
            use_norm=True, class_strategy="projection")})
for _name in ("timesfm3_zs", "sundial_base_zs", "moirai_1_1_r_small_zs"):
    _REGISTRY[_name] = _entry(_name, "official_frozen_foundation_adapter_connected", prediction="point" if _name == "timesfm3_zs" else "quantile",
        training="frozen_zero_shot", information="history_target_only", status="ADAPTER_CONNECTED",
        dependencies=("official package", "audited local weights", "schema3 foundation adapter"))
    _REGISTRY[_name].update(actual_weights_verified=False, actual_gpu_verified=False)
_REGISTRY["cyclenet_qr"] = _entry("cyclenet_qr", "validation_exploration_adapter_pending", role="validation_only_exploration",
    status="NOT_CONNECTED", dependencies=("audited CycleNet checkout", "schema3 adapter"))
_REGISTRY["conditional_empirical_quantiles"] = _entry("conditional_empirical_quantiles", "unique_train_target_empirical_quantiles",
    role="new_design_reference", training="train_only_distribution_fit", config=dict(minimum_support=200, timezone="Australia/Darwin", quantile_method="linear"),
    information="train_target_distribution+site+deterministic_target_local_month_hour", dependencies=("numpy", "pandas"))


def get_research_spec(model_id):
    if model_id == "softs_qr":
        raise ValueError("softs_qr is explicitly excluded from research baselines")
    if model_id not in _REGISTRY:
        raise ValueError(f"unknown research baseline: {model_id}")
    return deepcopy(_REGISTRY[model_id])


def research_baseline_registry():
    """Return detached identities, including explicit NOT_CONNECTED entries."""
    return deepcopy(_REGISTRY)
