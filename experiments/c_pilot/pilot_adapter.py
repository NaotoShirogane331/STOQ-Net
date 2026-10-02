"""Two explicit configuration changes; all numerical training code is original."""
from dataclasses import replace
from pilot_policy import validate_candidate


def build(signature, row):
    validate_candidate(row)
    from racepv.training.research_campaign import candidate_config
    from racepv.reliability_experiment import make_reliability_components
    cfg = replace(candidate_config(signature, synthetic_test=False),
                  router_temperature=row['router_temperature'])
    return make_reliability_components(cfg, 'control')


def trainer(model, objective, row, *, device, checkpoint_path=None, metadata=None):
    validate_candidate(row)
    from racepv.training.research_runtime import ResearchTrainer
    return ResearchTrainer(model, objective, model_id='C', device=device,
                           learning_rate=.001, weight_decay=row['weight_decay'],
                           max_epochs=1, min_epochs=1, patience=8, gradient_clip_norm=1.,
                           mixed_precision=True, separate_state_optimizer=True,
                           selection_metric='daylight_qcrps', checkpoint_path=checkpoint_path,
                           checkpoint_metadata=metadata or {})
