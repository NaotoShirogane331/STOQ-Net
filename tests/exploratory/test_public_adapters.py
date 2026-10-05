"""Published entry points, synthetic tensors only; no experiment dataset reads."""
import json
from pathlib import Path
import sys
import pytest
import torch
from racepv.exploratory.c3 import c3_adapter
from racepv.training.research_runtime import ResearchTrainer

def signature():
    return c3_adapter.load_signature_for_tests()

def batch():
    from .direct_history.fixtures import make_batch
    return make_batch()

@pytest.mark.parametrize('name', ['c2','c3','location_gap','direct_history'])
def test_build_trainer_and_real_synthetic_step(name):
    from racepv.exploratory.c2 import c2_adapter, cf_policy
    from racepv.exploratory.c3 import c3_policy
    from racepv.exploratory.location_gap import lg_adapter, lg_policy
    from racepv.exploratory.direct_history import dh_adapter, dh_policy
    module, policy = {'c2': (c2_adapter,cf_policy), 'c3': (c3_adapter,c3_policy),
        'location_gap': (lg_adapter,lg_policy), 'direct_history': (dh_adapter,dh_policy)}[name]
    row = policy.candidates()[0 if name == 'c2' else 1]
    torch.manual_seed(29)
    kwargs = {} if name in ('c2','c3') else {'backend': 'cpu-gru'}
    model, objective, *_ = module.build(signature(), row, **kwargs)
    runner = module.trainer(model, objective, row, device='cpu')
    assert isinstance(runner, ResearchTrainer)
    output = runner.predict_batch(batch())
    assert output['quantiles'].shape == (2,24,7)
    assert torch.isfinite(output['quantiles']).all()
    assert (output['quantiles'].diff(dim=-1) >= 0).all()
    result = runner._run_epoch([batch()], True)
    assert result['valid_count'] > 0
    assert torch.isfinite(torch.tensor(result['daylight_qcrps']))
    with pytest.raises(ValueError):
        module.build(signature(), dict(row,seed=999), **kwargs)

def test_windows_test_lock_shim_fails_closed():
    if sys.platform != 'win32':
        pytest.skip('Windows-only import shim, native Linux locks are unchanged')
    import fcntl
    with pytest.raises(RuntimeError, match='outside Windows synthetic tests'):
        fcntl.flock(None, fcntl.LOCK_EX)

def test_direct_history_dispatch_does_not_mutate_original():
    from racepv.exploratory.direct_history.dh_adapter import DirectHistoryTrainer
    assert DirectHistoryTrainer.predict_batch.__code__ is ResearchTrainer.predict_batch.__code__
    assert DirectHistoryTrainer._run_epoch.__code__ is ResearchTrainer._run_epoch.__code__
    assert ResearchTrainer.predict_batch.__globals__['CANDIDATES'] == ('F','C','M1','M3')
    assert DirectHistoryTrainer.predict_batch.__globals__['CANDIDATES'] == ('F','C','M1','M3','C-DH-linear')
