"""Fixed candidate dispatch; inherited original fit, criterion, AMP and optimizer steps."""
import functools,hashlib,json,types
from pathlib import Path
import torch
from racepv.reliability_experiment import ReliabilityControlRaCEPV
from racepv.training.research_runtime import ResearchTrainer
from .model_adapter import DirectHistoryC,CANDIDATE_ID
from .dh_policy import validate_candidate
RUNTIME_SHA256='75aad40737f02984189cefb5a1d3b558b30a7fffa45e9477a0a68c77cb9e5e4c'
def qualify_candidate_method(fn: types.FunctionType) -> types.FunctionType:
    if fn not in (ResearchTrainer.predict_batch, ResearchTrainer._run_epoch):
        raise ValueError('only the two original candidate-dispatch methods may be qualified')
    source_path = Path(fn.__code__.co_filename).resolve()
    expected = Path(__import__('racepv').__file__).resolve().parent / 'training/research_runtime.py'
    if (source_path != expected.resolve()
            or hashlib.sha256(source_path.read_bytes()).hexdigest() != RUNTIME_SHA256):
        raise RuntimeError('ResearchTrainer source identity changed')
    if fn.__globals__['CANDIDATES'] != ('F', 'C', 'M1', 'M3'):
        raise RuntimeError('original candidate registration changed')
    namespace = dict(fn.__globals__)
    namespace['CANDIDATES'] += (CANDIDATE_ID,)
    qualified = types.FunctionType(fn.__code__, namespace, fn.__name__, fn.__defaults__, fn.__closure__)
    qualified.__kwdefaults__ = None if fn.__kwdefaults__ is None else dict(fn.__kwdefaults__)
    return functools.update_wrapper(qualified, fn)

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

class DirectHistoryTrainer(ResearchTrainer):
    predict_batch=qualify_candidate_method(ResearchTrainer.predict_batch)
    _run_epoch=qualify_candidate_method(ResearchTrainer._run_epoch)

def build(signature,row,*,backend='cuda-mamba'):
    from racepv.training.research_campaign import candidate_config
    validate_candidate(row)
    if backend not in ('cuda-mamba','cpu-gru'):raise ValueError('explicit backend')
    cfg=candidate_config(signature,synthetic_test=backend=='cpu-gru')
    model=(ReliabilityControlRaCEPV if row['architecture']=='C' else DirectHistoryC)(cfg)
    if (cfg.backbone,model.backbone_name)!=({'cuda-mamba':('mamba','mamba_ssm'),'cpu-gru':('gru','causal_conv_gru')}[backend]):raise ValueError('no fallback')
    shared={k:v for k,v in model.state_dict().items() if not k.startswith('direct_history_head.')}
    h=_canonical_state_sha256(shared)
    head={k:v for k,v in model.state_dict().items() if k.startswith('direct_history_head.')}
    if any(bool(v.count_nonzero()) for v in head.values()):raise ValueError('new head not zero')
    if row['architecture']!='C' and set(head)!={'direct_history_head.weight','direct_history_head.bias'}:raise ValueError('head identity')
    proof=dict(canonical_shared_sha256=h,control_shared_sha256=h,candidate_shared_sha256=h,
        control_full_sha256=h,candidate_full_sha256=_canonical_state_sha256(model.state_dict()),
        zero_head_verified=True,extra_parameters=0 if row['architecture']=='C' else 776,
        shared_tensor_count=len(shared),shared_buffer_count=len(list(model.named_buffers())),
        hash_schema='sorted_named_tensor_dtype_shape_raw_bytes.v1',
        proof_kind='zero_head_shared_state_no_rng.v1')
    expected=(207014 if backend=='cuda-mamba' else 191846)+(0 if row['architecture']=='C' else 776)
    if sum(p.numel() for p in model.parameters())!=expected:raise ValueError('parameter count')
    return model,model.make_objective(),proof

def trainer(model,objective,row,*,device,checkpoint_path=None,metadata=None):
    validate_candidate(row)
    if device not in ('cpu','cuda'):raise ValueError('explicit device')
    if (device=='cuda')!=(model.config.backbone=='mamba'):raise ValueError('backend mismatch')
    kind=ResearchTrainer if row['architecture']=='C' else DirectHistoryTrainer
    return kind(model,objective,model_id=row['architecture'],device=device,
        learning_rate=.001,weight_decay=.0001,max_epochs=1,patience=8,min_epochs=1,
        gradient_clip_norm=1.,mixed_precision=device=='cuda',separate_state_optimizer=True,
        selection_metric='daylight_qcrps',checkpoint_path=checkpoint_path,
        checkpoint_metadata=metadata)

