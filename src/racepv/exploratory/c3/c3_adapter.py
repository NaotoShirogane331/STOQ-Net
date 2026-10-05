"""Narrow isolated candidate path; preserve original criterion, score and steps."""
import time
import torch
from racepv.training.research_runtime import ResearchTrainer,_model_inputs,masked_research_loss
from racepv.training.research_campaign import candidate_config
from racepv.reliability_a_loss import ReliabilityAObjective
from .c3_model import C3,C3Config

LOSS_PROTOCOL = dict(version='C3-rawloss-v1',training_forecast='original_qCRPS_on_raw_ordered_quantiles',
    evaluation='original_clamp_0_1.05_and_night_zero',selection='projected_validation_daylight_qcrps',
    state='original_detached_state_CE',extra_model_forward=False)

class C3ResearchTrainer(ResearchTrainer):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,model_id='C3',**kwargs)
    def predict_batch(self,batch):
        mask=batch['history_mask']
        if not bool(((mask==0)|(mask==1)).all()):raise ValueError('history mask is not binary')
        batch=dict(batch,history_mask=mask.bool());inputs=_model_inputs(batch)
        inputs['history_values']=torch.where(inputs['history_mask'].bool(),inputs['history_values'],0.)
        output=self.model(**inputs);q=output['quantiles']
        if q.shape!=(*batch['targets'].shape,7) or not bool(torch.isfinite(q).all()):raise ValueError('invalid C3 quantiles')
        if bool((q[...,1:]<q[...,:-1]).any()):raise ValueError('C3 quantile order')
        state=output['state_logits']
        if state.shape!=(*batch['targets'].shape,3) or not bool(torch.isfinite(state).all()):raise ValueError('invalid C3 state logits')
        output['raw_quantiles']=q
        output['quantiles']=torch.where(batch['daylight_mask'].bool()[...,None],q.clamp(0.,1.05),0.)
        return output
    def _run_epoch(self,loader,training):
        started=time.perf_counter();self.model.train(training);total=score=count=0
        raw_forecast_total=raw_objective_total=0.
        context=torch.enable_grad() if training else torch.inference_mode()
        with context:
            for raw_batch in loader:
                batch=self._move(raw_batch)
                with torch.autocast(device_type=self.device.type,dtype=torch.float16,enabled=self.use_amp):
                    output=self.predict_batch(batch)
                    result=masked_research_loss(output['quantiles'],batch,model_id='tcn_qr')
                    if result['skip']:continue
                    raw_loss,raw_terms=self._criterion(dict(output,quantiles=output['raw_quantiles']),batch)
                    loss,terms=(raw_loss,raw_terms) if training else self._criterion(output,batch)
                if not bool(torch.isfinite(loss)):raise FloatingPointError('non-finite C3 loss')
                if not bool(torch.isfinite(raw_loss)):raise FloatingPointError('non-finite C3 raw loss')
                if training:self._step_separated_losses(forecast_loss=terms['forecast_loss'],state_loss=terms['state_loss'])
                n=result['valid_count'];total+=float(loss.detach())*n;score+=float(result['daylight_qcrps'].detach())*n;count+=n
                raw_forecast_total+=float(raw_terms['forecast_loss'].detach())*n
                raw_objective_total+=float(raw_loss.detach())*n
        if not count:raise ValueError('no legal daylight target population')
        return dict(loss=total/count,daylight_qcrps=score/count,valid_count=float(count),
            raw_forecast_qcrps=raw_forecast_total/count,raw_objective_loss=raw_objective_total/count,
            elapsed_seconds=time.perf_counter()-started)

HISTORY=['power_norm','clear_sky_power_norm','clear_sky_residual','power_diff_1','power_diff_2','rolling_volatility_1h','rolling_volatility_3h','clear_sky_index','power_observed','power_imputed','ghi_z','temperature_z','humidity_z','wind_speed_z']
FUTURE=['clear_sky_power_norm','solar_elevation','solar_azimuth_sin','solar_azimuth_cos','hour_sin','hour_cos','day_of_year_sin','day_of_year_cos','lead_time_hours']
SITE=['latitude_scaled','longitude_scaled','log_capacity_kw_scaled','surface_tilt_scaled','surface_azimuth_scaled','has_geometry']
def load_signature_for_tests():
    import json
    from pathlib import Path
    return json.loads((Path(__file__).resolve().parent/'source-manifest.json').read_bytes())['feature_signature']
def build(signature,row):
    from .c3_policy import validate_candidate
    validate_candidate(row)
    if signature['history_columns']!=HISTORY or signature['future_known_columns']!=FUTURE or signature['site_metadata_names']!=SITE:
        raise ValueError('fixed feature order')
    original=candidate_config(signature,synthetic_test=False)
    if row['architecture']=='C':
        from racepv.reliability_experiment import make_reliability_components
        return make_reliability_components(original,'control')
    c=C3Config(**{k:getattr(original,k) for k in ('history_features','future_features','site_features','power_index','history_clear_sky_index','future_clear_sky_index')})
    return C3(c),ReliabilityAObjective(c.quantiles)
def trainer(model,objective,row,*,device,checkpoint_path=None,metadata=None):
    from .c3_policy import validate_candidate
    validate_candidate(row)
    klass=C3ResearchTrainer if row['architecture']=='C3' else ResearchTrainer
    kw={} if row['architecture']=='C3' else {'model_id':'C'}
    return klass(model,objective,device=device,learning_rate=.001,weight_decay=.0001,
        max_epochs=1,min_epochs=1,patience=8,gradient_clip_norm=1.,mixed_precision=True,separate_state_optimizer=True,
        selection_metric='daylight_qcrps',checkpoint_path=checkpoint_path,checkpoint_metadata=metadata or {},**kw)
