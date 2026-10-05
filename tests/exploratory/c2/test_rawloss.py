"""Regression target: loss-before-clamp dead zone, using real C2 and trainer."""
import sys
import unittest
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[3] / 'src/racepv/exploratory/c2'
from racepv.exploratory.c2.c2_model import C2, C2Config
from racepv.exploratory.c2.c2_adapter import C2ResearchTrainer
from racepv.reliability_a_loss import ReliabilityAObjective
from racepv.training.research_runtime import ResearchTrainer


class RawLossTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(29)

    def setup_case(self, correction=0., device='cpu', amp=False):
        model = C2(C2Config(history_features=14, future_features=9, site_features=6))
        with torch.no_grad():
            model.quantile_head.projection.weight.zero_()
            model.quantile_head.projection.bias[0] = correction
            model.quantile_head.projection.bias[1:].fill_(-2.)
        batch = dict(history_values=torch.full((2,192,14), .3),
            history_mask=torch.ones(2,192,14), future_known=torch.zeros(2,24,9),
            site_metadata=torch.zeros(2,6), targets=torch.full((2,24), .3),
            state_labels=torch.zeros(2,24,dtype=torch.long),
            target_mask=torch.ones(2,24,dtype=torch.bool), daylight_mask=torch.ones(2,24,dtype=torch.bool))
        batch['future_known'][...,0] = .4
        batch['daylight_mask'][:,0] = False
        batch['target_mask'][:,1] = False
        batch['target_mask'][:,2] = False
        batch['targets'][:,2] = float('nan')
        kwargs=dict(device=device,learning_rate=.001,weight_decay=.0001,max_epochs=1,min_epochs=1,
            patience=8,separate_state_optimizer=True,mixed_precision=amp,selection_metric='daylight_qcrps')
        objective=ReliabilityAObjective(model.config.quantiles)
        trainer=C2ResearchTrainer(model,objective,**kwargs)
        return model,trainer,batch,objective,kwargs

    def recovery(self,device='cpu',amp=False):
        self.amp_evidence=[]
        for correction,sign in ((-3.,-1.),(3.,1.)):
            with self.subTest(correction=correction,device=device):
                model,t,batch,_,_=self.setup_case(correction,device,amp)
                median=model.quantile_head.projection.bias
                before=float(median[0].detach());grads=[];calls=[]
                hook=t.forecast_optimizer.register_step_pre_hook(lambda opt,args,kwargs:grads.append(float(median.grad[0])))
                forward=model.register_forward_hook(lambda *args:calls.append(1))
                attempts=[]
                try:
                    for attempt in range(4 if amp else 1):
                        scale=t.scaler.get_scale()
                        forecast_before=[p.detach().clone() for p in t.forecast_parameters]
                        metrics=t._run_epoch([batch],True)
                        attempts.append(dict(scale_before=scale,scale_after=t.scaler.get_scale(),forecast_steps=len(grads)))
                        self.assertEqual(len(calls),attempt+1,'exactly one forward per attempted optimizer step')
                        if grads:break
                        self.assertTrue(amp,'CPU step cannot be skipped')
                        self.assertEqual(t.scaler.get_scale(),scale/2,'skip must be the original dynamic overflow protection')
                        for p,old in zip(t.forecast_parameters,forecast_before):
                            torch.testing.assert_close(p,old,rtol=0,atol=0)
                finally:hook.remove();forward.remove()
                self.assertEqual(len(grads),1)
                self.assertTrue(torch.isfinite(torch.tensor(grads)).all())
                self.assertGreater(sign*grads[0],1e-6,'out-of-range forecast needs a correctly directed nonzero gradient')
                self.assertGreater(sign*(before-float(median[0].detach())),1e-5,'real optimizer step must move toward the target, not merely weight decay')
                self.assertEqual(metrics['valid_count'],42.)
                self.assertTrue(all(torch.isfinite(p).all() for p in model.parameters()))
                self.amp_evidence.append(dict(correction=correction,attempts=attempts,gradient=grads[0],
                    bias_before=before,bias_after=float(median[0].detach()),production_scaler_unchanged=True))

    def test_negative_and_over_cap_forecasts_recover_cpu(self):
        self.recovery()

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA-specific witness is run in bounded GPU probe')
    def test_negative_and_over_cap_forecasts_recover_amp(self):
        self.recovery('cuda',True)

    def test_raw_graph_and_projected_interface_are_both_preserved(self):
        model,t,batch,obj,_=self.setup_case(-3.)
        model.eval();output=t.predict_batch(batch)
        self.assertIn('raw_quantiles',output,'raw training tensor must retain its graph')
        self.assertTrue(output['raw_quantiles'].requires_grad)
        self.assertEqual(output['quantiles'].count_nonzero().item(),0)
        raw=dict(output,quantiles=output['raw_quantiles'])
        _,terms=t._criterion(raw,batch)
        raw['quantiles'].retain_grad();terms['forecast_loss'].backward()
        legal=batch['target_mask']&batch['daylight_mask']&torch.isfinite(batch['targets'])
        self.assertTrue((raw['quantiles'].grad[legal]<0).all())
        self.assertEqual(raw['quantiles'].grad[~legal].count_nonzero().item(),0)
        self.assertTrue(all(p.grad is None for p in model.state_head.parameters()))
        model.zero_grad(set_to_none=True)
        output=t.predict_batch(batch);_,terms=t._criterion(dict(output,quantiles=output['raw_quantiles']),batch)
        terms['state_loss'].backward()
        state={id(p) for p in model.state_head.parameters()}
        self.assertTrue(all(p.grad is None for p in model.parameters() if id(p) not in state))

    def test_in_range_loss_and_all_validation_metrics_match_original(self):
        for correction in (0.,-3.,3.):
            with self.subTest(correction=correction):
                model,t,batch,obj,kwargs=self.setup_case(correction)
                old=ResearchTrainer(model,obj,model_id='C',**kwargs)
                model.eval();output=t.predict_batch(batch);previous=old.predict_batch(batch)
                torch.testing.assert_close(output['quantiles'],previous['quantiles'],rtol=0,atol=0)
                a=t._run_epoch([batch],False);b=old._run_epoch([batch],False)
                for key in ('loss','daylight_qcrps','valid_count'):self.assertAlmostEqual(a[key],b[key],places=7)
                self.assertEqual(a['valid_count'],42.)
                if correction==0.:
                    self.assertIn('raw_quantiles',output)
                    _,raw=t._criterion(dict(output,quantiles=output['raw_quantiles']),batch)
                    _,projected=t._criterion(output,batch)
                    torch.testing.assert_close(raw['forecast_loss'],projected['forecast_loss'],rtol=0,atol=0)


if __name__=='__main__':unittest.main(verbosity=2)
