"""New C2 behavioral acceptance; synthetic only, no dataset loading."""
import importlib
from pathlib import Path
import sys
import unittest
import torch
ROOT=Path(__file__).resolve().parents[3] / 'src/racepv/exploratory/c2'

class C2Tests(unittest.TestCase):
    def setUp(self):
        self.assertTrue((ROOT/'c2_model.py').exists(),'C2 implementation missing')
        self.m=importlib.import_module('racepv.exploratory.c2.c2_model')
        torch.set_num_threads(1)
    def build(self,variant='multiscale'):
        return self.m.C2(self.m.C2Config(history_features=14,future_features=9,site_features=6,
            power_index=0,history_clear_sky_index=1,future_clear_sky_index=0,variant=variant))
    def inputs(self,b=2):
        x=torch.randn(b,192,14)*.1+.3
        mask=torch.ones_like(x,dtype=torch.bool)
        future=torch.randn(b,24,9)*.1;future[...,0]=.4
        return x,mask,future,torch.zeros(b,6)
    def test_masked_pool_does_not_treat_missing_as_zero(self):
        x=torch.tensor([[[2.,8.],[100.,10.],[7.,12.],[9.,14.]]])
        mask=torch.tensor([[[1,0],[0,0],[1,1],[1,0]]],dtype=torch.bool)
        y,c=self.m.masked_pool(x,mask,2)
        torch.testing.assert_close(y,torch.tensor([[[2.,0.],[8.,12.]]]))
        torch.testing.assert_close(c,torch.tensor([[[.5,0.],[1.,.5]]]))
    def test_missing_power_does_not_create_negative_solar_residual(self):
        model=self.build();x,mask,f,s=self.inputs()
        mask[...,0]=False;x[...,0]=999
        values,valid=model.target_inputs(x,mask)
        self.assertEqual(values.count_nonzero().item(),0)
        self.assertFalse(valid.any())
    def test_allmissing_is_finite_ordered_and_no_false_experts(self):
        model=self.build().eval();x,m,f,s=self.inputs();m[:]=False;x[:]=float('nan')
        out=model(x,m,f,s);q=out['quantiles']
        self.assertEqual(tuple(q.shape),(2,24,7))
        self.assertEqual(tuple(out['state_logits'].shape),(2,24,3))
        self.assertTrue(torch.isfinite(q).all());self.assertTrue((q[...,1:]>=q[...,:-1]).all())
        self.assertIsNone(model.model_metadata.get('active_experts'))
        self.assertNotIn('gate_weights',out)
    def test_paired_parameters_exact_and_all_branches_trainable(self):
        torch.manual_seed(29);a=self.build()
        torch.manual_seed(29);b=self.build('single_scale')
        self.assertEqual(list(a.state_dict()),list(b.state_dict()))
        for k in a.state_dict():torch.testing.assert_close(a.state_dict()[k],b.state_dict()[k],rtol=0,atol=0)
        x,m,f,s=self.inputs();out=a(x,m,f,s)
        out['quantiles'].square().mean().backward()
        for i,branch in enumerate(a.branches):
            self.assertTrue(any(p.grad is not None and p.grad.abs().sum()>0 for p in branch.parameters()),str(i))
        self.assertTrue(all(p.grad is None for p in a.state_head.parameters()))
        a.zero_grad(set_to_none=True);a(x,m,f,s)['state_logits'].square().mean().backward()
        state={id(p) for p in a.state_head.parameters()}
        self.assertTrue(all(p.grad is None for p in a.parameters() if id(p) not in state))
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum()>0 for p in a.state_head.parameters()))
    def test_reject_future_weather_and_bad_mask(self):
        model=self.build();x,m,f,s=self.inputs()
        with self.assertRaises(ValueError):model(x,m,f,s,future_weather=torch.zeros(2,24,7))
        with self.assertRaises(ValueError):model(x,m.float()+.1,f,s)
    def test_adapter_preserves_original_score_and_optimizer_partition(self):
        self.assertTrue((ROOT/'c2_adapter.py').exists(),'C2 adapter missing')
        from racepv.exploratory.c2.c2_adapter import C2ResearchTrainer
        from racepv.reliability_a_loss import ReliabilityAObjective
        from racepv.training.research_runtime import ResearchTrainer
        x,m,f,s=self.inputs();batch=dict(history_values=x,history_mask=m.float(),future_known=f,site_metadata=s,
            targets=torch.full((2,24),.3),state_labels=torch.zeros(2,24,dtype=torch.long),
            target_mask=torch.ones(2,24,dtype=torch.bool),daylight_mask=torch.ones(2,24,dtype=torch.bool))
        batch['daylight_mask'][:,0]=False
        model=self.build();kw=dict(device='cpu',learning_rate=.001,weight_decay=.0001,max_epochs=1,
            min_epochs=1,patience=8,separate_state_optimizer=True,mixed_precision=False,selection_metric='daylight_qcrps')
        objective=ReliabilityAObjective(model.config.quantiles)
        t=C2ResearchTrainer(model,objective,**kw)
        old=ResearchTrainer(model,objective,model_id='C',**kw)
        model.eval();new=t.predict_batch(batch);original=old.predict_batch(batch)
        torch.testing.assert_close(new['quantiles'],original['quantiles'],rtol=0,atol=0)
        self.assertEqual(new['quantiles'][:,0].count_nonzero().item(),0)
        self.assertTrue(((new['quantiles']>=0)&(new['quantiles']<=1.05)).all())
        a=t._run_epoch([batch],False);b=old._run_epoch([batch],False)
        for k in ('loss','daylight_qcrps','valid_count'):self.assertAlmostEqual(a[k],b[k],places=7)
        self.assertEqual(t.model_id,'C2')
        fg={id(p) for p in t.forecast_parameters};sg={id(p) for p in model.state_head.parameters()}
        self.assertFalse(fg&sg);self.assertEqual(fg|sg,{id(p) for p in model.parameters() if p.requires_grad})

if __name__=='__main__':unittest.main()
