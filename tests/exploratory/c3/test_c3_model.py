import pathlib,sys,unittest,copy,ast
ROOT=pathlib.Path(__file__).resolve().parents[3] / 'src/racepv/exploratory/c3'
def make_fixture(batch=2,all_valid=True):
 import torch
 return dict(history_values=torch.randn(batch,192,14),history_mask=torch.full((batch,192,14),all_valid,dtype=torch.bool),future_known=torch.randn(batch,24,9),site_metadata=torch.randn(batch,6))
class ModelTests(unittest.TestCase):
 def setup_model(self):
  self.assertTrue((ROOT/'c3_model.py').exists(),'C3 model not implemented')
  import torch
  from racepv.exploratory.c3.c3_model import C3,C3Config
  torch.manual_seed(12);return C3(C3Config(14,9,6,0,1,0)).eval()
 def test_shape_order_finite_and_contract(self):
  m=self.setup_model();import torch
  x=make_fixture();o=m(**x)
  self.assertEqual(o['quantiles'].shape,(2,24,7));self.assertEqual(o['state_logits'].shape,(2,24,3))
  self.assertTrue(torch.isfinite(o['quantiles']).all());self.assertTrue((o['quantiles'][...,1:]>=o['quantiles'][...,:-1]).all())
  for key,val in [('future_weather',torch.zeros(1)),('future_weather_mask',torch.zeros(1)),('history_values',torch.zeros(2,191,14)),('history_mask',torch.full((2,192,14),.2)),('future_known',torch.full((2,24,9),float('nan'))),('site_metadata',torch.full((2,6),float('inf')))]:
   with self.assertRaises(ValueError):m(**dict(x,**{key:val}))
 def test_missingness_invariance(self):
  m=self.setup_model();import torch
  x=make_fixture();x['history_mask'][:,::2]=False
  a=m(**x)['quantiles'];y=copy.deepcopy(x);y['history_values'][:,::2]=float('nan')
  torch.testing.assert_close(a,m(**y)['quantiles'],rtol=0,atol=0)
  y['history_values'][:,::2]=float('inf');torch.testing.assert_close(a,m(**y)['quantiles'],rtol=0,atol=0)
 def test_all_missing_finite_and_observed_zero(self):
  m=self.setup_model();import torch
  x=make_fixture(all_valid=False);x['history_values'].fill_(float('nan'))
  self.assertTrue(torch.isfinite(m(**x)['quantiles']).all())
  x['history_values'].zero_();v,valid=m.sanitize_history(x['history_values'],x['history_mask']);self.assertFalse(valid.any())
  x['history_mask'].fill_(True);v,valid=m.sanitize_history(x['history_values'],x['history_mask']);self.assertTrue(valid.all())
 def test_state_gradient_isolation_and_partition(self):
  m=self.setup_model();o=m(**make_fixture());o['state_logits'].square().mean().backward()
  for n,p in m.named_parameters():self.assertEqual(p.grad is not None,n.startswith('state_head.'))
  from racepv.exploratory.c3 import c3_adapter
  from racepv.exploratory.c3.c3_policy import candidates
  t=c3_adapter.trainer(m,None,candidates()[1],device='cpu')
  groups=[{id(p) for g in opt.param_groups for p in g['params']} for opt in (t.forecast_optimizer,t.state_optimizer)]
  self.assertFalse(groups[0]&groups[1]);self.assertEqual(groups[0]|groups[1],{id(p) for p in m.parameters()})
 def test_explicit_padding_not_causal(self):
  self.setup_model();import torch
  from racepv.exploratory.c3.timexer_core import FullAttention
  from types import SimpleNamespace
  a=FullAttention(True,attention_dropout=0,output_attention=True)
  q=torch.zeros(1,3,1,2);v=torch.arange(3.).view(1,3,1,1).expand(-1,-1,-1,2).clone()
  mask=SimpleNamespace(mask=torch.zeros(1,1,3,3,dtype=torch.bool))
  out,w=a(q,q,v,mask);self.assertGreater(w[0,0,0,2].item(),0)
  mask.mask[...,2]=True
  out,_=a(q,q,v,mask);v[:,2]=1e4
  torch.testing.assert_close(out,a(q,q,v,mask)[0])
  with self.assertRaises(ValueError):a(q,q,v,None)
 def test_official_core_parity(self):
  self.setup_model();import torch,numpy,math
  from racepv.exploratory.c3 import timexer_core as core
  from types import SimpleNamespace
  ns=dict(torch=torch,nn=torch.nn,F=torch.nn.functional,np=numpy,math=math,sqrt=math.sqrt)
  for file,names in [('layers/Embed.py',('PositionalEmbedding',)),('layers/SelfAttention_Family.py',('FullAttention','AttentionLayer')),('models/TimeXer.py',('Encoder','EncoderLayer','FlattenHead'))]:
   tree=ast.parse((ROOT/'vendor/timexer_upstream'/file).read_text())
   for node in tree.body:
    if isinstance(node,ast.ClassDef) and node.name in names:exec(compile(ast.Module(body=[node],type_ignores=[]),file,'exec'),ns)
  def build(d):
   return d['Encoder']([d['EncoderLayer'](d['AttentionLayer'](d['FullAttention'](True,attention_dropout=0),64,4),d['AttentionLayer'](d['FullAttention'](True,attention_dropout=0),64,4),64,128,0,'gelu') for _ in range(2)],torch.nn.LayerNorm(64)).eval()
  a=build(vars(core));b=build(ns);b.load_state_dict(a.state_dict())
  x=torch.randn(2,13,64);cross=torch.randn(2,14,64)
  xm=SimpleNamespace(mask=torch.zeros(2,1,13,13,dtype=torch.bool));cm=SimpleNamespace(mask=torch.zeros(2,1,1,14,dtype=torch.bool))
  torch.testing.assert_close(a(x,cross,xm,cm),b(x,cross,xm,cm),rtol=0,atol=0)
if __name__=='__main__':unittest.main()
