"""TimeXer-PV task adaptation; upstream numerical encoder classes unchanged."""
from dataclasses import dataclass
from types import SimpleNamespace
import torch
from torch import nn
from .timexer_core import Encoder,EncoderLayer,FullAttention,AttentionLayer,PositionalEmbedding,FlattenHead
from racepv.models.components import OrderedQuantileHead
@dataclass(frozen=True)
class C3Config:
    history_features:int
    future_features:int
    site_features:int
    power_index:int
    history_clear_sky_index:int
    future_clear_sky_index:int
    lookback:int=192
    horizon:int=24
    patch_len:int=16
    hidden_size:int=64
    layers:int=2
    heads:int=4
    d_ff:int=128
    dropout:float=.1
    latent_per_step:int=16
    quantiles:tuple=(.05,.1,.25,.5,.75,.9,.95)
    def __post_init__(self):
        if tuple(vars(self).values())!=(14,9,6,0,1,0,192,24,16,64,2,4,128,.1,16,(.05,.1,.25,.5,.75,.9,.95)):
            raise ValueError('fixed C3 configuration')
class C3(nn.Module):
    model_metadata={'model_id':'C3-TimeXer-PV','source':'Time-Series-Library pinned MIT core','use_norm':False,'state_detached':True}
    def __init__(self,config):
        super().__init__();self.config=config;c=config
        self.patch_value=nn.Linear(16,64,bias=False);self.patch_mask=nn.Linear(16,64,bias=False)
        nn.init.zeros_(self.patch_mask.weight)
        self.global_token=nn.Parameter(torch.randn(1,1,64))
        self.position=PositionalEmbedding(64);self.dropout=nn.Dropout(.1)
        self.variable_value=nn.Linear(192,64);self.variable_mask=nn.Linear(192,64,bias=False)
        nn.init.zeros_(self.variable_mask.weight)
        self.variable_id=nn.Parameter(torch.empty(1,13,64));nn.init.normal_(self.variable_id,std=.02)
        self.encoder=Encoder([EncoderLayer(
            AttentionLayer(FullAttention(True,attention_dropout=.1),64,4),
            AttentionLayer(FullAttention(True,attention_dropout=.1),64,4),
            64,128,.1,'gelu') for _ in range(2)],nn.LayerNorm(64))
        self.flatten_head=FlattenHead(1,64*13,24*16,.1)
        self.condition=nn.Sequential(nn.Linear(31,64),nn.GELU(),nn.LayerNorm(64))
        self.quantile_head=OrderedQuantileHead(64,c.quantiles,.05)
        self.state_head=nn.Sequential(nn.Linear(64,64),nn.GELU(),nn.Linear(64,3))
    @staticmethod
    def sanitize_history(values,mask):
        if not bool(((mask==0)|(mask==1)).all()):raise ValueError('binary history mask required')
        valid=mask.bool()&torch.isfinite(values)
        return torch.where(valid,values,0.),valid
    def forward(self,history_values,history_mask,future_known,site_metadata,future_weather=None,future_weather_mask=None):
        if future_weather is not None or future_weather_mask is not None:raise ValueError('future observed weather forbidden')
        b=history_values.shape[0]
        if (history_values.shape!=(b,192,14) or history_mask.shape!=history_values.shape or
            future_known.shape!=(b,24,9) or site_metadata.shape!=(b,6)):
            raise ValueError('C3 fixed input shape')
        if not bool(torch.isfinite(future_known).all()) or not bool(torch.isfinite(site_metadata).all()):
            raise ValueError('nonfinite known future/site data')
        values,valid=self.sanitize_history(history_values,history_mask)
        pv=values[:,:,0].unfold(1,16,16);pm=valid[:,:,0].unfold(1,16,16)
        patch=self.patch_value(pv)+self.patch_mask(pm.to(pv.dtype))+self.position(pv)
        patch_valid=pm.any(-1)
        target=self.dropout(torch.cat([patch,self.global_token.expand(b,-1,-1)],dim=1))
        vv=values[:,:,1:].transpose(1,2);vm=valid[:,:,1:].transpose(1,2)
        variables=self.dropout(self.variable_value(vv)+self.variable_mask(vm.to(vv.dtype))+self.variable_id)
        variables=torch.cat([variables,torch.zeros(b,1,64,device=variables.device,dtype=variables.dtype)],dim=1)
        target_valid=torch.cat([patch_valid,torch.ones(b,1,device=valid.device,dtype=torch.bool)],dim=1)
        variable_valid=torch.cat([vm.any(-1),torch.ones(b,1,device=valid.device,dtype=torch.bool)],dim=1)
        xm=SimpleNamespace(mask=(~target_valid)[:,None,None,:].expand(b,1,13,13))
        cm=SimpleNamespace(mask=(~variable_valid)[:,None,None,:])
        encoded=self.encoder(target,variables,xm,cm)
        encoded=encoded*target_valid[:,:,None].to(encoded.dtype)
        latent=self.flatten_head(encoded.transpose(1,2).unsqueeze(1)).reshape(b,24,16)
        hidden=self.condition(torch.cat([latent,future_known,site_metadata[:,None,:].expand(b,24,6)],dim=-1))
        return dict(quantiles=self.quantile_head(hidden,future_known[:,:,0]),state_logits=self.state_head(hidden.detach()))
