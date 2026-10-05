"""Solar-reference conditioned latent multi-resolution C2, not old-C experts."""
from dataclasses import dataclass,asdict
import torch
from torch import nn
import torch.nn.functional as F
from racepv.models.components import OrderedQuantileHead

@dataclass(frozen=True)
class C2Config:
    history_features:int
    future_features:int
    site_features:int
    power_index:int=0
    history_clear_sky_index:int=1
    future_clear_sky_index:int=0
    variant:str='multiscale'
    history_length:int=192
    horizon:int=24
    quantiles:tuple=(.05,.10,.25,.50,.75,.90,.95)
    def __post_init__(self):
        if self.variant not in ('multiscale','single_scale'):raise ValueError('fixed variant')
        if self.history_length!=192 or self.horizon!=24 or self.quantiles!=(.05,.10,.25,.50,.75,.90,.95):raise ValueError('fixed shape/quantiles')
        if not 0<=self.power_index<self.history_features or not 0<=self.history_clear_sky_index<self.history_features:raise ValueError('history indices')
        if self.power_index==self.history_clear_sky_index or not 0<=self.future_clear_sky_index<self.future_features:raise ValueError('solar index')

def masked_pool(values,valid,scale):
    if values.ndim!=3 or valid.shape!=values.shape or scale not in (1,2,4) or values.shape[1]%scale:raise ValueError('unpadding aligned mask/pool contract')
    b,l,c=values.shape
    mask=valid.bool()&torch.isfinite(values)
    x=torch.where(mask,values,0.).reshape(b,l//scale,scale,c)
    m=mask.reshape(b,l//scale,scale,c).to(values.dtype)
    n=m.sum(2)
    return x.sum(2)/n.clamp_min(1),n/scale

class TemporalBlock(nn.Module):
    def __init__(self,dilation):
        super().__init__();self.dilation=dilation
        self.norm=nn.LayerNorm(64);self.conv=nn.Conv1d(64,64,5,dilation=dilation,groups=64)
        self.mlp=nn.Sequential(nn.Linear(64,128),nn.GELU(),nn.Linear(128,64))
        self.drop=nn.Dropout(.1)
    def forward(self,x):
        z=self.norm(x).transpose(1,2)
        z=self.conv(F.pad(z,(4*self.dilation,0))).transpose(1,2)
        return x+self.drop(self.mlp(z))

class Branch(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj=nn.Linear(4,64)
        self.blocks=nn.Sequential(TemporalBlock(1),TemporalBlock(2))
    def forward(self,x,coverage):
        return self.blocks(self.proj(torch.cat((x,coverage),-1)))

class C2(nn.Module):
    def __init__(self,config):
        super().__init__();self.config=config
        self.scales=(1,2,4) if config.variant=='multiscale' else (1,1,1)
        self.covariate_indices=tuple(i for i in range(config.history_features) if i!=config.power_index)
        self.branches=nn.ModuleList([Branch() for _ in range(3)])
        self.covariate_projection=nn.Linear(384,64)
        self.variable_embedding=nn.Parameter(torch.empty(len(self.covariate_indices),64))
        nn.init.normal_(self.variable_embedding,std=.02)
        self.covariate_attention=nn.MultiheadAttention(64,4,dropout=.1,batch_first=True)
        self.covariate_norm=nn.LayerNorm(64)
        self.future_projection=nn.Linear(config.future_features+config.site_features+1,64)
        self.future_attention=nn.MultiheadAttention(64,4,dropout=.1,batch_first=True)
        self.future_norm=nn.LayerNorm(64)
        self.fusion=nn.Sequential(nn.Linear(192,64),nn.GELU(),nn.LayerNorm(64))
        self.quantile_head=OrderedQuantileHead(64,config.quantiles,.05)
        self.state_head=nn.Sequential(nn.Linear(64,64),nn.GELU(),nn.Linear(64,3))
    @property
    def model_metadata(self):
        return dict(model_id='C2',model_revision='solar-conditioned-latent-multiscale-v1',variant=self.config.variant,
            scales=list(self.scales),prediction_head='single_ordered_quantile',active_experts=None,
            state_representation='detached_fused_latent',forecast_selection_metric='daylight_qcrps',
            config=asdict(self.config),formal_ready=False)
    def target_inputs(self,values,mask):
        p=values[...,self.config.power_index];b=values[...,self.config.history_clear_sky_index]
        mp=mask[...,self.config.power_index].bool()&torch.isfinite(p)
        mb=mask[...,self.config.history_clear_sky_index].bool()&torch.isfinite(b)
        mr=mp&mb
        return torch.stack((torch.where(mp,p,0.),torch.where(mr,p-b,0.)),-1),torch.stack((mp,mr),-1)
    def forward(self,history_values,history_mask,future_known,site_metadata,future_weather=None,future_weather_mask=None):
        c=self.config
        if future_weather is not None or future_weather_mask is not None:raise ValueError('future weather forbidden in C2 pilot')
        b=history_values.shape[0]
        if history_values.shape!=(b,192,c.history_features) or history_mask.shape!=history_values.shape:raise ValueError('history shape')
        if future_known.shape!=(b,24,c.future_features) or site_metadata.shape!=(b,c.site_features):raise ValueError('future/site shape')
        if not bool(((history_mask==0)|(history_mask==1)).all()):raise ValueError('binary history mask')
        mask=history_mask.bool()&torch.isfinite(history_values)
        x=torch.where(mask,history_values,0.)
        future=torch.nan_to_num(future_known,nan=0.,posinf=0.,neginf=0.)
        site=torch.nan_to_num(site_metadata,nan=0.,posinf=0.,neginf=0.)
        tv,tm=self.target_inputs(x,mask)
        cov=x[...,self.covariate_indices].transpose(1,2)
        cm=mask[...,self.covariate_indices].transpose(1,2)
        tokens=self.covariate_projection(torch.cat((cov,cm.to(cov.dtype)),-1))+self.variable_embedding
        tokens=torch.cat((tokens,torch.zeros(b,1,64,device=x.device,dtype=tokens.dtype)),1)
        padding=torch.cat((~cm.any(-1),torch.zeros(b,1,device=x.device,dtype=torch.bool)),1)
        lead=torch.arange(1,25,device=x.device,dtype=x.dtype).view(1,24,1).expand(b,-1,-1)/24
        query=self.future_projection(torch.cat((future,site[:,None,:].expand(-1,24,-1),lead),-1))
        decoded=[]
        for scale,branch in zip(self.scales,self.branches):
            pooled,coverage=masked_pool(tv,tm,scale)
            hidden=branch(pooled,coverage)
            z,_=self.covariate_attention(hidden,tokens,tokens,key_padding_mask=padding,need_weights=False)
            hidden=self.covariate_norm(hidden+z)
            z,_=self.future_attention(query,hidden,hidden,need_weights=False)
            decoded.append(self.future_norm(query+z))
        fused=self.fusion(torch.cat(decoded,-1))
        return dict(quantiles=self.quantile_head(fused,future[...,c.future_clear_sky_index]),
                    state_logits=self.state_head(fused.detach()))
