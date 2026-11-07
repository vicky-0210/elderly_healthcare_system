"""Compact five-second dense prediction heads from trailing optical context."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from wearable_ppg.preprocessing.optical import MORPH,MOD,QUALITY

PRIMARY=("cnn_transformer","tcn_gru","wavelet_ssm","quality_cross_attention")
CONTROLS=("morphology_only","modulation_only","fixed_fusion","quality_only","context_10","context_20","lag10")


class TemporalBlock(nn.Module):
    def __init__(self,w,d):
        super().__init__()
        self.filter=nn.Conv1d(w,w,3,dilation=d,groups=w)
        self.mix=nn.Conv1d(w,w,1)
        self.norm=nn.LayerNorm(w)
        self.pad=2*d

    def forward(self,x):
        h=self.filter(F.pad(x.transpose(1,2),(self.pad,0)))
        return self.norm(x+self.mix(F.silu(h)).transpose(1,2))


class SelectiveScan(nn.Module):
    def __init__(self,w):
        super().__init__()
        self.gains=nn.Linear(w,3*w); self.output=nn.Linear(w,w)

    def forward(self,x):
        u,dt,g=self.gains(x).chunk(3,-1)
        a=torch.exp(-F.softplus(dt-1)); h=(1-a)*u
        n=1
        while n<x.shape[1]:
            h=torch.cat((h[:,:n],h[:,n:]+a[:,n:]*h[:,:-n]),1)
            a=torch.cat((a[:,:n],a[:,n:]*a[:,:-n]),1); n*=2
        return x+self.output(h*g.sigmoid())


class RespiratoryNet(nn.Module):
    def __init__(self,name,w=32):
        super().__init__()
        self.name=name
        self.opt=nn.Linear(MORPH,w); self.mod=nn.Linear(len(MOD),w)
        self.quality=nn.Linear(len(QUALITY),w)
        self.fuse=nn.Linear(3*w,w)
        self.pulse_temporal=TemporalBlock(w,1)
        if name=="cnn_transformer":
            self.short=nn.Conv1d(w,w,3,padding=1); self.long=nn.Conv1d(w,w,7,padding=3)
            self.encoder=nn.TransformerEncoderLayer(w,4,2*w,.1,batch_first=True)
            pos=torch.arange(30)[:,None]*torch.exp(-torch.arange(0,w,2)*math.log(10000)/w)
            self.register_buffer("pos",torch.stack((pos.sin(),pos.cos()),-1).flatten(1))
        elif name=="tcn_gru":
            self.time=nn.Sequential(TemporalBlock(w,2),TemporalBlock(w,4)); self.gru=nn.GRU(w,w,batch_first=True)
        elif name=="wavelet_ssm":
            self.bands=nn.Linear(3*w,w); self.scan=SelectiveScan(w)
        else:
            self.slow=nn.Sequential(TemporalBlock(w,2),TemporalBlock(w,4),TemporalBlock(w,8))
            self.cross=nn.MultiheadAttention(w,4,batch_first=True,dropout=.05)
            self.gate=nn.Linear(3*w,w); self.norm=nn.LayerNorm(w)
        self.head=nn.Sequential(nn.LayerNorm(w),nn.Linear(w,w),nn.SiLU(),nn.Dropout(.15),nn.Linear(w,5))

    def forward(self,x):
        a=F.silu(self.opt(x[:,:,:MORPH])); b=F.silu(self.mod(x[:,:,MORPH:-len(QUALITY)]))
        q=F.silu(self.quality(x[:,:,-len(QUALITY):]))
        if self.name in PRIMARY[:3]:
            h=self.pulse_temporal(self.fuse(torch.cat((a,b,q),-1)))
            if self.name=="cnn_transformer":
                c=h.transpose(1,2); h=self.encoder(h+self.pos[:h.shape[1]]+F.silu(self.short(c)+self.long(c)).transpose(1,2))
            elif self.name=="tcn_gru": h=self.gru(self.time(h))[0]
            else:
                c=h.transpose(1,2)
                low=F.avg_pool1d(F.pad(c,(3,0),mode="replicate"),4,1)
                trend=F.avg_pool1d(F.pad(c,(7,0),mode="replicate"),8,1)
                h=self.scan(self.bands(torch.cat((c-low,low-trend,trend),1).transpose(1,2)))
            h=h[:,-5:]
        else:
            if self.name=="morphology_only": b=torch.zeros_like(b); q=torch.zeros_like(q)
            if self.name=="modulation_only": a=torch.zeros_like(a); q=torch.zeros_like(q)
            if self.name=="quality_only": a=torch.zeros_like(a); b=torch.zeros_like(b)
            pulse=self.pulse_temporal(a); slow=self.slow(b)
            query=slow[:,-5:]; anchor=pulse[:,-5:]
            context=self.cross(query,pulse,pulse,need_weights=False)[0]
            gate=.5 if self.name=="fixed_fusion" else self.gate(torch.cat((anchor,query,q[:,-5:]),-1)).sigmoid()
            h=self.norm(gate*(anchor+context)+(1-gate)*query+q[:,-5:])
        return self.head(h)
