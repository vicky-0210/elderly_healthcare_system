"""Temporal morphology encoders and optional privileged lead attention."""
import torch
from torch import nn
from torch.nn import functional as F


class Residual(nn.Module):
    def __init__(self, width, dilation):
        super().__init__()
        self.paths = nn.ModuleList(nn.Conv1d(width, width//2, k, padding=dilation*(k//2),
                                             dilation=dilation) for k in (3, 7))
        self.post = nn.Sequential(nn.BatchNorm1d(width), nn.SiLU(), nn.Conv1d(width,width,1))
    def forward(self,x):
        return F.silu(x+self.post(torch.cat([branch(x) for branch in self.paths],1)))


class MorphNet(nn.Module):
    def __init__(self,c):
        super().__init__()
        self.c=c
        leads=12 if c.get("teacher") else 1
        width=48 if leads==12 else 32
        patch=c.get("hz",360)//20
        self.patch=patch
        if c.get("teacher")=="lead_attention":
            self.shared=nn.Linear(patch,24)
            self.identity=nn.Parameter(torch.randn(12,24)*.05)
            self.attend=nn.MultiheadAttention(24,4,batch_first=True)
            self.mix=nn.Linear(24,width)
        else:
            self.stem=nn.Conv1d(leads,width,patch,stride=patch)
        count=1 if c.get("teacher")=="concat" else (4 if c.get("deep") or leads==12 else 2)
        self.blocks=nn.Sequential(*(Residual(width,2**k) for k in range(count)))
        self.pool=nn.Linear(2*width,48)
        self.event=nn.Linear(48,1)
        self.aux=nn.Linear(48,4)
        self.rebuild=nn.Conv1d(width,patch,1)
        if c.get("fusion"):
            self.beat=nn.Sequential(nn.Conv1d(1,8,7,stride=3),nn.SiLU(),nn.AdaptiveMaxPool1d(6),nn.Flatten())
            self.clinical=nn.Linear(14,48)
            self.gate=nn.Linear(2,48)
        self.register_buffer("center",torch.zeros(16))
        self.register_buffer("scale",torch.ones(16))
        self.register_buffer("offset",torch.tensor(0.))
        self.register_buffer("spread",torch.tensor(1.))

    def encode(self,x,mask=None):
        if self.c.get("teacher")=="lead_attention":
            z=self.shared(x.unfold(-1,self.patch,self.patch))+self.identity[None,:,None,:]
            # Summarize each lead before contextualizing it with the other available leads.
            desc=z.mean(2)
            context=self.attend(desc,desc,desc,key_padding_mask=None if mask is None else ~mask,need_weights=False)[0]
            weights=(desc*context).sum(-1)/24**.5
            if mask is not None: weights=weights.masked_fill(~mask,-1e4)
            h=self.mix((weights.softmax(1)[:,:,None,None]*z).sum(1)).transpose(1,2)
        else:
            h=self.stem((x-self.offset)/self.spread if mask is None else x*mask[:,:,None])
        return self.blocks(h)

    def forward(self,x,features,beat,mask=None):
        h=self.encode(x,mask)
        z=self.pool(torch.cat([h.mean(-1),h.amax(-1)],1)).tanh()
        if self.c.get("fusion"):
            f=((features-self.center)/self.scale).clamp(-10,10)
            extra=torch.zeros_like(z)
            if self.c["fusion"] in ("all","no_quality","beat"): extra=extra+self.beat(beat)
            if self.c["fusion"] in ("all","no_quality","features"): extra=extra+self.clinical(f[:,:14])
            gate=self.gate(f[:,14:]).sigmoid() if self.c["fusion"]!="no_quality" else .5
            z=z+gate*extra
        return self.event(z).flatten(),z,self.aux(z)
