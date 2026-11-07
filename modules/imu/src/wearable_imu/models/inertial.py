"""Compact window encoders for weakly supervised trial classification."""
import math
import torch
from torch import nn
from torch.nn import functional as F


class CausalConv(nn.Conv1d):
    def forward(self,x):
        left=(self.kernel_size[0]-1)*self.dilation[0]
        return super().forward(F.pad(x,(left,0)))


class Dilated(nn.Module):
    def __init__(self,w,d):
        super().__init__()
        self.conv=CausalConv(w,w,3,dilation=d)
        self.mix=nn.Conv1d(w,w,1)
        self.norm=nn.GroupNorm(1,w)

    def forward(self,x):
        return self.norm(x+self.mix(F.silu(self.conv(x))))


class MultiScale(nn.Module):
    def __init__(self,w,kernels):
        super().__init__()
        self.paths=nn.ModuleList([CausalConv(w,w,k,groups=w) for k in kernels])
        self.mix=nn.Conv1d(len(kernels)*w,w,1)

    def forward(self,x):
        return x+F.silu(self.mix(torch.cat([p(x) for p in self.paths],1)))


class InceptionModule(nn.Module):
    def __init__(self,w):
        super().__init__()
        self.bottle=nn.Conv1d(w,w//2,1)
        self.filters=nn.ModuleList([CausalConv(w//2,w//4,k) for k in (10,20,40)])
        self.pool=nn.Conv1d(w,w//4,1)
        self.norm=nn.BatchNorm1d(w)

    def forward(self,x):
        b=self.bottle(x)
        p=self.pool(F.max_pool1d(F.pad(x,(2,0)),3,stride=1))
        return F.relu(self.norm(torch.cat([f(b) for f in self.filters]+[p],1)))


class InceptionTime(nn.Module):
    def __init__(self,w):
        super().__init__()
        self.groups=nn.ModuleList([nn.Sequential(*(InceptionModule(w) for _ in range(3))) for _ in range(2)])

    def forward(self,x):
        for group in self.groups: x=F.relu(x+group(x))
        return x


class SelectiveSSM(nn.Module):
    """Input-dependent diagonal selective scan with a depthwise causal conv and SiLU gate."""
    def __init__(self,w,state=3):
        super().__init__()
        self.project=nn.Conv1d(w,2*w,1)
        self.local=CausalConv(w,w,4,groups=w)
        self.parameters_in=nn.Conv1d(w,w+2*state,1)
        self.log_a=nn.Parameter(torch.arange(1,state+1).float().log().repeat(w,1))
        self.skip=nn.Parameter(torch.ones(w))
        self.out=nn.Conv1d(w,w,1)

    def forward(self,x):
        u,gate=self.project(x).chunk(2,dim=1)
        u=F.silu(self.local(u)).transpose(1,2)
        w,n=self.log_a.shape
        dt,b,c=torch.split(self.parameters_in(u.transpose(1,2)).transpose(1,2),[w,n,n],dim=-1)
        dt=F.softplus(dt-2)[...,None]
        alpha=torch.exp(-dt*self.log_a.exp())
        h=dt*b[:,:,None,:]*u[...,None]
        # Associative prefix evaluation of h[t]=alpha[t]*h[t-1]+input[t].
        hop=1
        while hop<x.shape[-1]:
            h=torch.cat((h[:,:hop],h[:,hop:]+alpha[:,hop:]*h[:,:-hop]),dim=1)
            alpha=torch.cat((alpha[:,:hop],alpha[:,hop:]*alpha[:,:-hop]),dim=1)
            hop*=2
        y=(h*c[:,:,None,:]).sum(-1)+u*self.skip
        return x+self.out(y.transpose(1,2)*F.silu(gate))


def trunk(c,w,kind,kernels,dilations):
    layers=[nn.Conv1d(c,w,1)]
    if kind in ("cnn","cnn_tcn"):
        layers.extend((CausalConv(w,w,5),nn.SiLU(),CausalConv(w,w,3),nn.SiLU()))
    if kind in ("multi","multi_tcn"):
        layers.append(MultiScale(w,kernels))
    if kind in ("tcn","cnn_tcn","multi_tcn"):
        layers.extend(Dilated(w,d) for d in dilations)
    if kind=="inception": layers.append(InceptionTime(w))
    return nn.Sequential(*layers)


class FallNet(nn.Module):
    def __init__(self, model,names,*,width=12,fusion=None,pool=None,kernels=(3,5,9),dilations=(1,2,4)):
        super().__init__()
        self.model=model
        self.dual=model in ("dual_cnn_tcn","dm_tcn","cross_modal")
        self.recurrent=None
        self.transformer=None
        self.cross=None
        self.ssm=None
        self.se=None
        if self.dual:
            a=[i for i,n in enumerate(names) if n.startswith("a_")]
            g=[i for i,n in enumerate(names) if n.startswith("g_")]
            if not a or not g: raise ValueError("Dual model needs both sensors")
            self.register_buffer("ai",torch.tensor(a)); self.register_buffer("gi",torch.tensor(g))
            kind={"dm_tcn":"multi_tcn","dual_cnn_tcn":"cnn_tcn","cross_modal":"cnn"}[model]
            self.acc=trunk(len(a),width,kind,kernels,dilations)
            self.gyro=trunk(len(g),width,kind,kernels,dilations)
            self.fusion=fusion or {"dm_tcn":"gate","dual_cnn_tcn":"concat","cross_modal":"cross"}[model]
            if self.fusion in ("concat","gate"): self.combine=nn.Conv1d(2*width,width,1)
            if self.fusion=="attention": self.combine=nn.Conv1d(2*width,2,1)
            if self.fusion=="cross": self.cross=nn.MultiheadAttention(width,2,batch_first=True)
        else:
            kind={"lstm":"linear","gru":"linear","bigru":"linear","inception":"inception",
                  "multicnn_tcn":"multi_tcn","multitcn_attention":"multi_tcn","tcn_attention":"tcn",
                  "tcn":"tcn","cnn_tcn":"cnn_tcn"}.get(model,"cnn")
            self.encoder=trunk(len(names),width,kind,kernels,dilations)
            if model in ("lstm","gru","bigru","cnn_lstm","cnn_gru","cnn_bigru"):
                bi=model.endswith("bigru")
                cell=nn.LSTM if model.endswith("lstm") else nn.GRU
                self.recurrent=cell(width,width//2 if bi else width,batch_first=True,bidirectional=bi)
            if model=="cnn_transformer":
                self.transformer=nn.TransformerEncoderLayer(width,2,width*2,.10,batch_first=True)
                pos=torch.arange(128)[:,None]*torch.exp(-torch.arange(0,width,2)*math.log(10000)/width)
                self.register_buffer("pos",torch.stack((pos.sin(),pos.cos()),dim=-1).flatten(1))
            if model=="cnn_ssm": self.ssm=SelectiveSSM(width)
            if model=="cnn_channel": self.se=nn.Sequential(nn.Linear(width,width//2),nn.SiLU(),nn.Linear(width//2,width),nn.Sigmoid())
        self.pool=pool or ("attention" if self.dual or model in ("tcn_attention","multitcn_attention") else "avgmax")
        if self.pool=="attention": self.score=nn.Sequential(nn.Conv1d(width,width//2,1),nn.Tanh(),nn.Conv1d(width//2,1,1))
        h=2*width if self.pool=="avgmax" else width
        self.head=nn.Sequential(nn.LayerNorm(h),nn.Dropout(.12),nn.Linear(h,1))

    def forward(self,x):
        if self.dual:
            a,g=self.acc(x.index_select(1,self.ai)),self.gyro(x.index_select(1,self.gi))
            if self.fusion=="add": h=.5*(a+g)
            elif self.fusion=="cross":
                u,v=a.transpose(1,2),g.transpose(1,2)
                h=(u+self.cross(u,v,v,need_weights=False)[0]).transpose(1,2)
            else:
                weights=self.combine(torch.cat((a,g),dim=1))
                if self.fusion=="concat": h=weights
                elif self.fusion=="attention":
                    mix=weights.softmax(1); h=mix[:,:1]*a+mix[:,1:]*g
                else:
                    mix=weights.sigmoid(); h=mix*a+(1-mix)*g
        else:
            h=self.encoder(x)
            if self.recurrent is not None: h=self.recurrent(h.transpose(1,2))[0].transpose(1,2)
            if self.transformer is not None: h=self.transformer(h.transpose(1,2)+self.pos[:h.shape[-1]]).transpose(1,2)
            if self.ssm is not None: h=self.ssm(h)
            if self.se is not None: h=h*self.se(h.mean(-1))[:,:,None]
        if self.pool=="avg": z=h.mean(-1)
        elif self.pool=="max": z=h.amax(-1)
        elif self.pool=="attention": z=(h*self.score(h).softmax(-1)).sum(-1)
        else: z=torch.cat((h.mean(-1),h.amax(-1)),dim=1)
        return self.head(z).squeeze(-1)
