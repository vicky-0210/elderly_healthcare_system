"""Matched compact backbones with a shared physical-voltage bypass.

The SSM is a real diagonal, time-invariant state-space convolution, NOT Mamba.
This independent implementation uses stable real exponential modes and a
zero-order-hold discretization. S4D background: arxiv.org/abs/2206.11893.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F


class Stem(nn.Module):
    def __init__(self, channels=1, width=64):
        super().__init__()
        self.branches=nn.ModuleList([nn.Conv1d(channels,8,k,stride=4,padding=k//2) for k in (9,25,65)])
        self.reduce=nn.Sequential(nn.GELU(),nn.Conv1d(24,width,7,stride=3,padding=3),nn.GELU())
    def forward(self,x): return self.reduce(torch.cat([b(x) for b in self.branches],dim=1))


class ConvBlock(nn.Module):
    def __init__(self,width,dilation=1):
        super().__init__()
        self.block=nn.Sequential(nn.GroupNorm(1,width),nn.Conv1d(width,width,7,padding=3*dilation,dilation=dilation,groups=width),
                                 nn.Conv1d(width,2*width,1),nn.GELU(),nn.Dropout(.1),nn.Conv1d(2*width,width,1))
    def forward(self,x): return x+self.block(x)


class AttentionBlock(nn.Module):
    def __init__(self,width,heads=4):
        super().__init__()
        self.width=width; self.heads=heads
        self.norm1=nn.LayerNorm(width); self.norm2=nn.LayerNorm(width)
        self.qkv=nn.Linear(width,3*width); self.proj=nn.Linear(width,width)
        self.ff=nn.Sequential(nn.Linear(width,2*width),nn.GELU(),nn.Dropout(.1),nn.Linear(2*width,width))
        self.dropout=nn.Dropout(.1)
        self.quality_strength=nn.Parameter(torch.tensor(.1))
    def forward(self,x,quality=None):
        z=x.transpose(1,2); b,t,d=z.shape
        qkv=self.qkv(self.norm1(z)).reshape(b,t,3,self.heads,d//self.heads).permute(2,0,3,1,4)
        q,k,v=qkv.unbind(0)
        bias=None
        if quality is not None:
            bias=self.quality_strength.clamp(0,1)*quality.clamp(.05,1).log()[:,None,None,:]
        att=F.scaled_dot_product_attention(q,k,v,attn_mask=bias,dropout_p=.1 if self.training else 0)
        z=z+self.dropout(self.proj(att.transpose(1,2).reshape(b,t,d)))
        return (z+self.dropout(self.ff(self.norm2(z)))).transpose(1,2)


class DiagonalSSM(nn.Module):
    def __init__(self,width,states=16):
        super().__init__()
        # Continuous poles < 0; trainable dt; real stable exponential mixtures.
        self.log_decay=nn.Parameter(torch.linspace(-2,2,states).repeat(width,1))
        self.log_dt=nn.Parameter(torch.full((width,1),math.log(.05)))
        self.C=nn.Parameter(torch.randn(width,states)/math.sqrt(states))
        self.skip=nn.Parameter(torch.ones(width))
        self.norm=nn.GroupNorm(1,width)
        self.ff=nn.Sequential(nn.Conv1d(width,2*width,1),nn.GELU(),nn.Dropout(.1),nn.Conv1d(2*width,width,1))
    def kernel(self,length):
        a=-self.log_decay.exp()
        dt=self.log_dt.exp().clamp(1e-4,1)
        abar=torch.exp(dt*a)
        bbar=torch.expm1(dt*a)/a
        steps=torch.arange(length,device=a.device,dtype=a.dtype)
        return torch.sum((self.C*bbar)[...,None]*abar[...,None]**steps,dim=1)
    def forward(self,x):
        # FFT in float32 even when surrounding convolutions use AMP.
        z=self.norm(x)
        with torch.autocast(device_type=x.device.type,enabled=False):
            u=z.float(); length=u.shape[-1]
            kernel=self.kernel(length)
            nfft=1<<(2*length-1).bit_length()
            yf=torch.fft.rfft(u,n=nfft)*torch.fft.rfft(kernel,n=nfft)[None]
            y=torch.fft.irfft(yf,n=nfft)[...,:length]+self.skip[None,:,None]*u
        return x+self.ff(F.gelu(y.to(x.dtype)))


class ECGClassifier(nn.Module):
    def __init__(self,architecture="cnn",width=64,input_channels=1,quality_bias=True,voltage_bypass=True):
        super().__init__()
        if architecture not in {"cnn","cnn_transformer","cnn_ssm","cnn_bigru"}: raise ValueError(architecture)
        self.architecture=architecture; self.quality_bias=quality_bias; self.voltage_bypass=voltage_bypass
        self.stem=Stem(input_channels,width)
        if architecture=="cnn": self.blocks=nn.ModuleList([ConvBlock(width,2**i) for i in range(4)])
        elif architecture=="cnn_transformer": self.blocks=nn.ModuleList([AttentionBlock(width) for _ in range(2)])
        elif architecture=="cnn_ssm": self.blocks=nn.ModuleList([DiagonalSSM(width) for _ in range(3)])
        else:
            self.blocks=nn.ModuleList()
            self.gru=nn.GRU(width,width//2,num_layers=2,batch_first=True,bidirectional=True,dropout=.1)
        self.voltage=nn.Sequential(nn.Conv1d(1,8,15,stride=6,padding=7),nn.GELU(),
                                   nn.Conv1d(8,16,9,stride=2,padding=4),nn.GELU(),nn.AdaptiveAvgPool1d(1))
        self.feature=nn.Sequential(nn.Linear(16,32),nn.GELU())
        self.head=nn.Sequential(nn.Linear(width*2+16+32+16,64),nn.GELU(),nn.Dropout(.2),nn.Linear(64,1))
        self.register_buffer("feature_center",torch.zeros(16))
        self.register_buffer("feature_scale",torch.ones(16))
        self.config={"architecture":architecture,"width":width,"input_channels":input_channels,
                     "quality_bias":quality_bias,"voltage_bypass":voltage_bypass}
    def forward(self,x,features):
        if x.ndim==2: x=x[:,None,:]
        lead=x[:,1:2] if x.shape[1]>1 else x
        shape=(x-x.mean(-1,keepdim=True))/x.std(-1,keepdim=True).clamp_min(.02)
        z=self.stem(shape)
        q=None
        if self.architecture=="cnn_transformer":
            # Local reliability from acquisition-visible signal, no label input.
            d=F.pad((lead[:,:,1:]-lead[:,:,:-1]).abs(),(1,0))
            # The research/export contract is 3,600 samples -> 300 tokens.
            # This equals adaptive pooling for that shape and exports cleanly.
            q=torch.exp(-(F.avg_pool1d(d,12,stride=12).squeeze(1)-.10).clamp_min(0)/.25)
            steps=torch.arange(z.shape[-1],device=z.device,dtype=z.dtype)
            frequency=torch.exp(torch.arange(0,z.shape[1],2,device=z.device,dtype=z.dtype)*(-math.log(10000)/z.shape[1]))
            position=torch.zeros_like(z[:1]); position[:,0::2]=torch.sin(frequency[:,None]*steps)
            position[:,1::2]=torch.cos(frequency[:,None]*steps)
            z=z+position*.1
        for block in self.blocks:
            z=block(z,q if self.quality_bias else None) if isinstance(block,AttentionBlock) else block(z)
        if self.architecture=="cnn_bigru": z=self.gru(z.transpose(1,2))[0].transpose(1,2)
        pooled=torch.cat([z.mean(-1),z.amax(-1)],dim=1)
        v=self.voltage(lead).squeeze(-1)
        f=(features-self.feature_center)/self.feature_scale
        if not self.voltage_bypass:
            v=torch.zeros_like(v); f=torch.zeros_like(f)
        return self.head(torch.cat([pooled,v,self.feature(f),f],dim=1)).squeeze(-1)


def make_model(architecture,features=None,**kwargs):
    model=ECGClassifier(architecture,**kwargs)
    if features is not None:
        f=torch.as_tensor(features,dtype=torch.float32)
        model.feature_center.copy_(f.mean(0))
        model.feature_scale.copy_(f.std(0).clamp_min(.02))
    return model
