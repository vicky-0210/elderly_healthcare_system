"""VC-BiMamba: voltage-conditioned bidirectional selective state-space model for Lead II LVH.

Design rationale
----------------
ECG LVH criteria (Sokolow-Lyon, Cornell) are amplitude criteria, but deep models
normally receive amplitude-normalised waveforms so that shape learning is stable.
The existing baseline keeps physical voltage in a separate late-fusion branch.
VC-BiMamba instead injects physical millivolt information *inside* the sequence
model: the selective step size (delta) and a FiLM modulation of every Mamba-2 block
are conditioned on per-token voltage descriptors computed from the raw mV lead.
The sequence model can therefore decide how long to retain information based on
how large the local QRS deflection actually is.

Components
----------
1. Multiscale convolutional stem + dilated depthwise blocks (local morphology).
2. Bidirectional Mamba-2 (state-space duality, SSD) mixers with shared projections
   and direction-specific convolutions, decays and step sizes (global context
   over the whole 10 s window; non-causal because the window is buffered).
3. Voltage-conditioned selectivity (VCS): token voltage descriptors -> delta offset
   per head and FiLM scale/shift. Zero-initialised, so training starts from a plain
   bidirectional Mamba-2 model and learns how much voltage control to use.
4. Gated attention pooling + mean/max pooling, fused with the physical-voltage branch
   and the same-lead morphology features used by every baseline.

The SSD scan follows the chunked "minimal" reference algorithm of Mamba-2
(Dao & Gu, ICML 2024, arXiv:2405.21060) written in pure PyTorch, so it exports to
ONNX and runs on CPU-only edge devices without custom CUDA kernels.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
from wearable_ecg.models.backbones import Stem, ConvBlock

TOKEN = 12  # 3,600 samples -> 300 tokens, identical to the baseline stem


def segsum(x):
    """Stable segment sums: out[..., i, j] = sum_{j<k<=i} x[..., k], -inf above diagonal."""
    t = x.size(-1)
    x = x[..., None].expand(*x.shape, t)
    mask = torch.tril(torch.ones(t, t, dtype=torch.bool, device=x.device), -1)
    x = x.masked_fill(~mask, 0)
    out = torch.cumsum(x, dim=-2)
    mask = torch.tril(torch.ones(t, t, dtype=torch.bool, device=x.device), 0)
    return out.masked_fill(~mask, -torch.inf)


def ssd_scan(x, a, b, c, chunk):
    """Mamba-2 SSD. x:(B,L,H,P) a:(B,L,H) b,c:(B,L,N) -> y:(B,L,H,P)."""
    bsz, length, heads, p = x.shape
    n = b.shape[-1]
    pad = (-length) % chunk
    if pad:
        x = F.pad(x, (0, 0, 0, 0, 0, pad)); a = F.pad(a, (0, 0, 0, pad))
        b = F.pad(b, (0, 0, 0, pad)); c = F.pad(c, (0, 0, 0, pad))
    nc = x.shape[1] // chunk
    x = x.reshape(bsz, nc, chunk, heads, p)
    b = b.reshape(bsz, nc, chunk, n); c = c.reshape(bsz, nc, chunk, n)
    a = a.reshape(bsz, nc, chunk, heads).permute(0, 3, 1, 2)  # (B,H,C,L)
    a_cum = torch.cumsum(a, dim=-1)
    # 1. intra-chunk (diagonal blocks)
    decay = torch.exp(segsum(a))  # (B,H,C,L,L)
    scores = torch.einsum("bcln,bcsn->bcls", c, b)
    y_diag = torch.einsum("bcls,bhcls,bcshp->bclhp", scores, decay, x)
    # 2. chunk states
    decay_states = torch.exp(a_cum[..., -1:] - a_cum)  # (B,H,C,L)
    states = torch.einsum("bcln,bhcl,bclhp->bchpn", b, decay_states, x)
    # 3. inter-chunk recurrence
    states = torch.cat([torch.zeros_like(states[:, :1]), states], dim=1)
    decay_chunk = torch.exp(segsum(F.pad(a_cum[..., -1], (1, 0))))  # (B,H,C+1,C+1)
    states = torch.einsum("bhzc,bchpn->bzhpn", decay_chunk, states)[:, :-1]
    # 4. state -> output
    y_off = torch.einsum("bcln,bchpn,bhcl->bclhp", c, states, torch.exp(a_cum))
    y = (y_diag + y_off).reshape(bsz, nc * chunk, heads, p)
    return y[:, :length]


class VoltageTokens(nn.Module):
    """Per-token physical-voltage descriptors from the raw mV lead (no labels)."""
    def forward(self, lead):
        x = lead - lead.median(dim=-1, keepdim=True).values
        hi = F.max_pool1d(x, TOKEN, TOKEN); lo = -F.max_pool1d(-x, TOKEN, TOKEN)
        p2p = hi - lo
        slope = F.max_pool1d((x[..., 1:] - x[..., :-1]).abs().mul(360 / 1000), TOKEN, TOKEN,
                             ceil_mode=True)[..., :p2p.shape[-1]]
        # record-level context: robust large-deflection amplitude, broadcast to tokens
        rec = torch.quantile(p2p, .95, dim=-1, keepdim=True).expand_as(p2p)
        desc = torch.cat([hi, -lo, p2p, slope, rec], dim=1)  # (B,5,L) in mV and mV/ms
        return torch.sign(desc) * torch.log1p(desc.abs() * 4)


class BiMamba2(nn.Module):
    def __init__(self, width, expand=2, headdim=16, states=16, chunk=50, voltage_dim=5, conditioned=True,
                 bidirectional=True):
        super().__init__()
        inner = width * expand; self.inner = inner; self.heads = inner // headdim; self.headdim = headdim
        self.states = states; self.chunk = chunk; self.conditioned = conditioned; self.bidirectional = bidirectional
        self.norm = nn.LayerNorm(width)
        self.in_proj = nn.Linear(width, 2 * inner + 2 * states + self.heads, bias=False)
        dirs = 2 if bidirectional else 1
        conv_dim = inner + 2 * states
        self.conv = nn.ModuleList([nn.Conv1d(conv_dim, conv_dim, 4, padding=3, groups=conv_dim) for _ in range(dirs)])
        # dt initialised log-uniformly in [1e-3, 1e-1] as in Mamba
        dt = torch.exp(torch.rand(dirs, self.heads) * (math.log(.1) - math.log(1e-3)) + math.log(1e-3))
        self.dt_bias = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))
        self.A_log = nn.Parameter(torch.log(torch.empty(dirs, self.heads).uniform_(1, 16)))
        self.D = nn.Parameter(torch.ones(dirs, self.heads))
        self.out_norm = nn.LayerNorm(inner)
        self.out_proj = nn.Linear(inner, width, bias=False)
        self.drop = nn.Dropout(.1)
        if conditioned:
            self.film = nn.Conv1d(voltage_dim, 2 * width, 3, padding=1)
            self.vdt = nn.Conv1d(voltage_dim, dirs * self.heads, 3, padding=1)
            for m in (self.film, self.vdt): nn.init.zeros_(m.weight); nn.init.zeros_(m.bias)

    def scan(self, xbc, dt_raw, d):
        length = xbc.shape[1]
        xbc = F.silu(self.conv[d](xbc.transpose(1, 2))[..., :length].transpose(1, 2))
        x, b, c = xbc.split([self.inner, self.states, self.states], dim=-1)
        dt = F.softplus(dt_raw + self.dt_bias[d])  # (B,L,H)
        a = -torch.exp(self.A_log[d]) * dt
        xh = x.reshape(*x.shape[:2], self.heads, self.headdim)
        y = ssd_scan(xh * dt[..., None], a, b, c, self.chunk) + xh * self.D[d][:, None]
        return y.reshape(*x.shape[:2], self.inner)

    def forward(self, z, volt):
        # z: (B,D,L) channels-first, volt: (B,V,L)
        with torch.autocast(device_type=z.device.type, enabled=False):
            z = z.float(); h = self.norm(z.transpose(1, 2))
            vdt = 0.
            if self.conditioned:
                scale, shift = self.film(volt.float()).chunk(2, dim=1)
                h = h * (1 + scale.transpose(1, 2)) + shift.transpose(1, 2)
                vdt = self.vdt(volt.float()).transpose(1, 2)  # (B,L,dirs*H)
            gate, xbc, dt = self.in_proj(h).split([self.inner, self.inner + 2 * self.states, self.heads], dim=-1)
            dts = [dt + (vdt[..., i * self.heads:(i + 1) * self.heads] if self.conditioned else 0.)
                   for i in range(2 if self.bidirectional else 1)]
            y = self.scan(xbc, dts[0], 0)
            if self.bidirectional:
                y = y + self.scan(xbc.flip(1), dts[1].flip(1), 1).flip(1)
                y = y * .5
            y = self.out_norm(y) * F.silu(gate)
            return z + self.drop(self.out_proj(y)).transpose(1, 2)


class AttentionPool(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.score = nn.Sequential(nn.Conv1d(width, width // 2, 1), nn.Tanh(), nn.Conv1d(width // 2, 1, 1))
    def forward(self, z):
        w = torch.softmax(self.score(z).float(), dim=-1).to(z.dtype)
        return (z * w).sum(-1)


class VCBiMamba(nn.Module):
    """Same forward contract as ECGClassifier: forward(x [B,3600] mV, features [B,16])."""
    def __init__(self, width=64, conv_blocks=2, mamba_blocks=2, conditioned=True, bidirectional=True,
                 expand=2, states=16, input_channels=1, architecture="vc_bimamba"):
        super().__init__()
        self.stem = Stem(input_channels, width)
        self.local = nn.ModuleList([ConvBlock(width, 2 ** i) for i in range(conv_blocks)])
        self.vtok = VoltageTokens()
        self.mixers = nn.ModuleList([BiMamba2(width, expand=expand, states=states, conditioned=conditioned,
                                              bidirectional=bidirectional) for _ in range(mamba_blocks)])
        self.final_norm = nn.GroupNorm(1, width)
        self.pool = AttentionPool(width)
        self.voltage = nn.Sequential(nn.Conv1d(1, 8, 15, stride=6, padding=7), nn.GELU(),
                                     nn.Conv1d(8, 16, 9, stride=2, padding=4), nn.GELU(), nn.AdaptiveAvgPool1d(1))
        self.feature = nn.Sequential(nn.Linear(16, 32), nn.GELU())
        self.head = nn.Sequential(nn.Linear(width * 3 + 16 + 32 + 16, 64), nn.GELU(), nn.Dropout(.2), nn.Linear(64, 1))
        self.register_buffer("feature_center", torch.zeros(16))
        self.register_buffer("feature_scale", torch.ones(16))
        self.config = {"architecture": architecture, "width": width, "conv_blocks": conv_blocks,
                       "mamba_blocks": mamba_blocks, "conditioned": conditioned, "bidirectional": bidirectional,
                       "expand": expand, "states": states, "input_channels": input_channels}

    def forward(self, x, features):
        if x.ndim == 2: x = x[:, None, :]
        lead = x[:, 1:2] if x.shape[1] > 1 else x
        shape = (x - x.mean(-1, keepdim=True)) / x.std(-1, keepdim=True).clamp_min(.02)
        z = self.stem(shape)
        for block in self.local: z = block(z)
        volt = self.vtok(lead.float())[..., :z.shape[-1]]
        for mixer in self.mixers: z = mixer(z, volt)
        z = self.final_norm(z)
        pooled = torch.cat([z.mean(-1), z.amax(-1), self.pool(z)], dim=1)
        v = self.voltage(lead).squeeze(-1)
        f = (features - self.feature_center) / self.feature_scale
        return self.head(torch.cat([pooled, v, self.feature(f), f], dim=1)).squeeze(-1)


VARIANTS = {
    "vc_bimamba": {},
    "vc_bimamba_nocond": {"conditioned": False},
    "vc_bimamba_unidir": {"bidirectional": False},
    "vc_bimamba_noconv": {"conv_blocks": 0, "mamba_blocks": 3},
}


def build(config):
    config = dict(config)
    return VCBiMamba(**config)


def make_novel(architecture, features=None):
    model = VCBiMamba(architecture=architecture, **VARIANTS[architecture])
    if features is not None:
        f = torch.as_tensor(features, dtype=torch.float32)
        model.feature_center.copy_(f.mean(0)); model.feature_scale.copy_(f.std(0).clamp_min(.02))
    return model
