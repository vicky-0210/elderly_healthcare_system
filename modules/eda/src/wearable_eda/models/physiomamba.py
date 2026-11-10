"""PhysioMamba: artifact-aware, context-conditioned cross-modal Mamba-2 for wrist stress detection.

Inputs are the same 60 s, 4 Hz, 21-channel full-wrist windows (EDA + derived tonic/phasic,
contact quality, ACC, skin temperature, BVP descriptors) and 39 window features used by
every WESAD baseline in this project.

Novel elements
--------------
1. Modality tokenisation: EDA, motion, temperature and cardiac (BVP) channel groups are
   embedded separately per second and fused by a reliability gate whose logits receive a
   log-quality prior, so poor electrode contact down-weights the EDA stream token by token.
2. Artifact-aware selectivity: the Mamba-2 step size (delta) of every head receives an
   offset predicted from per-token motion and contact quality. The model can learn to
   shrink delta (hold its state) during movement artefacts instead of absorbing them.
3. Physiological-context conditioning: the 39 handcrafted SCL/SCR/HR/temperature features
   produce a FiLM modulation and a global delta offset, so window-level physiology steers
   which temporal events the selective scan retains.
4. Bidirectional Mamba-2 (SSD) mixers in pure PyTorch (ONNX/CPU friendly) followed by
   attention pooling, fused with a residual feature MLP.
5. Modality dropout during training (whole groups zeroed) for robustness to missing sensors.
All conditioning layers are zero-initialised: training starts from an unconditioned BiMamba.
"""
from __future__ import annotations
import math
import torch
from torch import nn
from torch.nn import functional as F

GROUPS = {
    "eda": ("raw", "clean", "tonic", "phasic", "relative_tonic", "relative_phasic", "tonic_slope",
            "phasic_slope", "phasic_energy", "quality", "artifact", "missing"),
    "motion": ("acc_magnitude", "acc_dynamic", "acc_rms", "acc_jerk", "motion"),
    "temperature": ("temperature",),
    "cardiac": ("bvp_rms", "bvp_rate", "bvp_pulse"),
}


def segsum(x):
    t = x.size(-1)
    x = x[..., None].expand(*x.shape, t)
    mask = torch.tril(torch.ones(t, t, dtype=torch.bool, device=x.device), -1)
    out = torch.cumsum(x.masked_fill(~mask, 0), dim=-2)
    mask = torch.tril(torch.ones(t, t, dtype=torch.bool, device=x.device), 0)
    return out.masked_fill(~mask, -torch.inf)


def ssd_scan(x, a, b, c, chunk):
    """Mamba-2 SSD (Dao & Gu 2024). x:(B,L,H,P) a:(B,L,H) b,c:(B,L,N)."""
    bsz, length, heads, p = x.shape
    n = b.shape[-1]
    pad = (-length) % chunk
    if pad:
        x = F.pad(x, (0, 0, 0, 0, 0, pad)); a = F.pad(a, (0, 0, 0, pad))
        b = F.pad(b, (0, 0, 0, pad)); c = F.pad(c, (0, 0, 0, pad))
    nc = x.shape[1] // chunk
    x = x.reshape(bsz, nc, chunk, heads, p)
    b = b.reshape(bsz, nc, chunk, n); c = c.reshape(bsz, nc, chunk, n)
    a = a.reshape(bsz, nc, chunk, heads).permute(0, 3, 1, 2)
    a_cum = torch.cumsum(a, dim=-1)
    y_diag = torch.einsum("bcls,bhcls,bcshp->bclhp", torch.einsum("bcln,bcsn->bcls", c, b), torch.exp(segsum(a)), x)
    states = torch.einsum("bcln,bhcl,bclhp->bchpn", b, torch.exp(a_cum[..., -1:] - a_cum), x)
    states = torch.cat([torch.zeros_like(states[:, :1]), states], dim=1)
    states = torch.einsum("bhzc,bchpn->bzhpn", torch.exp(segsum(F.pad(a_cum[..., -1], (1, 0)))), states)[:, :-1]
    y_off = torch.einsum("bcln,bchpn,bhcl->bclhp", c, states, torch.exp(a_cum))
    return (y_diag + y_off).reshape(bsz, nc * chunk, heads, p)[:, :length]


class ConditionedBiMamba2(nn.Module):
    def __init__(self, width, token_ctx, global_ctx, expand=2, headdim=16, states=16, chunk=20):
        super().__init__()
        inner = width * expand; self.inner = inner; self.heads = inner // headdim; self.headdim = headdim
        self.states = states; self.chunk = chunk
        self.norm = nn.LayerNorm(width)
        self.in_proj = nn.Linear(width, 2 * inner + 2 * states + self.heads, bias=False)
        conv_dim = inner + 2 * states
        self.conv = nn.ModuleList([nn.Conv1d(conv_dim, conv_dim, 4, padding=3, groups=conv_dim) for _ in range(2)])
        dt = torch.exp(torch.rand(2, self.heads) * (math.log(.1) - math.log(1e-3)) + math.log(1e-3))
        self.dt_bias = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))
        self.A_log = nn.Parameter(torch.log(torch.empty(2, self.heads).uniform_(1, 16)))
        self.D = nn.Parameter(torch.ones(2, self.heads))
        self.out_norm = nn.LayerNorm(inner)
        self.out_proj = nn.Linear(inner, width, bias=False)
        self.drop = nn.Dropout(.15)
        self.token_dt = nn.Linear(token_ctx, 2 * self.heads)       # artifact-aware selectivity
        self.global_dt = nn.Linear(global_ctx, 2 * self.heads)     # physiological-context selectivity
        self.film = nn.Linear(global_ctx, 2 * width)
        for m in (self.token_dt, self.global_dt, self.film): nn.init.zeros_(m.weight); nn.init.zeros_(m.bias)

    def scan(self, xbc, dt_raw, d):
        length = xbc.shape[1]
        xbc = F.silu(self.conv[d](xbc.transpose(1, 2))[..., :length].transpose(1, 2))
        x, b, c = xbc.split([self.inner, self.states, self.states], dim=-1)
        dt = F.softplus(dt_raw + self.dt_bias[d])
        xh = x.reshape(*x.shape[:2], self.heads, self.headdim)
        y = ssd_scan(xh * dt[..., None], -torch.exp(self.A_log[d]) * dt, b, c, self.chunk) + xh * self.D[d][:, None]
        return y.reshape(*x.shape[:2], self.inner)

    def forward(self, h, token_ctx, global_ctx):
        z = self.norm(h)
        scale, shift = self.film(global_ctx)[:, None].chunk(2, dim=-1)
        z = z * (1 + scale) + shift
        gate, xbc, dt = self.in_proj(z).split([self.inner, self.inner + 2 * self.states, self.heads], dim=-1)
        offset = self.token_dt(token_ctx) + self.global_dt(global_ctx)[:, None]
        fwd, bwd = offset.chunk(2, dim=-1)
        y = .5 * (self.scan(xbc, dt + fwd, 0) + self.scan(xbc.flip(1), (dt + bwd).flip(1), 1).flip(1))
        return h + self.drop(self.out_proj(self.out_norm(y) * F.silu(gate)))


class PhysioMamba(nn.Module):
    def __init__(self, channels, n_features, n_classes=2, width=48, layers=2, modality_dropout=.1,
                 conditioned=True, gated=True):
        super().__init__()
        self.channels = list(channels); self.width = width; self.modality_dropout = modality_dropout
        self.conditioned = conditioned; self.gated = gated
        self.groups = {g: [self.channels.index(c) for c in names if c in self.channels] for g, names in GROUPS.items()}
        self.groups = {g: idx for g, idx in self.groups.items() if idx}
        self.embed = nn.ModuleDict({g: nn.Sequential(nn.Conv1d(len(idx), width, 4, stride=4), nn.GELU(),
                                                     nn.Conv1d(width, width, 3, padding=1)) for g, idx in self.groups.items()})
        self.modality = nn.Parameter(torch.zeros(len(self.groups), width))
        self.gate = nn.Linear(width, 1)
        self.q_idx = self.channels.index("quality") if "quality" in self.channels else None
        self.m_idx = self.channels.index("motion") if "motion" in self.channels else None
        self.features = nn.Sequential(nn.Linear(n_features, 64), nn.GELU(), nn.Dropout(.2), nn.Linear(64, width), nn.LayerNorm(width))
        self.mixers = nn.ModuleList([ConditionedBiMamba2(width, 2, width) for _ in range(layers)])
        self.final = nn.LayerNorm(width)
        self.attn = nn.Linear(width, 1)
        self.classifier = nn.Sequential(nn.LayerNorm(3 * width), nn.Dropout(.2), nn.Linear(3 * width, n_classes))
        self.config = dict(width=width, layers=layers, modality_dropout=modality_dropout,
                           conditioned=conditioned, gated=gated)

    def forward(self, x, f):
        tokens = x.shape[-1] // 4
        quality = F.avg_pool1d(x[:, self.q_idx:self.q_idx + 1], 4, 4).transpose(1, 2) if self.q_idx is not None else x.new_ones(x.shape[0], tokens, 1)
        motion = F.avg_pool1d(x[:, self.m_idx:self.m_idx + 1], 4, 4).transpose(1, 2) if self.m_idx is not None else x.new_zeros(x.shape[0], tokens, 1)
        streams = []
        for i, (g, idx) in enumerate(self.groups.items()):
            s = self.embed[g](x[:, idx]).transpose(1, 2) + self.modality[i]
            if self.training and self.modality_dropout:
                keep = (torch.rand(x.shape[0], 1, 1, device=x.device) > self.modality_dropout).to(s.dtype)
                s = s * keep
            streams.append(s)
        stack = torch.stack(streams, 2)  # (B,L,M,D)
        if self.gated:
            logits = self.gate(stack).squeeze(-1)
            if "eda" in self.groups:
                prior = torch.zeros_like(logits); prior[..., list(self.groups).index("eda")] = quality.squeeze(-1).clamp(.05, 1).log()
                logits = logits + prior
            h = (stack * logits.softmax(-1).unsqueeze(-1)).sum(2)
        else:
            h = stack.mean(2)
        g = self.features(f)
        token_ctx = torch.cat([quality, motion.clamp(-30, 30) / 10], -1)
        if not self.conditioned:
            token_ctx = torch.zeros_like(token_ctx); gctx = torch.zeros_like(g)
        else:
            gctx = g
        for mixer in self.mixers: h = mixer(h, token_ctx, gctx)
        h = self.final(h)
        pooled = (h * self.attn(h).softmax(1)).sum(1)
        return self.classifier(torch.cat([pooled, h.mean(1), g], 1))


VARIANTS = {
    "physiomamba": {},
    "physiomamba_nocond": {"conditioned": False},
    "physiomamba_nogate": {"gated": False},
}
