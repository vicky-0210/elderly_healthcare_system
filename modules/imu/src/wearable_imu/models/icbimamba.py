"""IC-BiMamba: impact-conditioned bidirectional Mamba-2 for waist-IMU fall detection.

Each 20 Hz token carries two views of the same 50 ms:
  * the 24 gravity-relative / wavelet IPMV descriptors (standard-scaled, asinh), and
  * the 10 raw 200 Hz samples of all six axes plus |a| and |g| (a learned patch embedding),
so native-rate impact transients reach the encoder without filtering.

The novelty is in the state-space block: impact descriptors (peak acceleration, jerk,
peak angular rate, angular-rate change, dynamic acceleration, tilt) modulate both the
features (FiLM) and the selective step size dt of every Mamba-2 head. A large impact
therefore shortens the memory horizon (the state is "reset" around the event) while
quiet segments keep long memory of posture before and after. Both directions are
scanned because a fall is defined by what happens before (pre-impact descent) and
after (post-impact lying/stillness) the impact; the 4 s window trades latency for that
context. All conditioning layers are zero-initialised, so training starts from a plain
bidirectional Mamba-2 and learns how much to rely on impact evidence.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F

CONDITION = ("a_peak", "a_jerk", "g_peak", "g_change", "a_dynamic", "a_tilt")
PATCH = 10  # raw 200 Hz samples per 20 Hz token


def segsum(x):
    t = x.size(-1)
    x = x[..., None].expand(*x.shape, t)
    mask = torch.tril(torch.ones(t, t, dtype=torch.bool, device=x.device), -1)
    x = x.masked_fill(~mask, 0)
    out = torch.cumsum(x, dim=-2)
    mask = torch.tril(torch.ones(t, t, dtype=torch.bool, device=x.device), 0)
    return out.masked_fill(~mask, -torch.inf)


def ssd_scan(x, a, b, c, chunk):
    """Mamba-2 SSD (chunked, exact). x:(B,L,H,P) a:(B,L,H) b,c:(B,L,N) -> (B,L,H,P)."""
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
    decay = torch.exp(segsum(a))
    scores = torch.einsum("bcln,bcsn->bcls", c, b)
    y_diag = torch.einsum("bcls,bhcls,bcshp->bclhp", scores, decay, x)
    decay_states = torch.exp(a_cum[..., -1:] - a_cum)
    states = torch.einsum("bcln,bhcl,bclhp->bchpn", b, decay_states, x)
    states = torch.cat([torch.zeros_like(states[:, :1]), states], dim=1)
    decay_chunk = torch.exp(segsum(F.pad(a_cum[..., -1], (1, 0))))
    states = torch.einsum("bhzc,bchpn->bzhpn", decay_chunk, states)[:, :-1]
    y_off = torch.einsum("bcln,bchpn,bhcl->bclhp", c, states, torch.exp(a_cum))
    return (y_diag + y_off).reshape(bsz, nc * chunk, heads, p)[:, :length]


class ICBiMamba2(nn.Module):
    """Bidirectional Mamba-2 block whose FiLM and per-head step size follow impact evidence."""
    def __init__(self, width, cond_dim, expand=2, headdim=16, states=16, chunk=20, conditioned=True,
                 bidirectional=True, drop=.1):
        super().__init__()
        inner = width * expand; self.inner = inner; self.heads = inner // headdim; self.headdim = headdim
        self.states = states; self.chunk = chunk; self.conditioned = conditioned; self.bidirectional = bidirectional
        dirs = 2 if bidirectional else 1
        self.norm = nn.LayerNorm(width)
        self.in_proj = nn.Linear(width, 2 * inner + 2 * states + self.heads, bias=False)
        conv_dim = inner + 2 * states
        self.conv = nn.ModuleList([nn.Conv1d(conv_dim, conv_dim, 4, padding=3, groups=conv_dim) for _ in range(dirs)])
        dt = torch.exp(torch.rand(dirs, self.heads) * (math.log(.1) - math.log(1e-3)) + math.log(1e-3))
        self.dt_bias = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))
        self.A_log = nn.Parameter(torch.log(torch.empty(dirs, self.heads).uniform_(1, 16)))
        self.D = nn.Parameter(torch.ones(dirs, self.heads))
        self.out_norm = nn.LayerNorm(inner)
        self.out_proj = nn.Linear(inner, width, bias=False)
        self.drop = nn.Dropout(drop)
        if conditioned:
            self.film = nn.Conv1d(cond_dim, 2 * width, 3, padding=1)
            self.cdt = nn.Conv1d(cond_dim, dirs * self.heads, 3, padding=1)
            for m in (self.film, self.cdt): nn.init.zeros_(m.weight); nn.init.zeros_(m.bias)

    def scan(self, xbc, dt_raw, d):
        length = xbc.shape[1]
        xbc = F.silu(self.conv[d](xbc.transpose(1, 2))[..., :length].transpose(1, 2))
        x, b, c = xbc.split([self.inner, self.states, self.states], dim=-1)
        dt = F.softplus(dt_raw + self.dt_bias[d])
        a = -torch.exp(self.A_log[d]) * dt
        xh = x.reshape(*x.shape[:2], self.heads, self.headdim)
        y = ssd_scan(xh * dt[..., None], a, b, c, self.chunk) + xh * self.D[d][:, None]
        return y.reshape(*x.shape[:2], self.inner)

    def forward(self, z, cond):
        h = self.norm(z.transpose(1, 2))
        cdt = 0.
        if self.conditioned:
            scale, shift = self.film(cond).chunk(2, dim=1)
            h = h * (1 + scale.transpose(1, 2)) + shift.transpose(1, 2)
            cdt = self.cdt(cond).transpose(1, 2)
        gate, xbc, dt = self.in_proj(h).split([self.inner, self.inner + 2 * self.states, self.heads], dim=-1)
        dts = [dt + (cdt[..., i * self.heads:(i + 1) * self.heads] if self.conditioned else 0.)
               for i in range(2 if self.bidirectional else 1)]
        y = self.scan(xbc, dts[0], 0)
        if self.bidirectional:
            y = .5 * (y + self.scan(xbc.flip(1), dts[1].flip(1), 1).flip(1))
        y = self.out_norm(y) * F.silu(gate)
        return z + self.drop(self.out_proj(y)).transpose(1, 2)


class RawPatch(nn.Module):
    """Native 200 Hz samples -> one embedding per 50 ms token (no filtering)."""
    def __init__(self, width):
        super().__init__()
        self.embed = nn.Linear(PATCH * 8, width)

    def forward(self, raw):
        # raw: (B,L,PATCH*6) in g and deg/s
        b, l, _ = raw.shape
        r = raw.float().reshape(b, l, PATCH, 6)
        acc, gyro = r[..., :3], r[..., 3:]
        mags = torch.stack((acc.norm(dim=-1) - 1, gyro.norm(dim=-1) / 100), -1)
        feats = torch.cat((acc / 2, gyro / 250, mags), -1)
        return self.embed(torch.asinh(feats * 2).reshape(b, l, -1)).transpose(1, 2)


class ICBiMamba(nn.Module):
    def __init__(self, names, width=48, depth=3, raw=True, conditioned=True, bidirectional=True, pool="attnmax"):
        super().__init__()
        self.register_buffer("ci", torch.tensor([names.index(n) for n in CONDITION]))
        self.token = nn.Conv1d(len(names), width, 1)
        self.raw = RawPatch(width) if raw else None
        self.local = nn.ModuleList([nn.Conv1d(width, width, k, padding=k // 2, groups=width) for k in (3, 5, 9)])
        self.mix = nn.Conv1d(3 * width, width, 1)
        self.blocks = nn.ModuleList([ICBiMamba2(width, len(CONDITION), conditioned=conditioned,
                                                bidirectional=bidirectional) for _ in range(depth)])
        self.pool = pool
        self.score = nn.Sequential(nn.Conv1d(width, width // 2, 1), nn.Tanh(), nn.Conv1d(width // 2, 1, 1))
        h = 2 * width if pool == "attnmax" else width
        self.head = nn.Sequential(nn.LayerNorm(h), nn.Dropout(.15), nn.Linear(h, 1))

    def forward(self, x, raw=None):
        # x: (B,C,L) scaled IPMV tokens; raw: (B,L,60) physical samples
        cond = x.index_select(1, self.ci)
        z = self.token(x)
        if self.raw is not None: z = z + self.raw(raw)
        z = z + F.silu(self.mix(torch.cat([f(z) for f in self.local], 1)))
        for block in self.blocks: z = block(z, cond)
        attn = (z * self.score(z).softmax(-1)).sum(-1)
        feat = torch.cat((attn, z.amax(-1)), 1) if self.pool == "attnmax" else attn
        return self.head(feat).squeeze(-1)
