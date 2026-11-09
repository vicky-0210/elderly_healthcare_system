"""Models. Baseline 1D-CNN first; multi-scale CNN / Transformer / cross-channel attention come later."""
import torch
import torch.nn as nn


def block(cin, cout, k):
    return nn.Sequential(nn.Conv1d(cin, cout, k, padding=k // 2), nn.BatchNorm1d(cout), nn.ReLU(),
                         nn.MaxPool1d(2))


class BaselineCNN(nn.Module):
    """4 conv blocks (T/16 resolution) -> global avg+max pool -> linear."""

    def __init__(self, in_ch, n_classes, width=32, drop=0.3):
        super().__init__()
        w = width
        self.features = nn.Sequential(block(in_ch, w, 7), block(w, 2 * w, 7), block(2 * w, 4 * w, 5),
                                      block(4 * w, 4 * w, 5))
        self.head = nn.Sequential(nn.Dropout(drop), nn.Linear(8 * w, n_classes))

    def forward(self, x):
        h = self.features(x)
        return self.head(torch.cat([h.mean(-1), h.amax(-1)], 1))


class MultiScaleBlock(nn.Module):
    """3 parallel convs (short / medium / long kernel) = 3 temporal scales, concatenated, then pooled by 2."""

    def __init__(self, cin, cout_each, ks=(3, 7, 15)):
        super().__init__()
        self.branches = nn.ModuleList([nn.Conv1d(cin, cout_each, k, padding=k // 2) for k in ks])
        self.post = nn.Sequential(nn.BatchNorm1d(cout_each * len(ks)), nn.ReLU(), nn.MaxPool1d(2))

    def forward(self, x):
        return self.post(torch.cat([b(x) for b in self.branches], 1))


class HybridNet(nn.Module):
    """Diagram model: multi-scale CNN encoder -> temporal Transformer -> cross-channel attention -> prediction.

    The CNN + Transformer are SHARED over input channels (each channel is encoded on its own, plus a learned channel
    embedding), giving one vector per channel; self-attention ACROSS channels then fuses them (PPG components).
    """

    def __init__(self, in_ch, n_classes, w=12, d=36, heads=4, layers=2, T=1500, drop=0.3):
        super().__init__()
        # strided stem (patch embedding, /4) keeps CPU cost low, then 2 multi-scale blocks (/4) -> T/16 tokens, 3w = d
        self.enc = nn.Sequential(nn.Conv1d(1, 3 * w, 8, stride=4, padding=2), nn.BatchNorm1d(3 * w), nn.ReLU(),
                                 MultiScaleBlock(3 * w, w), MultiScaleBlock(3 * w, w))
        self.pos = nn.Parameter(torch.zeros(1, T // 16 + 1, d))
        nn.init.normal_(self.pos, std=0.02)
        self.chan_emb = nn.Parameter(torch.zeros(in_ch, 1, d))
        nn.init.normal_(self.chan_emb, std=0.02)
        layer = nn.TransformerEncoderLayer(d, heads, 2 * d, dropout=0.1, batch_first=True, norm_first=True)
        self.temporal = nn.TransformerEncoder(layer, layers)
        self.cross = nn.MultiheadAttention(d, heads, batch_first=True)
        self.norm = nn.LayerNorm(d)
        self.head = nn.Sequential(nn.Dropout(drop), nn.Linear(2 * d, n_classes))
        self.in_ch = in_ch

    def forward(self, x):
        B, C, T = x.shape
        h = self.enc(x.reshape(B * C, 1, T)).transpose(1, 2)               # (B*C, T/16, d)
        h = h + self.pos[:, :h.shape[1]] + self.chan_emb.repeat(B, 1, 1)[:, :1]
        h = self.temporal(h).mean(1).reshape(B, C, -1)                      # one vector per channel
        a, _ = self.cross(h, h, h)                                          # attention across channels
        h = self.norm(h + a)
        return self.head(torch.cat([h.mean(1), h.amax(1)], 1))


def build(name, in_ch, n_classes):
    if name == "cnn":
        return BaselineCNN(in_ch, n_classes)
    if name == "hybrid":
        return HybridNet(in_ch, n_classes)
    raise ValueError(name)
