"""Compact wrist arousal networks. Inputs are end-of-window 4-Hz observations."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class DilatedBlock(nn.Module):
    def __init__(self, width, dilation):
        super().__init__()
        self.pad = 2 * dilation
        self.conv = nn.Conv1d(width, width, 3, dilation=dilation)
        self.out = nn.Conv1d(width, width, 1)
        self.norm = nn.LayerNorm(width)

    def forward(self, x):
        h = self.conv(F.pad(x, (self.pad, 0)))
        return self.norm((x + self.out(F.gelu(h))).transpose(1, 2)).transpose(1, 2)


class TemporalBranch(nn.Module):
    """Learn embeddings of four consecutive native samples per token."""
    def __init__(self, channels, width, kind):
        super().__init__()
        self.patch = nn.Conv1d(channels, width, 4, stride=4)
        self.kind = kind
        if kind == 'cnn':
            self.layers = nn.Sequential(nn.Conv1d(width, width, 5, padding=2), nn.GELU(),
                                        nn.Conv1d(width, width, 3, padding=1), nn.GELU())
        elif kind == 'cnn_tcn':
            self.layers = nn.Sequential(nn.Conv1d(width,width,3,padding=1), nn.GELU(),
                                        DilatedBlock(width,2), DilatedBlock(width,4))
        elif kind == 'tcn':
            self.layers = nn.Sequential(*(DilatedBlock(width, d) for d in (1, 2, 4)))
        elif kind == 'multiscale':
            self.branches = nn.ModuleList(DilatedBlock(width, d) for d in (1, 4, 12))
            self.mix = nn.Conv1d(3 * width, width, 1)

    def forward(self, x):
        x = self.patch(x)
        if self.kind == 'multiscale':
            return self.mix(torch.cat([path(x) for path in self.branches], 1)).transpose(1, 2)
        if self.kind != 'linear':
            x = self.layers(x)
        return x.transpose(1, 2)


class ComponentRefiner(nn.Module):
    """Shared pointwise refinement of continuously estimated tonic/phasic states.

    The pointwise operation commutes with window extraction. Reconstruction is
    measured in physical conductance and standardized by training clean-EDA IQR.
    """
    def __init__(self, channels, centers, scales):
        super().__init__()
        self.indices = [channels.index(n) for n in ('clean', 'tonic', 'phasic')]
        self.register_buffer('centers', torch.as_tensor(centers).float()[self.indices][None, :, None])
        self.register_buffer('scales', torch.as_tensor(scales).float()[self.indices][None, :, None])
        self.adjust = nn.Sequential(nn.Conv1d(3, 8, 1), nn.Tanh(), nn.Conv1d(8, 3, 1))

    def components(self, x):
        standardized = x[:, self.indices]
        physical = standardized * self.scales + self.centers
        clean, tonic, phasic = physical.unbind(1)
        scale = self.scales[0, 0, 0]
        corrections = self.adjust(standardized).tanh() * scale * .25
        scl = (tonic + corrections[:, 0]).clamp_min(0.)
        scr = F.softplus(8 * (phasic + corrections[:, 1]) / scale) * scale / 8
        residual = corrections[:, 2]
        return scl, scr, residual

    def forward(self, x):
        scl, scr, residual = self.components(x)
        scale = self.scales[0, 0, 0]
        clean = x[:, self.indices[0]] * scale + self.centers[0, 0, 0]
        reconstruction = ((clean - scl - scr - residual) / scale).abs().mean()
        roughness = torch.diff(scl / scale, n=2, dim=-1).abs().mean()
        physics = reconstruction + .1 * roughness + .05 * (residual / scale).abs().mean() + .01 * (scr / scale).mean()
        adjusted = x.clone()
        adjusted[:, self.indices[1]] = (scl - self.centers[0, 1]) / self.scales[0, 1]
        adjusted[:, self.indices[2]] = (scr - self.centers[0, 2]) / self.scales[0, 2]
        return adjusted, physics


class WristNet(nn.Module):
    def __init__(self, cfg, channels, n_features, n_classes, centers, scales):
        super().__init__()
        self.cfg = cfg
        self.channels = channels
        self.kind = cfg['model']
        width = cfg.get('width', 16)
        self.width = width
        self.pg = self.kind in ('pg', 'physics')
        self.split = self.pg or self.kind in ('tonic_phasic', 'tonic_attention')
        self.q_idx = channels.index('quality') if 'quality' in channels else None
        self.refiner = ComponentRefiner(channels, centers, scales) if self.kind == 'physics' else None
        self.hand = cfg.get('hand', self.pg or self.kind == 'hybrid_features')
        if self.hand:
            self.hand_encoder = nn.Sequential(nn.Linear(n_features, width), nn.GELU(), nn.LayerNorm(width))
        if self.split:
            self.tonic_idx = [i for i,n in enumerate(channels) if n in ('raw','clean','tonic','relative_tonic','tonic_slope')]
            self.phasic_idx = [i for i,n in enumerate(channels) if n in ('phasic','relative_phasic','phasic_slope','phasic_energy')]
            self.ctx_idx = [i for i,n in enumerate(channels) if i not in self.tonic_idx + self.phasic_idx]
            if not self.phasic_idx:  # decomposition ablation: both branches see raw/clean
                self.phasic_idx = list(self.tonic_idx)
            self.tonic = TemporalBranch(len(self.tonic_idx), width, 'tcn')
            self.phasic = TemporalBranch(len(self.phasic_idx), width, 'multiscale' if cfg.get('multiscale', True) else 'tcn')
            self.cross = nn.MultiheadAttention(width, 2, batch_first=True) if self.kind=='tonic_attention' or cfg.get('fusion')=='cross' else None
            self.use_context = self.pg and bool(self.ctx_idx)
            if self.use_context:
                self.context = TemporalBranch(len(self.ctx_idx), width, 'linear')
            branches = 2 + int(self.use_context) + int(self.hand)
            self.fusion = cfg.get('fusion', 'quality' if self.pg else 'concat')
            self.concat = nn.Linear(branches * width, width) if self.fusion in ('concat','cross') else None
            self.gate = nn.Linear(width * branches + (self.fusion=='quality'), branches) if self.fusion in ('quality','gated') else None
            self.branch_logits = nn.Parameter(torch.zeros(branches)) if self.fusion=='weighted' else None
        else:
            conv = {'cnn':'cnn','cnn_bilstm':'cnn','cnn_transformer':'cnn','cnn_tcn_attention':'cnn_tcn','tcn':'tcn',
                    'tcn_attention':'tcn','multitcn':'multiscale','multitcn_gate':'multiscale',
                    'hybrid_features':'multiscale','multitcn_gru':'multiscale',
                    'multitcn_transformer':'multiscale','reliable':'tcn'}.get(self.kind,'linear')
            self.eda_idx = [i for i,n in enumerate(channels) if n not in ('acc_magnitude','acc_dynamic','acc_rms','acc_jerk','motion','quality','artifact','missing','temperature','bvp_rms','bvp_rate','bvp_pulse')]
            self.encoder = TemporalBranch(len(self.eda_idx) if self.kind=='reliable' else len(channels),width,conv)
            if self.kind=='reliable':
                self.context_idx = [i for i in range(len(channels)) if i not in self.eda_idx]
                self.context = TemporalBranch(len(self.context_idx), width, 'tcn')
                self.context_gate = nn.Linear(2 * width + 1, width)
        self.temporal = None
        aggregator = cfg.get('aggregation', '')
        if self.kind in ('lstm','bilstm','gru','bigru','cnn_bilstm','multitcn_gru') or aggregator in ('gru','bigru'):
            bidirectional = self.kind in ('bilstm','bigru','cnn_bilstm') or aggregator=='bigru'
            cell = nn.LSTM if self.kind in ('lstm','bilstm','cnn_bilstm') else nn.GRU
            self.temporal = cell(width, width//2 if bidirectional else width, batch_first=True, bidirectional=bidirectional)
        if self.kind in ('transformer','cnn_transformer','multitcn_transformer') or aggregator=='transformer':
            self.temporal = nn.TransformerEncoderLayer(width, 2, 2*width, .1, batch_first=True, activation='gelu')
            positions = torch.arange(180).float()[:,None]
            freq = torch.exp(-torch.arange(0,width,2).float()*9.21034/width)
            encoding = torch.zeros(180,width)
            encoding[:,0::2], encoding[:,1::2] = torch.sin(positions*freq),torch.cos(positions*freq)
            self.register_buffer('position',encoding)
        self.attentive = self.split or self.kind in ('cnn_tcn_attention','tcn_attention','multitcn_gate','reliable','hybrid_features')
        self.attn = nn.Linear(width,1) if self.attentive else None
        self.feature_gate = nn.Linear(width,width) if self.kind=='multitcn_gate' else None
        out_width = 2*width if self.hand and not self.split else width
        self.classifier = nn.Sequential(nn.LayerNorm(out_width),nn.Dropout(.15),nn.Linear(out_width,n_classes))

    def sequence(self, x, f):
        penalty = x.new_zeros(())
        if self.refiner is not None:
            x, penalty = self.refiner(x)
        quality = F.avg_pool1d(x[:,self.q_idx:self.q_idx+1],4,stride=4).transpose(1,2) if self.q_idx is not None else x.new_ones(x.shape[0], x.shape[-1]//4,1)
        if self.split:
            tonic, phasic = self.tonic(x[:,self.tonic_idx]), self.phasic(x[:,self.phasic_idx])
            if self.cross is not None:
                tonic = tonic + self.cross(tonic,phasic,phasic,need_weights=False)[0]
            parts = [tonic,phasic]
            if self.use_context:
                parts.append(self.context(x[:,self.ctx_idx]))
            if self.hand:
                parts.append(self.hand_encoder(f)[:,None].expand_as(tonic))
            joined = torch.cat(parts,-1)
            if self.concat is not None:
                h = self.concat(joined)
            else:
                if self.gate is None:
                    weights = self.branch_logits.softmax(0)[None,None,:,None]
                else:
                    gate_input = torch.cat([joined,quality],-1) if self.fusion=='quality' else joined
                    logits = self.gate(gate_input)
                    if self.fusion=='quality':
                        # A weak physiologic prior reduces EDA branch odds during poor contact.
                        prior = torch.cat([quality.clamp(.05,1).log().expand(-1,-1,2),torch.zeros_like(logits[:,:,2:])],-1)
                        logits = logits + prior
                    weights = logits.softmax(-1).unsqueeze(-1)
                h = (torch.stack(parts,2)*weights).sum(2)
        elif self.kind=='reliable':
            h = self.encoder(x[:,self.eda_idx]); context = self.context(x[:,self.context_idx])
            gate = torch.sigmoid(self.context_gate(torch.cat([h,context,quality],-1))) * (.25+.75*quality)
            h = gate*h+(1-gate)*context
        else:
            h = self.encoder(x)
        if isinstance(self.temporal,(nn.LSTM,nn.GRU)):
            h = self.temporal(h)[0]
        elif self.temporal is not None:
            h = self.temporal(h+self.position[:h.size(1)])
        if self.feature_gate is not None:
            h = h*torch.sigmoid(self.feature_gate(h.mean(1)))[:,None]
        return h,penalty

    def forward(self,x,f):
        h, penalty = self.sequence(x,f)
        pooled = (h*self.attn(h).softmax(1)).sum(1) if self.attentive else h.mean(1)
        if self.hand and not self.split:
            pooled = torch.cat([pooled,self.hand_encoder(f)],1)
        return self.classifier(pooled),penalty


