"""Within-training-subject masked reconstruction and component-preserving contrast."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def pretrain(model, x, f, groups, starts, permitted, method, generator, epochs=2, batch=128):
    """All modalities and engineered features are subject-partitioned before this call."""
    target_idx = [model.channels.index(k) for k in ("raw","tonic","phasic")]
    if method == "masked":
        head = nn.Linear(model.width, 4*len(target_idx)).to(x.device)
    else:
        head = nn.Sequential(nn.Linear(model.width,model.width),nn.GELU(),
                             nn.Linear(model.width,model.width)).to(x.device)
    optim = torch.optim.AdamW([*model.parameters(),*head.parameters()],lr=.002,weight_decay=.001)
    history = []
    for epoch in range(1,epochs+1):
        order = permitted[torch.randperm(len(permitted),device=x.device,generator=generator)]
        total, count = 0., 0
        for idx in order.split(batch):
            if len(idx) < 2:
                continue
            a, b = x[idx], f[idx]
            if method == "masked":
                # Hide the same patch across all correlated component/quality views.
                mask = torch.rand(len(a),a.shape[-1]//4,device=x.device,generator=generator) < .25
                masked = a.masked_fill(mask.repeat_interleave(4,dim=1)[:,None,:],0.)
                h,_ = model.sequence(masked,torch.zeros_like(b))
                target = a[:,target_idx].unfold(2,4,4).permute(0,2,1,3).flatten(2)
                loss = F.smooth_l1_loss(head(h)[mask],target[mask])
            else:
                views = []
                # Same small positive gain for all component channels, plus aligned
                # short temporal masks. Quality flags keep their physical meaning.
                continuous = [i for i,n in enumerate(model.channels) if n not in ("quality","artifact","missing")]
                for _ in range(2):
                    view = a.clone()
                    gain = .97 + .06*torch.rand(len(a),1,1,device=x.device,generator=generator)
                    view[:,continuous] *= gain
                    mask = torch.rand(len(a),1,a.size(-1)//4,device=x.device,generator=generator) < .06
                    view = view.masked_fill(mask.repeat_interleave(4,-1),0.)
                    h,_ = model.sequence(view,torch.zeros_like(b))
                    views.append(F.normalize(head(h.mean(1)),dim=1))
                similarity = (views[0] @ views[1].T)/.2
                overlapping = groups[idx,None].eq(groups[idx][None,:]) & ((starts[idx,None]-starts[idx][None,:]).abs() < a.shape[-1]/4)
                overlapping.fill_diagonal_(False)
                similarity = similarity.masked_fill(overlapping,-1e4)
                target = torch.arange(len(idx),device=x.device)
                loss = .5*(F.cross_entropy(similarity,target)+F.cross_entropy(similarity.T,target))
            optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.)
            optim.step()
            total += loss.detach()*len(idx)
            count += len(idx)
        history.append(dict(phase="ssl",epoch=epoch,loss=float(total/count)))
    return history
