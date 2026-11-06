"""Development-only synthetic GPU smoke test; no clinical performance claims."""
from pathlib import Path
import sys,time,json
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
import torch
from wearable_ecg.models.backbones import ECGClassifier

if __name__=="__main__":
    torch.set_num_threads(2)
    device="cuda" if torch.cuda.is_available() else "cpu"
    for name in ("cnn","cnn_transformer","cnn_ssm"):
        m=ECGClassifier(name).to(device)
        optimizer=torch.optim.AdamW(m.parameters(),lr=.0003)
        x=torch.randn(32,3600,device=device); f=torch.randn(32,16,device=device)
        scaler=torch.amp.GradScaler("cuda",enabled=device=="cuda")
        times=[]
        for i in range(12):
            if device=="cuda": torch.cuda.synchronize()
            t=time.perf_counter(); optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device,dtype=torch.float16,enabled=device=="cuda"):
                loss=m(x,f).square().mean()
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
            if device=="cuda": torch.cuda.synchronize()
            times.append(time.perf_counter()-t)
        print(json.dumps({"architecture":name,"parameters":sum(p.numel() for p in m.parameters()),
                          "batch_32_training_seconds":sum(times[3:])/len(times[3:]),
                          "finite_loss":bool(torch.isfinite(loss)),
                          "gpu_peak_mb":torch.cuda.max_memory_allocated()/1e6 if device=="cuda" else None}),flush=True)
        del m,optimizer,x,f
        if device=="cuda": torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
