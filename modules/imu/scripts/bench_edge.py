"""Profile complete 1-hop updates and verify a synthetic golden sequence on the target."""
import argparse,json,platform,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
import numpy as np
from wearable_imu.inference.predictor import IMUPredictor


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("export",type=Path); p.add_argument("--trt-engine",type=Path)
    p.add_argument("--build-reference",action="store_true"); p.add_argument("--iterations",type=int,default=300)
    p.add_argument("--output",type=Path); args=p.parse_args()
    if args.iterations<10: p.error("At least 10 timed hops are required")
    pred=IMUPredictor(args.export,args.trt_engine); file=args.export/"synthetic_reference.npz"
    if args.build_reference:
        if args.trt_engine: p.error("Build the golden vector with the host reference runtime")
        t=np.arange(10*pred.fs)/pred.fs
        x=np.column_stack((.4*np.sin(4*t),1+.3*np.cos(8*t),.5*np.sin(t),40*np.sin(5*t),55*np.cos(3*t),9*np.sin(t)))
        x[int(pred.fs*4.2):int(pred.fs*4.2)+3]+=[5.,-3.,2.,240.,55.,34.]
        reference=np.array([r["window_logit"] for r in pred.push(x,0.)])
        np.savez_compressed(file,physical=x,logits=reference)
    with np.load(file,allow_pickle=False) as data: x,expected=data["physical"],data["logits"]
    pred.reset(); actual=np.array([r["window_logit"] for r in pred.push(x,0.)])
    err=float(np.abs(actual-expected).max())
    flips=int(np.sum((actual>=pred.meta["threshold"])!=(expected>=pred.meta["threshold"])))
    if err>(.12 if args.trt_engine else .001) or flips: raise ValueError(f"Golden-vector disagreement: {err} logit, {flips} decisions")
    pred.reset(); count=pred.fs*pred.hop//20
    timing=[]; decisions=0
    for i in range(args.iterations+10):
        offset=i*count; block=x[(offset+np.arange(count))%len(x)]
        start=time.perf_counter(); rows=pred.push(block,offset/pred.fs); dt=(time.perf_counter()-start)*1000
        if i>=10:
            timing.append(dt); decisions+=len(rows)
    board=Path("/proc/device-tree/model")
    result=dict(model=pred.meta["experiment"],hardware=board.read_bytes().rstrip(b"\x00").decode() if board.exists() else platform.platform(),
        engine="TensorRT" if args.trt_engine else pred.meta["engine"],golden_max_logit_error=err,decisions_changed=flips,
        decisions=decisions,hops=len(timing),p50_ms=float(np.percentile(timing,50)),p95_ms=float(np.percentile(timing,95)),
        max_ms=max(timing),hop_ms=pred.hop/20*1000,missed_deadlines=sum(t>pred.hop/20*1000 for t in timing))
    if args.output: args.output.write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))

if __name__=="__main__": main()
