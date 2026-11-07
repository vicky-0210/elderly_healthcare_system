"""Read, calibrate and score real FIFO frames, or fit a device calibration."""
import argparse
import json
from pathlib import Path
import sys,time
import numpy as np

repo=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(repo/"modules/imu/src"),str(repo/"acquisition/src")]
from wearable_acquisition.drivers.mpu9250.calibration import apply,fit_calibration
from wearable_acquisition.drivers.mpu9250.device import MPU9250
from wearable_imu.inference.predictor import IMUPredictor


def capture_calibration(args):
    from smbus2 import SMBus
    if not 2<=args.seconds<=60: raise ValueError("Use 2–60 stationary seconds per pose")
    n=args.seconds*args.fs
    poses=[]
    with SMBus(args.bus) as bus:
        device=MPU9250(bus,args.address,args.fs)
        try:
            for face in ("+X","-X","+Y","-Y","+Z","-Z","gyro rest"):
                input(f"Hold the device still with {face} upward (any pose for gyro rest), then press Enter. ")
                device.restart()
                blocks=[]; count=0; deadline=time.monotonic()+args.seconds+5
                while count<n:
                    if time.monotonic()>deadline: raise TimeoutError(f"Timed out collecting {face}")
                    _,x,status=device.read()
                    if status=="waiting": time.sleep(.005); continue
                    if status!="ok": raise IOError(f"Interrupted {face} capture: {status}")
                    blocks.append(x); count+=len(x)
                poses.append(np.concatenate(blocks)[:n])
        finally:
            device.close()
    faces=np.stack(poses[:6])[:,:,:3]; rest=poses[-1][:,3:]
    fit_calibration(faces,rest)  # Check pose order and motion before saving the capture.
    np.savez_compressed(args.output,faces_g=faces,stationary_dps=rest)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    commands=p.add_subparsers(dest="cmd")
    fit=commands.add_parser("calibrate"); fit.add_argument("capture",type=Path); fit.add_argument("output",type=Path)
    capture=commands.add_parser("capture-calibration"); capture.add_argument("output",type=Path)
    capture.add_argument("--fs",type=int,choices=[100,200],default=200); capture.add_argument("--seconds",type=int,default=5)
    capture.add_argument("--bus",type=int,default=1); capture.add_argument("--address",type=lambda v:int(v,0),default=0x68)
    live=commands.add_parser("live"); live.add_argument("export",type=Path); live.add_argument("calibration",type=Path)
    live.add_argument("--bus",type=int,default=1); live.add_argument("--address",type=lambda v:int(v,0),default=0x68)
    live.add_argument("--trt-engine",type=Path)
    args=p.parse_args()
    if not args.cmd: p.error("Choose capture-calibration, calibrate or live")
    if args.cmd=="capture-calibration": capture_calibration(args); return
    if args.cmd=="calibrate":
        with np.load(args.capture,allow_pickle=False) as d: params=fit_calibration(d["faces_g"],d["stationary_dps"])
        args.output.write_text(json.dumps(params,indent=2)); return
    from smbus2 import SMBus
    predictor=IMUPredictor(args.export,args.trt_engine)
    calibration=json.loads(args.calibration.read_text())
    with SMBus(args.bus) as bus:
        device=MPU9250(bus,args.address,predictor.fs)
        try:
            while True:
                t,x,status=device.read()
                if status=="waiting": time.sleep(.005); continue
                if status!="ok":
                    predictor.reset(); print(json.dumps(dict(signal="imu",status=status)),flush=True); continue
                for row in predictor.push(apply(x,calibration),t):
                    print(json.dumps(dict(signal="imu",clock="fifo_nominal_monotonic",**row)),flush=True)
        except KeyboardInterrupt: pass
        finally: device.close()

if __name__=="__main__": main()
