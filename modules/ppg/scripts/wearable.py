import argparse,json,sys,time
from pathlib import Path
repo=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(repo/"modules/ppg/src"),str(repo/"acquisition/src"),str(repo/"packages/common/src")]
from wearable_acquisition.drivers.max30102.device import MAX30102
from wearable_ppg.inference.predictor import PPGPredictor

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("bundle",type=Path); p.add_argument("--bus",type=int,default=1)
    p.add_argument("--channel",choices=["ir","red"],default="ir"); p.add_argument("--polarity",type=int,choices=[-1,1],default=1)
    p.add_argument("--led-register",type=lambda v:int(v,0),default=0x24); args=p.parse_args()
    from smbus2 import SMBus
    engine=PPGPredictor(args.bundle,100,args.polarity)
    with SMBus(args.bus) as bus:
        optical=MAX30102(bus,led=args.led_register)
        try:
            while True:
                stamp,channels,status=optical.poll()
                if status=="waiting": time.sleep(.01); continue
                if status!="ok":
                    engine.reset(); print(json.dumps(dict(status=status)),flush=True); continue
                for row in engine.push(channels[:,int(args.channel=="ir")],stamp):
                    print(json.dumps(dict(signal="ppg",channel=args.channel,**row)),flush=True)
        except KeyboardInterrupt: pass
        finally: optical.close()
