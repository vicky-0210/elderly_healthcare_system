"""Trial-local tokens and windows. Only the bag carries a fall label."""
import hashlib
import json
from pathlib import Path
import os
import numpy as np

from .sisfall import CACHE,COUNTS_TO_PHYSICAL,load_cache
from wearable_imu.preprocessing import ipmv


def build_tokens(settings,cache=CACHE):
    manifest,counts=load_cache(cache)
    signature=hashlib.sha256(Path(ipmv.__file__).read_bytes()+Path(__file__).read_bytes()+
        (Path(cache)/"manifest.csv").read_bytes()+json.dumps(settings,sort_keys=True).encode()).hexdigest()[:18]
    folder=Path(cache)/"tokens"
    folder.mkdir(parents=True,exist_ok=True)
    path=folder/(signature+".npy")
    names=ipmv.channel_names(("db2","sym4"))
    lengths=manifest.n.to_numpy()//(200//ipmv.TOKEN_HZ)
    offsets=np.r_[0,np.cumsum(lengths)[:-1]]
    if not path.is_file():
        tmp=path.with_suffix(f".{os.getpid()}.tmp.npy")
        output=np.lib.format.open_memmap(tmp,mode="w+",dtype="f",shape=(int(lengths.sum()),len(names)))
        for row,start,length in zip(manifest.itertuples(),offsets,lengths):
            data=counts[row.offset:row.offset+row.n].astype("f")*COUNTS_TO_PHYSICAL
            if settings.get("fs",200)==100:
                from scipy.signal import sosfilt,sosfilt_zi,butter
                sos=butter(4,40,fs=200,output="sos")
                data=sosfilt(sos,data,axis=0,zi=sosfilt_zi(sos)[:,:,None]*data[0])[0][1::2]
            output[start:start+length]=ipmv.IPMV(**settings).push(data)
        output.flush()
        del output
        try:
            tmp.replace(path)
        except PermissionError:
            if not path.is_file(): raise
            tmp.unlink()
    return manifest,np.load(path,mmap_mode="r"),names,offsets,lengths


def window_index(lengths,offsets,seconds=2,overlap=.5,cap=8):
    width=int(seconds*ipmv.TOKEN_HZ)
    hop=int(round(width*(1-overlap)))
    if hop<1 or (lengths<width).any():
        raise ValueError("Window unsupported by trial lengths")
    each=[np.arange(o,o+n-width+1,hop,dtype=np.int64) for n,o in zip(lengths,offsets)]
    sizes=np.array([len(v) for v in each])
    starts=np.concatenate(each)
    group=np.repeat(np.arange(len(each)),sizes)
    # Uniformly cover each training trial. Repeating locations for short bags
    # does not change a max-MIL logit. Validation and test scan every window.
    first=np.r_[0,np.cumsum(sizes)[:-1]]
    subset=np.stack([o+np.rint(np.linspace(0,n-1,cap)).astype(int) for o,n in zip(first,sizes)])
    return starts,group,subset,width
