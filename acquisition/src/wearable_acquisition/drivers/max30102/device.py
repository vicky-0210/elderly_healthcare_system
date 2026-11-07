"""MAX30102 red/IR acquisition at 100 Hz using the Linux SMBus interface."""
import time
import numpy as np


class MAX30102:
    def __init__(self,bus,address=0x57,led=0x24,clock=time.monotonic):
        if not 1<=led<=255: raise ValueError("LED register outside supported range")
        self.bus,self.address,self.clock=bus,address,clock
        if self.read(0xff)!=0x15: raise IOError("Unexpected optical sensor part ID")
        self.write(0x09,0x40)
        for _ in range(100):
            if not self.read(0x09)&0x40: break
            time.sleep(.002)
        else: raise IOError("MAX30102 reset timed out")
        # 4096 nA, 100 samples/s, 411 us pulse width, 18-bit ADC.
        for r,v in ((0x08,0x0f),(0x0a,0x27),(0x0c,led),(0x0d,led)):
            self.write(r,v)
            if self.read(r)!=v: raise IOError(f"MAX30102 configuration mismatch at {r:#x}")
        self.write(0x09,0x03); self.clear()

    def write(self,r,v): self.bus.write_byte_data(self.address,r,v)
    def read(self,r): return self.bus.read_byte_data(self.address,r)
    def close(self): self.write(0x09,0x80)

    def clear(self):
        for reg in (0x04,0x05,0x06): self.write(reg,0)
        self.anchor=None; self.count=0; self.last=self.clock()

    def poll(self):
        now=self.clock()
        pointer=self.bus.read_i2c_block_data(self.address,0x04,3)
        wr,overflow,rd=pointer
        # With 32 slots pointer equality is ambiguous after a long host stall.
        if overflow or now-self.last>=.30:
            self.clear(); return None,None,"fifo_gap"
        self.last=now; count=(wr-rd)&31
        if not count: return None,None,"waiting"
        packet=bytearray()
        for i in range(0,count,5):
            n=6*min(5,count-i)
            block=self.bus.read_i2c_block_data(self.address,0x07,n)
            if len(block)!=n:
                self.clear(); return None,None,"short_read"
            packet.extend(block)
        fields=np.frombuffer(packet,dtype=np.uint8).astype(np.uint32).reshape(-1,2,3)
        optical=((fields[:,:,0]&3)<<16)|(fields[:,:,1]<<8)|fields[:,:,2]
        if self.anchor is None: self.anchor=now-(count-1)/100
        stamp=self.anchor+self.count/100; self.count+=count
        return stamp,optical,"ok"
