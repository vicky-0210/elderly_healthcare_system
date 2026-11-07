"""MPU-9250 I2C FIFO reader. No magnetometer or dataset code is involved."""
import time
import numpy as np


class MPU9250:
    def __init__(self,bus,address=0x68,fs=200,clock=time.monotonic):
        if fs not in (100,200): raise ValueError("Expected 100 or 200 Hz")
        self.bus,self.address,self.fs,self.clock=bus,address,fs,clock
        if bus.read_byte_data(address,0x75)!=0x71: raise IOError("WHO_AM_I does not identify an MPU-9250")
        self.write(0x6B,1); time.sleep(.05)
        for reg,value in [(0x6C,0),(0x1A,3),(0x1B,24),(0x1C,24),(0x1D,3),(0x19,1000//fs-1)]:
            self.write(reg,value)
            if bus.read_byte_data(address,reg)!=value: raise IOError(f"Configuration readback failed for {reg:#x}")
        self.write(0x38,0x10)  # FIFO overflow interrupt status
        self.restart()

    def write(self,reg,value): self.bus.write_byte_data(self.address,reg,value)

    def restart(self):
        self.write(0x23,0); self.write(0x6A,0x04); self.write(0x6A,0x40)
        self.write(0x23,0x78)  # accelerometer + three gyroscope axes
        self.origin=None; self.count=0

    def close(self):
        self.write(0x23,0); self.write(0x6A,0)

    def read(self):
        status=self.bus.read_byte_data(self.address,0x3a)
        size=int.from_bytes(bytes(self.bus.read_i2c_block_data(self.address,0x72,2)),"big")
        if status&0x10 or size>=512:
            self.restart(); return None,None,"fifo_overflow"
        frames=size//12
        if not frames: return None,None,"waiting"
        # The oldest FIFO sample is the first six signed big-endian axis values.
        now=self.clock()
        if self.origin is None: self.origin=now-(frames-1)/self.fs
        stamp=self.origin+self.count/self.fs
        data=bytearray()
        for i in range(0,frames,2):
            n=min(2,frames-i)*12
            block=self.bus.read_i2c_block_data(self.address,0x74,n)
            if len(block)!=n:
                self.restart(); return None,None,"short_fifo_read"
            data.extend(block)
        self.count+=frames
        physical=np.frombuffer(data,dtype=">i2").reshape(-1,6).astype(float)
        physical[:,:3]/=2048.; physical[:,3:]/=16.4
        return stamp,physical,"ok"
