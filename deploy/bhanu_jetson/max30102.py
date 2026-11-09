"""Minimal MAX30102 driver (I2C, SpO2 mode) + per-second SpO2 estimate. Needs smbus2. I2C bus 7 on Orin Nano 40-pin header
(pins 3/5 = SDA/SCL, check `i2cdetect -l`; address 0x57).

WARNING: the SpO2 here is the textbook ratio-of-ratios estimate  SpO2 = 110 - 25 R  (R = (AC_red/DC_red)/(AC_ir/DC_ir)), NOT a
calibrated medical value. The detector was trained on a hospital device's SpO2, so you must compare this sensor with a reference
oximeter and fix the mapping (A - B*R) before trusting alarms.
"""
import time

import numpy as np
from scipy.signal import butter, sosfiltfilt

ADDR, RATE = 0x57, 100
FINGER_MIN = 20000        # IR DC below this = no finger


class MAX30102:
    def __init__(self, bus=7):
        from smbus2 import SMBus
        self.i2c = SMBus(bus)
        assert self.i2c.read_byte_data(ADDR, 0xFF) == 0x15, "MAX30102 not found (PART_ID != 0x15)"
        w = lambda r, v: self.i2c.write_byte_data(ADDR, r, v)
        w(0x09, 0x40); time.sleep(0.1)          # reset
        w(0x08, 0x1F)                           # no averaging, FIFO rollover on
        w(0x09, 0x03)                           # SpO2 mode (red + IR)
        w(0x0A, 0x27)                           # ADC 4096 nA, 100 Hz, 411 us pulse
        w(0x0C, 0x3F); w(0x0D, 0x3F)            # LED current red / IR (~12 mA)
        w(0x04, 0); w(0x05, 0); w(0x06, 0)      # clear FIFO pointers
        self.red, self.ir = [], []

    def _drain(self):
        wr, rd = self.i2c.read_byte_data(ADDR, 0x04), self.i2c.read_byte_data(ADDR, 0x06)
        n = (wr - rd) % 32
        for _ in range(n):
            b = self.i2c.read_i2c_block_data(ADDR, 0x07, 6)
            self.red.append(((b[0] << 16 | b[1] << 8 | b[2]) & 0x3FFFF)); self.ir.append(((b[3] << 16 | b[4] << 8 | b[5]) & 0x3FFFF))

    def read_second(self):
        """Blocks until RATE samples are available. Returns (spo2_percent_or_nan, ir_samples (RATE,))."""
        while len(self.ir) < RATE:
            self._drain(); time.sleep(0.05)
        red, ir = np.array(self.red[:RATE], float), np.array(self.ir[:RATE], float)
        del self.red[:RATE], self.ir[:RATE]
        self.hist = getattr(self, "hist", {"red": np.zeros(0), "ir": np.zeros(0)})
        self.hist["red"], self.hist["ir"] = np.r_[self.hist["red"], red][-4 * RATE:], np.r_[self.hist["ir"], ir][-4 * RATE:]
        return spo2_from(self.hist["red"], self.hist["ir"]), ir


def spo2_from(red, ir):
    """SpO2 from the last <=4 s of red / IR. NaN when there is no finger or too little data."""
    if len(ir) < 3 * RATE or ir.mean() < FINGER_MIN:
        return np.nan
    sos = butter(2, [0.5, 5.0], btype="band", fs=RATE, output="sos")
    ac_r, ac_i = np.std(sosfiltfilt(sos, red)), np.std(sosfiltfilt(sos, ir))
    if ac_i < 1e-6 or red.mean() < 1:
        return np.nan
    R = (ac_r / red.mean()) / (ac_i / ir.mean())
    return float(np.clip(np.round(110.0 - 25.0 * R), 50, 100))
