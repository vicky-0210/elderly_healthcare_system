# MAX30102 acquisition and validation

The Linux I²C driver configures simultaneous red/IR acquisition at 100 Hz, reads the 18-bit FIFO, and preserves an estimated monotonic sample clock. A FIFO overflow, short read or extended polling stall resets the inference history. The frontend also resets after invalid or flat packets. Verify LED current, contact, placement, pulse polarity and ambient-light behavior for the actual board. [MAX30102 datasheet](https://www.analog.com/media/en/technical-documentation/data-sheets/max30102.pdf)

Copy the validated ONNX bundle to the board. Provide a board-compatible Python environment with NumPy, SciPy, ONNX Runtime with opset 17 support and smbus2. Run from the repository checkout, using the bus connected to the sensor.

~~~sh
python modules/ppg/scripts/wearable.py artifacts/ppg/exports/osasud_v1/cnn_transformer --bus 1 --channel ir
~~~

The research CLI prints JSON lines to stdout. Timestamp denotes the start of the scored second. Available-at denotes the end of the five-second block. The first block becomes available after 30 contiguous seconds. The LED register, optical channel and polarity are explicit CLI options. Record hardware settings with each capture. Keep red and IR synchronized for analysis. OSASUD supports training with one monitor waveform. It supplies no paired labels for supervised red/IR fusion.

The completed transfer study uses simulated rate, gain, polarity, noise and dropout changes on held-out Mindray traces. To measure hardware performance, collect simultaneous MAX30102 and reference respiratory recordings with independent synchronized clocks and expert labels. Audit drift, missingness, saturation and site-specific polarity. Use development participants for channel selection and calibration, freeze the configuration, and evaluate separate participants for event sensitivity, false events per usable hour, unavailable time and latency. Measure Jetson p50/p95 end-to-end timing, power and sustained FIFO behavior on the target board. Physical MAX30102 and Jetson measurements are pending. [Transfer results](../../../research/results/osasud_v1/transfer.csv)
