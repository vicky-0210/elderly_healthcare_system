# PPG environments

The training workstation used Python 3.12.14 on Windows, an NVIDIA RTX 3050 Laptop GPU, PyTorch 2.6.0+cu124, NumPy 2.2.6, SciPy 1.15.3, pandas 2.2.3, scikit-learn 1.6.1, Matplotlib 3.10.1, ONNX 1.17.0, ONNX Runtime 1.21.1 and pytest 8.3.5. Install the local common package and the PPG research extra in an isolated environment. Select the PyTorch wheel for the target's accelerator.

The streaming path requires the local common package, NumPy, SciPy and ONNX Runtime. Linux hardware acquisition additionally requires smbus2. Pin compatible aarch64 builds on the actual Jetson image and rerun parity and timing checks there. [Study commands](../README.md)
