# IMU environments

The completed research run used Python 3.12.14, PyTorch 2.6.0+cu124, a laptop RTX 3050 and the direct dependencies in [research/requirements.txt](research/requirements.txt). Choose the PyTorch build matching the workstation, then install that requirements file and the local common package with `python -m pip install -e packages/common`. The Windows research environment already exists under the team repository.

For the 100 Hz tree export, the inference path needs NumPy and SciPy. The export includes the wavelet filter coefficients and learned ensemble. Hardware acquisition additionally uses smbus2. Neural inference needs a compatible ONNX Runtime build or TensorRT 8 and PyCUDA. Install target-compatible versions for the board's JetPack and Python. Follow the [hardware validation procedure](../docs/hardware_validation.md) for a numerical reference check and measured device timings.
