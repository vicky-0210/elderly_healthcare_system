# EDA environments

[research.txt](research.txt) pins the evaluated Python packages. The study used Python 3.12.14 on Windows, with PyTorch 2.6.0+cu124 for GPU training. Choose the PyTorch build appropriate for the host. See the [EDA runbook](../README.md) for the install and verification sequence.

Raw-prefix inference requires `wearable-common`, NumPy, SciPy and pandas. The coefficient-JSON logistic export uses these dependencies directly. PyTorch is also required for neural exports. The research reference joblib files require their original estimator library and scikit-learn. Record package compatibility, memory and measured latency independently when targeting Jetson Nano.
