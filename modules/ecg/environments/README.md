# ECG environments

`research/requirements.txt` pins the direct dependencies used by the Windows CUDA workstation experiment. Every run also records a full `requirements.lock.txt`, Python/package versions and GPU identity. The local `.venv` is ignored by Git.

The inference implementation currently uses the same preprocessing and PyTorch dependencies; ONNX inference requires the matching feature/preprocessing front end. No original Jetson Nano environment is claimed. Establish a separate compatible JetPack/Python/TensorRT environment and run FP32/FP16 parity plus hardware tests before deployment.
