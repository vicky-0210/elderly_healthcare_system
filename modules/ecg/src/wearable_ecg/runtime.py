"""Small experiment utilities; never writes to a dataset directory."""
from __future__ import annotations
import hashlib
import json
import os
import platform
import random
from pathlib import Path
import numpy as np


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    def convert(x):
        if isinstance(x, Path): return str(x)
        if isinstance(x, np.ndarray): return x.tolist()
        if isinstance(x, np.generic): return x.item()
        raise TypeError(type(x).__name__)
    path.write_text(json.dumps(value, indent=2, default=convert, allow_nan=False) + "\n", encoding="utf-8")


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""): h.update(block)
    return h.hexdigest()


def separate_output(source, output):
    source, output = Path(source).resolve(), Path(output).resolve()
    if output == source or source in output.parents:
        raise ValueError(f"Refusing to write inside source dataset: {output}")
    return output


def seed_everything(seed):
    import torch
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # FFT/convolution implementations can still vary across platforms/versions.
    torch.set_num_threads(min(4, os.cpu_count() or 1))


def environment():
    import importlib.metadata
    import torch
    names = ["torch", "numpy", "scipy", "pandas", "scikit-learn", "PyWavelets", "matplotlib", "wfdb"]
    return {"python": platform.python_version(), "platform": platform.platform(),
            "packages": {n: importlib.metadata.version(n) for n in names},
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "device_memory_bytes": torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else None}
