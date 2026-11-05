# IMU environments

`research/` reserves tested training/evaluation dependency locks.

`inference/` reserves the minimal model/preprocessing runtime dependency specification.

Do not copy workstation dependency versions onto Jetson without checking the target stack. Shared packages must not import a training framework at import time. No framework or lockfile has been selected yet.
