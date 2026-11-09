#!/bin/bash
# Jetson Orin Nano (JetPack 5.x / 6.x, Ubuntu 20.04 / 22.04). Run from inside the jetson_deploy folder: bash setup_orin.sh
set -e
sudo apt-get update
sudo apt-get install -y python3-pip python3-numpy python3-scipy python3-pandas i2c-tools
python3 -m pip install --user smbus2
# PyTorch: install the NVIDIA build that matches your JetPack (https://docs.nvidia.com/deeplearning/frameworks/install-pytorch-jetson-platform/)
python3 -c "import torch" 2>/dev/null || echo "!! install the NVIDIA PyTorch wheel for your JetPack, then re-run"
sudo usermod -aG i2c $USER                 # log out/in once so I2C works without sudo
echo "I2C buses:"; i2cdetect -l
echo "Sensor check (replace 7 with your bus): sudo i2cdetect -y -r 7   -> expect 57"
echo "Test the install without a sensor:  python3 replay_demo.py sample_spo2.csv"
