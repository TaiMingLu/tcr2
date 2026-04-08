#!/bin/bash
# Install Python packages into the container overlay.
# uv is pre-installed in the base CUDA container.

set -e

echo "=== Installing Python packages for TCR environment ==="

# Install core packages - these are cached locally in /home/user/pylibs
# but we reinstall to ensure they're in the system Python
uv pip install --system \
    torch \
    numpy \
    scipy \
    scikit-learn \
    transformers \
    huggingface_hub \
    accelerate \
    peft \
    tqdm

echo "=== Package installation complete ==="
