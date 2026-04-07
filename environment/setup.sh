#!/bin/bash
# Install Python packages into the container overlay.
#
# All Python packages (PyTorch, transformers, etc.) go here, NOT in container.def.
# Only OS-level packages go in container.def.
#
# IMPORTANT: Install into the SYSTEM Python, not a virtual environment.
# Use: uv pip install --system <packages>

set -e

echo "=== Installing Python packages for TCR environment ==="

# Core ML dependencies
uv pip install --system \
    torch \
    torchvision \
    torchaudio \
    numpy \
    scikit-learn \
    scipy

# Transformers and related
uv pip install --system \
    transformers \
    accelerate \
    peft \
    huggingface_hub

# Data processing
uv pip install --system \
    pandas \
    tqdm

echo "=== Package installation complete ==="
