#!/bin/bash
# FAIL Environment Setup Script

set -e

echo "=== FAIL Environment Setup ==="

# Install PyTorch with CUDA 12.1
echo "Installing PyTorch..."
pip install torch==2.5.0 torchvision --index-url https://download.pytorch.org/whl/cu121

# Install Flash Attention 2
echo "Installing Flash Attention 2..."
pip install packaging ninja
pip install flash-attn==2.7.0.post2 --no-build-isolation

# Install linting tools (optional, for development)
echo "Installing linting tools..."
pip install -r requirements-lint.txt

# Install FAIL package
echo "Installing FAIL..."
pip install -e .

# Install additional dependencies
echo "Installing additional dependencies..."
pip install ml-collections absl-py inflect==6.0.4 pydantic==1.10.9 huggingface_hub==0.24.0 protobuf==3.20.0
pip install transformers==4.46.1
pip install diffusers==0.36.0
pip install peft==0.17.0
pip install huggingface_hub=0.36.2

echo "=== FAIL Environment Setup Complete ==="
