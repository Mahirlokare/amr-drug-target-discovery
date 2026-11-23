#!/usr/bin/env bash
# Render Build Script for AMR Drug Target Discovery
# Installs CPU PyTorch and required dependencies within Render limits
set -euo pipefail

echo "==> [Render Build] Upgrading pip and build tools..."
python -m pip install --upgrade pip setuptools wheel

echo "==> [Render Build] Installing PyTorch CPU wheel..."
pip install --no-cache-dir torch>=2.1.0 --index-url https://download.pytorch.org/whl/cpu

echo "==> [Render Build] Installing PyTorch Geometric & RDKit dependencies..."
pip install --no-cache-dir torch-geometric>=2.5.0
pip install --no-cache-dir -r requirements.txt

echo "==> [Render Build] Verifying installation & running test assertions..."
python -c "import torch, torch_geometric, rdkit, streamlit; print('All core libraries verified!')"

echo "==> [Render Build] Build completed successfully."
