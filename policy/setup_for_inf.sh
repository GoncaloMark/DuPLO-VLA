#!/bin/bash
# ============================================================
# setup_eval_env.sh — Run this ONCE on the cluster (login node)
#
# Usage:
#   chmod +x setup_eval_env.sh
#   ./setup_eval_env.sh
# ============================================================

set -e

REPO_DIR="/data/home/g.marques/storage/DuPLO-VLA"
VENV_DIR="${REPO_DIR}/eval_venv"

echo "============================================"
echo "Setting up evaluation environment"
echo "============================================"

if ! command -v mise &> /dev/null; then
    echo "Installing mise..."
    curl https://mise.jdx.dev/install.sh | sh
    # Add to PATH for this session
    export PATH="$HOME/.local/bin:$PATH"
    echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
    echo "  mise installed at $(which mise)"
else
    echo "  mise already installed: $(which mise)"
fi

eval "$(mise activate bash)"

echo ""
echo "Installing Python 3.10..."
mise use python@3.10
echo "  Python: $(python --version)"
echo "  Path: $(which python)"

echo ""
echo "Creating virtual environment at ${VENV_DIR}..."
if [ -d "$VENV_DIR" ]; then
    echo "  Removing existing venv..."
    rm -rf "$VENV_DIR"
fi
python -m venv "$VENV_DIR"
source "${VENV_DIR}/bin/activate"
pip install --upgrade pip setuptools wheel

echo ""
echo "Installing PyTorch..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

echo ""
echo "Setting up MuJoCo..."

MUJOCO_DIR="$HOME/.mujoco"
if [ ! -d "${MUJOCO_DIR}/mujoco210" ]; then
    echo "  Downloading MuJoCo 2.1.0..."
    mkdir -p "$MUJOCO_DIR"
    wget -q https://github.com/deepmind/mujoco/releases/download/2.1.0/mujoco210-linux-x86_64.tar.gz -O /tmp/mujoco210.tar.gz
    tar -xzf /tmp/mujoco210.tar.gz -C "$MUJOCO_DIR"
    rm /tmp/mujoco210.tar.gz
    echo "  MuJoCo installed at ${MUJOCO_DIR}/mujoco210"
else
    echo "  MuJoCo already at ${MUJOCO_DIR}/mujoco210"
fi

# module load anaconda3
# conda install -c conda-forge mesalib -y

# Set environment variables
export MUJOCO_PY_MUJOCO_PATH="${MUJOCO_DIR}/mujoco210"
export LD_LIBRARY_PATH="${MUJOCO_DIR}/mujoco210/bin:${LD_LIBRARY_PATH}"
export MUJOCO_GL=osmesa
# export C_INCLUDE_PATH="${CONDA_PREFIX}/include:${C_INCLUDE_PATH}"
# export LIBRARY_PATH="${CONDA_PREFIX}/lib:${LIBRARY_PATH}"
# export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH}"

echo "  Installing mujoco-py..."
pip install "numpy<2.0"
pip install "cython<3.0"
pip install mujoco-py==2.1.2.14
python -c "import mujoco_py"  # compiles against numpy 1.x headers
pip install "numpy>=2.0"      # upgrade back, compiled .so still works

# Trigger compilation
# echo "  Triggering mujoco-py compilation (this takes a minute)..."
# python -c "import mujoco_py" 2>&1 || true
# python -c "import mujoco_py; print('  mujoco-py OK')"

echo ""
echo "Installing MetaWorld..."
cd Metaworld && pip install -e . && cd ..

echo ""
echo "Installing project dependencies..."
pip install "zarr<3.0.0" wandb ipdb gpustat dm_control omegaconf hydra-core dill einops \
    diffusers numba moviepy imageio av matplotlib termcolor transformers \
    imageio-ffmpeg

# Pin numcodecs
pip install --no-binary=numcodecs numcodecs==0.14.0

# Install the policy package
cd "${REPO_DIR}/policy"
pip install -e .
cd "${REPO_DIR}"

# Verify everything 
echo ""
echo "============================================"
echo "Verification"
echo "============================================"
python -c "
import sys
print(f'Python:      {sys.version}')

import torch
print(f'PyTorch:     {torch.__version__}')
print(f'CUDA avail:  {torch.cuda.is_available()}')

import mujoco_py
print(f'mujoco-py:   OK')

import metaworld
print(f'MetaWorld:   OK')

import imageio
print(f'imageio:     OK')

import transformers
print(f'transformers: {transformers.__version__}')

print()
print('All dependencies verified!')
"

echo ""
echo "============================================"
echo "Setup complete!"
echo ""
echo "Next steps:"
echo "  1. Copy your checkpoint to ${REPO_DIR}/checkpoints/"
echo "  2. Submit the SLURM job: sbatch eval_job.sh"
echo "============================================"