#!/usr/bin/env bash
# Installation script for Booster T1 training environment.
# Run from within the BoosterT1_train directory:
#
#   cd BoosterT1_train
#   ./install.sh
#
# Expected structure after running:
#   <root>/
#   ├── hl_train/         ← venv created by this script
#   ├── IsaacLab/         ← cloned by this script
#   ├── booster_assets/   ← cloned by this script
#   └── BoosterT1_train/  ← this repo

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
VENV_DIR="$ROOT_DIR/hl_train"

echo "=================================================="
echo " Booster T1 Training - Installation"
echo " Root: $ROOT_DIR"
echo "=================================================="
echo ""

# --- Prerequisites ---
if ! command -v uv &>/dev/null; then
    echo "[ERROR] uv not found. Install with:"
    echo "  curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

if ! command -v git &>/dev/null; then
    echo "[ERROR] git not found. Install with: sudo apt install git"
    exit 1
fi

if ! command -v nvidia-smi &>/dev/null; then
    echo "[WARNING] nvidia-smi not found. Make sure NVIDIA drivers >= 525 are installed."
fi

# --- Virtual environment ---
if [ ! -d "$VENV_DIR" ]; then
    echo "[INFO] Creating venv with Python 3.11..."
    uv venv --python 3.11 "$VENV_DIR"
else
    echo "[INFO] Venv already exists at $VENV_DIR"
fi

source "$VENV_DIR/bin/activate"
uv pip install pip "setuptools<70" wheel -q

# --- Clone IsaacLab ---
ISAACLAB_DIR="$ROOT_DIR/IsaacLab"
if [ ! -d "$ISAACLAB_DIR" ]; then
    echo "[INFO] Cloning IsaacLab v2.2.0..."
    git clone --branch v2.2.0 --depth 1 https://github.com/isaac-sim/IsaacLab.git "$ISAACLAB_DIR"
else
    echo "[INFO] IsaacLab already present, skipping clone."
fi

# --- Clone booster_assets ---
BOOSTER_ASSETS_DIR="$ROOT_DIR/booster_assets"
if [ ! -d "$BOOSTER_ASSETS_DIR" ]; then
    echo "[INFO] Cloning booster_assets..."
    git clone https://github.com/BoosterRobotics/booster_assets.git "$BOOSTER_ASSETS_DIR"
else
    echo "[INFO] booster_assets already present, skipping clone."
fi

# --- Isaac Sim ---
if python -c "import isaacsim" &>/dev/null 2>&1; then
    echo "[INFO] isaacsim already installed, skipping."
else
    echo ""
    echo "[INFO] Installing Isaac Sim (~10GB download)..."
    echo "[INFO] You will be prompted to accept the NVIDIA Omniverse EULA."
    echo ""
    pip install "isaacsim[all,extscache]==5.0.0" --extra-index-url https://pypi.nvidia.com
fi

# --- Isaac Lab packages ---
# Note: --no-build-isolation needed because Isaac Sim upgrades setuptools,
# breaking pkg_resources which older dependencies (flatdict) require.
echo "[INFO] Installing Isaac Lab packages..."
uv pip install -e "$ISAACLAB_DIR/source/isaaclab" --no-build-isolation -q
uv pip install -e "$ISAACLAB_DIR/source/isaaclab_assets" --no-build-isolation -q
uv pip install -e "$ISAACLAB_DIR/source/isaaclab_rl" --no-build-isolation -q
uv pip install -e "$ISAACLAB_DIR/source/isaaclab_tasks" --no-build-isolation -q

# --- Booster packages ---
echo "[INFO] Installing booster_assets..."
uv pip install -e "$BOOSTER_ASSETS_DIR" --no-build-isolation -q

echo "[INFO] Installing PyTorch with Blackwell (sm_120) support..."
# Install with all CUDA 12.8 dependencies (required for sm_120/Blackwell nvrtc support)
pip install "torch==2.7.0+cu128" "torchvision==0.22.0+cu128" \
    --index-url https://download.pytorch.org/whl/cu128 \
    --force-reinstall -q
# Restore Isaac Sim's required package versions overwritten by torch
pip install "numpy==1.26.0" "pillow==11.2.1" "typing_extensions==4.12.2" \
    "filelock==3.13.1" "fsspec==2024.6.1" "markupsafe==2.1.3" \
    "networkx==3.3" "sympy==1.13.3" -q

echo "[INFO] Installing RL frameworks..."
pip install rsl-rl-lib==2.3.3 -q

echo "[INFO] Installing booster_train..."
uv pip install -e "$SCRIPT_DIR/source" --no-build-isolation -q

# --- Done ---
echo ""
echo "=================================================="
echo " Installation complete!"
echo ""
echo " Activate the environment:"
echo "   source $VENV_DIR/bin/activate"
echo ""
echo " On first run, Isaac Sim downloads extensions"
echo " (a few minutes). Test with:"
echo "   python -c \"from isaacsim import SimulationApp; \\"
echo "     app = SimulationApp({'headless': True}); \\"
echo "     print('Isaac Sim OK'); app.close()\""
echo ""
echo " Start training:"
echo "   cd $SCRIPT_DIR"
echo "   python scripts/rsl_rl/train.py --task Booster-T1-Locomotion-Flat-v0 --headless"
echo "=================================================="
