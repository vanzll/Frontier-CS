#!/bin/bash
# Set up environment for Agentic RL Algorithm Discovery evaluation.
#
# This script runs inside Docker before evaluation and:
# 1. Installs verl-agent from resources
# 2. Downloads ALFWorld data
# 3. Prepares training data
#
# Note: alfworld and textworld are installed via pyproject.toml (uv_project)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESOURCES_DIR="${SCRIPT_DIR}/resources"
VERL_AGENT_DIR="${RESOURCES_DIR}/verl-agent"

echo "=== Setting up Agentic RL Discovery Environment ==="

# Configure attention backend for vLLM 0.11.0
export VLLM_ATTENTION_BACKEND=FLASH_ATTN

# Install vLLM version required by verl-agent (from README)
echo "[setup] Installing vLLM 0.11.0 (verl-agent standard)..."
pip install "vllm==0.11.0" --quiet 2>/dev/null || pip install "vllm==0.11.0"

echo "[setup] vLLM 0.11.0 installed with FLASH_ATTN backend"

# Uninstall upstream verl first (installed by Dockerfile), then install verl-agent
echo "[setup] Uninstalling upstream verl..."
pip uninstall -y verl 2>/dev/null || true

echo "[setup] Installing verl-agent..."
cd "${VERL_AGENT_DIR}"

# Clear Python cache to ensure fresh code is used
echo "[setup] Clearing Python cache..."
find "${VERL_AGENT_DIR}" -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find "${VERL_AGENT_DIR}" -name "*.pyc" -delete 2>/dev/null || true

pip install -e . || { echo "[setup] ERROR: Failed to install verl-agent"; exit 1; }

# Verify verl-agent is installed correctly (should have AdvantageEstimator.CUSTOM)
python3 -c "from verl.trainer.ppo.ray_trainer import AdvantageEstimator; print(f'[setup] AdvantageEstimator.CUSTOM = {AdvantageEstimator.CUSTOM}')" || {
    echo "[setup] ERROR: verl-agent not installed correctly - AdvantageEstimator.CUSTOM not found"
    exit 1
}

# Download ALFWorld data using the official download command
echo "[setup] Downloading ALFWorld data..."
# The alfworld-download command downloads data to ~/.cache/alfworld
# Use -f flag to force download, ignore errors if already downloaded
alfworld-download -f 2>&1 || {
    echo "[setup] alfworld-download failed, trying alternative..."
    python3 -c "
import subprocess
import sys
# Try running as module
try:
    from alfworld.scripts import alfworld_download
    alfworld_download.main()
except Exception as e:
    print(f'Download script failed: {e}')
    sys.exit(1)
"
}

# Set ALFWORLD_DATA environment variable
export ALFWORLD_DATA="$HOME/.cache/alfworld"
echo "[setup] ALFWORLD_DATA=${ALFWORLD_DATA}"

# Verify data exists
if [ -d "$ALFWORLD_DATA/json_2.1.1" ]; then
    echo "[setup] ALFWorld data found at $ALFWORLD_DATA/json_2.1.1"
    ls -la "$ALFWORLD_DATA/json_2.1.1/" | head -5
else
    echo "[setup] ERROR: ALFWorld data not found at $ALFWORLD_DATA/json_2.1.1"
    echo "[setup] Available in $ALFWORLD_DATA:"
    ls -la "$ALFWORLD_DATA/" 2>/dev/null || echo "(directory doesn't exist)"
fi

echo "[setup] ALFWorld ready"

# Prepare verl-agent training data
echo "[setup] Preparing training data..."
cd "${VERL_AGENT_DIR}"
python3 -m examples.data_preprocess.prepare \
    --mode 'text' \
    --train_data_size 16 \
    --val_data_size 128 2>/dev/null || echo "[setup] Data preparation skipped (may already exist)"

echo "[setup] Environment setup complete!"
