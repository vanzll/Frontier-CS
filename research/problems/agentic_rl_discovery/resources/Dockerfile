# Dockerfile for Agentic RL Discovery problem
# Uses NVIDIA NGC PyTorch base image with vLLM and verl

# Start from NVIDIA official image (ubuntu-22.04 + cuda-12.6 + python-3.10)
FROM nvcr.io/nvidia/pytorch:24.08-py3

# Define environments
ENV MAX_JOBS=32
ENV VLLM_WORKER_MULTIPROC_METHOD=spawn
ENV DEBIAN_FRONTEND=noninteractive
ENV NODE_OPTIONS=""
ENV PIP_ROOT_USER_ACTION=ignore
ENV HF_HUB_ENABLE_HF_TRANSFER="1"

# Use xformers attention backend (more stable than flash_attn)
ENV VLLM_ATTENTION_BACKEND=XFORMERS

# Install systemctl and tini
RUN apt-get update && \
    apt-get install -y -o Dpkg::Options::="--force-confdef" systemd tini wget && \
    apt-get clean

# Upgrade pip
RUN python -m pip install --upgrade pip

# Uninstall nv-pytorch fork to avoid conflicts
RUN pip uninstall -y torch torchvision torchaudio \
    pytorch-quantization pytorch-triton torch-tensorrt \
    xgboost transformer_engine flash_attn apex megatron-core grpcio || true

# Install vllm 0.6.3 (verl-agent only supports 0.5.4 and 0.6.3)
RUN pip install --no-cache-dir \
    "vllm==0.6.3" \
    "tensordict==0.6.2" \
    torchdata \
    "transformers[hf_xet]>=4.51.0" \
    accelerate \
    datasets \
    peft \
    hf-transfer \
    "numpy<2.0.0" \
    "pyarrow>=15.0.0" \
    pandas \
    "ray[default]" \
    codetiming \
    hydra-core \
    pylatexenc \
    qwen-vl-utils \
    wandb \
    dill \
    pybind11 \
    liger-kernel

# Fix packages
RUN pip uninstall -y pynvml nvidia-ml-py || true && \
    pip install --no-cache-dir --upgrade \
    "nvidia-ml-py>=12.560.30" \
    "fastapi[standard]>=0.115.0" \
    "optree>=0.13.0" \
    "pydantic>=2.9" \
    "grpcio>=1.62.1"

# Install verl
RUN pip install --no-cache-dir "verl[vllm]" -U

# Install ALFWorld and textworld dependencies
RUN pip install --no-cache-dir \
    alfworld \
    textworld \
    "gymnasium>=0.29.0" \
    omegaconf \
    sentencepiece \
    einops

# Download ALFWorld data during build (to avoid downloading every evaluation)
RUN alfworld-download -f || echo "ALFWorld download warning (may be partial)"

# Set ALFWORLD_DATA environment variable
ENV ALFWORLD_DATA=/root/.cache/alfworld

# Verify core installations
RUN python3 -c "import torch; print(f'torch version: {torch.__version__}')" && \
    python3 -c "import vllm; print(f'vllm version: {vllm.__version__}')" && \
    python3 -c "import verl; print('verl available')" && \
    python3 -c "import alfworld; print('alfworld available')"

# Set working directory
WORKDIR /workspace

# Default command
CMD ["/bin/bash"]
