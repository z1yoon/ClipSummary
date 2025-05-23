# Stage 1: Base with CUDA 12.4 development support for RTX 5090
FROM nvidia/cuda:12.4.0-devel-ubuntu22.04 AS base

# Install Python and essential tools
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-venv \
    python3-pip \
    python3.10-dev \
    ffmpeg \
    git \
    wget \
    build-essential \
    --no-install-recommends \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.10 as default python
RUN ln -sf /usr/bin/python3.10 /usr/bin/python && \
    ln -sf /usr/bin/python3.10 /usr/bin/python3

# Create and activate virtual environment
ENV VIRTUAL_ENV=/opt/venv
RUN python -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Stage 2: Dependencies
FROM base AS dependencies

WORKDIR /app

# Set non-interactive frontend to avoid timezone prompt
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Singapore

# Set CUDA environment variables for RTX 5090
ENV CUDA_VISIBLE_DEVICES=0
ENV TORCH_CUDA_ARCH_LIST="5.0;6.0;7.0;7.5;8.0;8.6;9.0;12.0"
ENV CUDA_MODULE_LOADING=LAZY
ENV CUDA_LAUNCH_BLOCKING=0

# Copy requirements file
COPY requirements.txt .

# Install PyTorch stable version with CUDA 12.4 support for RTX 5090
RUN pip install --no-cache-dir torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124

# Install torchaudio separately (CPU version is fine for audio processing)
RUN pip install --no-cache-dir torchaudio==2.5.1

# Install remaining dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Verify RTX 5090 CUDA compatibility
RUN python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('CUDA version:', torch.version.cuda); print('PyTorch version:', torch.__version__); print('Device capability:', torch.cuda.get_device_capability(0) if torch.cuda.is_available() else 'N/A'); print('Device name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"

# Stage 3: Runtime
FROM dependencies AS runtime

WORKDIR /app

# Copy application files
COPY ./ai ./ai
COPY ./api ./api
COPY ./db ./db
COPY ./schemas ./schemas
COPY ./security ./security
COPY ./utils ./utils
COPY ./main.py ./
COPY ./logging.conf ./

# Create necessary directories
RUN mkdir -p uploads cache .cache/huggingface/hub && \
    chmod -R 777 uploads cache .cache

# Set environment variables for optimal RTX 5090 performance
ENV PYTHONUNBUFFERED=1
ENV TRANSFORMERS_CACHE=/app/.cache/huggingface
ENV HF_HOME=/app/.cache/huggingface
ENV TRANSFORMERS_OFFLINE=0
ENV CUDA_VISIBLE_DEVICES=0
ENV TORCH_CUDA_ARCH_LIST="5.0;6.0;7.0;7.5;8.0;8.6;9.0;12.0"
ENV CUDA_MODULE_LOADING=LAZY

# Run the application
CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--log-config", "logging.conf"]