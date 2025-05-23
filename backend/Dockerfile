# Stage 1: Base with CUDA 12.4 support for RTX 5090
FROM nvidia/cuda:12.4.0-base-ubuntu22.04 as base

# Install Python and essential tools
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-venv \
    python3-pip \
    ffmpeg \
    git \
    wget \
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
FROM base as dependencies

WORKDIR /app

# Set non-interactive frontend to avoid timezone prompt
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Singapore

# Copy requirements file
COPY requirements.txt .

# Install all Python dependencies from requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
# Verify CUDA setup
RUN python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('CUDA version:', torch.version.cuda); print('Device count:', torch.cuda.device_count()); print('PyTorch version:', torch.__version__); print('Compute capability:', torch.cuda.get_device_capability(0) if torch.cuda.is_available() else 'N/A')"

# Stage 3: Runtime
FROM dependencies as runtime

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

# Create necessary directories with wide-open permissions
RUN mkdir -p uploads cache .cache/huggingface/hub && \
    chmod -R 777 uploads cache .cache

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV TRANSFORMERS_CACHE=/app/.cache/huggingface
ENV HF_HOME=/app/.cache/huggingface
ENV TRANSFORMERS_OFFLINE=0
ENV CUDA_LAUNCH_BLOCKING=1
# Force device index
ENV CUDA_VISIBLE_DEVICES=0

# Run the application
CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--log-config", "logging.conf"]