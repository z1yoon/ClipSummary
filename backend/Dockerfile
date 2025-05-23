# Use official PyTorch image with CUDA 12.8 support for RTX 5090/5080
FROM pytorch/pytorch:2.7.0-cuda12.8-cudnn9-devel

# Install system dependencies including curl for healthcheck
RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    wget \
    build-essential \
    curl \
    --no-install-recommends \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Set environment variables for RTX 5090
ENV CUDA_VISIBLE_DEVICES=0
ENV PYTHONUNBUFFERED=1
ENV TRANSFORMERS_CACHE=/app/.cache/huggingface
ENV HF_HOME=/app/.cache/huggingface
ENV TRANSFORMERS_OFFLINE=0
# Fix cuDNN library path issues
ENV LD_LIBRARY_PATH=/opt/conda/lib/python3.11/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies (PyTorch is already included in base image)
RUN pip install --no-cache-dir -r requirements.txt

# Verify RTX 5090 CUDA compatibility
RUN python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('CUDA version:', torch.version.cuda); print('PyTorch version:', torch.__version__); print('Device capability:', torch.cuda.get_device_capability(0) if torch.cuda.is_available() else 'N/A'); print('Device name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"

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

# Run the application
CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--log-config", "logging.conf"]