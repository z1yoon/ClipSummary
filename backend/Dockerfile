# Use official PyTorch image with CUDA 12.8 support for RTX 5090/5080
FROM pytorch/pytorch:2.7.0-cuda12.8-cudnn9-devel

# Install system dependencies including curl for healthcheck
RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    wget \
    build-essential \
    curl \
    libsndfile1 \
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
# Fix cuDNN library path issues - add multiple paths for compatibility
ENV LD_LIBRARY_PATH=/opt/conda/lib/python3.11/site-packages/nvidia/cudnn/lib:/opt/conda/lib:/usr/local/cuda/lib64:$LD_LIBRARY_PATH
# Disable TF32 for better reproducibility
ENV TORCH_BACKENDS_CUDA_MATMUL_ALLOW_TF32=false
ENV TORCH_BACKENDS_CUDNN_ALLOW_TF32=false

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies with better error handling
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Fix PyTorch Lightning checkpoint compatibility issues
RUN python -c "
import torch
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
print('PyTorch version:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('CUDA version:', torch.version.cuda)
    print('Device capability:', torch.cuda.get_device_capability(0))
    print('Device name:', torch.cuda.get_device_name(0))
    # Test tensor operations
    x = torch.randn(5, 3).cuda()
    print('GPU tensor test passed')
"

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

# Add healthcheck
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the application
CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--log-config", "logging.conf"]