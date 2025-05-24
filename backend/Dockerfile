# Use official PyTorch image that matches our requirements (torch>=2.5.1)
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel

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

# Set environment variables for GPU optimization
ENV CUDA_VISIBLE_DEVICES=0
ENV PYTHONUNBUFFERED=1
ENV TRANSFORMERS_CACHE=/app/.cache/huggingface
ENV HF_HOME=/app/.cache/huggingface
ENV TRANSFORMERS_OFFLINE=0
# Updated library paths for PyTorch 2.5.1
ENV LD_LIBRARY_PATH=/opt/conda/lib/python3.11/site-packages/nvidia/cudnn/lib:/opt/conda/lib:/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies with better error handling
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Test PyTorch and CUDA setup with WhisperX
RUN python -c "import torch; import whisperx; print('PyTorch version:', torch.__version__); print('WhisperX version:', whisperx.__version__); print('CUDA available:', torch.cuda.is_available())"

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