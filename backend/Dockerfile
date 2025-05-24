# Use PyTorch 2.7.0 with CUDA 12.8 for RTX 5090 support (required)
FROM pytorch/pytorch:2.7.0-cuda12.8-cudnn9-devel

# Install system dependencies and cuDNN compatibility libraries
RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    wget \
    build-essential \
    curl \
    libsndfile1 \
    --no-install-recommends && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Fix cuDNN library compatibility (cuDNN 8 to cuDNN 9 mapping)
# This creates proper symbolic links for legacy cuDNN 8 library names
RUN mkdir -p /usr/local/cuda/lib64 && \
    # First, find the actual cuDNN 9 libraries
    CUDNN_LIB_PATH="/opt/conda/lib/python3.11/site-packages/nvidia/cudnn/lib" && \
    # Create specific symbolic links for the libraries that are commonly needed
    if [ -f "$CUDNN_LIB_PATH/libcudnn.so.9" ]; then \
        ln -sf "$CUDNN_LIB_PATH/libcudnn.so.9" /usr/local/cuda/lib64/libcudnn.so.8; \
    fi && \
    if [ -f "$CUDNN_LIB_PATH/libcudnn_ops_infer.so.9" ]; then \
        ln -sf "$CUDNN_LIB_PATH/libcudnn_ops_infer.so.9" /usr/local/cuda/lib64/libcudnn_ops_infer.so.8; \
    fi && \
    if [ -f "$CUDNN_LIB_PATH/libcudnn_ops_train.so.9" ]; then \
        ln -sf "$CUDNN_LIB_PATH/libcudnn_ops_train.so.9" /usr/local/cuda/lib64/libcudnn_ops_train.so.8; \
    fi && \
    if [ -f "$CUDNN_LIB_PATH/libcudnn_adv_infer.so.9" ]; then \
        ln -sf "$CUDNN_LIB_PATH/libcudnn_adv_infer.so.9" /usr/local/cuda/lib64/libcudnn_adv_infer.so.8; \
    fi && \
    if [ -f "$CUDNN_LIB_PATH/libcudnn_adv_train.so.9" ]; then \
        ln -sf "$CUDNN_LIB_PATH/libcudnn_adv_train.so.9" /usr/local/cuda/lib64/libcudnn_adv_train.so.8; \
    fi && \
    if [ -f "$CUDNN_LIB_PATH/libcudnn_cnn_infer.so.9" ]; then \
        ln -sf "$CUDNN_LIB_PATH/libcudnn_cnn_infer.so.9" /usr/local/cuda/lib64/libcudnn_cnn_infer.so.8; \
    fi && \
    if [ -f "$CUDNN_LIB_PATH/libcudnn_cnn_train.so.9" ]; then \
        ln -sf "$CUDNN_LIB_PATH/libcudnn_cnn_train.so.9" /usr/local/cuda/lib64/libcudnn_cnn_train.so.8; \
    fi && \
    # Verify the links were created
    ls -la /usr/local/cuda/lib64/libcudnn*

WORKDIR /app

# Set environment variables for GPU optimization
ENV CUDA_VISIBLE_DEVICES=0
ENV PYTHONUNBUFFERED=1
ENV TRANSFORMERS_CACHE=/app/.cache/huggingface
ENV HF_HOME=/app/.cache/huggingface
ENV TRANSFORMERS_OFFLINE=0
# Updated library paths - ensure both cuDNN 9 and compatibility links are found
ENV LD_LIBRARY_PATH=/opt/conda/lib/python3.11/site-packages/nvidia/cudnn/lib:/usr/local/cuda/lib64:/opt/conda/lib:$LD_LIBRARY_PATH

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies with better error handling
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

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

# Improved healthcheck that handles model loading gracefully
HEALTHCHECK --interval=30s --timeout=30s --start-period=180s --retries=5 \
    CMD curl -f --connect-timeout 5 --max-time 10 http://localhost:8000/health || exit 1

# Run the application
CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--log-config", "logging.conf"]