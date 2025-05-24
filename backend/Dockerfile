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
# Create proper symbolic links with absolute paths
RUN mkdir -p /usr/local/cuda/lib64 && \
    # Create symbolic links using absolute paths (not variables)
    ln -sf /opt/conda/lib/python3.11/site-packages/nvidia/cudnn/lib/libcudnn.so.9 /usr/local/cuda/lib64/libcudnn.so.8 && \
    ln -sf /opt/conda/lib/python3.11/site-packages/nvidia/cudnn/lib/libcudnn_ops.so.9 /usr/local/cuda/lib64/libcudnn_ops_infer.so.8 && \
    ln -sf /opt/conda/lib/python3.11/site-packages/nvidia/cudnn/lib/libcudnn_ops.so.9 /usr/local/cuda/lib64/libcudnn_ops_train.so.8 && \
    ln -sf /opt/conda/lib/python3.11/site-packages/nvidia/cudnn/lib/libcudnn_adv.so.9 /usr/local/cuda/lib64/libcudnn_adv_infer.so.8 && \
    ln -sf /opt/conda/lib/python3.11/site-packages/nvidia/cudnn/lib/libcudnn_adv.so.9 /usr/local/cuda/lib64/libcudnn_adv_train.so.8 && \
    ln -sf /opt/conda/lib/python3.11/site-packages/nvidia/cudnn/lib/libcudnn_cnn.so.9 /usr/local/cuda/lib64/libcudnn_cnn_infer.so.8 && \
    ln -sf /opt/conda/lib/python3.11/site-packages/nvidia/cudnn/lib/libcudnn_cnn.so.9 /usr/local/cuda/lib64/libcudnn_cnn_train.so.8 && \
    # Verify the links were created correctly
    echo "Checking cuDNN symbolic links:" && \
    ls -la /usr/local/cuda/lib64/libcudnn* && \
    echo "Verifying target files exist:" && \
    ls -la /opt/conda/lib/python3.11/site-packages/nvidia/cudnn/lib/libcudnn_ops.so.9

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

# Create necessary directories with proper permissions
RUN mkdir -p uploads cache .cache/huggingface/hub && \
    chmod -R 777 uploads cache .cache && \
    chown -R root:root uploads cache .cache

# Run the application
CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--log-config", "logging.conf"]