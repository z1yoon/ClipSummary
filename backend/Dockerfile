# Stage 1: Dependencies
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime as dependencies

WORKDIR /app

# Set non-interactive frontend to avoid timezone prompt
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Singapore

# Install only necessary system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    --no-install-recommends \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy only requirements file first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies, excluding PyTorch which is already installed
RUN pip install --no-cache-dir -r requirements.txt

# Stage 2: Runtime
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

# Run as root user (removing the user restriction)
# No USER directive means we stay as root

# Run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--log-config", "logging.conf"]