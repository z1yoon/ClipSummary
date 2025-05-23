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
# Using --no-cache-dir reduces the image size
RUN pip install --no-cache-dir -r requirements.txt

# Stage 2: Runtime
FROM dependencies as runtime

WORKDIR /app

# Copy installed dependencies from the dependencies stage
COPY --from=dependencies /usr/local/lib/python3.8/site-packages /usr/local/lib/python3.8/site-packages
COPY --from=dependencies /usr/local/bin /usr/local/bin

# Copy only the necessary application files
COPY ./ai ./ai
COPY ./api ./api
COPY ./db ./db
COPY ./schemas ./schemas
COPY ./security ./security
COPY ./utils ./utils
COPY ./main.py ./
COPY ./logging.conf ./

# Create necessary directories
RUN mkdir -p uploads cache .cache/huggingface

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV TRANSFORMERS_CACHE=/app/.cache/huggingface
ENV HF_HOME=/app/.cache/huggingface
ENV TRANSFORMERS_OFFLINE=0

# Set permissions for directories that need write access
RUN chmod 777 uploads cache .cache/huggingface

# Create a non-root user
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

# Run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--log-config", "logging.conf"]