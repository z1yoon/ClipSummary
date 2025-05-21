FROM python:3.9-slim

WORKDIR /app

# Install system dependencies including FFmpeg
RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application
COPY . .

# Create uploads directory
RUN mkdir -p uploads

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Create a non-root user
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

# Run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--log-config", "logging.conf"]