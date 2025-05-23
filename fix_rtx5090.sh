#!/bin/bash

# RTX 5090 CUDA Compatibility Fix Script
echo "🔧 Fixing RTX 5090 CUDA compatibility issues..."

# Stop the current containers
echo "📋 Stopping current containers..."
docker-compose down

# Remove old images to force rebuild
echo "🗑️  Removing old backend image..."
docker image rm clipsummary-backend 2>/dev/null || true
docker image rm $(docker images -q clipsummary-backend) 2>/dev/null || true

# Clean up any dangling images
echo "🧹 Cleaning up dangling images..."
docker image prune -f

# Rebuild backend with no cache to ensure fresh PyTorch installation
echo "🔨 Rebuilding backend with updated PyTorch for RTX 5090..."
docker-compose build --no-cache backend

# Check if build was successful
if [ $? -eq 0 ]; then
    echo "✅ Build successful!"
    
    # Start the services
    echo "🚀 Starting services..."
    docker-compose up -d
    
    # Wait a bit for the services to start
    echo "⏳ Waiting for services to start..."
    sleep 15
    
    # Check the logs for CUDA status
    echo "📊 Checking CUDA status in logs..."
    echo "========================================="
    docker-compose logs backend | grep -E "(CUDA|RTX|GPU|cpu|PyTorch|WhisperX)" | tail -15
    echo "========================================="
    
    echo ""
    echo "✅ Fix applied! The system should now:"
    echo "   - Use PyTorch 2.5.0+ with CUDA 12.4 support"
    echo "   - Automatically detect RTX 5090 compatibility"
    echo "   - Fall back to CPU if GPU issues persist"
    echo "   - Show better error messages for debugging"
    echo ""
    echo "📊 Monitor with: docker-compose logs -f backend"
    echo "🔍 Check CUDA status: docker-compose exec backend python -c \"import torch; print('CUDA available:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')\""
    
else
    echo "❌ Build failed! Please check the error messages above."
    echo "Common issues:"
    echo "   - Network connection problems"
    echo "   - Docker daemon not running"
    echo "   - Insufficient disk space"
fi