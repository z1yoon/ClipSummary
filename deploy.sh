#!/bin/bash
# ClipSummary Deployment Script for On-Premises Server

# Define variables
GIT_SHA=${1:-latest}
IMAGE_REGISTRY="ghcr.io"
REPO_OWNER=$(echo $GITHUB_REPOSITORY | cut -d'/' -f1)
BACKEND_IMAGE="$IMAGE_REGISTRY/$REPO_OWNER/clip-summary-backend:$GIT_SHA"
FRONTEND_IMAGE="$IMAGE_REGISTRY/$REPO_OWNER/clip-summary-frontend:$GIT_SHA"
APP_DIR=$(pwd)
LOG_FILE="$APP_DIR/deploy.log"
DATE=$(date '+%Y-%m-%d %H:%M:%S')

# Log function
log() {
  echo "[$DATE] $1" | tee -a $LOG_FILE
}

# Create log file if it doesn't exist
touch $LOG_FILE
log "Starting deployment process for commit $GIT_SHA"

# Check if docker and docker-compose are installed
if ! command -v docker &> /dev/null; then
  log "ERROR: Docker is not installed. Please install Docker first."
  exit 1
fi

if ! command -v docker-compose &> /dev/null; then
  log "ERROR: Docker Compose is not installed. Please install Docker Compose first."
  exit 1
fi

# Pull the latest images
log "Pulling latest images from registry"
docker pull $BACKEND_IMAGE || { log "Failed to pull backend image"; exit 1; }
docker pull $FRONTEND_IMAGE || { log "Failed to pull frontend image"; exit 1; }

# Tag images as latest for docker-compose
log "Tagging images for local use"
docker tag $BACKEND_IMAGE clip-summary-backend:latest
docker tag $FRONTEND_IMAGE clip-summary-frontend:latest

# Create directories for persistent data if they don't exist
log "Setting up directory structure"
mkdir -p $APP_DIR/uploads
mkdir -p $APP_DIR/logs

# Set correct permissions
log "Setting permissions"
chmod -R 755 $APP_DIR

# Stop and remove existing containers
log "Stopping and removing existing containers"
docker-compose -f docker-compose.prod.yml down || log "Warning: Failed to stop containers, may not exist yet"

# Start the application
log "Starting application with docker-compose"
docker-compose -f docker-compose.prod.yml up -d

# Check if containers are running
sleep 10
if docker-compose -f docker-compose.prod.yml ps | grep -q "Up"; then
  log "Deployment successful! Application is running."
  
  # Display container status
  docker-compose -f docker-compose.prod.yml ps
else
  log "ERROR: Deployment failed. Containers are not running."
  docker-compose -f docker-compose.prod.yml logs
  exit 1
fi

# Create a deployment record
echo "$DATE - Deployed commit $GIT_SHA" >> $APP_DIR/deployment_history.txt

log "Deployment process completed"