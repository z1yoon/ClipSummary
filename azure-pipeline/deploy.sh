#!/bin/bash
# Azure Container Apps deployment script for ClipSummary

# Exit on error
set -e

# Configuration variables
RESOURCE_GROUP="ClipSummaryRG"
LOCATION="westus2"
REGISTRY_NAME="clipsummaryregistry"
FRONTEND_IMAGE="clipsummary-frontend:latest"
BACKEND_IMAGE="clipsummary-backend:latest"
ENVIRONMENT_NAME="clipsummary-env"

# Get parameters or use defaults
YOUTUBE_API_KEY=${YOUTUBE_API_KEY:-""}
REDIS_CONNECTION_STRING=${REDIS_CONNECTION_STRING:-""}

# Print usage information
usage() {
  echo "Usage: $0 [options]"
  echo "Options:"
  echo "  -g, --resource-group    Resource group name (default: ClipSummaryRG)"
  echo "  -l, --location          Azure region (default: westus2)"
  echo "  -e, --environment       Container Apps environment name (default: clipsummary-env)"
  echo "  -r, --registry          Azure Container Registry name (default: clipsummaryregistry)"
  echo "  -y, --youtube-api-key   YouTube API key (required)"
  echo "  -h, --help              Show this help message"
  exit 1
}

# Parse arguments
while [[ $# -gt 0 ]]; do
  key="$1"
  case $key in
    -g|--resource-group)
      RESOURCE_GROUP="$2"
      shift
      shift
      ;;
    -l|--location)
      LOCATION="$2"
      shift
      shift
      ;;
    -e|--environment)
      ENVIRONMENT_NAME="$2"
      shift
      shift
      ;;
    -r|--registry)
      REGISTRY_NAME="$2"
      shift
      shift
      ;;
    -y|--youtube-api-key)
      YOUTUBE_API_KEY="$2"
      shift
      shift
      ;;
    -h|--help)
      usage
      ;;
    *)
      echo "Unknown option: $1"
      usage
      ;;
  esac
done

# Check for required YouTube API key
if [ -z "$YOUTUBE_API_KEY" ]; then
  echo "Error: YouTube API key is required."
  usage
fi

echo "=== ClipSummary Azure Deployment ==="
echo "Resource Group: $RESOURCE_GROUP"
echo "Location: $LOCATION"
echo "Environment: $ENVIRONMENT_NAME"
echo "Container Registry: $REGISTRY_NAME"

# Create resource group if it doesn't exist
if [ $(az group exists --name $RESOURCE_GROUP) = false ]; then
    echo "Creating resource group $RESOURCE_GROUP..."
    az group create --name $RESOURCE_GROUP --location $LOCATION
fi

# Create Azure Container Registry if it doesn't exist
if ! az acr show --name $REGISTRY_NAME --resource-group $RESOURCE_GROUP &>/dev/null; then
    echo "Creating Azure Container Registry $REGISTRY_NAME..."
    az acr create --resource-group $RESOURCE_GROUP --name $REGISTRY_NAME --sku Basic
fi

# Build and push images to ACR
echo "Building and pushing frontend image..."
az acr build --registry $REGISTRY_NAME --image $FRONTEND_IMAGE --file ../frontend/Dockerfile ../frontend

echo "Building and pushing backend image..."
az acr build --registry $REGISTRY_NAME --image $BACKEND_IMAGE --file ../backend/Dockerfile ../backend

# Get ACR credentials
REGISTRY_URL=$(az acr show --name $REGISTRY_NAME --query loginServer --output tsv)
REGISTRY_USERNAME=$(az acr credential show --name $REGISTRY_NAME --query username --output tsv)
REGISTRY_PASSWORD=$(az acr credential show --name $REGISTRY_NAME --query passwords[0].value --output tsv)

# Deploy Container Apps using ARM template
echo "Deploying Container Apps..."
az deployment group create \
  --resource-group $RESOURCE_GROUP \
  --template-file ./templates/container-apps.json \
  --parameters \
    frontendImage="$REGISTRY_URL/$FRONTEND_IMAGE" \
    backendImage="$REGISTRY_URL/$BACKEND_IMAGE" \
    containerRegistryUrl="$REGISTRY_URL" \
    containerRegistryUsername="$REGISTRY_USERNAME" \
    containerRegistryPassword="$REGISTRY_PASSWORD" \
    environmentName="$ENVIRONMENT_NAME" \
    youtubeApiKey="$YOUTUBE_API_KEY" \
    redisConnectionString="$REDIS_CONNECTION_STRING" \
    location="$LOCATION"

# Get the deployed URLs
FRONTEND_URL=$(az deployment group show --resource-group $RESOURCE_GROUP --name container-apps --query properties.outputs.frontendUrl.value --output tsv)
BACKEND_URL=$(az deployment group show --resource-group $RESOURCE_GROUP --name container-apps --query properties.outputs.backendUrl.value --output tsv)

echo "Deployment completed successfully!"
echo "Frontend URL: $FRONTEND_URL"
echo "Backend URL: $BACKEND_URL"