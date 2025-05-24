#!/usr/bin/env python3
"""
Model Downloader for ClipSummary

This script downloads all the necessary models for the ClipSummary application:
- WhisperX (Systran--faster-whisper-large-v2)
- Summarizers (facebook--bart-large-cnn)
- Translation models (NLLB for Chinese and Korean)

Models are saved to a "models" directory in the project root, which can be mounted
in the Docker container.
"""

import os
import sys
import logging
from pathlib import Path
from huggingface_hub import snapshot_download

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Define models to download
MODELS = [
    "Systran/faster-whisper-large-v2",   # WhisperX ASR model
    "facebook/bart-large-cnn",           # Summarization model
    "facebook/nllb-200-distilled-600M",  # NLLB translation model for Chinese and Korean
]

def download_model(model_id, output_dir):
    """
    Download a model from the Hugging Face Hub
    
    Args:
        model_id: Model identifier (e.g., "Systran/faster-whisper-large-v2")
        output_dir: Directory to save the model
    """
    logger.info(f"Downloading {model_id}...")
    
    # Convert model IDs with slashes to directory-friendly names
    model_dir_name = model_id.replace("/", "--")
    model_dir = os.path.join(output_dir, model_dir_name)
    
    try:
        # Create directory if it doesn't exist
        os.makedirs(model_dir, exist_ok=True)
        
        # Download model
        snapshot_download(
            repo_id=model_id,
            local_dir=model_dir,
            local_dir_use_symlinks=False  # Important for Docker volume mounts
        )
        
        logger.info(f"✓ Successfully downloaded {model_id} to {model_dir}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to download {model_id}: {str(e)}")
        return False

def main():
    """Main execution function"""
    # Create models directory in the project root
    script_dir = Path(__file__).resolve().parent
    models_dir = script_dir / "models"
    
    # Ensure the models directory exists
    os.makedirs(models_dir, exist_ok=True)
    
    logger.info(f"Downloading models to {models_dir}")
    
    # Download all models
    success_count = 0
    for model_id in MODELS:
        if download_model(model_id, models_dir):
            success_count += 1
    
    # Report results
    logger.info(f"Downloaded {success_count}/{len(MODELS)} models successfully")
    if success_count < len(MODELS):
        logger.warning("Some models failed to download. Check the logs above for details.")
        sys.exit(1)
    else:
        logger.info("All models downloaded successfully!")
        sys.exit(0)

if __name__ == "__main__":
    main()