#!/usr/bin/env python3
import os
import shutil
import logging
import sys
from pathlib import Path
import torch
from huggingface_hub import snapshot_download
from transformers import (
    BartTokenizer, 
    BartForConditionalGeneration, 
    MarianMTModel, 
    MarianTokenizer
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("model_downloader")

# Models to download
MODELS = {
    # WhisperX model
    "transcription": [
        "Systran/faster-whisper-large-v2"
    ],
    # Summarization model
    "summarization": [
        "facebook/bart-large-cnn"
    ],
    # Translation models
    "translation": [
        "Helsinki-NLP/opus-mt-en-ROMANCE",
        "Helsinki-NLP/opus-mt-en-es",
        "Helsinki-NLP/opus-mt-en-fr",
        "Helsinki-NLP/opus-mt-en-de",
        # English to Asian languages
        "Helsinki-NLP/opus-mt-en-zh",
        "Helsinki-NLP/opus-mt-en-ko",
        "Helsinki-NLP/opus-mt-en-jap",
        # Asian languages to English
        "Helsinki-NLP/opus-mt-zh-en",
        "Helsinki-NLP/opus-mt-ko-en",
        "Helsinki-NLP/opus-mt-jap-en",
        # Translations between Asian languages
        "Helsinki-NLP/opus-mt-zh-ko",
        "Helsinki-NLP/opus-mt-ko-zh",
        # Other languages
        "Helsinki-NLP/opus-mt-en-ru",
        "Helsinki-NLP/opus-mt-en-ar",
        "Helsinki-NLP/opus-mt-en-hi"
    ]
}

def create_huggingface_structure(models_dir):
    """Create the proper directory structure expected by HuggingFace in containers"""
    # Create basic HF structure
    hub_dir = models_dir / "hub"
    hub_dir.mkdir(exist_ok=True)
    
    # Create main subdirectories
    models_dir = hub_dir / "models"
    models_dir.mkdir(exist_ok=True)
    
    # Create directory for specific model namespaces
    for model_group in MODELS.values():
        for model_id in model_group:
            # Split model_id into namespace and name (e.g., "Helsinki-NLP/opus-mt-en-ROMANCE" -> "Helsinki-NLP", "opus-mt-en-ROMANCE")
            parts = model_id.split('/')
            if len(parts) == 2:
                namespace = parts[0]
                namespace_dir = models_dir / namespace
                namespace_dir.mkdir(exist_ok=True)

def download_model(model_id, model_type):
    """Download a specific model"""
    try:
        logger.info(f"Downloading {model_type} model: {model_id}")
        
        if model_type == "transcription":
            # Download WhisperX model using snapshot_download
            snapshot_download(
                repo_id=model_id,
                repo_type="model",
                local_files_only=False
            )
            logger.info(f"✅ Successfully downloaded {model_id}")
            
        elif model_type == "summarization":
            # Download BART model using transformers
            tokenizer = BartTokenizer.from_pretrained(model_id)
            model = BartForConditionalGeneration.from_pretrained(model_id)
            logger.info(f"✅ Successfully downloaded {model_id}")
            
        elif model_type == "translation":
            # Download translation model using transformers
            tokenizer = MarianTokenizer.from_pretrained(model_id)
            model = MarianMTModel.from_pretrained(model_id)
            logger.info(f"✅ Successfully downloaded {model_id}")
            
        return True
        
    except Exception as e:
        logger.error(f"❌ Error downloading {model_id}: {str(e)}")
        return False

def main():
    """Main function to download all models"""
    # Create models directory in the current working directory
    script_dir = Path(__file__).resolve().parent
    models_dir = script_dir / "models"
    models_dir.mkdir(exist_ok=True)
    
    # Set the HF_HOME environment variable to use our models directory
    os.environ["HF_HOME"] = str(models_dir)
    os.environ["TRANSFORMERS_CACHE"] = str(models_dir)
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(models_dir)
    
    logger.info(f"Using models directory: {models_dir}")
    logger.info(f"Set HuggingFace cache directory to: {models_dir}")
    
    # Create HuggingFace directory structure
    create_huggingface_structure(models_dir)
    
    # Check available device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    if device == "cuda":
        logger.info(f"CUDA Device: {torch.cuda.get_device_name(0)}")
        logger.info(f"Available VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
    # Download models
    success = True
    for model_type, model_list in MODELS.items():
        logger.info(f"=== Downloading {model_type} models ===")
        for model_id in model_list:
            if not download_model(model_id, model_type):
                success = False
    
    if success:
        logger.info("✨ All models downloaded successfully!")
    else:
        logger.warning("⚠️ Some models failed to download. See logs for details.")
    
    # Print instructions for Docker
    logger.info("\n=== Instructions ===")
    logger.info("1. The models have been downloaded to the './models' directory")
    logger.info("2. This directory is mounted to '/root/.cache/huggingface' in the Docker container")
    logger.info("3. When running in offline mode (TRANSFORMERS_OFFLINE=1), the models will be loaded from this directory")
    logger.info("4. Restart your Docker containers to use the downloaded models")

if __name__ == "__main__":
    main()