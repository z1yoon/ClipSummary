#!/usr/bin/env python3
import os
import sys
import logging
import subprocess
from pathlib import Path
import time

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
        # Asian languages to English
        "Helsinki-NLP/opus-mt-zh-en",
        "Helsinki-NLP/opus-mt-ko-en",
        # Translations between Asian languages
        "Helsinki-NLP/opus-mt-zh-ko",
        "Helsinki-NLP/opus-mt-ko-zh",
        # Other languages
        "Helsinki-NLP/opus-mt-en-ru",
        "Helsinki-NLP/opus-mt-en-ar"
    ]
}

def install_dependencies():
    """Install required dependencies for model downloading"""
    logger.info("Installing required dependencies...")
    
    # List of required packages
    packages = [
        "torch",
        "transformers",
        "huggingface_hub",
        "sentencepiece",
        "protobuf",
        "faster-whisper",
    ]
    
    # Install each package
    for package in packages:
        try:
            logger.info(f"Installing {package}...")
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", package
            ])
            logger.info(f"✅ Successfully installed {package}")
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Failed to install {package}: {str(e)}")
            if package == "sentencepiece":
                logger.info("Attempting to install sentencepiece with additional dependencies...")
                try:
                    # On some systems, sentencepiece requires additional setup
                    subprocess.check_call([
                        sys.executable, "-m", "pip", "install", "sentencepiece", "--no-binary", "sentencepiece"
                    ])
                    logger.info("✅ Successfully installed sentencepiece from source")
                except subprocess.CalledProcessError as e:
                    logger.error(f"❌ Failed to install sentencepiece from source: {str(e)}")

def download_model_files(model_id, output_dir):
    """Download model files using huggingface-cli"""
    try:
        logger.info(f"Downloading model: {model_id}")
        
        # Create huggingface-cli command
        cmd = [
            sys.executable, "-m", "huggingface_hub", "download",
            "--repo-id", model_id,
            "--local-dir", str(output_dir / model_id.replace("/", "--")),
            "--local-dir-use-symlinks", "False"
        ]
        
        # Run the command
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Process output in real-time
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                logger.info(output.strip())
        
        # Check for errors
        stderr = process.stderr.read()
        if stderr:
            logger.warning(stderr)
        
        # Check return code
        if process.returncode == 0:
            logger.info(f"✅ Successfully downloaded {model_id}")
            return True
        else:
            logger.error(f"❌ Failed to download {model_id}, return code: {process.returncode}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Error downloading {model_id}: {str(e)}")
        return False

def main():
    """Main function to download all models"""
    # First install required dependencies
    install_dependencies()
    
    # Create models directory in the current working directory
    script_dir = Path(__file__).resolve().parent
    models_dir = script_dir / "models"
    models_dir.mkdir(exist_ok=True)
    
    # Set environment variables for Hugging Face
    os.environ["HF_HOME"] = str(models_dir)
    os.environ["TRANSFORMERS_CACHE"] = str(models_dir)
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(models_dir)
    
    logger.info(f"Using models directory: {models_dir}")
    
    # Download models
    total_models = sum(len(models) for models in MODELS.values())
    successful_downloads = 0
    
    # Download each model
    for model_type, model_list in MODELS.items():
        logger.info(f"\n=== Downloading {model_type} models ===")
        for model_id in model_list:
            success = download_model_files(model_id, models_dir)
            if success:
                successful_downloads += 1
    
    # Print summary
    logger.info("\n=== Download Summary ===")
    logger.info(f"Total models: {total_models}")
    logger.info(f"Successfully downloaded: {successful_downloads}")
    logger.info(f"Failed: {total_models - successful_downloads}")
    
    if successful_downloads == total_models:
        logger.info("\n✨ All models downloaded successfully!")
    else:
        logger.warning(f"\n⚠️ {total_models - successful_downloads} models failed to download.")
    
    # Print instructions for Docker
    logger.info("\n=== How to Use ===")
    logger.info("1. The models have been downloaded to the './models' directory")
    logger.info("2. This directory is mounted to '/root/.cache/huggingface' in the Docker container")
    logger.info("3. When running in offline mode (TRANSFORMERS_OFFLINE=1), the models will be loaded from this directory")
    logger.info("4. Restart your Docker containers with: docker-compose down && docker-compose up -d")

if __name__ == "__main__":
    main()