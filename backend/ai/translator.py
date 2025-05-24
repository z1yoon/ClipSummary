import os
from pathlib import Path
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch
import logging
import time
from typing import Dict, List, Tuple, Any, Union
from utils.cache import update_processing_status

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Cache for loaded models
model_cache: Dict[str, Tuple[Any, Any]] = {}

def get_models_path() -> Path:
    """Get the path to the models directory."""
    # In Docker, models are mounted at /app/models
    # In development, they're in the project root/models
    if os.path.exists("/app/models"):
        return Path("/app/models")
    else:
        # Development path
        project_root = Path(__file__).resolve().parent.parent.parent
        return project_root / "models"

def get_model_path(target_lang: str) -> str:
    """Get the local path for the translation model."""
    if target_lang not in ['zh', 'ko']:
        raise ValueError(f"Unsupported language: {target_lang}. Only 'zh' (Chinese) and 'ko' (Korean) are supported.")
    
    models_path = get_models_path()
    # Use M2M100 model instead of NLLB for better language support
    model_dir = models_path / "facebook--m2m100_418M"
    
    if not model_dir.exists():
        raise FileNotFoundError(f"Model not found at {model_dir}. Please download the M2M100 model first.")
    
    return str(model_dir)

def load_translation_model(target_lang: str, upload_id: str = None) -> Tuple[Any, Any]:
    """Load M2M100 translation model from local path with detailed logging."""
    model_path = get_model_path(target_lang)
    
    # Check cache first (use model_path as cache key)
    if model_path in model_cache:
        logger.info(f"[{upload_id}] Using cached M2M100 translation model for {target_lang}")
        return model_cache[model_path]
    
    try:
        start_time = time.time()
        logger.info(f"[{upload_id}] Loading M2M100 translation model for {target_lang} from {model_path}")
        
        # Log CUDA status
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"[{upload_id}] Using device: {device}")
        if device == "cuda":
            logger.info(f"[{upload_id}] CUDA Device: {torch.cuda.get_device_name(0)}")
            logger.info(f"[{upload_id}] Available VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        
        # Load M2M100 tokenizer and model from local path
        tokenizer_start = time.time()
        if os.path.exists(model_path):
            tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        else:
            # Fallback to online model if local not available
            logger.warning(f"[{upload_id}] Local model not found, downloading M2M100 model...")
            tokenizer = AutoTokenizer.from_pretrained("facebook/m2m100_418M")
        
        tokenizer_time = time.time() - tokenizer_start
        logger.info(f"[{upload_id}] M2M100 Tokenizer loaded in {tokenizer_time:.2f} seconds")
        
        model_start = time.time()
        if os.path.exists(model_path):
            model = AutoModelForSeq2SeqLM.from_pretrained(model_path, local_files_only=True).to(device)
        else:
            # Fallback to online model if local not available
            model = AutoModelForSeq2SeqLM.from_pretrained("facebook/m2m100_418M").to(device)
        
        model_time = time.time() - model_start
        logger.info(f"[{upload_id}] M2M100 Model loaded in {model_time:.2f} seconds")
        
        # Cache the loaded model
        model_cache[model_path] = (model, tokenizer)
        
        total_time = time.time() - start_time
        logger.info(f"[{upload_id}] M2M100 translation model setup completed in {total_time:.2f} seconds")
        
        return model, tokenizer
        
    except Exception as e:
        error_msg = f"Failed to load M2M100 translation model from {model_path}: {str(e)}"
        logger.error(f"[{upload_id}] {error_msg}")
        raise RuntimeError(error_msg)

def translate_text(text: str, target_lang: str, upload_id: str = None, 
                  segment_index: int = None, total_segments: int = None) -> str:
    """Translate text using M2M100 model with proper language forcing."""
    try:
        # Load model
        model, tokenizer = load_translation_model(target_lang, upload_id)
        
        # Only log every 10th segment or for summary
        should_log = (segment_index is None or 
                     total_segments is None or 
                     segment_index % 10 == 0 or 
                     segment_index == 1 or 
                     segment_index == total_segments)
        
        if should_log:
            logger.info(f"[{upload_id}] Translating to {target_lang} - segment {segment_index}/{total_segments}")
        
        if segment_index is not None and total_segments is not None and upload_id:
            progress = 50 + ((segment_index / total_segments) * 40)
            # Only update status every 10 segments to reduce backend load
            if segment_index % 10 == 0:
                update_processing_status(
                    upload_id=upload_id,
                    status="processing",
                    progress=progress,
                    message=f"Translating to {target_lang} ({segment_index}/{total_segments})"
                )
        
        # Set source language to English for M2M100
        tokenizer.src_lang = "en"
        
        # Encode the input text
        encoded = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        
        # Move to device
        device = next(model.parameters()).device
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)
        
        # Generate translation using M2M100's get_lang_id method
        with torch.no_grad():
            generated_tokens = model.generate(
                input_ids,
                attention_mask=attention_mask,
                forced_bos_token_id=tokenizer.get_lang_id(target_lang),
                max_length=512,
                num_beams=4,
                length_penalty=0.6,
                early_stopping=True,
                do_sample=False
            )
        
        # Decode the translation
        translated_text = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
        
        # Log a sample translation for debugging
        if should_log and upload_id:
            logger.info(f"[{upload_id}] Translation sample: '{text[:50]}...' -> '{translated_text[:50]}...'")
        
        return translated_text
        
    except Exception as e:
        error_msg = f"Translation failed: {str(e)}"
        logger.error(f"[{upload_id}] {error_msg}")
        if upload_id:
            update_processing_status(
                upload_id=upload_id,
                status="error",
                progress=0,
                message=error_msg
            )
        raise RuntimeError(error_msg)

def batch_translate(texts: List[str], target_lang: str, upload_id: str = None) -> List[str]:
    """Batch translate multiple texts with progress tracking."""
    try:
        total_texts = len(texts)
        logger.info(f"[{upload_id}] Starting batch translation of {total_texts} texts to {target_lang}")
        start_time = time.time()
        
        translated_texts = []
        for i, text in enumerate(texts, 1):
            translated = translate_text(
                text, 
                target_lang, 
                upload_id=upload_id,
                segment_index=i,
                total_segments=total_texts
            )
            translated_texts.append(translated)
            
            if i % 10 == 0:
                logger.info(f"[{upload_id}] Translated {i}/{total_texts} segments")
        
        total_time = time.time() - start_time
        avg_time = total_time / total_texts
        logger.info(f"[{upload_id}] Batch translation completed:")
        logger.info(f"[{upload_id}] - Total time: {total_time:.2f} seconds")
        logger.info(f"[{upload_id}] - Average time per text: {avg_time:.2f} seconds")
        
        return translated_texts
        
    except Exception as e:
        error_msg = f"Batch translation failed: {str(e)}"
        logger.error(f"[{upload_id}] {error_msg}")
        if upload_id:
            update_processing_status(
                upload_id=upload_id,
                status="error",
                progress=0,
                message=error_msg
            )
        raise RuntimeError(error_msg)

def translate_summary(summary_text: str, target_lang: str, upload_id: str = None) -> str:
    """Translate summary text to target language."""
    try:
        logger.info(f"[{upload_id}] Translating summary to {target_lang}")
        
        if upload_id:
            update_processing_status(
                upload_id=upload_id,
                status="processing",
                progress=60,
                message=f"Translating summary to {target_lang}..."
            )
        
        translated_summary = translate_text(summary_text, target_lang, upload_id)
        
        logger.info(f"[{upload_id}] Summary translation completed")
        return translated_summary
        
    except Exception as e:
        error_msg = f"Summary translation failed: {str(e)}"
        logger.error(f"[{upload_id}] {error_msg}")
        return summary_text  # Return original if translation fails

def translate_subtitle_segments(segments: List[dict], target_lang: str, upload_id: str = None) -> List[dict]:
    """Translate subtitle segments to target language."""
    try:
        logger.info(f"[{upload_id}] Translating {len(segments)} subtitle segments to {target_lang}")
        
        if upload_id:
            update_processing_status(
                upload_id=upload_id,
                status="processing",
                progress=70,
                message=f"Translating subtitles to {target_lang}..."
            )
        
        # Extract texts for batch translation
        texts = [segment.get('text', '') for segment in segments]
        
        # Batch translate
        translated_texts = batch_translate(texts, target_lang, upload_id)
        
        # Create new segments with translated text
        translated_segments = []
        for i, segment in enumerate(segments):
            translated_segment = segment.copy()
            translated_segment['text'] = translated_texts[i]
            translated_segments.append(translated_segment)
        
        logger.info(f"[{upload_id}] Subtitle translation completed")
        return translated_segments
        
    except Exception as e:
        error_msg = f"Subtitle translation failed: {str(e)}"
        logger.error(f"[{upload_id}] {error_msg}")
        return segments  # Return original if translation fails

def get_supported_languages() -> Dict[str, str]:
    """Get list of supported translation languages."""
    return {
        'zh': 'Chinese',
        'ko': 'Korean'
    }