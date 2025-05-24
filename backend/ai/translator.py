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
    model_dir = models_path / "facebook--nllb-200-distilled-600M"
    
    if not model_dir.exists():
        raise FileNotFoundError(f"Model not found at {model_dir}. Please run download_models.py first.")
    
    return str(model_dir)

def load_translation_model(target_lang: str, upload_id: str = None) -> Tuple[Any, Any]:
    """Load translation model from local path with detailed logging."""
    model_path = get_model_path(target_lang)
    
    # Check cache first (use model_path as cache key)
    if model_path in model_cache:
        logger.info(f"[{upload_id}] Using cached translation model for {target_lang}")
        return model_cache[model_path]
    
    try:
        start_time = time.time()
        logger.info(f"[{upload_id}] Loading translation model for {target_lang} from {model_path}")
        
        # Log CUDA status
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"[{upload_id}] Using device: {device}")
        if device == "cuda":
            logger.info(f"[{upload_id}] CUDA Device: {torch.cuda.get_device_name(0)}")
            logger.info(f"[{upload_id}] Available VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        
        # Load NLLB tokenizer and model from local path
        tokenizer_start = time.time()
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        tokenizer_time = time.time() - tokenizer_start
        logger.info(f"[{upload_id}] Tokenizer loaded in {tokenizer_time:.2f} seconds")
        
        model_start = time.time()
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path, local_files_only=True).to(device)
        model_time = time.time() - model_start
        logger.info(f"[{upload_id}] Model loaded in {model_time:.2f} seconds")
        
        # Cache the loaded model
        model_cache[model_path] = (model, tokenizer)
        
        total_time = time.time() - start_time
        logger.info(f"[{upload_id}] Translation model setup completed in {total_time:.2f} seconds")
        
        return model, tokenizer
        
    except Exception as e:
        error_msg = f"Failed to load translation model from {model_path}: {str(e)}"
        logger.error(f"[{upload_id}] {error_msg}")
        raise RuntimeError(error_msg)

def translate_text(text: str, target_lang: str, upload_id: str = None, 
                  segment_index: int = None, total_segments: int = None) -> str:
    """Translate text with detailed logging and progress updates."""
    try:
        # Load model
        model, tokenizer = load_translation_model(target_lang, upload_id)
        
        # Log translation attempt
        start_time = time.time()
        text_length = len(text)
        logger.info(f"[{upload_id}] Translating text to {target_lang} (length: {text_length} chars)")
        
        if segment_index is not None and total_segments is not None:
            progress_msg = f"Translating segment {segment_index}/{total_segments} to {target_lang}"
            logger.info(f"[{upload_id}] {progress_msg}")
            if upload_id:
                progress = 50 + ((segment_index / total_segments) * 40)
                update_processing_status(
                    upload_id=upload_id,
                    status="processing",
                    progress=progress,
                    message=progress_msg
                )
        
        # Map target language to NLLB format
        nllb_lang_map = {
            'ko': 'kor_Kore',
            'zh': 'zho_Hans'
        }
        
        # Get target language code
        tgt_lang = nllb_lang_map.get(target_lang)
        if not tgt_lang:
            raise ValueError(f"Unsupported target language: {target_lang}")
        
        # Encode and translate using NLLB
        device = next(model.parameters()).device
        
        # Prepare inputs - NLLB doesn't need source language prefix in newer versions
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        input_ids = inputs["input_ids"].to(device)
        
        # Set target language for generation
        tokenizer.src_lang = "eng_Latn"
        tokenizer.tgt_lang = tgt_lang
        
        # Generate translation
        outputs = model.generate(
            input_ids, 
            max_length=512, 
            num_beams=4, 
            length_penalty=0.6,
            early_stopping=True
        )
        
        translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Log completion
        translation_time = time.time() - start_time
        output_length = len(translated_text)
        chars_per_second = text_length / translation_time if translation_time > 0 else 0
        
        logger.info(f"[{upload_id}] Translation completed:")
        logger.info(f"[{upload_id}] - Time taken: {translation_time:.2f} seconds")
        logger.info(f"[{upload_id}] - Input length: {text_length} chars")
        logger.info(f"[{upload_id}] - Output length: {output_length} chars")
        logger.info(f"[{upload_id}] - Speed: {chars_per_second:.2f} chars/second")
        
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