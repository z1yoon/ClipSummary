from transformers import MarianMTModel, MarianTokenizer
import torch
import logging
import time
import os
from typing import Dict, List, Optional, Tuple
from utils.cache import update_processing_status

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Cache for loaded models
model_cache: Dict[str, tuple[MarianMTModel, MarianTokenizer]] = {}

# Available models (based on what's actually downloaded)
AVAILABLE_MODELS = [
    'Helsinki-NLP/opus-mt-en-zh',  # English to Chinese
    'Helsinki-NLP/opus-mt-ko-en',  # Korean to English
    'Helsinki-NLP/opus-mt-mul-en', # Multiple languages to English
    'Helsinki-NLP/opus-mt-zh-en',  # Chinese to English
]

def get_model_name(source_lang: str, target_lang: str) -> Optional[str]:
    """
    Get the appropriate model name for the source and target language pair.
    
    Args:
        source_lang: Source language code (e.g., 'en', 'zh', 'ko')
        target_lang: Target language code (e.g., 'en', 'zh', 'ko')
    
    Returns:
        Model name or None if no appropriate model is available
    """
    # Specific language pair models
    lang_pair_models = {
        ('en', 'zh'): 'Helsinki-NLP/opus-mt-en-zh',
        ('zh', 'en'): 'Helsinki-NLP/opus-mt-zh-en',
        ('ko', 'en'): 'Helsinki-NLP/opus-mt-ko-en',
    }
    
    # Get the specific model for this language pair if available
    model_key = (source_lang, target_lang)
    if model_key in lang_pair_models and lang_pair_models[model_key] in AVAILABLE_MODELS:
        return lang_pair_models[model_key]
    
    # For translation TO English from various languages, use the multilingual model
    if target_lang == 'en' and 'Helsinki-NLP/opus-mt-mul-en' in AVAILABLE_MODELS:
        return 'Helsinki-NLP/opus-mt-mul-en'
    
    # No suitable model found
    return None

def check_model_available(model_name: str) -> bool:
    """Check if the model is in our list of available models."""
    if not model_name:
        return False
    return model_name in AVAILABLE_MODELS

def load_translation_model(source_lang: str, target_lang: str, upload_id: str = None) -> Optional[Tuple[MarianMTModel, MarianTokenizer]]:
    """Load translation model with detailed logging and fallback mechanisms."""
    model_name = get_model_name(source_lang, target_lang)
    
    # If model is not available for this language pair, return None
    if not model_name:
        logger.warning(f"[{upload_id}] Translation model for {source_lang} to {target_lang} is not available")
        return None
    
    # Check cache first
    if model_name in model_cache:
        logger.info(f"[{upload_id}] Using cached translation model for {source_lang} to {target_lang}")
        return model_cache[model_name]
    
    try:
        start_time = time.time()
        logger.info(f"[{upload_id}] Loading translation model for {source_lang} to {target_lang} ({model_name})")
        
        # Log CUDA status
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"[{upload_id}] Using device: {device}")
        if device == "cuda":
            logger.info(f"[{upload_id}] CUDA Device: {torch.cuda.get_device_name(0)}")
            logger.info(f"[{upload_id}] Available VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        
        # Load tokenizer and model
        tokenizer_start = time.time()
        tokenizer = MarianTokenizer.from_pretrained(model_name)
        tokenizer_time = time.time() - tokenizer_start
        logger.info(f"[{upload_id}] Tokenizer loaded in {tokenizer_time:.2f} seconds")
        
        model_start = time.time()
        model = MarianMTModel.from_pretrained(model_name).to(device)
        model_time = time.time() - model_start
        logger.info(f"[{upload_id}] Model loaded in {model_time:.2f} seconds")
        
        # Cache the loaded model
        model_cache[model_name] = (model, tokenizer)
        
        total_time = time.time() - start_time
        logger.info(f"[{upload_id}] Translation model setup completed in {total_time:.2f} seconds")
        
        return model, tokenizer
        
    except Exception as e:
        error_msg = f"Failed to load translation model: {str(e)}"
        logger.error(f"[{upload_id}] {error_msg}")
        # Return None instead of raising an exception
        return None

def translate_text(text: str, target_lang: str, source_lang: str = "en", upload_id: str = None, 
                  segment_index: int = None, total_segments: int = None) -> str:
    """Translate text with detailed logging and progress updates."""
    try:
        # Load model
        model_result = load_translation_model(source_lang, target_lang, upload_id)
        
        # If model couldn't be loaded, return original text
        if model_result is None:
            logger.warning(f"[{upload_id}] Translation from {source_lang} to {target_lang} skipped (model not available)")
            return text
            
        model, tokenizer = model_result
        
        # Log translation attempt
        start_time = time.time()
        text_length = len(text)
        logger.info(f"[{upload_id}] Translating text from {source_lang} to {target_lang} (length: {text_length} chars)")
        
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
        
        # Encode and translate
        device = next(model.parameters()).device
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        input_ids = inputs["input_ids"].to(device)
        
        # Generate translation
        outputs = model.generate(input_ids, max_length=512, num_beams=4, length_penalty=0.6)
        translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Log completion
        translation_time = time.time() - start_time
        output_length = len(translated_text)
        chars_per_second = text_length / translation_time
        
        logger.info(f"[{upload_id}] Translation completed:")
        logger.info(f"[{upload_id}] - Time taken: {translation_time:.2f} seconds")
        logger.info(f"[{upload_id}] - Input length: {text_length} chars")
        logger.info(f"[{upload_id}] - Output length: {output_length} chars")
        logger.info(f"[{upload_id}] - Speed: {chars_per_second:.2f} chars/second")
        
        return translated_text
        
    except Exception as e:
        error_msg = f"Translation failed: {str(e)}"
        logger.error(f"[{upload_id}] {error_msg}")
        logger.error(f"[{upload_id}] Returning original text due to translation error")
        
        # Instead of failing, return the original text
        return text

def batch_translate(texts: List[str], target_lang: str, source_lang: str = "en", upload_id: str = None) -> List[str]:
    """Batch translate multiple texts with progress tracking."""
    try:
        total_texts = len(texts)
        logger.info(f"[{upload_id}] Starting batch translation of {total_texts} texts from {source_lang} to {target_lang}")
        start_time = time.time()
        
        # First check if translation is available for this language pair
        model_result = load_translation_model(source_lang, target_lang, upload_id)
        if model_result is None:
            logger.warning(f"[{upload_id}] Translation from {source_lang} to {target_lang} skipped (model not available)")
            return texts  # Return original texts
        
        translated_texts = []
        for i, text in enumerate(texts, 1):
            translated = translate_text(
                text,
                target_lang,
                source_lang=source_lang,
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
        logger.warning(f"[{upload_id}] Returning original texts due to translation error")
        
        # Instead of failing, return original texts
        return texts