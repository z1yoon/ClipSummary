import logging
import time
from typing import Dict
from transformers import BartTokenizer, BartForConditionalGeneration
import torch
from utils.cache import update_processing_status

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Cache for loaded models
model_cache: Dict[str, tuple[BartForConditionalGeneration, BartTokenizer]] = {}

def load_summarization_model(upload_id: str = None):
    """Load BART summarization model with detailed logging."""
    model_name = "facebook/bart-large-cnn"
    
    # Check cache first
    if model_name in model_cache:
        logger.info(f"[{upload_id}] Using cached summarization model")
        return model_cache[model_name]
    
    try:
        start_time = time.time()
        logger.info(f"[{upload_id}] Loading summarization model ({model_name})")
        
        # Log CUDA status
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"[{upload_id}] Using device: {device}")
        if device == "cuda":
            logger.info(f"[{upload_id}] CUDA Device: {torch.cuda.get_device_name(0)}")
            logger.info(f"[{upload_id}] Available VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        
        # Load tokenizer and model
        tokenizer_start = time.time()
        tokenizer = BartTokenizer.from_pretrained(model_name)
        tokenizer_time = time.time() - tokenizer_start
        logger.info(f"[{upload_id}] Tokenizer loaded in {tokenizer_time:.2f} seconds")
        
        model_start = time.time()
        model = BartForConditionalGeneration.from_pretrained(model_name).to(device)
        model_time = time.time() - model_start
        logger.info(f"[{upload_id}] Model loaded in {model_time:.2f} seconds")
        
        # Cache the loaded model
        model_cache[model_name] = (model, tokenizer)
        
        total_time = time.time() - start_time
        logger.info(f"[{upload_id}] Summarization model setup completed in {total_time:.2f} seconds")
        
        return model, tokenizer
        
    except Exception as e:
        error_msg = f"Failed to load summarization model: {str(e)}"
        logger.error(f"[{upload_id}] {error_msg}")
        raise Exception(error_msg)

def generate_summary(text: str, max_sentences: int = 3, upload_id: str = None) -> str:
    """Generate summary with detailed logging and progress updates."""
    try:
        start_time = time.time()
        text_length = len(text)
        logger.info(f"[{upload_id}] Starting summary generation for text of length {text_length}")
        
        if upload_id:
            update_processing_status(
                upload_id=upload_id,
                status="processing",
                progress=45,
                message="Loading summarization model..."
            )
        
        # Load model
        model, tokenizer = load_summarization_model(upload_id)
        
        # Log model loaded status
        if upload_id:
            update_processing_status(
                upload_id=upload_id,
                status="processing",
                progress=47,
                message="Generating summary..."
            )
        
        # Tokenize and generate summary
        tokenize_start = time.time()
        inputs = tokenizer(text, return_tensors="pt", max_length=1024, truncation=True)
        input_ids = inputs["input_ids"].to(next(model.parameters()).device)
        tokenize_time = time.time() - tokenize_start
        logger.info(f"[{upload_id}] Text tokenized in {tokenize_time:.2f} seconds")
        
        # Generate summary
        generate_start = time.time()
        summary_ids = model.generate(
            input_ids,
            max_length=150,
            min_length=40,
            num_beams=4,
            length_penalty=2.0,
            early_stopping=True
        )
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        generate_time = time.time() - generate_start
        
        # Log completion
        total_time = time.time() - start_time
        summary_length = len(summary)
        sentences = summary.split('.')
        num_sentences = len([s for s in sentences if s.strip()])
        
        logger.info(f"[{upload_id}] Summary generation completed:")
        logger.info(f"[{upload_id}] - Time taken: {total_time:.2f} seconds")
        logger.info(f"[{upload_id}] - Input length: {text_length} chars")
        logger.info(f"[{upload_id}] - Summary length: {summary_length} chars")
        logger.info(f"[{upload_id}] - Number of sentences: {num_sentences}")
        logger.info(f"[{upload_id}] - Compression ratio: {summary_length/text_length:.2%}")
        
        if upload_id:
            update_processing_status(
                upload_id=upload_id,
                status="processing",
                progress=50,
                message=f"Summary generated ({num_sentences} sentences)"
            )
        
        return summary
        
    except Exception as e:
        error_msg = f"Summary generation failed: {str(e)}"
        logger.error(f"[{upload_id}] {error_msg}")
        if upload_id:
            update_processing_status(
                upload_id=upload_id,
                status="error",
                progress=0,
                message=error_msg
            )
        raise Exception(error_msg)