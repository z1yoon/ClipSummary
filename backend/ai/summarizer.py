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
        
        # Clean and validate input text
        if not text or not text.strip():
            logger.warning(f"[{upload_id}] Empty or invalid input text provided")
            return "No transcript available for summarization."
        
        # Clean the text - remove excessive punctuation and whitespace
        cleaned_text = text.strip()
        
        # Remove excessive punctuation patterns that might confuse the model
        import re
        cleaned_text = re.sub(r'[^\w\s.,!?;:\'"()-]', ' ', cleaned_text)  # Remove unusual characters
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text)  # Normalize whitespace
        cleaned_text = re.sub(r'[.,!?;:]{3,}', '.', cleaned_text)  # Remove excessive punctuation
        
        # Check if text is mostly punctuation (garbage input)
        words = re.findall(r'\b\w+\b', cleaned_text)
        if len(words) < 10:
            logger.warning(f"[{upload_id}] Text appears to be mostly punctuation or too short")
            return "The transcript appears to be incomplete or corrupted."
        
        text_length = len(cleaned_text)
        logger.info(f"[{upload_id}] Starting summary generation for cleaned text of length {text_length}")
        
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
        
        # Tokenize with proper truncation for long texts
        tokenize_start = time.time()
        
        # For very long texts, truncate smartly by taking first and last parts
        if len(cleaned_text) > 3000:
            # Take first 2000 chars and last 1000 chars
            truncated_text = cleaned_text[:2000] + " ... " + cleaned_text[-1000:]
            logger.info(f"[{upload_id}] Text truncated for summarization: {len(truncated_text)} chars")
        else:
            truncated_text = cleaned_text
        
        inputs = tokenizer(
            truncated_text, 
            return_tensors="pt", 
            max_length=1024, 
            truncation=True,
            padding=True
        )
        input_ids = inputs["input_ids"].to(next(model.parameters()).device)
        attention_mask = inputs["attention_mask"].to(next(model.parameters()).device)
        
        tokenize_time = time.time() - tokenize_start
        logger.info(f"[{upload_id}] Text tokenized in {tokenize_time:.2f} seconds")
        
        # Generate summary with better parameters
        generate_start = time.time()
        with torch.no_grad():
            summary_ids = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_length=min(150, max_sentences * 30),  # Adjust based on requested sentences
                min_length=max(30, max_sentences * 8),    # Ensure minimum content
                num_beams=4,
                length_penalty=2.0,
                early_stopping=True,
                no_repeat_ngram_size=3,  # Avoid repetition
                do_sample=False  # Use deterministic generation
            )
        
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        generate_time = time.time() - generate_start
        
        # Post-process summary
        summary = summary.strip()
        
        # If summary is too short or seems corrupted, provide fallback
        if len(summary) < 20 or len(re.findall(r'\b\w+\b', summary)) < 5:
            logger.warning(f"[{upload_id}] Generated summary appears corrupted, using fallback")
            summary = f"This video contains a {len(words)}-word transcript discussing various topics."
        
        # Ensure summary ends with proper punctuation
        if summary and not summary[-1] in '.!?':
            summary += '.'
        
        # Log completion
        total_time = time.time() - start_time
        summary_length = len(summary)
        sentences = [s.strip() for s in summary.split('.') if s.strip()]
        num_sentences = len(sentences)
        
        logger.info(f"[{upload_id}] Summary generation completed:")
        logger.info(f"[{upload_id}] - Time taken: {total_time:.2f} seconds")
        logger.info(f"[{upload_id}] - Input length: {text_length} chars")
        logger.info(f"[{upload_id}] - Summary length: {summary_length} chars")
        logger.info(f"[{upload_id}] - Number of sentences: {num_sentences}")
        logger.info(f"[{upload_id}] - Summary preview: {summary[:100]}...")
        
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
        
        # Return a fallback summary instead of raising exception
        return "Summary generation failed. Please try processing the video again."