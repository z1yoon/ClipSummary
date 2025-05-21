import torch
import whisperx
import os
import time
import logging
from typing import Dict, Any
from utils.cache import update_processing_status

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Cache device and models to avoid reloading
device = "cuda" if torch.cuda.is_available() else "cpu"
compute_type = "float16" if device == "cuda" else "float32"
asr_model = None
alignment_model = None

def load_models():
    """Load models if not already loaded"""
    global asr_model, alignment_model
    
    if asr_model is None:
        # Load the main ASR model
        logger.info("Loading WhisperX ASR model...")
        model_load_start = time.time()
        asr_model = whisperx.load_model(
            "large-v2", 
            device=device, 
            compute_type=compute_type,
            language="en"  # Default language, will be auto-detected
        )
        model_load_time = time.time() - model_load_start
        logger.info(f"ASR model loaded in {model_load_time:.2f} seconds")
    
    # No need to pre-load alignment model as it's language-specific
    # and will be loaded based on detected language
    
    return asr_model

def transcribe_audio(audio_path: str, upload_id: str = None) -> Dict[str, Any]:
    """
    Transcribe audio using WhisperX with word-level timestamps and detailed logging
    
    Args:
        audio_path: Path to audio file
        upload_id: Optional upload ID for progress tracking
        
    Returns:
        Dict containing transcription results with timestamps
    """
    try:
        start_time = time.time()
        logger.info(f"[{upload_id}] Starting WhisperX transcription process")
        
        # Log CUDA availability
        cuda_available = torch.cuda.is_available()
        logger.info(f"[{upload_id}] Using device: {device}")
        if cuda_available:
            logger.info(f"[{upload_id}] CUDA Device: {torch.cuda.get_device_name(0)}")
            logger.info(f"[{upload_id}] Available VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

        # Log audio file details
        audio_size = os.path.getsize(audio_path) / (1024 * 1024)  # Size in MB
        logger.info(f"[{upload_id}] Audio file size: {audio_size:.2f} MB")

        if upload_id:
            update_processing_status(
                upload_id=upload_id,
                status="processing",
                progress=20,
                message="Loading WhisperX model..."
            )

        # Load models
        model = load_models()

        # Load audio
        audio = whisperx.load_audio(audio_path)
        
        # Log audio duration if upload_id provided
        if upload_id:
            duration = len(audio) / 16000  # Convert samples to seconds
            logger.info(f"[{upload_id}] Audio duration: {duration:.2f} seconds")
            update_processing_status(
                upload_id=upload_id,
                status="processing",
                progress=25,
                message=f"Transcribing {duration:.1f} seconds of audio..."
            )
        
        # Transcribe with batch size appropriate for the device
        batch_size = 16 if device == "cuda" else 8
        transcribe_start = time.time()
        logger.info(f"[{upload_id}] Starting transcription...")
        result = model.transcribe(audio, batch_size=batch_size)
        transcribe_time = time.time() - transcribe_start
        logger.info(f"[{upload_id}] Transcription completed in {transcribe_time:.2f} seconds")
        
        # Get the detected language
        language_code = result["language"]
        
        if upload_id:
            logger.info(f"[{upload_id}] Detected language: {language_code}")
            update_processing_status(
                upload_id=upload_id,
                status="processing",
                progress=35,
                message="Aligning transcription with audio..."
            )
        
        # Load alignment model for the detected language
        align_model_load_start = time.time()
        logger.info(f"[{upload_id}] Loading alignment model...")
        alignment_model, metadata = whisperx.load_align_model(
            language_code=language_code,
            device=device
        )
        align_model_load_time = time.time() - align_model_load_start
        logger.info(f"[{upload_id}] Alignment model loaded in {align_model_load_time:.2f} seconds")
        
        # Align the transcription with the audio to get word-level timestamps
        align_start = time.time()
        logger.info(f"[{upload_id}] Aligning timestamps...")
        result = whisperx.align(
            result["segments"],
            alignment_model,
            metadata,
            audio,
            device,
            return_char_alignments=False
        )
        align_time = time.time() - align_start
        logger.info(f"[{upload_id}] Alignment completed in {align_time:.2f} seconds")
        
        if upload_id:
            segments_count = len(result.get("segments", []))
            logger.info(f"[{upload_id}] Alignment complete. Generated {segments_count} segments")
        
        # Log final statistics
        total_time = time.time() - start_time
        num_segments = len(result["segments"])
        total_duration = result["segments"][-1]["end"] if result["segments"] else 0
        words_per_second = sum(len(s["text"].split()) for s in result["segments"]) / total_duration if total_duration > 0 else 0

        logger.info(f"[{upload_id}] Transcription Statistics:")
        logger.info(f"[{upload_id}] - Total processing time: {total_time:.2f} seconds")
        logger.info(f"[{upload_id}] - Number of segments: {num_segments}")
        logger.info(f"[{upload_id}] - Audio duration: {total_duration:.2f} seconds")
        logger.info(f"[{upload_id}] - Processing speed: {total_duration/total_time:.2f}x realtime")
        logger.info(f"[{upload_id}] - Words per second: {words_per_second:.2f}")

        if upload_id:
            update_processing_status(
                upload_id=upload_id,
                status="processing",
                progress=40,
                message=f"Transcription completed: {num_segments} segments processed"
            )

        # Return the aligned result with word-level timestamps
        return result
        
    except Exception as e:
        error_msg = f"Transcription failed: {str(e)}"
        logger.error(f"[{upload_id}] {error_msg}")
        if upload_id:
            update_processing_status(
                upload_id=upload_id,
                status="error",
                progress=0,
                message=error_msg
            )
        return {
            "error": str(e),
            "segments": []
        }