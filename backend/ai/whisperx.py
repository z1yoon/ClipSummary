import torch
import whisperx
import os
import time
import logging
import gc
import threading
from typing import Dict, Any
from utils.cache import update_processing_status

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize device variable before any potential exceptions
device = "cpu"  # Default to CPU in case check fails

# Check CUDA availability and capabilities
try:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        compute_capability = torch.cuda.get_device_capability()
        torch_version = torch.__version__
        cuda_version = torch.version.cuda
        logger.info(f"CUDA available: device={torch.cuda.get_device_name(0)}, "
                   f"compute capability={compute_capability}, "
                   f"torch={torch_version}, CUDA={cuda_version}")
        
        # Check if compute capability is supported
        compute_type = "float16"
    else:
        logger.warning("CUDA not available, using CPU")
        compute_type = "int8"
except Exception as e:
    logger.warning(f"Error checking CUDA capabilities: {str(e)}, falling back to CPU")
    device = "cpu"
    compute_type = "int8"

# Fall back to CPU if needed
if os.environ.get("WHISPERX_DEVICE", "").lower() == "cpu":
    logger.info("Forcing CPU usage based on environment variable")
    device = "cpu"
    compute_type = "int8"

# Default model size - balance between accuracy and performance
DEFAULT_MODEL = "large-v2" if device == "cuda" else "medium"

# Get HuggingFace token from environment variables for speaker diarization
HF_TOKEN = os.environ.get("HUGGINGFACE_TOKEN", None)

# Cache paths - both the local and repo_id versions for better compatibility
MODEL_PATH = os.environ.get("TRANSFORMERS_CACHE", "/app/.cache/huggingface")
MODEL_ID = "Systran/faster-whisper-large-v2"
MODEL_DIR = os.path.join(MODEL_PATH, MODEL_ID.replace("/", "--"))

# Global variable to store the ASR model once loaded
asr_model = None

# Track model loading state
model_loading_lock = threading.Lock()
model_loading_state = {
    "is_loading": False,
    "start_time": None,
    "progress": 0,
    "message": ""
}

if HF_TOKEN:
    logger.info("HUGGINGFACE_TOKEN found, speaker diarization will be available")
else:
    logger.warning("HUGGINGFACE_TOKEN not found, speaker diarization may not work properly")

def is_model_loading():
    """Check if WhisperX model is currently being loaded"""
    return model_loading_state["is_loading"]

def wait_for_model(timeout=300):
    """
    Wait for the model to be loaded
    
    Args:
        timeout: Maximum time to wait in seconds
        
    Returns:
        bool: True if model was loaded successfully, False if timed out
    """
    global asr_model
    start_time = time.time()
    
    while time.time() - start_time < timeout:
        if asr_model is not None:
            return True
        if not model_loading_state["is_loading"]:
            # If not loading and still None, something went wrong
            return False
        time.sleep(1)
    
    return False  # Timeout

def load_models():
    """
    Load WhisperX models into memory
    
    This function loads the ASR model and sets the global asr_model variable.
    It uses a lock to prevent multiple threads from loading the model simultaneously.
    """
    global asr_model
    global model_loading_state
    
    # If model is already loaded, nothing to do
    if asr_model is not None:
        logger.info("WhisperX model is already loaded")
        return
    
    # Use a lock to prevent multiple threads from loading simultaneously
    acquired = model_loading_lock.acquire(blocking=False)
    if not acquired:
        logger.info("Another thread is already loading the WhisperX model")
        return
    
    try:
        model_loading_state["is_loading"] = True
        model_loading_state["start_time"] = time.time()
        model_loading_state["progress"] = 10
        model_loading_state["message"] = "Starting WhisperX model load"
        
        logger.info(f"Loading WhisperX model ({DEFAULT_MODEL}) on {device}...")
        
        # Progress updates
        model_loading_state["progress"] = 20
        model_loading_state["message"] = "Loading model files..."
        
        # Load the actual model - first check if model is local
        model_loading_state["progress"] = 50
        model_loading_state["message"] = f"Initializing WhisperX {DEFAULT_MODEL} model..."
        
        # Check if model directory exists
        if os.path.exists(MODEL_DIR) and not os.environ.get("TRANSFORMERS_OFFLINE", "0") == "0":
            logger.info(f"Loading WhisperX model from local cache: {MODEL_DIR}")
            try:
                # Load the model with path to local directory
                asr_model = whisperx.load_model(
                    MODEL_DIR, 
                    device, 
                    compute_type=compute_type,
                    local_files_only=True
                )
            except Exception as local_error:
                logger.warning(f"Failed to load model from local cache: {str(local_error)}")
                logger.info("Attempting to download model...")
                # Try downloading the model (requires TRANSFORMERS_OFFLINE=0)
                asr_model = whisperx.load_model(
                    DEFAULT_MODEL, 
                    device, 
                    compute_type=compute_type,
                    local_files_only=False
                )
        else:
            # Fall back to loading by model name (will download if needed)
            logger.info(f"Loading WhisperX model by name: {DEFAULT_MODEL}")
            
            # If we're encountering CUDA issues, fall back to CPU
            try:
                asr_model = whisperx.load_model(
                    DEFAULT_MODEL, 
                    device, 
                    compute_type=compute_type,
                    local_files_only=False
                )
            except RuntimeError as cuda_error:
                if "CUDA" in str(cuda_error):
                    logger.warning(f"CUDA error loading model, falling back to CPU: {str(cuda_error)}")
                    device = "cpu"
                    compute_type = "int8"
                    asr_model = whisperx.load_model(
                        DEFAULT_MODEL, 
                        device, 
                        compute_type=compute_type,
                        local_files_only=False
                    )
                else:
                    raise
        
        model_loading_state["progress"] = 100
        model_loading_state["message"] = "Model loaded successfully"
        
        # Log completion
        total_time = time.time() - model_loading_state["start_time"]
        logger.info(f"WhisperX model loaded successfully in {total_time:.2f}s on {device}")
        
    except Exception as e:
        logger.error(f"Failed to load WhisperX model: {str(e)}")
        model_loading_state["progress"] = 0
        model_loading_state["message"] = f"Error: {str(e)}"
    finally:
        model_loading_state["is_loading"] = False
        model_loading_lock.release()

def transcribe_audio(audio_path: str, upload_id: str = None, diarize: bool = True) -> Dict[str, Any]:
    """
    Transcribe audio using WhisperX with word-level timestamps and speaker diarization
    
    Args:
        audio_path: Path to audio file
        upload_id: Optional upload ID for progress tracking
        diarize: Whether to perform speaker diarization
        
    Returns:
        Dict containing transcription results with timestamps and speaker labels
    """
    # Make sure to use the global device and compute_type
    global device, compute_type
    
    # Ensure device is initialized even if something failed during module initialization
    if 'device' not in globals() or device is None:
        device = "cpu"
        compute_type = "int8"
        logger.warning(f"[{upload_id}] Device was not initialized, falling back to CPU")
    
    try:
        start_time = time.time()
        logger.info(f"[{upload_id}] Starting WhisperX transcription")
        
        # Log device info
        logger.info(f"[{upload_id}] Using device: {device}")
        if device == "cuda":
            logger.info(f"[{upload_id}] CUDA Device: {torch.cuda.get_device_name(0)}")
        
        if upload_id:
            update_processing_status(
                upload_id=upload_id,
                status="processing",
                progress=10,
                message="Loading WhisperX model..."
            )
        
        # 1. Load model and transcribe - prefer local cache
        logger.info(f"[{upload_id}] Loading WhisperX model ({DEFAULT_MODEL}) on {device}...")
        
        try:
            # First try loading from local model directory
            model = None
            use_cpu_fallback = False
            
            try:
                if os.path.exists(MODEL_DIR):
                    logger.info(f"[{upload_id}] Loading WhisperX model from local cache: {MODEL_DIR}")
                    model = whisperx.load_model(
                        MODEL_DIR,
                        device, 
                        compute_type=compute_type,
                        local_files_only=False
                    )
                else:
                    # Fall back to model name - offline will fail if not downloaded
                    logger.info(f"[{upload_id}] Loading WhisperX model by name: {DEFAULT_MODEL}")
                    model = whisperx.load_model(
                        DEFAULT_MODEL, 
                        device, 
                        compute_type=compute_type,
                        local_files_only=False
                    )
            except RuntimeError as e:
                if "CUDA" in str(e):
                    logger.warning(f"[{upload_id}] CUDA error: {str(e)}")
                    logger.info(f"[{upload_id}] Falling back to CPU for transcription")
                    use_cpu_fallback = True
                else:
                    raise
                    
            # Fall back to CPU if CUDA failed
            if use_cpu_fallback:
                model = whisperx.load_model(
                    DEFAULT_MODEL,
                    "cpu",
                    compute_type="int8",
                    local_files_only=False
                )
                
        except Exception as e:
            logger.error(f"[{upload_id}] Error loading model: {str(e)}")
            # Add debug info
            logger.info(f"[{upload_id}] Model directory exists: {os.path.exists(MODEL_DIR)}")
            logger.info(f"[{upload_id}] Model directory contents: {os.listdir(MODEL_PATH) if os.path.exists(MODEL_PATH) else 'not accessible'}")
            raise
        
        if upload_id:
            update_processing_status(
                upload_id=upload_id,
                status="processing",
                progress=30,
                message="Transcribing audio..."
            )
        
        # Load and transcribe audio
        audio = whisperx.load_audio(audio_path)
        batch_size = 16 if device == "cuda" else 8
        result = model.transcribe(audio, batch_size=batch_size)
        
        # Log detected language
        language_code = result["language"]
        logger.info(f"[{upload_id}] Detected language: {language_code}")
        
        # Free up GPU memory
        del model
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()
        
        if upload_id:
            update_processing_status(
                upload_id=upload_id,
                status="processing",
                progress=50,
                message="Aligning transcription with audio..."
            )
        
        # 2. Align whisper output
        logger.info(f"[{upload_id}] Loading alignment model...")
        try:
            model_a, metadata = whisperx.load_align_model(language_code=language_code, device=device)
        except RuntimeError as e:
            if "CUDA" in str(e):
                logger.warning(f"[{upload_id}] CUDA error loading alignment model, falling back to CPU")
                model_a, metadata = whisperx.load_align_model(language_code=language_code, device="cpu")
            else:
                raise
                
        result = whisperx.align(
            result["segments"],
            model_a,
            metadata,
            audio,
            device if model_a.device.type == "cuda" else "cpu",  # Use same device as model
            return_char_alignments=False
        )
        logger.info(f"[{upload_id}] Alignment complete")
        
        # Free up GPU memory
        del model_a
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()
        
        # 3. Speaker diarization (if requested and token available)
        if diarize and HF_TOKEN:
            if upload_id:
                update_processing_status(
                    upload_id=upload_id,
                    status="processing",
                    progress=70,
                    message="Identifying speakers..."
                )
            
            try:
                logger.info(f"[{upload_id}] Running speaker diarization...")
                # Try CPU if CUDA is problematic
                diarize_device = device
                try:
                    # Updated to use the 3.3.3 API for diarization
                    diarize_model = whisperx.DiarizationPipeline(
                        use_auth_token=HF_TOKEN,
                        device=diarize_device
                    )
                except RuntimeError as e:
                    if "CUDA" in str(e):
                        logger.warning(f"[{upload_id}] CUDA error in diarization, falling back to CPU")
                        diarize_device = "cpu"
                        diarize_model = whisperx.DiarizationPipeline(
                            use_auth_token=HF_TOKEN,
                            device=diarize_device
                        )
                    else:
                        raise
                
                # Run diarization
                diarize_segments = diarize_model(audio)
                
                # Assign speakers to words/segments
                result = whisperx.assign_word_speakers(diarize_segments, result)
                logger.info(f"[{upload_id}] Speaker diarization complete")
                
                # Count unique speakers
                speaker_set = set()
                for segment in result["segments"]:
                    if "speaker" in segment:
                        speaker_set.add(segment["speaker"])
                
                logger.info(f"[{upload_id}] Identified {len(speaker_set)} unique speakers")
                
                # Free up GPU memory
                del diarize_model
                gc.collect()
                if device == "cuda":
                    torch.cuda.empty_cache()
            
            except Exception as e:
                logger.error(f"[{upload_id}] Speaker diarization failed: {str(e)}")
                logger.info(f"[{upload_id}] Continuing with transcription without speaker identification")
        
        else:
            if not diarize:
                logger.info(f"[{upload_id}] Speaker diarization skipped by request")
            elif not HF_TOKEN:
                logger.warning(f"[{upload_id}] Speaker diarization skipped: No Hugging Face token provided")
        
        # Log completion time
        end_time = time.time()
        duration = end_time - start_time
        segment_count = len(result["segments"]) if "segments" in result else 0
        logger.info(f"[{upload_id}] Transcription completed, {segment_count} segments generated")
        logger.info(f"[{upload_id}] Processing took {duration:.2f} seconds")
        
        if upload_id:
            update_processing_status(
                upload_id=upload_id,
                status="completed",
                progress=100,
                message="Transcription complete"
            )
        
        return result
    
    except Exception as e:
        logger.error(f"[{upload_id}] Transcription failed: {str(e)}")
        if upload_id:
            update_processing_status(
                upload_id=upload_id,
                status="failed",
                progress=0,
                message=f"Processing failed: {str(e)}"
            )
        raise