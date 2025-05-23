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

# Set fixed values to avoid any reference errors
DEFAULT_MODEL = "large-v2"
DEFAULT_COMPUTE_TYPE = "float16"

# Force CUDA visible devices to ensure GPU is used
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Improved CUDA detection and device selection
def detect_best_device():
    """Detect the best available device for computation"""
    if not torch.cuda.is_available():
        logger.warning("CUDA is not available - using CPU")
        return "cpu", "int8"
    
    try:
        # Test if CUDA actually works with a simple operation
        test_tensor = torch.tensor([1.0, 2.0, 3.0])
        test_result = test_tensor.cuda()
        _ = test_result * 2  # Simple operation to verify CUDA works
        del test_tensor, test_result
        torch.cuda.empty_cache()
        
        device_name = torch.cuda.get_device_name(0)
        logger.info(f"CUDA test successful - using GPU: {device_name}")
        return "cuda", "float16"
        
    except Exception as e:
        logger.warning(f"CUDA test failed: {str(e)}")
        logger.info("CUDA is available but not working properly - falling back to CPU")
        return "cpu", "int8"

# Detect the best device at startup
DEFAULT_DEVICE, DEFAULT_COMPUTE_TYPE = detect_best_device()

# Log configuration
logger.info(f"WhisperX configured with: model={DEFAULT_MODEL}, device={DEFAULT_DEVICE}")

# More detailed CUDA check
if torch.cuda.is_available():
    try:
        device_name = torch.cuda.get_device_name(0)
        cuda_version = torch.version.cuda
        torch_version = torch.__version__
        device_count = torch.cuda.device_count()
        current_device = torch.cuda.current_device()
        
        logger.info(f"CUDA is available: {device_count} device(s)")
        logger.info(f"Current CUDA device: {current_device} - {device_name}")
        logger.info(f"CUDA version: {cuda_version}, PyTorch: {torch_version}")
        
        # Check if this is an RTX 5090 with potential compatibility issues
        if "RTX 5090" in device_name:
            logger.warning("RTX 5090 detected - checking CUDA compatibility...")
            logger.info("If you encounter CUDA errors, the PyTorch version may need updating")
        
    except Exception as e:
        logger.error(f"Error during CUDA verification: {str(e)}")
        logger.info("Will fall back to CPU processing")
else:
    logger.warning("CUDA NOT AVAILABLE - using CPU processing")

# Get HuggingFace token from environment variables for speaker diarization
HF_TOKEN = os.environ.get("HUGGINGFACE_TOKEN", None)

# Cache paths
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
    global asr_model, model_loading_state
    
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
        
        logger.info(f"Loading WhisperX model ({DEFAULT_MODEL}) on {DEFAULT_DEVICE}...")
        
        # Progress updates
        model_loading_state["progress"] = 20
        model_loading_state["message"] = "Loading model files..."
        
        # Load the actual model
        model_loading_state["progress"] = 50
        model_loading_state["message"] = f"Initializing WhisperX {DEFAULT_MODEL} model..."
        
        try:
            # Try with GPU first
            logger.info(f"Attempting to load model with GPU")
            asr_model = whisperx.load_model(
                DEFAULT_MODEL, 
                DEFAULT_DEVICE, 
                compute_type=DEFAULT_COMPUTE_TYPE,
                local_files_only=False
            )
            logger.info("Successfully loaded model with GPU")
        except Exception as e:
            # If GPU fails, try CPU
            logger.warning(f"GPU load failed: {str(e)}")
            logger.info("Falling back to CPU")
            try:
                asr_model = whisperx.load_model(
                    DEFAULT_MODEL, 
                    "cpu", 
                    compute_type="int8",
                    local_files_only=False
                )
                logger.info("Successfully loaded model with CPU")
            except Exception as cpu_e:
                logger.error(f"CPU load also failed: {str(cpu_e)}")
                raise
            
        model_loading_state["progress"] = 100
        model_loading_state["message"] = "Model loaded successfully"
        
        # Log completion
        total_time = time.time() - model_loading_state["start_time"]
        logger.info(f"WhisperX model loaded successfully in {total_time:.2f}s")
        
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
    try:
        start_time = time.time()
        logger.info(f"[{upload_id}] Starting WhisperX transcription")
        
        if upload_id:
            update_processing_status(
                upload_id=upload_id,
                status="processing",
                progress=10,
                message="Loading WhisperX model..."
            )
        
        # 1. Load model and transcribe - use fixed constants
        logger.info(f"[{upload_id}] Loading WhisperX model ({DEFAULT_MODEL})...")
        
        # Double-check CUDA status before loading model
        if not torch.cuda.is_available():
            logger.error(f"[{upload_id}] CUDA is not available! Cannot use GPU.")
        else:
            logger.info(f"[{upload_id}] CUDA is available: {torch.cuda.get_device_name(0)}")
        
        try:
            # First try with GPU
            logger.info(f"[{upload_id}] Attempting to load model with GPU (CUDA)")
            model = whisperx.load_model(
                DEFAULT_MODEL, 
                DEFAULT_DEVICE, 
                compute_type=DEFAULT_COMPUTE_TYPE,
                local_files_only=False
            )
            logger.info(f"[{upload_id}] Successfully loaded WhisperX model with GPU")
        except Exception as e:
            # Log detailed error
            logger.error(f"[{upload_id}] GPU load failed with error: {str(e)}")
            logger.error(f"[{upload_id}] Error type: {type(e).__name__}")
            
            # Show traceback for better debugging
            import traceback
            tb = traceback.format_exc()
            logger.error(f"[{upload_id}] Error traceback: {tb}")
            
            logger.info(f"[{upload_id}] Falling back to CPU")
            model = whisperx.load_model(
                DEFAULT_MODEL,
                "cpu",
                compute_type="int8",
                local_files_only=False
            )
            logger.info(f"[{upload_id}] Successfully loaded WhisperX model with CPU")
        
        if upload_id:
            update_processing_status(
                upload_id=upload_id,
                status="processing",
                progress=30,
                message="Transcribing audio..."
            )
        
        # Load and transcribe audio
        audio = whisperx.load_audio(audio_path)
        batch_size = 16  # Use a fixed batch size
        result = model.transcribe(audio, batch_size=batch_size)
        
        # Log detected language
        language_code = result["language"]
        logger.info(f"[{upload_id}] Detected language: {language_code}")
        
        # Free up GPU memory
        del model
        gc.collect()
        torch.cuda.empty_cache()
        
        if upload_id:
            update_processing_status(
                upload_id=upload_id,
                status="processing",
                progress=50,
                message="Aligning transcription with audio..."
            )
        
        # 2. Align whisper output - try with GPU first, then fall back to CPU if needed
        logger.info(f"[{upload_id}] Loading alignment model...")
        try:
            model_a, metadata = whisperx.load_align_model(language_code=language_code, device=DEFAULT_DEVICE)
            align_device = DEFAULT_DEVICE
        except Exception as e:
            logger.warning(f"[{upload_id}] GPU alignment failed: {str(e)}")
            model_a, metadata = whisperx.load_align_model(language_code=language_code, device="cpu")
            align_device = "cpu"
                
        result = whisperx.align(
            result["segments"],
            model_a,
            metadata,
            audio,
            align_device,
            return_char_alignments=False
        )
        logger.info(f"[{upload_id}] Alignment complete")
        
        # Free up memory
        del model_a
        gc.collect()
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
                try:
                    # Try GPU first
                    diarize_model = whisperx.DiarizationPipeline(
                        use_auth_token=HF_TOKEN,
                        device=DEFAULT_DEVICE
                    )
                except Exception as e:
                    logger.warning(f"[{upload_id}] GPU diarization failed: {str(e)}")
                    diarize_model = whisperx.DiarizationPipeline(
                        use_auth_token=HF_TOKEN,
                        device="cpu"
                    )
                
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
                
                # Free up memory
                del diarize_model
                gc.collect()
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