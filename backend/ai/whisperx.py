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

# Force GPU usage - RTX 5090 with PyTorch 2.7.0 + CUDA 12.8
DEFAULT_MODEL = "large-v2"
DEFAULT_DEVICE = "cuda"
DEFAULT_COMPUTE_TYPE = "float16"

# Force CUDA visible devices to ensure GPU is used
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Verify RTX 5090 CUDA compatibility
def verify_rtx5090_cuda():
    """Verify CUDA works properly with RTX 5090"""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available! RTX 5090 requires CUDA support.")
    
    device_name = torch.cuda.get_device_name(0)
    cuda_version = torch.version.cuda
    torch_version = torch.__version__
    
    logger.info(f"CUDA device: {device_name}")
    logger.info(f"CUDA version: {cuda_version}")
    logger.info(f"PyTorch version: {torch_version}")
    
    # Test CUDA functionality
    try:
        test_tensor = torch.tensor([1.0, 2.0, 3.0]).cuda()
        result = test_tensor * 2
        assert result.is_cuda, "Tensor not on CUDA device"
        del test_tensor, result
        torch.cuda.empty_cache()
        logger.info(f"RTX 5090 CUDA verification successful")
        return True
    except Exception as e:
        raise RuntimeError(f"RTX 5090 CUDA test failed: {str(e)}")

# Verify RTX 5090 CUDA at startup
verify_rtx5090_cuda()

# Log configuration
logger.info(f"WhisperX configured with: model={DEFAULT_MODEL}, device={DEFAULT_DEVICE}")

# Detailed CUDA information
device_name = torch.cuda.get_device_name(0)
cuda_version = torch.version.cuda
torch_version = torch.__version__
device_count = torch.cuda.device_count()
current_device = torch.cuda.current_device()
capability = torch.cuda.get_device_capability(0)

logger.info(f"CUDA device count: {device_count}")
logger.info(f"Current CUDA device: {current_device} - {device_name}")
logger.info(f"CUDA version: {cuda_version}, PyTorch: {torch_version}")
logger.info(f"Device compute capability: {capability[0]}.{capability[1]}")

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
    """Load WhisperX models into memory on RTX 5090"""
    global asr_model, model_loading_state
    
    if asr_model is not None:
        logger.info("WhisperX model is already loaded")
        return
    
    acquired = model_loading_lock.acquire(blocking=False)
    if not acquired:
        logger.info("Another thread is already loading the WhisperX model")
        return
    
    try:
        model_loading_state["is_loading"] = True
        model_loading_state["start_time"] = time.time()
        model_loading_state["progress"] = 10
        model_loading_state["message"] = "Starting WhisperX model load"
        
        logger.info(f"Loading WhisperX model ({DEFAULT_MODEL}) on RTX 5090...")
        
        model_loading_state["progress"] = 50
        model_loading_state["message"] = f"Initializing WhisperX {DEFAULT_MODEL} model on RTX 5090..."
        
        # Load model on RTX 5090
        asr_model = whisperx.load_model(
            DEFAULT_MODEL, 
            DEFAULT_DEVICE, 
            compute_type=DEFAULT_COMPUTE_TYPE,
            local_files_only=False
        )
        
        model_loading_state["progress"] = 100
        model_loading_state["message"] = "Model loaded successfully on RTX 5090"
        
        total_time = time.time() - model_loading_state["start_time"]
        logger.info(f"WhisperX model loaded successfully on RTX 5090 in {total_time:.2f}s")
        
    except Exception as e:
        logger.error(f"Failed to load WhisperX model on RTX 5090: {str(e)}")
        model_loading_state["progress"] = 0
        model_loading_state["message"] = f"RTX 5090 Error: {str(e)}"
        raise RuntimeError(f"RTX 5090 model loading failed: {str(e)}")
    finally:
        model_loading_state["is_loading"] = False
        model_loading_lock.release()

def transcribe_audio(audio_path: str, upload_id: str = None, diarize: bool = True) -> Dict[str, Any]:
    """Transcribe audio using WhisperX on RTX 5090"""
    try:
        start_time = time.time()
        logger.info(f"[{upload_id}] Starting WhisperX transcription on RTX 5090")
        
        if upload_id:
            update_processing_status(
                upload_id=upload_id,
                status="processing",
                progress=10,
                message="Loading WhisperX model on RTX 5090..."
            )
        
        # Load model on RTX 5090
        logger.info(f"[{upload_id}] Loading WhisperX model ({DEFAULT_MODEL}) on RTX 5090...")
        
        model = whisperx.load_model(
            DEFAULT_MODEL, 
            DEFAULT_DEVICE, 
            compute_type=DEFAULT_COMPUTE_TYPE,
            local_files_only=False
        )
        logger.info(f"[{upload_id}] Successfully loaded WhisperX model on RTX 5090")
        
        if upload_id:
            update_processing_status(
                upload_id=upload_id,
                status="processing",
                progress=30,
                message="Transcribing audio on RTX 5090..."
            )
        
        # Load and transcribe audio
        audio = whisperx.load_audio(audio_path)
        batch_size = 16
        result = model.transcribe(audio, batch_size=batch_size)
        
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
                message="Aligning transcription on RTX 5090..."
            )
        
        # Load alignment model on RTX 5090
        logger.info(f"[{upload_id}] Loading alignment model on RTX 5090...")
        model_a, metadata = whisperx.load_align_model(language_code=language_code, device=DEFAULT_DEVICE)
                
        result = whisperx.align(
            result["segments"],
            model_a,
            metadata,
            audio,
            DEFAULT_DEVICE,
            return_char_alignments=False
        )
        logger.info(f"[{upload_id}] Alignment complete on RTX 5090")
        
        # Free up memory
        del model_a
        gc.collect()
        torch.cuda.empty_cache()
        
        # Speaker diarization on RTX 5090
        if diarize and HF_TOKEN:
            if upload_id:
                update_processing_status(
                    upload_id=upload_id,
                    status="processing",
                    progress=70,
                    message="Identifying speakers on RTX 5090..."
                )
            
            logger.info(f"[{upload_id}] Running speaker diarization on RTX 5090...")
            diarize_model = whisperx.DiarizationPipeline(
                use_auth_token=HF_TOKEN,
                device=DEFAULT_DEVICE
            )
            
            diarize_segments = diarize_model(audio)
            result = whisperx.assign_word_speakers(diarize_segments, result)
            logger.info(f"[{upload_id}] Speaker diarization complete on RTX 5090")
            
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
        
        else:
            if not diarize:
                logger.info(f"[{upload_id}] Speaker diarization skipped by request")
            elif not HF_TOKEN:
                logger.warning(f"[{upload_id}] Speaker diarization skipped: No Hugging Face token provided")
        
        # Log completion time
        end_time = time.time()
        duration = end_time - start_time
        segment_count = len(result["segments"]) if "segments" in result else 0
        logger.info(f"[{upload_id}] RTX 5090 transcription completed, {segment_count} segments generated")
        logger.info(f"[{upload_id}] Processing took {duration:.2f} seconds on RTX 5090")
        
        if upload_id:
            update_processing_status(
                upload_id=upload_id,
                status="completed",
                progress=100,
                message="RTX 5090 transcription complete"
            )
        
        return result
    
    except Exception as e:
        logger.error(f"[{upload_id}] RTX 5090 transcription failed: {str(e)}")
        if upload_id:
            update_processing_status(
                upload_id=upload_id,
                status="failed",
                progress=0,
                message=f"RTX 5090 processing failed: {str(e)}"
            )
        raise