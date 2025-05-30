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

# GPU configuration for RTX 5090 with PyTorch 2.5.1 + CUDA 12.4
DEFAULT_MODEL = "large-v2"
DEFAULT_DEVICE = "cuda"
DEFAULT_COMPUTE_TYPE = "float16"

# Force CUDA visible devices to ensure GPU is used
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Global model cache
_model_cache = {}
_alignment_cache = {}
_diarization_cache = {}

def preload_whisperx_model():
    """Pre-load WhisperX model to reduce first-request latency."""
    try:
        logger.info("Pre-loading WhisperX model for faster processing...")
        
        # Check if CUDA is available
        device = "cuda" if torch.cuda.is_available() else "cpu"
        compute_type = "float16" if device == "cuda" else "int8"
        
        logger.info(f"Using device: {device}, compute_type: {compute_type}")
        
        # Load main transcription model
        model = whisperx.load_model(DEFAULT_MODEL, device, compute_type=compute_type)
        _model_cache['transcription'] = model
        
        logger.info("WhisperX model pre-loaded successfully")
        
    except Exception as e:
        logger.error(f"Failed to pre-load WhisperX model: {e}")

def load_models():
    """Load all required models for processing"""
    preload_whisperx_model()

# Verify RTX 5090 CUDA
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
    
    # Test CUDA functionality with better error handling
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
try:
    verify_rtx5090_cuda()
except Exception as e:
    logger.error(f"RTX 5090 CUDA verification failed: {e}")
    # Continue anyway, might work in CPU mode
    DEFAULT_DEVICE = "cpu"
    DEFAULT_COMPUTE_TYPE = "int8"
    logger.warning("Falling back to CPU mode due to CUDA issues")

# Log configuration
logger.info(f"WhisperX configured with: model={DEFAULT_MODEL}, device={DEFAULT_DEVICE}")

# Detailed CUDA information
device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
cuda_version = torch.version.cuda if torch.cuda.is_available() else "N/A"
torch_version = torch.__version__
device_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
current_device = torch.cuda.current_device() if torch.cuda.is_available() else "CPU"
capability = torch.cuda.get_device_capability(0) if torch.cuda.is_available() else (0, 0)

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
    """Load WhisperX models into memory on RTX 5090 with error handling"""
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
        
        logger.info(f"Loading WhisperX model ({DEFAULT_MODEL}) on {DEFAULT_DEVICE}...")
        
        model_loading_state["progress"] = 50
        model_loading_state["message"] = f"Initializing WhisperX {DEFAULT_MODEL} model on {DEFAULT_DEVICE}..."
        
        # Load model with better error handling
        try:
            asr_model = whisperx.load_model(
                DEFAULT_MODEL, 
                DEFAULT_DEVICE, 
                compute_type=DEFAULT_COMPUTE_TYPE,
                local_files_only=False
            )
        except Exception as model_error:
            logger.error(f"Failed to load model on {DEFAULT_DEVICE}: {model_error}")
            if DEFAULT_DEVICE == "cuda":
                logger.info("Falling back to CPU mode...")
                asr_model = whisperx.load_model(
                    DEFAULT_MODEL, 
                    "cpu", 
                    compute_type="int8",
                    local_files_only=False
                )
            else:
                raise
        
        model_loading_state["progress"] = 100
        model_loading_state["message"] = f"Model loaded successfully on {DEFAULT_DEVICE}"
        
        total_time = time.time() - model_loading_state["start_time"]
        logger.info(f"WhisperX model loaded successfully in {total_time:.2f}s")
        
    except Exception as e:
        logger.error(f"Failed to load WhisperX model: {str(e)}")
        model_loading_state["progress"] = 0
        model_loading_state["message"] = f"Model loading error: {str(e)}"
        raise RuntimeError(f"Model loading failed: {str(e)}")
    finally:
        model_loading_state["is_loading"] = False
        model_loading_lock.release()

def transcribe_audio(audio_path: str, upload_id: str = None, diarize: bool = True) -> Dict[str, Any]:
    """Transcribe audio using WhisperX with improved error handling"""
    try:
        start_time = time.time()
        logger.info(f"[{upload_id}] Starting WhisperX transcription on {DEFAULT_DEVICE}")
        
        if upload_id:
            update_processing_status(
                upload_id=upload_id,
                status="processing",
                progress=10,
                message=f"Loading WhisperX model on {DEFAULT_DEVICE}..."
            )
        
        # Load model with fallback handling
        logger.info(f"[{upload_id}] Loading WhisperX model ({DEFAULT_MODEL}) on {DEFAULT_DEVICE}...")
        
        try:
            model = whisperx.load_model(
                DEFAULT_MODEL, 
                DEFAULT_DEVICE, 
                compute_type=DEFAULT_COMPUTE_TYPE,
                local_files_only=False
            )
            current_device = DEFAULT_DEVICE
        except Exception as e:
            logger.warning(f"[{upload_id}] Failed to load on {DEFAULT_DEVICE}: {e}")
            logger.info(f"[{upload_id}] Falling back to CPU mode...")
            model = whisperx.load_model(
                DEFAULT_MODEL, 
                "cpu", 
                compute_type="int8",
                local_files_only=False
            )
            current_device = "cpu"
        
        logger.info(f"[{upload_id}] Successfully loaded WhisperX model on {current_device}")
        
        if upload_id:
            update_processing_status(
                upload_id=upload_id,
                status="processing",
                progress=30,
                message=f"Transcribing audio on {current_device}..."
            )
        
        # Load and transcribe audio
        audio = whisperx.load_audio(audio_path)
        batch_size = 16 if current_device == "cuda" else 4
        result = model.transcribe(audio, batch_size=batch_size)
        
        language_code = result["language"]
        logger.info(f"[{upload_id}] Detected language: {language_code}")
        
        # Free up memory
        del model
        gc.collect()
        if current_device == "cuda":
            torch.cuda.empty_cache()
        
        if upload_id:
            update_processing_status(
                upload_id=upload_id,
                status="processing",
                progress=50,
                message=f"Aligning transcription on {current_device}..."
            )
        
        # Load alignment model
        logger.info(f"[{upload_id}] Loading alignment model on {current_device}...")
        model_a, metadata = whisperx.load_align_model(language_code=language_code, device=current_device)
                
        result = whisperx.align(
            result["segments"],
            model_a,
            metadata,
            audio,
            current_device,
            return_char_alignments=False
        )
        logger.info(f"[{upload_id}] Alignment complete on {current_device}")
        
        # Free up memory
        del model_a
        gc.collect()
        if current_device == "cuda":
            torch.cuda.empty_cache()
        
        # Speaker diarization
        if diarize and HF_TOKEN:
            if upload_id:
                update_processing_status(
                    upload_id=upload_id,
                    status="processing",
                    progress=70,
                    message=f"Identifying speakers on {current_device}..."
                )
            
            logger.info(f"[{upload_id}] Running speaker diarization on {current_device}...")
            try:
                # Updated diarization implementation for WhisperX 3.3.4
                diarize_model = whisperx.DiarizationPipeline(
                    use_auth_token=HF_TOKEN,
                    device=current_device
                )
                
                # Run diarization on the audio
                diarize_segments = diarize_model(audio)
                
                # Assign speakers to words/segments
                result = whisperx.assign_word_speakers(diarize_segments, result)
                logger.info(f"[{upload_id}] Speaker diarization complete on {current_device}")
                
                # Count unique speakers
                speaker_set = set()
                for segment in result["segments"]:
                    if "speaker" in segment:
                        speaker_set.add(segment["speaker"])
                
                logger.info(f"[{upload_id}] Identified {len(speaker_set)} unique speakers: {list(speaker_set)}")
                
                # Free up memory
                del diarize_model
                gc.collect()
                if current_device == "cuda":
                    torch.cuda.empty_cache()
                    
            except AttributeError as attr_error:
                if "DiarizationPipeline" in str(attr_error):
                    logger.warning(f"[{upload_id}] Speaker diarization failed: WhisperX version doesn't support DiarizationPipeline")
                    logger.info(f"[{upload_id}] Trying alternative diarization method...")
                    
                    try:
                        # Alternative method for newer WhisperX versions
                        import pyannote.audio
                        from pyannote.audio import Pipeline
                        
                        # Load diarization pipeline directly from pyannote
                        diarize_model = Pipeline.from_pretrained(
                            "pyannote/speaker-diarization-3.1",
                            use_auth_token=HF_TOKEN
                        )
                        diarize_model.to(torch.device(current_device))
                        
                        # Convert audio path to proper format for pyannote
                        import tempfile
                        import shutil
                        
                        # Create temporary wav file if needed
                        temp_wav = None
                        if not audio_path.endswith('.wav'):
                            temp_wav = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
                            temp_wav.close()
                            
                            # Convert to wav using ffmpeg
                            import subprocess
                            subprocess.run([
                                'ffmpeg', '-i', audio_path, '-ar', '16000', 
                                '-ac', '1', '-y', temp_wav.name
                            ], check=True, capture_output=True)
                            diarize_audio_path = temp_wav.name
                        else:
                            diarize_audio_path = audio_path
                        
                        # Run diarization
                        diarization_result = diarize_model(diarize_audio_path)
                        
                        # Convert pyannote output to WhisperX format
                        diarize_segments = []
                        for turn, _, speaker in diarization_result.itertracks(yield_label=True):
                            diarize_segments.append({
                                "start": turn.start,
                                "end": turn.end,
                                "speaker": speaker
                            })
                        
                        # Assign speakers to WhisperX segments
                        for segment in result["segments"]:
                            segment_start = segment.get("start", 0)
                            segment_end = segment.get("end", 0)
                            
                            # Find the speaker for this segment
                            best_speaker = None
                            max_overlap = 0
                            
                            for diarize_seg in diarize_segments:
                                # Calculate overlap between segment and diarization
                                overlap_start = max(segment_start, diarize_seg["start"])
                                overlap_end = min(segment_end, diarize_seg["end"])
                                overlap = max(0, overlap_end - overlap_start)
                                
                                if overlap > max_overlap:
                                    max_overlap = overlap
                                    best_speaker = diarize_seg["speaker"]
                            
                            if best_speaker:
                                segment["speaker"] = best_speaker
                        
                        # Count unique speakers
                        speaker_set = set()
                        for segment in result["segments"]:
                            if "speaker" in segment:
                                speaker_set.add(segment["speaker"])
                        
                        logger.info(f"[{upload_id}] Identified {len(speaker_set)} unique speakers using pyannote: {list(speaker_set)}")
                        
                        # Clean up temporary file
                        if temp_wav:
                            os.unlink(temp_wav.name)
                        
                        # Free up memory
                        del diarize_model
                        gc.collect()
                        if current_device == "cuda":
                            torch.cuda.empty_cache()
                            
                    except Exception as pyannote_error:
                        logger.warning(f"[{upload_id}] Alternative diarization also failed: {pyannote_error}")
                        logger.info(f"[{upload_id}] Continuing without speaker diarization")
                else:
                    raise attr_error
                    
            except Exception as diarize_error:
                logger.warning(f"[{upload_id}] Speaker diarization failed: {diarize_error}")
                logger.info(f"[{upload_id}] Continuing without speaker diarization")
        
        else:
            if not diarize:
                logger.info(f"[{upload_id}] Speaker diarization skipped by request")
            elif not HF_TOKEN:
                logger.warning(f"[{upload_id}] Speaker diarization skipped: No Hugging Face token provided")
        
        # Log completion time
        end_time = time.time()
        duration = end_time - start_time
        segment_count = len(result["segments"]) if "segments" in result else 0
        logger.info(f"[{upload_id}] Transcription completed on {current_device}, {segment_count} segments generated")
        logger.info(f"[{upload_id}] Processing took {duration:.2f} seconds")
        
        if upload_id:
            update_processing_status(
                upload_id=upload_id,
                status="completed",
                progress=100,
                message=f"Transcription complete on {current_device}"
            )
        
        # Note: Multi-language subtitle generation will be handled by the processing pipeline
        # after this function returns, since this is a sync function
        
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