import torch
import whisperx
import os
from typing import Dict, Any

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
        asr_model = whisperx.load_model(
            "large-v2", 
            device=device, 
            compute_type=compute_type,
            language="en"  # Default language, will be auto-detected
        )
    
    # No need to pre-load alignment model as it's language-specific
    # and will be loaded based on detected language
    
    return asr_model

def transcribe_audio(audio_path: str) -> Dict[str, Any]:
    """
    Transcribe audio using WhisperX with word-level timestamps
    
    Args:
        audio_path: Path to audio file
        
    Returns:
        Dict containing transcription results with timestamps
    """
    try:
        # Load models
        model = load_models()
        
        # Load audio
        audio = whisperx.load_audio(audio_path)
        
        # Transcribe with batch size appropriate for the device
        batch_size = 16 if device == "cuda" else 8
        result = model.transcribe(audio, batch_size=batch_size)
        
        # Get the detected language
        language_code = result["language"]
        
        # Load alignment model for the detected language
        alignment_model, metadata = whisperx.load_align_model(
            language_code=language_code,
            device=device
        )
        
        # Align the transcription with the audio to get word-level timestamps
        result = whisperx.align(
            result["segments"],
            alignment_model,
            metadata,
            audio,
            device,
            return_char_alignments=False  # Set to True if you need character-level alignment
        )
        
        # Return the aligned result with word-level timestamps
        return result
        
    except Exception as e:
        # Log error and return a formatted error message
        print(f"Error in transcription: {str(e)}")
        return {
            "error": str(e),
            "segments": []
        }