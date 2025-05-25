from fastapi import APIRouter, UploadFile, File, Form, HTTPException, BackgroundTasks, status, Request, Depends
from fastapi.responses import JSONResponse
from typing import List, Optional
import os
import time
import json
import uuid
import sqlite3
import threading
import traceback
import logging
import subprocess
import ffmpeg
import tempfile

# Change relative imports to absolute imports
from ai.whisperx import transcribe_audio, load_models, asr_model, is_model_loading, wait_for_model
from ai.summarizer import generate_summary
from ai.translator import translate_text
from utils.cache import cache_result, get_cached_result, get_redis_client
from utils.azure_storage import azure_storage
from api.auth import get_current_user
from pydantic import BaseModel

router = APIRouter()

# Configure logging for this module
logger = logging.getLogger(__name__)

# Global variable to track model loading state
whisperx_loading_state = {
    "is_loading": False,
    "progress": 0,
    "message": "",
    "start_time": None,
    "completed": False,
    "error": None
}

def update_loading_status(progress, message):
    """Update the global WhisperX loading status"""
    global whisperx_loading_state
    whisperx_loading_state["progress"] = progress
    whisperx_loading_state["message"] = message
    logger.info(f"WhisperX loading: {progress}% - {message}")

def background_load_whisperx():
    """Load WhisperX in a background thread with progress updates"""
    global whisperx_loading_state
    
    # Mark as loading
    whisperx_loading_state["is_loading"] = True
    whisperx_loading_state["start_time"] = time.time()
    whisperx_loading_state["completed"] = False
    whisperx_loading_state["error"] = None
    
    try:
        # Initialize loading with progress updates
        update_loading_status(5, "Initializing WhisperX model loading...")
        time.sleep(1)  # Give time for UI to update
        
        update_loading_status(10, "Preparing model resources...")
        time.sleep(1)
        
        # Actual model loading happens here
        update_loading_status(15, "Loading WhisperX ASR model (large-v2)...")
        
        # Check if model is already being loaded or loaded
        if is_model_loading():
            logger.info("WhisperX model is already being loaded by another thread, waiting for completion...")
            update_loading_status(20, "Waiting for model to complete loading in another process...")
            
            # Wait for the loading to complete
            if wait_for_model(timeout=300):  # Wait up to 5 minutes
                update_loading_status(100, "Model loading completed by another process")
                whisperx_loading_state["completed"] = True
            else:
                raise Exception("Timed out waiting for model to load in another process")
        else:
            # This will trigger the progress reporter in the load_models function
            load_models()
        
        # Model loaded successfully
        elapsed = time.time() - whisperx_loading_state["start_time"]
        update_loading_status(100, f"WhisperX model loaded successfully in {elapsed:.1f}s")
        whisperx_loading_state["completed"] = True
        
    except Exception as e:
        error_msg = f"Error loading WhisperX model: {str(e)}"
        whisperx_loading_state["error"] = error_msg
        logger.error(error_msg, exc_info=True)
    finally:
        whisperx_loading_state["is_loading"] = False

def ensure_whisperx_loaded():
    """Ensure WhisperX model is loaded, start loading if needed"""
    # Check if model is already loaded
    if asr_model is not None:
        return True
    
    global whisperx_loading_state
    
    # If model is being loaded by another thread, don't start a new loading thread
    if is_model_loading():
        logger.info("WhisperX model is already being loaded by another thread")
        return False
        
    # If not already loading, start background loading
    if not whisperx_loading_state["is_loading"] and not whisperx_loading_state["completed"]:
        logger.info("Starting WhisperX model loading in background thread")
        loading_thread = threading.Thread(target=background_load_whisperx)
        loading_thread.daemon = True
        loading_thread.start()
    
    return False  # Model not yet loaded

@router.get("/whisperx-status")
async def get_whisperx_loading_status():
    """Get the current status of WhisperX model loading"""
    global whisperx_loading_state
    
    # Check if model is already loaded via direct inspection
    if asr_model is not None and not whisperx_loading_state["completed"]:
        whisperx_loading_state["completed"] = True
        whisperx_loading_state["progress"] = 100
        whisperx_loading_state["message"] = "WhisperX model is loaded and ready"
    
    # Calculate elapsed time if loading
    elapsed = None
    if whisperx_loading_state["start_time"] and whisperx_loading_state["is_loading"]:
        elapsed = time.time() - whisperx_loading_state["start_time"]
    
    return {
        "is_loading": whisperx_loading_state["is_loading"],
        "progress": whisperx_loading_state["progress"],
        "message": whisperx_loading_state["message"],
        "completed": whisperx_loading_state["completed"],
        "error": whisperx_loading_state["error"],
        "elapsed_seconds": elapsed
    }

# Processing status tracking with Redis
def update_processing_status(upload_id: str, status: str, progress: float = 0, message: str = ""):
    """Update the processing status in Redis"""
    status_data = {
        "status": status,
        "upload_id": upload_id,
        "progress": progress,
        "message": message,
        "updated_at": time.time()
    }
    
    # Cache with a longer TTL (3 days) to keep processing history
    cache_result(f"upload:{upload_id}:status", status_data, ttl=259200)
    
    # Also save to file system as fallback
    try:
        status_dir = f"uploads/{upload_id}"
        os.makedirs(status_dir, exist_ok=True)
        
        with open(f"{status_dir}/status.json", "w") as f:
            json.dump(status_data, f)
    except Exception as e:
        print(f"Error saving status to file: {str(e)}")

# Data models for signed URL uploads
class GenerateUploadUrlRequest(BaseModel):
    filename: str
    file_size: int
    languages: str = "en"
    summary_length: int = 3

class ConfirmUploadRequest(BaseModel):
    upload_id: str
    filename: str

@router.post("/generate-upload-url")
async def generate_upload_url(
    request: GenerateUploadUrlRequest,
    current_user: dict = Depends(get_current_user)
):
    """Generate a secure signed URL for direct upload to Azure Blob Storage."""
    try:
        # Validate file type
        if not request.filename.lower().endswith(('.mp4', '.mkv', '.webm', '.avi', '.mov')):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Unsupported file format. Please upload MP4, MKV, WEBM, AVI, or MOV."
            )
        
        # Generate a unique ID for this upload
        upload_id = str(uuid.uuid4())
        
        # Generate signed URL for Azure Blob Storage
        signed_url_data = azure_storage.generate_upload_url(
            upload_id=upload_id,
            filename=request.filename,
            expiry_hours=2  # 2 hour expiry for upload
        )
        
        # Store upload metadata for processing after upload
        upload_metadata = {
            "upload_id": upload_id,
            "filename": request.filename,
            "file_size": request.file_size,
            "languages": request.languages.split(","),
            "summary_length": request.summary_length,
            "user_id": current_user.id,
            "user_name": current_user.username,
            "created_at": time.time(),
            "status": "url_generated",
            "blob_name": signed_url_data["blob_name"]
        }
        
        # Cache the metadata
        cache_result(f"upload:{upload_id}:metadata", upload_metadata, ttl=7200)  # 2 hours
        
        # Create local directory for processing files (will be used later)
        upload_dir = f"uploads/{upload_id}"
        os.makedirs(upload_dir, exist_ok=True)
        
        # Save metadata to file as backup
        with open(f"{upload_dir}/metadata.json", "w") as f:
            json.dump(upload_metadata, f)
        
        logger.info(f"Generated signed upload URL for user {current_user.username}: {upload_id}")
        
        # Return response compatible with frontend expectations
        return {
            "upload_id": upload_id,
            "base_url": signed_url_data["base_url"],
            "sas_token": signed_url_data["sas_token"],
            "blob_name": signed_url_data["blob_name"],
            "expires_at": signed_url_data["expires_at"],
            "account_name": signed_url_data["account_name"],
            "container_name": signed_url_data["container_name"],
            "upload_type": signed_url_data["upload_type"],
            "message": "Upload URL generated. Upload your file directly to the provided URL."
        }
        
    except Exception as e:
        logger.error(f"Error generating upload URL: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate upload URL: {str(e)}"
        )

@router.post("/confirm-upload")
async def confirm_upload(
    request: ConfirmUploadRequest,
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(get_current_user)
):
    """Confirm that file upload to Azure Blob Storage is complete and start processing."""
    try:
        upload_id = request.upload_id
        filename = request.filename
        
        # Get upload metadata
        metadata = get_cached_result(f"upload:{upload_id}:metadata")
        if not metadata:
            # Try to load from file
            metadata_file = f"uploads/{upload_id}/metadata.json"
            if os.path.exists(metadata_file):
                with open(metadata_file, "r") as f:
                    metadata = json.load(f)
            else:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Upload session not found or expired"
                )
        
        # Verify user owns this upload
        if metadata["user_id"] != current_user.id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied"
            )
        
        # Verify the file was uploaded to Azure Blob Storage
        if not azure_storage.verify_blob_exists(upload_id, filename):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="File not found in cloud storage. Please ensure upload completed successfully."
            )
        
        # Get actual file size from Azure
        actual_size = azure_storage.get_blob_size(upload_id, filename)
        if not actual_size or actual_size == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Uploaded file is empty or corrupted"
            )
        
        logger.info(f"Confirmed upload {upload_id}: {filename} ({actual_size} bytes)")
        
        # Create video info
        video_info = {
            "filename": filename,
            "upload_time": time.time(),
            "languages_requested": metadata["languages"],
            "summary_length": metadata["summary_length"],
            "user_id": current_user.id,
            "user_name": current_user.username,
            "file_size": actual_size,
            "storage_location": "azure_blob",
            "processing_state": "upload_confirmed"
        }
        
        # Save video record to database
        try:
            conn = sqlite3.connect("clipsummary.db")
            cursor = conn.cursor()
            video_id = str(uuid.uuid4())
            cursor.execute(
                """INSERT INTO videos (id, user_id, title, filename, upload_id, status, is_youtube) 
                VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (video_id, current_user.id, filename, filename, upload_id, "processing", False)
            )
            conn.commit()
            conn.close()
            video_info["video_id"] = video_id
        except Exception as e:
            logger.error(f"Error saving to database: {str(e)}")
        
        # Save video info
        upload_dir = f"uploads/{upload_id}"
        with open(f"{upload_dir}/info.json", "w") as f:
            json.dump(video_info, f)
        
        # Cache video info
        cache_result(f"video:{upload_id}:info", video_info)
        
        # Set initial processing status
        update_processing_status(
            upload_id=upload_id,
            status="processing",
            progress=5,
            message="Upload confirmed. Starting video processing..."
        )
        
        # Schedule background processing with Azure Blob Storage
        background_tasks.add_task(
            process_azure_video,
            upload_id=upload_id,
            filename=filename,
            languages=metadata["languages"],
            summary_length=metadata["summary_length"],
            user_id=current_user.id
        )
        
        logger.info(f"Started processing for upload {upload_id}")
        
        return {
            "success": True,
            "upload_id": upload_id,
            "filename": filename,
            "file_size": actual_size,
            "status": "processing",
            "message": "Upload confirmed. Video processing started.",
            "video_info": video_info
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error confirming upload: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to confirm upload: {str(e)}"
        )

async def process_azure_video(
    upload_id: str,
    filename: str,
    languages: List[str],
    summary_length: int,
    user_id: str
):
    """Background task to process video stored in Azure Blob Storage."""
    import traceback
    
    try:
        logger.info(f"[{upload_id}] Starting Azure video processing for {filename}")
        
        # Verify blob exists with retry logic
        blob_verified = False
        for attempt in range(3):
            try:
                if azure_storage.verify_blob_exists(upload_id, filename):
                    blob_verified = True
                    break
                else:
                    if attempt < 2:
                        logger.warning(f"[{upload_id}] Blob not found, retrying in {2**attempt} seconds...")
                        await asyncio.sleep(2**attempt)
                    else:
                        raise Exception(f"Video file not found in Azure Blob Storage after 3 attempts")
            except Exception as e:
                if attempt < 2:
                    logger.warning(f"[{upload_id}] Blob verification failed (attempt {attempt+1}): {str(e)}")
                    await asyncio.sleep(2**attempt)
                else:
                    raise Exception(f"Failed to verify blob existence: {str(e)}")
        
        if not blob_verified:
            raise Exception(f"Video file not found in Azure Blob Storage")

        file_size = azure_storage.get_blob_size(upload_id, filename)
        logger.info(f"[{upload_id}] Processing video from Azure ({file_size/1024/1024:.2f} MB)")

        # Update status: downloading for processing
        update_processing_status(
            upload_id=upload_id,
            status="processing",
            progress=10,
            message=f"Downloading video from cloud storage ({file_size/1024/1024:.2f} MB)..."
        )

        # Create temporary local file for processing
        upload_dir = f"uploads/{upload_id}"
        os.makedirs(upload_dir, exist_ok=True)
        local_video_path = f"{upload_dir}/{filename}"

        # Download video from Azure Blob Storage with retry logic
        download_success = False
        for attempt in range(3):
            try:
                await azure_storage.download_video(upload_id, filename, local_video_path)
                
                # Verify download completed successfully
                if os.path.exists(local_video_path) and os.path.getsize(local_video_path) > 0:
                    actual_size = os.path.getsize(local_video_path)
                    if actual_size >= file_size * 0.95:  # Allow 5% tolerance
                        download_success = True
                        logger.info(f"[{upload_id}] Downloaded video for processing ({actual_size} bytes)")
                        break
                    else:
                        logger.warning(f"[{upload_id}] Download size mismatch: {actual_size} vs expected {file_size}")
                        if os.path.exists(local_video_path):
                            os.remove(local_video_path)
                
                if attempt < 2:
                    logger.warning(f"[{upload_id}] Download incomplete, retrying in {3*(attempt+1)} seconds...")
                    await asyncio.sleep(3*(attempt+1))
                    
            except Exception as e:
                if attempt < 2:
                    logger.warning(f"[{upload_id}] Download attempt {attempt+1} failed: {str(e)}")
                    await asyncio.sleep(3*(attempt+1))
                else:
                    raise Exception(f"Failed to download video from Azure after 3 attempts: {str(e)}")
        
        if not download_success:
            raise Exception(f"Failed to download video from Azure Blob Storage")

        # Update status: extracting audio
        update_processing_status(
            upload_id=upload_id,
            status="processing",
            progress=20,
            message="Extracting audio from video..."
        )

        # Extract audio from video with enhanced error handling
        audio_path = f"{upload_dir}/audio.wav"
        
        for attempt in range(2):
            try:
                logger.info(f"[{upload_id}] Extracting audio (attempt {attempt+1})")
                (
                    ffmpeg
                    .input(local_video_path)
                    .output(audio_path, acodec='pcm_s16le', ac=1, ar='16k')
                    .run(quiet=True, overwrite_output=True)
                )
                
                # Verify audio extraction
                if os.path.exists(audio_path) and os.path.getsize(audio_path) > 0:
                    break
                elif attempt == 0:
                    logger.warning(f"[{upload_id}] Audio extraction failed, retrying...")
                    await asyncio.sleep(2)
                else:
                    raise Exception("Audio extraction produced empty file")
                    
            except ffmpeg.Error as e:
                error_message = e.stderr.decode() if e.stderr else str(e)
                if attempt == 0:
                    logger.warning(f"[{upload_id}] FFmpeg attempt 1 failed: {error_message}")
                    await asyncio.sleep(2)
                else:
                    logger.error(f"[{upload_id}] FFmpeg error after retries: {error_message}")
                    raise Exception(f"Failed to extract audio after retries: {error_message}")

        # Verify final audio extraction
        if not os.path.exists(audio_path) or os.path.getsize(audio_path) == 0:
            raise Exception("Audio extraction failed - no audio file produced")

        # Update status: transcribing
        update_processing_status(
            upload_id=upload_id,
            status="processing",
            progress=30,
            message="Transcribing audio with WhisperX..."
        )

        # Transcribe with WhisperX with retry logic
        transcript = None
        for attempt in range(2):
            try:
                transcript = transcribe_audio(audio_path, upload_id)
                
                if transcript and "segments" in transcript and transcript["segments"]:
                    logger.info(f"[{upload_id}] Transcription completed: {len(transcript['segments'])} segments")
                    break
                elif attempt == 0:
                    logger.warning(f"[{upload_id}] Transcription attempt 1 failed, retrying...")
                    await asyncio.sleep(5)
                else:
                    if "error" in transcript:
                        raise Exception(f"Transcription failed: {transcript['error']}")
                    else:
                        raise Exception("Transcription failed: No segments generated")
                        
            except Exception as e:
                if attempt == 0:
                    logger.warning(f"[{upload_id}] Transcription attempt 1 failed: {str(e)}")
                    await asyncio.sleep(5)
                else:
                    raise Exception(f"Transcription failed after retries: {str(e)}")

        if not transcript or "segments" not in transcript or not transcript["segments"]:
            raise Exception("Transcription failed: No valid transcript produced")

        # Save transcript
        transcript_dir = f"{upload_dir}/subtitles"
        os.makedirs(transcript_dir, exist_ok=True)

        with open(f"{transcript_dir}/en.json", "w") as f:
            json.dump({"segments": transcript["segments"], "language": "en"}, f)

        # Update status: generating summary
        update_processing_status(
            upload_id=upload_id,
            status="processing",
            progress=50,
            message=f"Generating summary from transcript..."
        )

        # Generate summary with retry logic
        summary = None
        for attempt in range(2):
            try:
                transcript_text = ' '.join([segment['text'] for segment in transcript['segments']])
                if not transcript_text.strip():
                    raise Exception("Empty transcript text")
                    
                summary = generate_summary(transcript_text, max_sentences=summary_length)
                
                if summary and summary.strip():
                    logger.info(f"[{upload_id}] Summary generated ({len(summary)} characters)")
                    break
                elif attempt == 0:
                    logger.warning(f"[{upload_id}] Summary generation attempt 1 failed, retrying...")
                    await asyncio.sleep(3)
                else:
                    raise Exception("Summary generation produced empty result")
                    
            except Exception as e:
                if attempt == 0:
                    logger.warning(f"[{upload_id}] Summary generation attempt 1 failed: {str(e)}")
                    await asyncio.sleep(3)
                else:
                    raise Exception(f"Summary generation failed after retries: {str(e)}")

        if not summary or not summary.strip():
            raise Exception("Summary generation failed: No valid summary produced")

        # Prepare results
        result = {
            "upload_id": upload_id,
            "filename": filename,
            "transcript": transcript,
            "summary": {"en": summary},
            "translations": {}
        }

        # Handle translations with improved error handling
        total_languages = len([lang for lang in languages if lang != "en"])
        if total_languages > 0:
            for i, lang in enumerate([l for l in languages if l != "en"], 1):
                try:
                    progress = 50 + (i / total_languages * 40)
                    
                    update_processing_status(
                        upload_id=upload_id,
                        status="processing", 
                        progress=progress,
                        message=f"Translating to {lang} ({i}/{total_languages})..."
                    )
                    
                    # Translate summary with retry
                    translated_summary = None
                    for attempt in range(2):
                        try:
                            translated_summary = translate_text(summary, target_lang=lang)
                            if translated_summary and translated_summary.strip():
                                break
                            elif attempt == 0:
                                await asyncio.sleep(2)
                        except Exception as e:
                            if attempt == 0:
                                logger.warning(f"[{upload_id}] Summary translation to {lang} failed, retrying: {str(e)}")
                                await asyncio.sleep(2)
                            else:
                                raise e
                    
                    if not translated_summary:
                        raise Exception(f"Failed to translate summary to {lang}")
                    
                    # Translate transcript with error handling
                    translated_segments = []
                    segment_errors = 0
                    max_segment_errors = min(5, len(transcript["segments"]) // 4)  # Allow up to 25% failures
                    
                    for segment in transcript["segments"]:
                        try:
                            translated_text = translate_text(segment["text"], target_lang=lang)
                            if translated_text and translated_text.strip():
                                translated_segments.append({
                                    "start": segment["start"],
                                    "end": segment["end"],
                                    "text": translated_text
                                })
                            else:
                                # Keep original text if translation fails
                                segment_errors += 1
                                translated_segments.append({
                                    "start": segment["start"],
                                    "end": segment["end"],
                                    "text": segment["text"]  # Fallback to original
                                })
                                
                        except Exception as e:
                            segment_errors += 1
                            logger.warning(f"[{upload_id}] Failed to translate segment: {str(e)}")
                            
                            # Keep original text as fallback
                            translated_segments.append({
                                "start": segment["start"],
                                "end": segment["end"],
                                "text": segment["text"]
                            })
                            
                            if segment_errors > max_segment_errors:
                                logger.warning(f"[{upload_id}] Too many translation errors for {lang}, stopping")
                                break
                    
                    # Save translated transcript
                    with open(f"{transcript_dir}/{lang}.json", "w") as f:
                        json.dump({"segments": translated_segments, "language": lang}, f)
                    
                    result["translations"][lang] = {
                        "summary": translated_summary,
                        "transcript": translated_segments
                    }
                    
                    logger.info(f"[{upload_id}] Completed translation to {lang} ({segment_errors} segment errors)")
                    
                except Exception as e:
                    logger.warning(f"[{upload_id}] Translation to {lang} failed: {str(e)}")
                    # Continue with other languages instead of failing completely

        # Extract metadata and thumbnail
        update_processing_status(
            upload_id=upload_id,
            status="processing",
            progress=95,
            message="Finalizing video metadata..."
        )

        try:
            from utils.helpers import get_video_metadata, extract_thumbnail

            metadata = get_video_metadata(local_video_path)

            thumbnail_path = f"{upload_dir}/thumbnail.jpg"
            extract_thumbnail(local_video_path, thumbnail_path)

            # Update info.json
            info_path = f"{upload_dir}/info.json"
            if os.path.exists(info_path):
                with open(info_path, "r") as f:
                    info = json.load(f)
            else:
                info = {}

            info.update({
                "thumbnail": f"/uploads/{upload_id}/thumbnail.jpg",
                "duration": metadata.get("duration", 0),
                "title": filename,
                "metadata": metadata,
                "azure_blob_url": azure_storage.get_blob_url(upload_id, filename)
            })

            with open(info_path, "w") as f:
                json.dump(info, f)

            cache_result(f"video:{upload_id}:info", info)

        except Exception as e:
            logger.warning(f"[{upload_id}] Error extracting metadata: {str(e)}")

        # Save results
        with open(f"{upload_dir}/result.json", "w") as f:
            json.dump(result, f)

        cache_result(f"upload:{upload_id}:result", result)

        # Clean up local video file to save space (keep audio for potential reprocessing)
        try:
            if os.path.exists(local_video_path):
                os.remove(local_video_path)
                logger.info(f"[{upload_id}] Cleaned up local video file")
        except Exception as e:
            logger.warning(f"[{upload_id}] Failed to clean up local video: {str(e)}")

        # Update final status
        update_processing_status(
            upload_id=upload_id,
            status="completed",
            progress=100,
            message="Processing completed successfully."
        )

        logger.info(f"[{upload_id}] Processing completed successfully")

    except Exception as e:
        error_message = str(e)
        error_traceback = traceback.format_exc()

        logger.error(f"[{upload_id}] Processing failed: {error_message}")
        logger.error(f"[{upload_id}] Full traceback:\n{error_traceback}")

        # Update status to failed
        try:
            update_processing_status(
                upload_id=upload_id,
                status="failed",
                progress=0,
                message=error_message
            )
        except Exception as status_error:
            logger.error(f"[{upload_id}] Failed to update status: {str(status_error)}")

        # Log error
        try:
            upload_dir = f"uploads/{upload_id}"
            os.makedirs(upload_dir, exist_ok=True)
            with open(f"{upload_dir}/error.log", "w") as f:
                f.write(f"{error_message}\n\nFull traceback:\n{error_traceback}")
        except Exception as log_error:
            logger.error(f"[{upload_id}] Failed to write error log: {str(log_error)}")

# Keep legacy endpoints for backward compatibility but mark as deprecated
@router.post("/video")
async def upload_video_legacy(
    request: Request,
    file: UploadFile = File(...),
    languages: str = Form("en"),
    summary_length: int = Form(3),
    background_tasks: BackgroundTasks = None,
    current_user: dict = Depends(get_current_user)
):
    """Legacy video upload endpoint - DEPRECATED. Use signed URL approach instead."""
    raise HTTPException(
        status_code=status.HTTP_410_GONE,
        detail="This upload method is deprecated for security reasons. Please use the signed URL upload method via /generate-upload-url endpoint."
    )

# Remove chunked upload endpoints as they're no longer needed
# The signed URL approach handles large files much more efficiently

@router.get("/ping")
async def ping():
    """Simple ping endpoint for network diagnostics"""
    return {"status": "ok", "timestamp": time.time()}

@router.get("/status/{upload_id}")
async def get_upload_status(
    upload_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Get the current processing status of an upload"""
    try:
        # First check Redis cache
        status_data = get_cached_result(f"upload:{upload_id}:status")
        
        if not status_data:
            # Fallback to file system
            status_file = f"uploads/{upload_id}/status.json"
            if os.path.exists(status_file):
                with open(status_file, "r") as f:
                    status_data = json.load(f)
            else:
                # Check if it's completed but status not found
                result_file = f"uploads/{upload_id}/result.json"
                if os.path.exists(result_file):
                    return {
                        "status": "completed",
                        "upload_id": upload_id,
                        "progress": 100,
                        "message": "Processing completed",
                        "updated_at": time.time()
                    }
                
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Upload not found"
                )
        
        # Verify user has access to this upload
        video_info = get_cached_result(f"video:{upload_id}:info")
        if video_info and video_info.get("user_id") != current_user.id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied"
            )
        
        return status_data
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting upload status: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get upload status"
        )