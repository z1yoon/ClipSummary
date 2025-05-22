from fastapi import APIRouter, UploadFile, File, Form, HTTPException, BackgroundTasks, status, Request, Depends
from fastapi.responses import JSONResponse
import os
import uuid
import shutil
import json
import time
import subprocess
from typing import List, Optional
import sqlite3
import threading
import logging

# Change relative imports to absolute imports
from ai.whisperx import transcribe_audio, load_models, asr_model, is_model_loading, wait_for_model
from ai.summarizer import generate_summary
from ai.translator import translate_text
from utils.cache import cache_result, get_cached_result, get_redis_client
from api.auth import get_current_user

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

@router.post("/video")
async def upload_video(
    request: Request,
    file: UploadFile = File(...),
    languages: str = Form("en"),
    summary_length: int = Form(3),
    background_tasks: BackgroundTasks = None,
    current_user: dict = Depends(get_current_user)
):
    """Upload a video file for transcription, summarization, and translation."""
    try:
        print(f"Received upload request for file: {file.filename} from user: {current_user['username']}")
        
        # Validate file type
        if not file.filename.lower().endswith(('.mp4', '.mkv', '.webm', '.avi', '.mov')):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Unsupported file format. Please upload MP4, MKV, WEBM, AVI, or MOV."
            )
        
        # Generate a unique ID for this upload
        upload_id = str(uuid.uuid4())
        
        # Create directory for this upload
        upload_dir = f"uploads/{upload_id}"
        os.makedirs(upload_dir, exist_ok=True)
        
        try:
            # Save the uploaded file in chunks to avoid memory issues with large files
            file_path = f"{upload_dir}/{file.filename}"
            
            # Write file in chunks to handle large files
            with open(file_path, "wb") as buffer:
                # Use a larger chunk size for better performance with large files
                chunk_size = 10 * 1024 * 1024  # 10MB chunks for better performance
                while True:
                    try:
                        chunk = await file.read(chunk_size)
                        if not chunk:
                            break
                        buffer.write(chunk)
                        # Ensure chunks are flushed to disk regularly
                        buffer.flush()
                        os.fsync(buffer.fileno())
                    except Exception as chunk_error:
                        print(f"Error processing chunk: {str(chunk_error)}")
                        # Clean up partial file
                        buffer.close()
                        if os.path.exists(file_path):
                            os.remove(file_path)
                        raise HTTPException(
                            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail=f"Error during file upload: {str(chunk_error)}"
                        )
            
            print(f"File saved to {file_path}, file size: {os.path.getsize(file_path)} bytes")
        except Exception as e:
            print(f"Error saving file: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error saving file: {str(e)}"
            )
        
        # Extract metadata and thumbnail immediately
        video_info = {
            "filename": file.filename,
            "upload_time": time.time(),
            "languages_requested": languages.split(","),
            "summary_length": summary_length,
            "user_id": current_user["id"],
            "user_name": current_user["username"],
            "processing_state": "metadata_only",  # Flag to indicate only metadata is available
        }
        
        # Extract video thumbnail and basic metadata synchronously
        try:
            from utils.helpers import get_video_metadata, extract_thumbnail
            
            # Get basic metadata (quick)
            metadata = get_video_metadata(file_path)
            
            # Extract thumbnail
            thumbnail_path = f"{upload_dir}/thumbnail.jpg"
            extract_thumbnail(file_path, thumbnail_path)
            
            # Add thumbnail and metadata to video_info
            video_info.update({
                "thumbnail": f"/uploads/{upload_id}/thumbnail.jpg",
                "duration": metadata.get("duration", 0),
                "title": file.filename,
                "metadata": metadata
            })
        except Exception as e:
            print(f"Error extracting initial metadata: {str(e)}")
            # Continue even if metadata extraction fails
        
        # Save video record to database
        try:
            conn = sqlite3.connect("clipsummary.db")
            cursor = conn.cursor()
            video_id = str(uuid.uuid4())
            cursor.execute(
                """INSERT INTO videos (id, user_id, title, filename, upload_id, status, is_youtube) 
                VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (video_id, current_user["id"], file.filename, file.filename, upload_id, "metadata_ready", False)
            )
            conn.commit()
            conn.close()
            video_info["video_id"] = video_id
        except Exception as e:
            print(f"Error saving to database: {str(e)}")
        
        # Save video info
        with open(f"{upload_dir}/info.json", "w") as f:
            json.dump(video_info, f)
        
        # Cache metadata for immediate access
        cache_result(f"video:{upload_id}:info", video_info)
        
        # Create a symlink or copy the video file to make it accessible for streaming
        video_stream_path = f"uploads/{upload_id}/video.mp4"
        if not os.path.exists(video_stream_path) or os.path.getsize(video_stream_path) == 0:
            print(f"[{upload_id}] Creating video stream symlink/copy")
            try:
                # If symlink exists but is broken, remove it
                if os.path.islink(video_stream_path) and not os.path.exists(os.readlink(video_stream_path)):
                    print(f"[{upload_id}] Removing broken symlink")
                    os.unlink(video_stream_path)
                elif os.path.exists(video_stream_path) and os.path.getsize(video_stream_path) == 0:
                    print(f"[{upload_id}] Removing empty video file")
                    os.unlink(video_stream_path)
                
                # For large files (>1GB), avoid copying and use hard link if possible
                file_size_gb = os.path.getsize(file_path) / (1024*1024*1024)
                if file_size_gb > 1:
                    print(f"[{upload_id}] Large file detected ({file_size_gb:.2f} GB), using hard link if possible")
                    try:
                        # Try hard link first (more efficient)
                        os.link(file_path, video_stream_path)
                        print(f"[{upload_id}] Hard link created successfully")
                    except Exception as link_error:
                        print(f"[{upload_id}] Hard link failed: {str(link_error)}, trying symlink")
                        try:
                            # Try symlink as backup
                            os.symlink(file_path, video_stream_path)
                            print(f"[{upload_id}] Symlink created successfully")
                        except Exception as symlink_error:
                            print(f"[{upload_id}] Symlink failed: {str(symlink_error)}, using chunked copy")
                            # Fall back to chunked copying for large files
                            try:
                                chunk_size = 64 * 1024 * 1024  # 64MB chunks
                                copied_size = 0
                                total_size = os.path.getsize(file_path)
                                
                                with open(file_path, 'rb') as src, open(video_stream_path, 'wb') as dst:
                                    # Copy in chunks and report progress
                                    while True:
                                        chunk = src.read(chunk_size)
                                        if not chunk:
                                            break
                                        dst.write(chunk)
                                        dst.flush()
                                        os.fsync(dst.fileno())  # Ensure data is written to disk
                                        
                                        copied_size += len(chunk)
                                        progress_pct = (copied_size / total_size) * 100
                                        print(f"[{upload_id}] Copying: {progress_pct:.1f}% complete ({copied_size/(1024*1024):.1f}MB / {total_size/(1024*1024):.1f}MB)")
                                    
                                print(f"[{upload_id}] File copy completed successfully")
                            except Exception as copy_error:
                                print(f"[{upload_id}] Chunked copy failed: {str(copy_error)}")
                                raise
                else:
                    # For smaller files, use regular symlink or copy
                    try:
                        # Try symlink first
                        os.symlink(file_path, video_stream_path)
                        print(f"[{upload_id}] Symlink created successfully")
                    except Exception as symlink_error:
                        print(f"[{upload_id}] Symlink failed: {str(symlink_error)}, falling back to copy")
                        # If symlink fails, copy the file
                        shutil.copy(file_path, video_stream_path)
                        print(f"[{upload_id}] File copy completed successfully")
                
                # Verify the file exists and has content
                if not os.path.exists(video_stream_path):
                    raise Exception("Failed to create video.mp4 link or copy - file does not exist")
                
                stream_file_size = os.path.getsize(video_stream_path)
                if stream_file_size == 0:
                    raise Exception("Failed to create video.mp4 link or copy - file is empty")
                    
                print(f"[{upload_id}] Successfully created video.mp4 ({stream_file_size/(1024*1024):.1f} MB)")
                
            except Exception as e:
                detailed_error = f"Error creating video stream file: {str(e)}, Video path exists: {os.path.exists(file_path)}, Video path size: {os.path.getsize(file_path) if os.path.exists(file_path) else 'N/A'}"
                print(f"[{upload_id}] {detailed_error}")
                
                # Check disk space as a possible cause
                try:
                    import shutil
                    disk_usage = shutil.disk_usage(os.path.dirname(video_stream_path))
                    free_space_gb = disk_usage.free / (1024*1024*1024)
                    video_size_gb = os.path.getsize(file_path) / (1024*1024*1024) if os.path.exists(file_path) else 0
                    print(f"[{upload_id}] Free disk space: {free_space_gb:.2f}GB, Video size: {video_size_gb:.2f}GB")
                    
                    if free_space_gb < video_size_gb + 1:  # +1GB buffer
                        raise Exception(f"Insufficient disk space: {free_space_gb:.2f}GB free, need at least {video_size_gb + 1:.2f}GB")
                except Exception as disk_error:
                    print(f"[{upload_id}] Error checking disk space: {str(disk_error)}")
                
                raise Exception(f"Failed to create video.mp4: {str(e)}")
        
        # Set initial processing status
        update_processing_status(
            upload_id=upload_id,
            status="metadata_ready",
            progress=5,
            message="Video uploaded successfully. Metadata and video preview are available."
        )
        
        # Schedule full processing as a background task (don't make user wait)
        background_tasks.add_task(
            process_uploaded_video,
            video_path=file_path,
            upload_id=upload_id,
            filename=file.filename,
            languages=languages.split(","),
            summary_length=summary_length,
            user_id=current_user["id"]
        )
        
        print(f"Upload processed successfully. ID: {upload_id}")
        
        return JSONResponse(
            status_code=status.HTTP_202_ACCEPTED,
            content={
                "status": "metadata_ready",
                "upload_id": upload_id,
                "filename": file.filename,
                "metadata": video_info,
                "message": "Your video is ready for viewing. Transcript and summary will be processed in the background.",
                "redirectUrl": f"/video.html?id={upload_id}&metadata_only=true"
            }
        )
        
    except HTTPException as e:
        # Re-raise HTTP exceptions as they already have status codes
        print(f"HTTP Exception in upload: {str(e.detail)}")
        raise
    except Exception as e:
        print(f"Unexpected error during upload: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred during upload: {str(e)}"
        )

@router.get("/status/{upload_id}")
async def get_upload_status(upload_id: str):
    """Check the status of uploaded video processing."""
    # Try to get status from Redis first
    cache_key = f"upload:{upload_id}:status"
    status_data = get_cached_result(cache_key)
    
    if status_data:
        # Make sure we include progress in the response
        if 'progress' not in status_data:
            status_data['progress'] = 0
        # Add startTime for frontend estimation if missing
        if 'startTime' not in status_data:
            status_data['startTime'] = int(time.time() * 1000)
        return status_data
    
    # Fallback to checking files if Redis data is not available
    result_file = f"uploads/{upload_id}/result.json"
    status_file = f"uploads/{upload_id}/status.json"
    
    # Check if status file exists
    if os.path.exists(status_file):
        try:
            with open(status_file, "r") as f:
                data = json.load(f)
                # Make sure we include progress in the response
                if 'progress' not in data:
                    data['progress'] = 0
                # Add startTime for frontend estimation if missing
                if 'startTime' not in data:
                    data['startTime'] = int(time.time() * 1000)
                return data
        except Exception as e:
            print(f"Error reading status file: {str(e)}")
    
    # Check if the result file exists
    if os.path.exists(result_file):
        status_data = {
            "status": "completed",
            "upload_id": upload_id,
            "progress": 100,
            "message": "Processing completed",
            "result_url": f"/api/upload/result/{upload_id}"
        }
        # Cache this status
        cache_result(cache_key, status_data)
        return status_data
        
    # Check if there was an error
    error_file = f"uploads/{upload_id}/error.log"
    if os.path.exists(error_file):
        try:
            with open(error_file, "r") as f:
                error_message = f.read()
            
            status_data = {
                "status": "failed",
                "upload_id": upload_id,
                "progress": 0,
                "message": f"Processing failed: {error_message}",
                "error": error_message
            }
            # Cache this status
            cache_result(cache_key, status_data)
            return status_data
        except Exception as e:
            print(f"Error reading error file: {str(e)}")
    
    # Check if processing has actually started by checking info.json
    info_file = f"uploads/{upload_id}/info.json"
    if os.path.exists(info_file):
        # If info file exists but no status, assume processing is still ongoing
        status_data = {
            "status": "processing",
            "upload_id": upload_id,
            "progress": 5,  # Assume minimal progress 
            "message": "Your video is being processed...",
            "startTime": int(time.time() * 1000)
        }
        cache_result(cache_key, status_data, ttl=30)  # Short TTL so it will be refreshed
        return status_data
        
    # If no status info found, return a default response
    return {
        "status": "processing",
        "upload_id": upload_id,
        "progress": 0,
        "message": "Processing status not available.",
        "startTime": int(time.time() * 1000)
    }

@router.get("/result/{upload_id}")
async def get_upload_result(upload_id: str):
    """Get the result of uploaded video processing."""
    # Try to get from cache first
    cache_key = f"upload:{upload_id}:result"
    cached_result = get_cached_result(cache_key)
    
    if cached_result:
        return cached_result
    
    # If not in cache, try to get from file
    result_file = f"uploads/{upload_id}/result.json"
    if os.path.exists(result_file):
        with open(result_file, "r") as f:
            result = json.load(f)
        
        # Cache for future requests
        cache_result(cache_key, result)
        
        return result
    else:
        # Check status to give better error message
        status_data = await get_upload_status(upload_id)
        
        if status_data["status"] == "processing":
            detail = "Result not ready yet. The video is still being processed."
        elif status_data["status"] == "failed":
            detail = f"Processing failed: {status_data.get('error', 'Unknown error')}"
        else:
            detail = "Result not found. The video may still be processing or an error occurred."
        
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=detail
        )

async def process_uploaded_video(
    video_path: str,
    upload_id: str,
    filename: str,
    languages: List[str],
    summary_length: int,
    user_id: str
):
    """Background task to process uploaded video."""
    try:
        # Check if the source video file exists
        if not os.path.exists(video_path) or os.path.getsize(video_path) == 0:
            print(f"[{upload_id}] Source video not found at {video_path}, checking for alternatives")
            
            # Try to find any video file in the upload directory
            upload_dir = f"uploads/{upload_id}"
            video_files = [f for f in os.listdir(upload_dir) if f.endswith(('.mp4', '.mkv', '.webm', '.avi', '.mov'))]
            
            if video_files:
                # Use the first video file found
                video_path = os.path.join(upload_dir, video_files[0])
                print(f"[{upload_id}] Found alternative video: {video_path}")
            else:
                raise Exception(f"No video file found in upload directory: {upload_dir}")
        
        # Get file size for logging
        file_size = os.path.getsize(video_path)
        print(f"[{upload_id}] Starting processing of {filename} ({file_size/1024/1024:.2f} MB)")

        # Create a reference file instead of symlink or copy for large video files
        video_stream_path = f"uploads/{upload_id}/video.mp4"
        original_video_path = video_path
        
        # Create a video.path reference file that contains the path to the original
        # This avoids file copying/linking issues in Docker environments
        reference_path = f"uploads/{upload_id}/video.path"
        
        # Save reference to the original file
        with open(reference_path, "w") as ref_file:
            ref_file.write(original_video_path)
            print(f"[{upload_id}] Created reference to original video at: {original_video_path}")
        
        # For FFmpeg processing, we'll use the original path directly
        processing_video_path = original_video_path
        
        # Update status: starting audio extraction
        update_processing_status(
            upload_id=upload_id,
            status="processing",
            progress=10,
            message=f"Extracting audio from video ({file_size/1024/1024:.2f} MB)..."
        )
        
        # Extract audio from video
        import ffmpeg
        audio_path = f"uploads/{upload_id}/audio.wav"
        
        try:
            print(f"[{upload_id}] Extracting audio to {audio_path}")
            
            # Try standard extraction first
            (
                ffmpeg
                .input(processing_video_path)
                .output(audio_path, acodec='pcm_s16le', ac=1, ar='16k')
                .run(quiet=False, overwrite_output=True, capture_stderr=True)
            )
            print(f"[{upload_id}] Audio extraction completed successfully")
        except ffmpeg.Error as e:
            # Log the detailed error
            error_message = e.stderr.decode() if e.stderr else str(e)
            print(f"[{upload_id}] FFmpeg error details: {error_message}")
            
            # Try alternative extraction method with more compatible settings
            try:
                print(f"[{upload_id}] Attempting alternative ffmpeg extraction method...")
                subprocess.run([
                    'ffmpeg',
                    '-i', processing_video_path,
                    '-vn',  # Disable video
                    '-acodec', 'pcm_s16le',
                    '-ar', '16000',
                    '-ac', '1',
                    '-y',  # Overwrite output files
                    audio_path
                ], check=True, capture_output=True)
                print(f"[{upload_id}] Alternative extraction method succeeded")
            except subprocess.CalledProcessError as sub_err:
                error_output = sub_err.stderr.decode() if sub_err.stderr else str(sub_err)
                print(f"[{upload_id}] Alternative extraction also failed: {error_output}")
                raise Exception(f"Failed to extract audio: {error_output}")

        # Verify audio file exists and has content
        if not os.path.exists(audio_path) or os.path.getsize(audio_path) == 0:
            raise Exception("Audio extraction failed: The audio file was not created or is empty")
        
        audio_size = os.path.getsize(audio_path)
        print(f"[{upload_id}] Starting transcription of {audio_size/1024/1024:.2f} MB audio")
        
        # Transcribe with WhisperX, passing upload_id for progress updates
        transcript = transcribe_audio(audio_path, upload_id)
        
        print(f"[{upload_id}] Transcription completed, {len(transcript.get('segments', []))} segments generated")
        
        # Check if transcription failed
        if not transcript or "segments" not in transcript or not transcript["segments"]:
            if "error" in transcript:
                raise Exception(f"Transcription failed: {transcript['error']}")
            else:
                raise Exception("Transcription failed: No segments were generated")
        
        # Save original transcript to file
        transcript_dir = f"uploads/{upload_id}/subtitles"
        os.makedirs(transcript_dir, exist_ok=True)
        
        with open(f"{transcript_dir}/en.json", "w") as f:
            json.dump({"segments": transcript["segments"], "language": "en"}, f)
        
        # Update status: generating summary
        update_processing_status(
            upload_id=upload_id,
            status="processing",
            progress=50,
            message=f"Generating summary from {len(transcript['segments'])} segments..."
        )
        
        print(f"[{upload_id}] Generating summary")
        
        # Generate summary (English)
        transcript_text = ' '.join([segment['text'] for segment in transcript['segments']])
        summary = generate_summary(transcript_text, max_sentences=summary_length)
        
        print(f"[{upload_id}] Summary generated, length: {len(summary)} characters")
        
        # Save summary to file
        with open(f"uploads/{upload_id}/summary.txt", "w") as f:
            f.write(summary)
        
        # Prepare results with the original language (English)
        result = {
            "upload_id": upload_id,
            "filename": filename,
            "transcript": transcript,
            "summary": {
                "en": summary
            },
            "translations": {}
        }
        
        # Update status: starting translations
        total_languages = len([lang for lang in languages if lang != "en"])
        current_language_index = 0
        
        # Translate to requested languages
        for lang in languages:
            if lang != "en":  # Skip English as it's already done
                current_language_index += 1
                progress = 50 + (current_language_index / total_languages * 40)
                
                update_processing_status(
                    upload_id=upload_id,
                    status="processing",
                    progress=progress,
                    message=f"Translating to {lang} ({current_language_index}/{total_languages})..."
                )
                
                print(f"[{upload_id}] Translating to {lang}")
                
                # Translate summary
                translated_summary = translate_text(summary, target_lang=lang)
                
                # Translate transcript segments
                translated_segments = []
                total_segments = len(transcript["segments"])
                for i, segment in enumerate(transcript["segments"], 1):
                    if i % 10 == 0:  # Update progress every 10 segments
                        update_processing_status(
                            upload_id=upload_id,
                            status="processing",
                            progress=progress,
                            message=f"Translating to {lang} ({i}/{total_segments} segments)..."
                        )
                    
                    translated_text = translate_text(segment["text"], target_lang=lang)
                    translated_segments.append({
                        "start": segment["start"],
                        "end": segment["end"],
                        "text": translated_text
                    })
                
                # Save translated transcript to file
                with open(f"{transcript_dir}/{lang}.json", "w") as f:
                    json.dump({"segments": translated_segments, "language": lang}, f)
                
                # Add to result
                result["translations"][lang] = {
                    "summary": translated_summary,
                    "transcript": translated_segments
                }
                
                print(f"[{upload_id}] Completed translation to {lang}")
        
        # Update status: finalizing
        update_processing_status(
            upload_id=upload_id,
            status="processing",
            progress=95,
            message="Extracting video metadata and thumbnail..."
        )
        
        # Extract video thumbnail and metadata
        try:
            from utils.helpers import get_video_metadata, extract_thumbnail
            
            print(f"[{upload_id}] Extracting video metadata and thumbnail")
            
            # Get video metadata
            metadata = get_video_metadata(video_path)
            
            # Extract thumbnail
            thumbnail_path = f"uploads/{upload_id}/thumbnail.jpg"
            extract_thumbnail(video_path, thumbnail_path)
            
            print(f"[{upload_id}] Metadata and thumbnail extraction complete")
            
            # Add thumbnail and metadata to info.json
            info_path = f"uploads/{upload_id}/info.json"
            if os.path.exists(info_path):
                with open(info_path, "r") as f:
                    info = json.load(f)
            else:
                info = {}
            
            info.update({
                "thumbnail": f"/uploads/{upload_id}/thumbnail.jpg",
                "duration": metadata.get("duration", 0),
                "title": filename,
                "metadata": metadata
            })
            
            with open(info_path, "w") as f:
                json.dump(info, f)
            
            # Cache this info
            cache_result(f"video:{upload_id}:info", info)
            
        except Exception as e:
            print(f"[{upload_id}] Error extracting video metadata: {str(e)}")
        
        # Save full results
        print(f"[{upload_id}] Saving final results")
        with open(f"uploads/{upload_id}/result.json", "w") as f:
            json.dump(result, f)
        
        # Cache the result
        cache_result(f"upload:{upload_id}:result", result)
        
        # Update status to completed
        update_processing_status(
            upload_id=upload_id,
            status="completed",
            progress=100,
            message="Processing completed successfully."
        )
        
        print(f"[{upload_id}] Processing completed successfully")
        
    except Exception as e:
        # Update status to failed
        error_message = str(e)
        print(f"[{upload_id}] Processing failed: {error_message}")
        
        update_processing_status(
            upload_id=upload_id,
            status="failed",
            progress=0,
            message=f"Processing failed: {error_message}"
        )
        
        # Log the error
        with open(f"uploads/{upload_id}/error.log", "w") as f:
            f.write(error_message)