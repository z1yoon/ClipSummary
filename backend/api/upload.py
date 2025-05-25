from fastapi import APIRouter, UploadFile, File, Form, HTTPException, BackgroundTasks, status, Request, Depends
from fastapi.responses import JSONResponse
from typing import List, Optional
import asyncio
import aiofiles
import os
import time
import json
import uuid
import shutil
import sqlite3
import threading
import traceback
import logging
import subprocess
import ffmpeg
import math
from concurrent.futures import ThreadPoolExecutor

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
def update_processing_status(upload_id: str, status: str, progress: float = 0, message: str = "", error: str = None):
    """Update the processing status in Redis"""
    status_data = {
        "status": status,
        "upload_id": upload_id,
        "progress": progress,
        "message": message,
        "updated_at": time.time()
    }
    
    if error:
        status_data["error"] = error
    
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
        print(f"Received upload request for file: {file.filename} from user: {current_user.username}")
        
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
            
            # Optimize chunk size based on file size for better performance
            file_size = 0
            if hasattr(file, 'size') and file.size:
                file_size = file.size
            
            # Use much larger chunks for very large files to reduce overhead
            if file_size > 5 * 1024 * 1024 * 1024:  # Files > 5GB
                chunk_size = 100 * 1024 * 1024  # 100MB chunks for very large files
            elif file_size > 2 * 1024 * 1024 * 1024:  # Files > 2GB
                chunk_size = 75 * 1024 * 1024   # 75MB chunks for large files
            elif file_size > 1024 * 1024 * 1024:  # Files > 1GB
                chunk_size = 50 * 1024 * 1024   # 50MB chunks for large files
            elif file_size > 100 * 1024 * 1024:  # Files > 100MB
                chunk_size = 20 * 1024 * 1024   # 20MB chunks for medium files
            else:
                chunk_size = 10 * 1024 * 1024   # 10MB chunks for smaller files
            
            print(f"Using chunk size: {chunk_size / (1024*1024):.1f}MB for file size: {file_size / (1024*1024):.1f}MB")
            
            # Write file using optimized streaming approach with buffering
            import io
            buffer_size = chunk_size * 2  # Double buffering for better performance
            
            with open(file_path, "wb", buffering=buffer_size) as buffer:
                total_written = 0
                chunks_written = 0
                
                while True:
                    try:
                        chunk = await file.read(chunk_size)
                        if not chunk:
                            break
                        buffer.write(chunk)
                        total_written += len(chunk)
                        chunks_written += 1
                        
                        # Flush less frequently for large files to reduce I/O overhead
                        flush_interval = 10 if file_size > 2 * 1024 * 1024 * 1024 else 5
                        if chunks_written % flush_interval == 0:
                            buffer.flush()
                            
                        # Only sync to disk every 500MB for very large files
                        if total_written % (500 * 1024 * 1024) == 0:
                            import os
                            os.fsync(buffer.fileno())
                            
                        # Log progress less frequently for large files
                        log_interval = 200 * 1024 * 1024 if file_size > 2 * 1024 * 1024 * 1024 else 50 * 1024 * 1024
                        if file_size > 0 and total_written % log_interval == 0:
                            progress = (total_written / file_size) * 100
                            print(f"Upload progress: {progress:.1f}% ({total_written/(1024*1024):.1f}MB)")
                        
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
                
                # Final flush and sync
                buffer.flush()
                os.fsync(buffer.fileno())
            
            actual_size = os.path.getsize(file_path)
            print(f"File saved to {file_path}, actual size: {actual_size} bytes ({actual_size/(1024*1024):.1f}MB)")
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
            "user_id": current_user.id,
            "user_name": current_user.username,
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
                (video_id, current_user.id, file.filename, file.filename, upload_id, "metadata_ready", False)
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
            user_id=current_user.id
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

@router.post("/status/update")
async def update_upload_status(
    request: dict,
    current_user: dict = Depends(get_current_user)
):
    """Update the processing status of an upload."""
    try:
        upload_id = request.get("upload_id")
        status = request.get("status", "processing")
        progress = request.get("progress", 0)
        message = request.get("message", "")
        
        if not upload_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="upload_id is required"
            )
        
        # Update the processing status
        update_processing_status(
            upload_id=upload_id,
            status=status,
            progress=progress,
            message=message
        )
        
        return {
            "success": True,
            "upload_id": upload_id,
            "status": status,
            "progress": progress,
            "message": message
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update status: {str(e)}"
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
    """Background task to process uploaded video with automatic chunking for large files."""
    # Import at the top to avoid conflicts
    import traceback
    import ffmpeg
    from utils.cache import update_processing_status, cache_result
    from ai.whisperx import transcribe_audio
    from ai.summarizer import generate_summary
    from ai.translator import translate_text
    from utils.helpers import get_video_metadata, extract_thumbnail
    
    try:
        print(f"[{upload_id}] Starting background processing task for {filename}")
        
        # Check if the source video file exists
        if not os.path.exists(video_path) or os.path.getsize(video_path) == 0:
            print(f"[{upload_id}] Source video not found at {video_path}, checking for alternatives")
            
            # Try to find any video file in the upload directory
            upload_dir = f"uploads/{upload_id}"
            if not os.path.exists(upload_dir):
                raise Exception(f"Upload directory not found: {upload_dir}")
                
            video_files = [f for f in os.listdir(upload_dir) if f.endswith(('.mp4', '.mkv', '.webm', '.avi', '.mov'))]
            
            if video_files:
                # Use the first video file found
                video_path = os.path.join(upload_dir, video_files[0])
                print(f"[{upload_id}] Found alternative video: {video_path}")
            else:
                raise Exception(f"No video file found in upload directory: {upload_dir}")
        else:
            print(f"[{upload_id}] Source video found: {video_path}")
        
        # Get file size and check if we should use chunked processing
        file_size = os.path.getsize(video_path)
        file_size_gb = file_size / (1024 * 1024 * 1024)
        
        print(f"[{upload_id}] Video file size: {file_size_gb:.2f}GB")
        
        # Use chunked processing for files larger than 5GB
        if file_size_gb > 5:
            print(f"[{upload_id}] Large file detected ({file_size_gb:.2f}GB), using chunked processing")
            return await process_large_video_in_chunks(
                video_path, upload_id, filename, languages, summary_length, user_id
            )
        
        # For files <= 5GB, use the standard processing method
        print(f"[{upload_id}] Using standard processing for {file_size_gb:.2f}GB file")
        
        # For FFmpeg processing, we'll use the original path directly
        processing_video_path = video_path
        
        # Update status: starting audio extraction
        update_processing_status(
            upload_id=upload_id,
            status="processing",
            progress=10,
            message=f"Extracting audio from video ({file_size/1024/1024:.2f} MB)..."
        )
        
        # Extract audio from video with enhanced error handling
        audio_path = f"uploads/{upload_id}/audio.wav"
        
        try:
            print(f"[{upload_id}] Checking video for audio streams...")
            
            # First, probe the video to check if it has audio streams
            try:
                probe = ffmpeg.probe(processing_video_path)
                audio_streams = [stream for stream in probe['streams'] if stream['codec_type'] == 'audio']
                
                if not audio_streams:
                    print(f"[{upload_id}] No audio streams found in video")
                    raise Exception("The uploaded video does not contain any audio tracks. Please upload a video with audio content for transcription and summarization.")
                
                print(f"[{upload_id}] Found {len(audio_streams)} audio stream(s)")
                
                # Log audio stream details for debugging
                for i, stream in enumerate(audio_streams):
                    codec = stream.get('codec_name', 'unknown')
                    duration = stream.get('duration', 'unknown')
                    print(f"[{upload_id}] Audio stream {i}: codec={codec}, duration={duration}")
                
            except ffmpeg.Error as probe_error:
                print(f"[{upload_id}] FFmpeg probe failed: {str(probe_error)}")
                # Continue with extraction attempt - probe might fail but extraction could work
            
            print(f"[{upload_id}] Extracting audio to {audio_path}")
            
            # Try standard extraction first
            try:
                (
                    ffmpeg
                    .input(processing_video_path)
                    .output(audio_path, acodec='pcm_s16le', ac=1, ar='16k')
                    .run(quiet=False, overwrite_output=True, capture_stderr=True)
                )
                print(f"[{upload_id}] Standard audio extraction completed successfully")
            except ffmpeg.Error as e:
                error_message = e.stderr.decode() if e.stderr else str(e)
                print(f"[{upload_id}] Standard FFmpeg extraction failed: {error_message}")
                
                # Check for common "no audio" error patterns
                no_audio_patterns = [
                    "does not contain any stream",
                    "Output file #0 does not contain any stream",
                    "No such file or directory",
                    "Stream specifier ':a' does not match any streams"
                ]
                
                if any(pattern in error_message for pattern in no_audio_patterns):
                    raise Exception("The video file does not contain any audio stream that can be extracted. Please ensure your video has audio content.")
                
                # Try alternative extraction methods
                print(f"[{upload_id}] Trying alternative extraction methods...")
                
                # Method 1: Explicit audio stream mapping
                try:
                    print(f"[{upload_id}] Method 1: Explicit audio stream mapping...")
                    subprocess.run([
                        'ffmpeg', '-i', processing_video_path,
                        '-map', '0:a:0',  # Map first audio stream explicitly
                        '-vn',  # Disable video
                        '-acodec', 'pcm_s16le',
                        '-ar', '16000', '-ac', '1',
                        '-y', audio_path
                    ], check=True, capture_output=True, text=True)
                    print(f"[{upload_id}] Method 1 succeeded")
                except subprocess.CalledProcessError as sub_err:
                    error_output = sub_err.stderr if sub_err.stderr else str(sub_err)
                    print(f"[{upload_id}] Method 1 failed: {error_output}")
                    
                    # Method 2: Force WAV format with any available audio
                    try:
                        print(f"[{upload_id}] Method 2: Force WAV with any audio...")
                        subprocess.run([
                            'ffmpeg', '-i', processing_video_path,
                            '-f', 'wav', '-vn',
                            '-acodec', 'pcm_s16le',
                            '-ar', '16000', '-ac', '1',
                            '-y', audio_path
                        ], check=True, capture_output=True, text=True)
                        print(f"[{upload_id}] Method 2 succeeded")
                    except subprocess.CalledProcessError as sub_err2:
                        error_output2 = sub_err2.stderr if sub_err2.stderr else str(sub_err2)
                        print(f"[{upload_id}] Method 2 failed: {error_output2}")
                        
                        # Check if all methods failed due to no audio
                        combined_errors = error_message + error_output + error_output2
                        if any(pattern in combined_errors for pattern in no_audio_patterns):
                            raise Exception("The video file does not contain any audio stream. Please upload a video with audio content for transcription.")
                        else:
                            raise Exception(f"Failed to extract audio after trying multiple methods. Last error: {error_output2}")

        except Exception as audio_error:
            # Re-raise our custom error messages
            raise audio_error

        # Verify audio file was created and has content
        if not os.path.exists(audio_path):
            raise Exception("Audio extraction failed: No audio file was created. The video may not contain audio.")
            
        audio_size = os.path.getsize(audio_path)
        if audio_size == 0:
            raise Exception("Audio extraction failed: The extracted audio file is empty. The video may not contain audio content.")
        
        # Check for suspiciously small audio files
        if audio_size < 1024:  # Less than 1KB
            print(f"[{upload_id}] Warning: Very small audio file ({audio_size} bytes) - may indicate no actual audio content")
            
        print(f"[{upload_id}] Audio extraction successful: {audio_size/1024/1024:.2f} MB audio file created")
        
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
        error_traceback = traceback.format_exc()
        print(f"[{upload_id}] Processing failed: {error_message}")
        print(f"[{upload_id}] Full traceback:\n{error_traceback}")
        
        # Update status to failed - use the cache version to avoid conflicts
        try:
            update_processing_status(
                upload_id=upload_id,
                status="failed",
                progress=0,
                message=error_message,
                error=error_message
            )
        except Exception as status_error:
            print(f"[{upload_id}] Failed to update status: {str(status_error)}")
        
        # Log the error
        try:
            with open(f"uploads/{upload_id}/error.log", "w") as f:
                f.write(f"{error_message}\n\nFull traceback:\n{error_traceback}")
        except Exception as log_error:
            print(f"[{upload_id}] Failed to write error log: {str(log_error)}")

# Chunked upload data models and session storage
from pydantic import BaseModel

class ChunkedUploadInit(BaseModel):
    upload_id: str
    filename: str
    total_size: int
    total_chunks: int
    languages: str = "en"
    summary_length: int = 3

class ChunkedUploadFinalize(BaseModel):
    upload_id: str

# In-memory storage for chunked upload sessions
chunked_uploads = {}

@router.post("/init-chunked")
async def init_chunked_upload(
    init_data: ChunkedUploadInit,
    current_user: dict = Depends(get_current_user)
):
    """Initialize a chunked upload session."""
    try:
        upload_id = init_data.upload_id
        
        # Create upload directory
        upload_dir = f"uploads/{upload_id}"
        os.makedirs(upload_dir, exist_ok=True)
        
        # Create chunks directory
        chunks_dir = f"{upload_dir}/chunks"
        os.makedirs(chunks_dir, exist_ok=True)
        
        # Store session info
        chunked_uploads[upload_id] = {
            "filename": init_data.filename,
            "total_size": init_data.total_size,
            "total_chunks": init_data.total_chunks,
            "languages": init_data.languages,
            "summary_length": init_data.summary_length,
            "user_id": current_user.id,
            "chunks_received": set(),
            "created_at": time.time()
        }
        
        print(f"Initialized chunked upload {upload_id}: {init_data.filename} ({init_data.total_size} bytes, {init_data.total_chunks} chunks)")
        
        return {
            "success": True,
            "upload_id": upload_id,
            "message": "Chunked upload session initialized"
        }
        
    except Exception as e:
        print(f"Error initializing chunked upload: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to initialize chunked upload: {str(e)}"
        )

@router.post("/chunk")
async def upload_chunk(
    chunk: UploadFile = File(...),
    chunk_index: int = Form(...),
    upload_id: str = Form(...),
    current_user: dict = Depends(get_current_user)
):
    """Upload a single chunk of a file."""
    import asyncio
    import aiofiles
    
    try:
        # Verify session exists
        if upload_id not in chunked_uploads:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Upload session not found or expired"
            )
        
        session = chunked_uploads[upload_id]
        
        # Verify user owns this upload
        if session["user_id"] != current_user.id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied"
            )
        
        # Save chunk using async file operations to prevent blocking
        chunk_path = f"uploads/{upload_id}/chunks/chunk_{chunk_index:06d}"
        
        # Read chunk content asynchronously
        content = await chunk.read()
        content_size = len(content)
        
        # Write chunk asynchronously to prevent blocking other requests
        async with aiofiles.open(chunk_path, "wb") as f:
            await f.write(content)
            # Note: aiofiles doesn't support fsync(), but the async write is sufficient
        
        # Mark chunk as received (this is thread-safe for basic operations)
        session["chunks_received"].add(chunk_index)
        
        # Log less frequently to reduce I/O overhead
        if chunk_index % 10 == 0 or chunk_index < 5:
            print(f"Received chunk {chunk_index} for upload {upload_id} ({content_size} bytes)")
        
        return {
            "success": True,
            "chunk_index": chunk_index,
            "upload_id": upload_id,
            "chunks_received": len(session["chunks_received"]),
            "total_chunks": session["total_chunks"]
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error uploading chunk {chunk_index} for {upload_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to upload chunk: {str(e)}"
        )

@router.post("/finalize-chunked")
async def finalize_chunked_upload(
    finalize_data: ChunkedUploadFinalize,
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(get_current_user)
):
    """Finalize a chunked upload by combining all chunks."""
    try:
        upload_id = finalize_data.upload_id
        
        # Verify session exists
        if upload_id not in chunked_uploads:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Upload session not found or expired"
            )
        
        session = chunked_uploads[upload_id]
        
        # Verify user owns this upload
        if session["user_id"] != current_user.id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied"
            )
        
        # Verify all chunks received
        expected_chunks = set(range(session["total_chunks"]))
        if session["chunks_received"] != expected_chunks:
            missing_chunks = expected_chunks - session["chunks_received"]
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Missing chunks: {sorted(missing_chunks)}"
            )
        
        print(f"Finalizing chunked upload {upload_id}: combining {session['total_chunks']} chunks")
        
        # Combine chunks into final file
        chunks_dir = f"uploads/{upload_id}/chunks"
        final_path = f"uploads/{upload_id}/{session['filename']}"
        
        with open(final_path, "wb") as final_file:
            for chunk_index in range(session["total_chunks"]):
                chunk_path = f"{chunks_dir}/chunk_{chunk_index:06d}"
                
                if os.path.exists(chunk_path):
                    with open(chunk_path, "rb") as chunk_file:
                        final_file.write(chunk_file.read())
                else:
                    raise Exception(f"Chunk {chunk_index} not found")
        
        # Verify final file size
        final_size = os.path.getsize(final_path)
        if final_size != session["total_size"]:
            raise Exception(f"File size mismatch: expected {session['total_size']}, got {final_size}")
        
        print(f"Successfully combined chunks for {upload_id}: {final_size} bytes")
        
        # Clean up chunks directory
        try:
            shutil.rmtree(chunks_dir)
        except Exception as e:
            print(f"Warning: Failed to clean up chunks directory: {str(e)}")
        
        # Extract metadata and thumbnail immediately
        video_info = {
            "filename": session["filename"],
            "upload_time": time.time(),
            "languages_requested": session["languages"].split(","),
            "summary_length": session["summary_length"],
            "user_id": current_user.id,
            "user_name": current_user.username,
            "processing_state": "metadata_only",
        }
        
        # Extract basic metadata and thumbnail
        try:
            from utils.helpers import get_video_metadata, extract_thumbnail
            
            metadata = get_video_metadata(final_path)
            
            thumbnail_path = f"uploads/{upload_id}/thumbnail.jpg"
            extract_thumbnail(final_path, thumbnail_path)
            
            video_info.update({
                "thumbnail": f"/uploads/{upload_id}/thumbnail.jpg",
                "duration": metadata.get("duration", 0),
                "title": session["filename"],
                "metadata": metadata
            })
        except Exception as e:
            print(f"Error extracting initial metadata: {str(e)}")
        
        # Save to database
        try:
            conn = sqlite3.connect("clipsummary.db")
            cursor = conn.cursor()
            video_id = str(uuid.uuid4())
            cursor.execute(
                """INSERT INTO videos (id, user_id, title, filename, upload_id, status, is_youtube) 
                VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (video_id, current_user.id, session["filename"], session["filename"], upload_id, "metadata_ready", False)
            )
            conn.commit()
            conn.close()
            video_info["video_id"] = video_id
        except Exception as e:
            print(f"Error saving to database: {str(e)}")
        
        # Save video info
        with open(f"uploads/{upload_id}/info.json", "w") as f:
            json.dump(video_info, f)
        
        # Cache metadata
        cache_result(f"video:{upload_id}:info", video_info)
        
        # Create video.mp4 symlink/hardlink for streaming
        video_stream_path = f"uploads/{upload_id}/video.mp4"
        try:
            if os.path.exists(video_stream_path):
                os.unlink(video_stream_path)
            
            # Use hard link for efficiency
            os.link(final_path, video_stream_path)
            print(f"Created hard link for streaming: {video_stream_path}")
        except Exception as e:
            # Fall back to copy if hard link fails
            try:
                shutil.copy(final_path, video_stream_path)
                print(f"Created copy for streaming: {video_stream_path}")
            except Exception as copy_error:
                print(f"Warning: Failed to create streaming file: {str(copy_error)}")
        
        # Set initial status
        update_processing_status(
            upload_id=upload_id,
            status="metadata_ready",
            progress=5,
            message="Chunked upload completed. Video ready for viewing."
        )
        
        # Schedule background processing
        background_tasks.add_task(
            process_uploaded_video,
            video_path=final_path,
            upload_id=upload_id,
            filename=session["filename"],
            languages=session["languages"].split(","),
            summary_length=session["summary_length"],
            user_id=current_user.id
        )
        
        # Clean up session
        del chunked_uploads[upload_id]
        
        print(f"Chunked upload {upload_id} finalized and processing scheduled")
        
        return {
            "success": True,
            "upload_id": upload_id,
            "filename": session["filename"],
            "size": final_size,
            "metadata": video_info,
            "message": "Upload completed successfully. Processing started in background."
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error finalizing chunked upload: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to finalize upload: {str(e)}"
        )

async def process_large_video_in_chunks(
    video_path: str,
    upload_id: str,
    filename: str,
    languages: List[str],
    summary_length: int,
    user_id: str,
    chunk_duration_minutes: int = 1  # Split into 1-minute chunks
):
    """Process large videos by splitting them into smaller chunks for faster processing."""
    try:
        file_size = os.path.getsize(video_path)
        file_size_gb = file_size / (1024 * 1024 * 1024)
        
        print(f"[{upload_id}] Processing large video ({file_size_gb:.2f}GB) in chunks")
        
        # Get video duration to calculate number of chunks
        from utils.helpers import get_video_metadata
        metadata = get_video_metadata(video_path)
        duration_seconds = metadata.get("duration", 0)
        duration_minutes = duration_seconds / 60
        
        if duration_minutes <= chunk_duration_minutes:
            # Video is short enough, process normally
            return await process_uploaded_video(video_path, upload_id, filename, languages, summary_length, user_id)
        
        # Calculate number of chunks
        num_chunks = math.ceil(duration_minutes / chunk_duration_minutes)
        chunk_duration_seconds = chunk_duration_minutes * 60
        
        print(f"[{upload_id}] Splitting {duration_minutes:.1f} minute video into {num_chunks} chunks of {chunk_duration_minutes} minute(s) each")
        
        update_processing_status(
            upload_id=upload_id,
            status="processing",
            progress=5,
            message=f"Splitting large video into {num_chunks} chunks for faster processing..."
        )
        
        # Create chunks directory
        chunks_dir = f"uploads/{upload_id}/video_chunks"
        os.makedirs(chunks_dir, exist_ok=True)
        
        # Split video into chunks
        chunk_paths = []
        for i in range(num_chunks):
            start_time = i * chunk_duration_seconds
            chunk_path = f"{chunks_dir}/chunk_{i:03d}.mp4"
            
            try:
                # Extract chunk using FFmpeg
                (
                    ffmpeg
                    .input(video_path, ss=start_time, t=chunk_duration_seconds)
                    .output(chunk_path, vcodec='copy', acodec='copy')
                    .run(quiet=True, overwrite_output=True)
                )
                
                if os.path.exists(chunk_path) and os.path.getsize(chunk_path) > 0:
                    chunk_paths.append(chunk_path)
                    print(f"[{upload_id}] Created chunk {i+1}/{num_chunks}: {os.path.getsize(chunk_path)/1024/1024:.1f}MB")
                else:
                    print(f"[{upload_id}] Warning: Chunk {i+1} is empty or missing")
                    
            except Exception as e:
                print(f"[{upload_id}] Error creating chunk {i+1}: {str(e)}")
        
        if not chunk_paths:
            raise Exception("Failed to create any video chunks")
        
        print(f"[{upload_id}] Successfully created {len(chunk_paths)} video chunks")
        
        # Process chunks in parallel
        update_processing_status(
            upload_id=upload_id,
            status="processing",
            progress=15,
            message=f"Processing {len(chunk_paths)} video chunks in parallel..."
        )
        
        # Use ThreadPoolExecutor to process chunks in parallel
        max_workers = min(4, len(chunk_paths))  # Limit concurrent processing
        chunk_results = []
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all chunk processing tasks
            future_to_chunk = {}
            for i, chunk_path in enumerate(chunk_paths):
                future = executor.submit(
                    process_video_chunk,
                    chunk_path,
                    upload_id,
                    i,
                    len(chunk_paths)
                )
                future_to_chunk[future] = (i, chunk_path)
            
            # Collect results as they complete
            completed_futures = []
            for future in future_to_chunk:
                completed_futures.append(future)
            
            # Wait for all futures to complete
            for future in completed_futures:
                try:
                    chunk_index, chunk_path = future_to_chunk[future]
                    result = future.result()  # This will block until the future completes
                    chunk_results.append((chunk_index, result))
                    
                    # Update progress
                    progress = 15 + (len(chunk_results) / len(chunk_paths)) * 60
                    update_processing_status(
                        upload_id=upload_id,
                        status="processing",
                        progress=progress,
                        message=f"Processed chunk {len(chunk_results)}/{len(chunk_paths)}"
                    )
                    
                except Exception as e:
                    print(f"[{upload_id}] Error processing chunk: {str(e)}")
        
        # Sort results by chunk index
        chunk_results.sort(key=lambda x: x[0])
        
        # Merge transcription results
        update_processing_status(
            upload_id=upload_id,
            status="processing",
            progress=80,
            message="Merging chunk transcriptions..."
        )
        
        merged_transcript = merge_chunk_transcripts([result[1] for result in chunk_results])
        
        # Generate summary from merged transcript
        update_processing_status(
            upload_id=upload_id,
            status="processing",
            progress=85,
            message="Generating summary from merged transcript..."
        )
        
        from ai.summarizer import generate_summary
        from ai.translator import translate_text
        
        transcript_text = ' '.join([segment['text'] for segment in merged_transcript['segments']])
        summary = generate_summary(transcript_text, max_sentences=summary_length)
        
        # Handle translations
        result = {
            "upload_id": upload_id,
            "filename": filename,
            "transcript": merged_transcript,
            "summary": {"en": summary},
            "translations": {}
        }
        
        # Process translations
        for lang in languages:
            if lang != "en":
                update_processing_status(
                    upload_id=upload_id,
                    status="processing",
                    progress=90,
                    message=f"Translating to {lang}..."
                )
                
                translated_summary = translate_text(summary, target_lang=lang)
                translated_segments = []
                
                for segment in merged_transcript["segments"]:
                    translated_text = translate_text(segment["text"], target_lang=lang)
                    translated_segments.append({
                        "start": segment["start"],
                        "end": segment["end"],
                        "text": translated_text
                    })
                
                result["translations"][lang] = {
                    "summary": translated_summary,
                    "transcript": translated_segments
                }
        
        # Clean up chunk files
        try:
            shutil.rmtree(chunks_dir)
            print(f"[{upload_id}] Cleaned up video chunks directory")
        except Exception as e:
            print(f"[{upload_id}] Warning: Failed to clean up chunks: {str(e)}")
        
        # Save and finalize results
        with open(f"uploads/{upload_id}/result.json", "w") as f:
            json.dump(result, f)
        
        cache_result(f"upload:{upload_id}:result", result)
        
        update_processing_status(
            upload_id=upload_id,
            status="completed",
            progress=100,
            message="Large video processing completed successfully."
        )
        
        print(f"[{upload_id}] Large video chunk processing completed successfully")
        
    except Exception as e:
        error_message = f"Error processing large video in chunks: {str(e)}"
        print(f"[{upload_id}] {error_message}")
        update_processing_status(
            upload_id=upload_id,
            status="failed",
            progress=0,
            message=error_message,
            error=error_message
        )
        raise

def process_video_chunk(chunk_path: str, upload_id: str, chunk_index: int, total_chunks: int):
    """Process a single video chunk (extract audio and transcribe)."""
    try:
        print(f"[{upload_id}] Processing chunk {chunk_index + 1}/{total_chunks}")
        
        # Extract audio from chunk
        audio_path = f"{chunk_path.replace('.mp4', '.wav')}"
        
        (
            ffmpeg
            .input(chunk_path)
            .output(audio_path, acodec='pcm_s16le', ac=1, ar='16k')
            .run(quiet=True, overwrite_output=True)
        )
        
        if not os.path.exists(audio_path) or os.path.getsize(audio_path) == 0:
            raise Exception(f"Failed to extract audio from chunk {chunk_index}")
        
        # Transcribe chunk
        from ai.whisperx import transcribe_audio
        transcript = transcribe_audio(audio_path, f"{upload_id}_chunk_{chunk_index}")
        
        # Clean up chunk audio file
        try:
            os.remove(audio_path)
        except Exception:
            pass
        
        print(f"[{upload_id}] Completed processing chunk {chunk_index + 1}/{total_chunks}")
        return transcript
        
    except Exception as e:
        print(f"[{upload_id}] Error processing chunk {chunk_index}: {str(e)}")
        raise

def merge_chunk_transcripts(chunk_transcripts):
    """Merge multiple chunk transcripts into a single transcript with correct timing."""
    merged_segments = []
    time_offset = 0
    
    for i, transcript in enumerate(chunk_transcripts):
        if not transcript or "segments" not in transcript:
            continue
            
        for segment in transcript["segments"]:
            # Adjust timing based on chunk position
            adjusted_segment = {
                "start": segment["start"] + time_offset,
                "end": segment["end"] + time_offset,
                "text": segment["text"]
            }
            merged_segments.append(adjusted_segment)
        
        # Update time offset for next chunk (assume 1 minute chunks)
        if transcript["segments"]:
            last_segment = transcript["segments"][-1]
            time_offset = last_segment["end"] + time_offset
    
    return {
        "segments": merged_segments,
        "language": "en"
    }