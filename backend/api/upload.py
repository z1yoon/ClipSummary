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

# Change relative imports to absolute imports
from ai.whisperx import transcribe_audio
from ai.summarizer import generate_summary
from ai.translator import translate_text
from utils.cache import cache_result, get_cached_result, get_redis_client
from api.auth import get_current_user

router = APIRouter()

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
        
        # Save video record to database
        try:
            conn = sqlite3.connect("clipsummary.db")
            cursor = conn.cursor()
            video_id = str(uuid.uuid4())
            cursor.execute(
                """INSERT INTO videos (id, user_id, title, filename, upload_id, status, is_youtube) 
                VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (video_id, current_user["id"], file.filename, file.filename, upload_id, "processing", False)
            )
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"Error saving to database: {str(e)}")
        
        # Parse languages from form data
        language_list = languages.split(",")
        
        # Set initial processing status
        update_processing_status(
            upload_id=upload_id,
            status="processing",
            progress=0,
            message="Upload received. Starting processing for large video file."
        )
        
        # Process the video in the background
        if background_tasks:
            background_tasks.add_task(
                process_uploaded_video,
                video_path=file_path,
                upload_id=upload_id,
                filename=file.filename,
                languages=language_list,
                summary_length=summary_length,
                user_id=current_user["id"]
            )
        
        # Save basic video info
        video_info = {
            "filename": file.filename,
            "upload_time": time.time(),
            "languages_requested": language_list,
            "summary_length": summary_length,
            "user_id": current_user["id"],
            "user_name": current_user["username"]
        }
        
        with open(f"{upload_dir}/info.json", "w") as f:
            json.dump(video_info, f)
        
        # Cache initial metadata
        cache_result(f"video:{upload_id}:info", video_info)
        
        print(f"Upload processed successfully. ID: {upload_id}")
        
        return JSONResponse(
            status_code=status.HTTP_202_ACCEPTED,
            content={
                "status": "processing",
                "upload_id": upload_id,
                "filename": file.filename,
                "message": "Your video is being processed. For long videos, this may take significant time.",
                "redirectUrl": f"/video.html?id={upload_id}"
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
        # Get file size for logging
        file_size = os.path.getsize(video_path)
        print(f"[{upload_id}] Starting processing of {filename} ({file_size/1024/1024:.2f} MB)")

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
                .input(video_path)
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
                    '-i', video_path,
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

        # Create a symlink or copy the video file to make it accessible for streaming
        video_stream_path = f"uploads/{upload_id}/video.mp4"
        if not os.path.exists(video_stream_path):
            print(f"[{upload_id}] Creating video stream symlink/copy")
            try:
                os.symlink(video_path, video_stream_path)
            except:
                shutil.copy(video_path, video_stream_path)
        
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