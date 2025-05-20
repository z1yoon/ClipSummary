from fastapi import APIRouter, UploadFile, File, Form, HTTPException, BackgroundTasks, status
from fastapi.responses import JSONResponse
import os
import uuid
import shutil
from typing import List, Optional

# Change relative imports to absolute imports
from ai.whisperx import transcribe_audio
from ai.summarizer import generate_summary
from ai.translator import translate_text
from utils.cache import cache_result

router = APIRouter()

@router.post("/video")
async def upload_video(
    file: UploadFile = File(...),
    languages: str = Form("en"),
    summary_length: int = Form(3),
    background_tasks: BackgroundTasks = None
):
    """Upload a video file for transcription, summarization, and translation."""
    try:
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
        
        # Save the uploaded file
        file_path = f"{upload_dir}/{file.filename}"
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Parse languages from form data
        language_list = languages.split(",")
        
        # Process the video in the background
        background_tasks.add_task(
            process_uploaded_video,
            video_path=file_path,
            upload_id=upload_id,
            filename=file.filename,
            languages=language_list,
            summary_length=summary_length
        )
        
        return JSONResponse(
            status_code=status.HTTP_202_ACCEPTED,
            content={
                "status": "processing",
                "upload_id": upload_id,
                "filename": file.filename,
                "message": "Your video is being processed. Check status with the provided upload_id."
            }
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred during upload: {str(e)}"
        )

@router.get("/status/{upload_id}")
async def get_upload_status(upload_id: str):
    """Check the status of uploaded video processing."""
    # Check if the result file exists
    result_file = f"uploads/{upload_id}/result.json"
    if os.path.exists(result_file):
        return {
            "status": "completed",
            "upload_id": upload_id,
            "result_url": f"/api/upload/result/{upload_id}"
        }
    # Check if there was an error
    error_file = f"uploads/{upload_id}/error.log"
    if os.path.exists(error_file):
        with open(error_file, "r") as f:
            error_message = f.read()
        return {
            "status": "failed",
            "upload_id": upload_id,
            "error": error_message
        }
    # Otherwise, still processing
    return {
        "status": "processing",
        "upload_id": upload_id,
        "message": "Your video is still being processed."
    }

@router.get("/result/{upload_id}")
async def get_upload_result(upload_id: str):
    """Get the result of uploaded video processing."""
    result_file = f"uploads/{upload_id}/result.json"
    if os.path.exists(result_file):
        import json
        with open(result_file, "r") as f:
            result = json.load(f)
        return result
    else:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Result not found. The video may still be processing or an error occurred."
        )

async def process_uploaded_video(
    video_path: str,
    upload_id: str,
    filename: str,
    languages: List[str],
    summary_length: int
):
    """Background task to process uploaded video."""
    try:
        # Extract audio from video
        import ffmpeg
        audio_path = f"uploads/{upload_id}/audio.wav"
        
        (
            ffmpeg
            .input(video_path)
            .output(audio_path, acodec='pcm_s16le', ac=1, ar='16k')
            .run(quiet=True, overwrite_output=True)
        )
        
        # Transcribe with WhisperX
        transcript = transcribe_audio(audio_path)
        
        # Generate summary (English)
        transcript_text = ' '.join([segment['text'] for segment in transcript['segments']])
        summary = generate_summary(transcript_text, max_sentences=summary_length)
        
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
        
        # Translate to requested languages
        for lang in languages:
            if lang != "en":  # Skip English as it's already done
                result["translations"][lang] = {
                    "summary": translate_text(summary, target_lang=lang),
                    "transcript": [
                        {
                            "start": segment["start"],
                            "end": segment["end"],
                            "text": translate_text(segment["text"], target_lang=lang)
                        }
                        for segment in transcript["segments"]
                    ]
                }
        
        # Save results
        import json
        with open(f"uploads/{upload_id}/result.json", "w") as f:
            json.dump(result, f)
        
        # Clean up
        # We'll leave the original video and audio for now, but in production
        # you might want to delete them after processing or move to storage
        
    except Exception as e:
        # Log the error
        with open(f"uploads/{upload_id}/error.log", "w") as f:
            f.write(str(e))