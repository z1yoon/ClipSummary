from fastapi import APIRouter, Depends, HTTPException, Request, BackgroundTasks, File, UploadFile, Form, Query, Path
from fastapi.responses import JSONResponse, StreamingResponse
from typing import List, Optional
import os
import time
import logging
from .videos import (
    get_video_info,
    get_videos_by_user,
    get_random_videos,
    delete_video
)
from .upload import (
    upload_video_file,
    process_video_file,
    get_processing_status,
    create_video_entry,
)
from .youtube import download_youtube_video
from .users import get_current_user
from .auth import router as auth_router
from db.database import get_db
from db.models import User, Video
from sqlalchemy.orm import Session
from schemas.videos import VideoCreate, VideoResponse

# Import AI modules
from ai import translator

# Configure logging
logger = logging.getLogger(__name__)

# Create the main router
router = APIRouter()

# Include authentication routes
router.include_router(auth_router, prefix="/auth", tags=["Authentication"])

# --- Video Routes ---

@router.get("/videos", response_model=List[VideoResponse], tags=["Videos"])
async def list_videos(
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
    limit: int = 20,
    offset: int = 0
):
    """Get a list of videos for the current user"""
    videos = get_videos_by_user(db, user_id=user.id, limit=limit, offset=offset)
    return videos

@router.get("/videos/{video_id}", response_model=VideoResponse, tags=["Videos"])
async def get_video(
    video_id: str,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get information about a specific video"""
    video = get_video_info(db, video_id, user.id)
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    return video

@router.delete("/videos/{video_id}", tags=["Videos"])
async def remove_video(
    video_id: str,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Delete a video and its associated files"""
    success = delete_video(db, video_id, user.id)
    if not success:
        raise HTTPException(status_code=404, detail="Video not found")
    return {"status": "success", "message": "Video deleted successfully"}

@router.get("/videos/discover/random", response_model=List[VideoResponse], tags=["Discovery"])
async def discover_random_videos(
    limit: int = 5,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user)
):
    """Get a random selection of public videos for discovery"""
    videos = get_random_videos(db, limit=limit, exclude_user_id=user.id)
    return videos

# --- Upload Routes ---

@router.post("/upload/file", tags=["Upload"])
async def upload_file(
    request: Request,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    title: str = Form(None),
    description: str = Form(None),
    is_public: bool = Form(False),
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Upload a video file for processing"""
    try:
        # Generate a unique ID for the video
        upload_id = await upload_video_file(file)
        
        # Create entry in database
        video = create_video_entry(
            db=db,
            user_id=user.id,
            upload_id=upload_id,
            title=title or file.filename,
            description=description or "",
            is_public=is_public
        )
        
        # Process video in background
        background_tasks.add_task(
            process_video_file,
            upload_id=upload_id,
            db=db,
            video_id=video.id
        )
        
        return {
            "status": "success",
            "message": "File uploaded successfully",
            "upload_id": upload_id,
            "video_id": video.id
        }
    except Exception as e:
        logger.error(f"Upload failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@router.post("/upload/youtube", tags=["Upload"])
async def upload_youtube(
    request: Request,
    background_tasks: BackgroundTasks,
    youtube_url: str = Form(...),
    title: str = Form(None),
    description: str = Form(None),
    is_public: bool = Form(False),
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Upload a YouTube video for processing"""
    try:
        # Download YouTube video and get upload ID
        upload_id = await download_youtube_video(youtube_url)
        
        # Create entry in database
        video = create_video_entry(
            db=db,
            user_id=user.id,
            upload_id=upload_id,
            title=title or "YouTube Video",
            description=description or "",
            is_public=is_public
        )
        
        # Process video in background
        background_tasks.add_task(
            process_video_file,
            upload_id=upload_id,
            db=db,
            video_id=video.id
        )
        
        return {
            "status": "success",
            "message": "YouTube video downloaded successfully",
            "upload_id": upload_id,
            "video_id": video.id
        }
    except Exception as e:
        logger.error(f"YouTube upload failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"YouTube upload failed: {str(e)}")

@router.get("/upload/status/{upload_id}", tags=["Upload"])
async def check_upload_status(
    upload_id: str,
    user: User = Depends(get_current_user)
):
    """Check the processing status of an uploaded video"""
    status_info = get_processing_status(upload_id)
    if not status_info:
        raise HTTPException(status_code=404, detail="Upload not found")
    return status_info

# --- Translation and Language Routes ---

@router.get("/videos/{video_id}/subtitles/{language}", tags=["Videos"])
async def get_video_subtitles(
    video_id: str,
    language: str,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get subtitles for a video in the specified language"""
    video = get_video_info(db, video_id, user.id)
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    
    # Here you would retrieve the subtitles for the specified language
    # For now, we'll return a stub
    return {
        "status": "success",
        "language": language,
        "segments": []  # This would be populated with actual subtitle segments
    }

@router.get("/available-languages", response_model=dict, tags=["Translation"])
async def get_available_languages(
    user: User = Depends(get_current_user)
):
    """Get list of available source and target languages for translation"""
    # Check what translation models are actually available
    source_languages = []
    target_languages = []
    
    # Add languages based on available models
    for model_name in translator.AVAILABLE_MODELS:
        if model_name == "Helsinki-NLP/opus-mt-en-zh":
            source_languages.append({"code": "en", "name": "English"})
            target_languages.append({"code": "zh", "name": "Chinese"})
        
        elif model_name == "Helsinki-NLP/opus-mt-zh-en":
            source_languages.append({"code": "zh", "name": "Chinese"})
            target_languages.append({"code": "en", "name": "English"})
        
        elif model_name == "Helsinki-NLP/opus-mt-ko-en":
            source_languages.append({"code": "ko", "name": "Korean"})
            target_languages.append({"code": "en", "name": "English"})
        
        elif model_name == "Helsinki-NLP/opus-mt-mul-en":
            # Multiple languages are supported for translation to English
            multi_source_languages = [
                {"code": "fr", "name": "French"},
                {"code": "es", "name": "Spanish"},
                {"code": "de", "name": "German"},
                {"code": "it", "name": "Italian"},
                {"code": "pt", "name": "Portuguese"},
                {"code": "ru", "name": "Russian"},
                {"code": "ja", "name": "Japanese"},
                {"code": "ar", "name": "Arabic"}
            ]
            # Add all multi-language sources if not already present
            for lang in multi_source_languages:
                if not any(l["code"] == lang["code"] for l in source_languages):
                    source_languages.append(lang)
            
            # Make sure English is in target languages
            if not any(l["code"] == "en" for l in target_languages):
                target_languages.append({"code": "en", "name": "English"})
    
    # Remove duplicates if any
    unique_source_codes = set()
    unique_source_languages = []
    for lang in source_languages:
        if lang["code"] not in unique_source_codes:
            unique_source_codes.add(lang["code"])
            unique_source_languages.append(lang)
    
    unique_target_codes = set()
    unique_target_languages = []
    for lang in target_languages:
        if lang["code"] not in unique_target_codes:
            unique_target_codes.add(lang["code"])
            unique_target_languages.append(lang)
    
    return {
        "source_languages": unique_source_languages,
        "target_languages": unique_target_languages,
        "available_translations": [
            {"source": "en", "target": "zh", "model": "Helsinki-NLP/opus-mt-en-zh"},
            {"source": "zh", "target": "en", "model": "Helsinki-NLP/opus-mt-zh-en"},
            {"source": "ko", "target": "en", "model": "Helsinki-NLP/opus-mt-ko-en"}
        ] + ([{"source": "mul", "target": "en", "model": "Helsinki-NLP/opus-mt-mul-en"}] 
             if "Helsinki-NLP/opus-mt-mul-en" in translator.AVAILABLE_MODELS else [])
    }

@router.post("/videos/{video_id}/translate", tags=["Translation"])
async def translate_video_content(
    video_id: str,
    request: Request,
    background_tasks: BackgroundTasks,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Translate video content (transcript and subtitles) to the target language
    """
    try:
        # Parse request body
        body = await request.json()
        source_lang = body.get("source_language", "auto")
        target_lang = body.get("target_language")
        
        if not target_lang:
            raise HTTPException(status_code=400, detail="Target language is required")
        
        # Get the video
        video = get_video_info(db, video_id, user.id)
        if not video:
            raise HTTPException(status_code=404, detail="Video not found")
        
        # If source is auto, use the detected language from the video
        if source_lang == "auto":
            source_lang = video.detected_language or "en"
        
        # Check if we can translate from source to target language
        model_result = translator.get_model_name(source_lang, target_lang)
        if not model_result:
            raise HTTPException(
                status_code=400, 
                detail=f"Translation from {source_lang} to {target_lang} is not supported"
            )
        
        # Generate a unique translation job ID
        import uuid
        translation_job_id = f"trans_{str(uuid.uuid4())[:8]}"
        
        # Store initial status in cache
        from utils.cache import update_processing_status
        update_processing_status(
            upload_id=translation_job_id,
            status="processing",
            progress=10,
            message=f"Starting translation from {source_lang} to {target_lang}"
        )
        
        # Start translation in the background
        background_tasks.add_task(
            process_translation,
            video_id=video_id,
            source_lang=source_lang,
            target_lang=target_lang,
            translation_job_id=translation_job_id,
            db=db
        )
        
        return {
            "status": "processing",
            "message": f"Started translating from {source_lang} to {target_lang}",
            "job_id": translation_job_id
        }
        
    except Exception as e:
        logger.error(f"Translation request failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Translation request failed: {str(e)}")

@router.get("/videos/{video_id}/translate/status", tags=["Translation"])
async def get_translation_status(
    video_id: str,
    job_id: str = Query(..., description="Translation job ID"),
    user: User = Depends(get_current_user)
):
    """Check status of a translation job"""
    from utils.cache import get_processing_status
    
    status_info = get_processing_status(job_id)
    if not status_info:
        raise HTTPException(status_code=404, detail="Translation job not found")
    
    return status_info

@router.get("/videos/{video_id}/subtitles/{language}", tags=["Videos"])
async def get_video_subtitles(
    video_id: str,
    language: str,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get subtitles for a video in the specified language"""
    video = get_video_info(db, video_id, user.id)
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    
    # Get the subtitle file path
    subtitle_path = f"uploads/{video.upload_id}/subtitles_{language}.json"
    
    # If the subtitles don't exist for this language but the source language exists,
    # notify that translation is required
    if not os.path.exists(subtitle_path):
        # Check if original language subtitles exist
        original_subtitle_path = f"uploads/{video.upload_id}/subtitles_{video.detected_language}.json"
        if os.path.exists(original_subtitle_path):
            return {
                "status": "needs_translation",
                "message": f"Subtitles need to be translated to {language}",
                "detected_language": video.detected_language
            }
        else:
            return {
                "status": "error",
                "message": "No subtitles available for this video"
            }
    
    # Read and return the subtitles
    import json
    try:
        with open(subtitle_path, "r") as f:
            subtitles = json.load(f)
        
        return {
            "status": "success",
            "language": language,
            "segments": subtitles.get("segments", [])
        }
    except Exception as e:
        logger.error(f"Error reading subtitles: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to read subtitles")

# --- Helper function for background translation ---
async def process_translation(
    video_id: str,
    source_lang: str,
    target_lang: str,
    translation_job_id: str,
    db: Session
):
    """Background task for translating video content"""
    from utils.cache import update_processing_status
    
    try:
        # Get video info
        video = db.query(Video).filter(Video.id == video_id).first()
        if not video:
            update_processing_status(
                upload_id=translation_job_id,
                status="failed",
                progress=0,
                message="Video not found"
            )
            return
        
        # Update status
        update_processing_status(
            upload_id=translation_job_id,
            status="processing",
            progress=20,
            message=f"Reading source subtitles in {source_lang}"
        )
        
        # Read source subtitles
        source_subtitle_path = f"uploads/{video.upload_id}/subtitles_{source_lang}.json"
        if not os.path.exists(source_subtitle_path):
            update_processing_status(
                upload_id=translation_job_id,
                status="failed",
                progress=0,
                message=f"Source subtitles not found for {source_lang}"
            )
            return
        
        # Load source subtitles
        import json
        try:
            with open(source_subtitle_path, "r") as f:
                source_subtitles = json.load(f)
            
            segments = source_subtitles.get("segments", [])
            if not segments:
                update_processing_status(
                    upload_id=translation_job_id,
                    status="failed",
                    progress=0,
                    message="No subtitle segments found"
                )
                return
            
            # Extract texts for translation
            texts = [segment.get("text", "") for segment in segments]
            
            # Update status
            update_processing_status(
                upload_id=translation_job_id,
                status="processing",
                progress=40,
                message=f"Translating {len(texts)} segments from {source_lang} to {target_lang}"
            )
            
            # Translate texts
            translated_texts = translator.batch_translate(
                texts=texts,
                source_lang=source_lang,
                target_lang=target_lang,
                upload_id=translation_job_id
            )
            
            # Update status
            update_processing_status(
                upload_id=translation_job_id,
                status="processing",
                progress=80,
                message="Creating translated subtitle file"
            )
            
            # Create translated subtitles
            translated_subtitles = {
                "language": target_lang,
                "segments": []
            }
            
            for i, segment in enumerate(segments):
                if i < len(translated_texts):
                    translated_segment = segment.copy()
                    translated_segment["text"] = translated_texts[i]
                    translated_subtitles["segments"].append(translated_segment)
            
            # Save translated subtitles
            target_subtitle_path = f"uploads/{video.upload_id}/subtitles_{target_lang}.json"
            with open(target_subtitle_path, "w") as f:
                json.dump(translated_subtitles, f, ensure_ascii=False, indent=2)
            
            # Update status
            update_processing_status(
                upload_id=translation_job_id,
                status="completed",
                progress=100,
                message=f"Successfully translated subtitles to {target_lang}"
            )
            
        except Exception as e:
            logger.error(f"Error in translation process: {str(e)}")
            update_processing_status(
                upload_id=translation_job_id,
                status="failed", 
                progress=0,
                message=f"Translation failed: {str(e)}"
            )
            
    except Exception as e:
        logger.error(f"Translation process failed: {str(e)}")
        update_processing_status(
            upload_id=translation_job_id,
            status="failed",
            progress=0,
            message=f"Process failed: {str(e)}"
        )

# --- System Health Routes ---
# ...existing code...
