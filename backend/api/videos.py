from fastapi import APIRouter, HTTPException, Depends, Path, Query, status
from typing import Dict, List, Optional
from pydantic import BaseModel
import os
import json
import time
from utils.helpers import get_video_metadata
from ai.translator import translate_text
from utils.cache import get_cached_result, cache_result
from api.auth import get_current_user

router = APIRouter()

class SubtitleSegment(BaseModel):
    start: float
    end: float
    text: str

class VideoDetails(BaseModel):
    id: str
    title: str
    url: str
    duration: int
    thumbnail: Optional[str]
    summary: Optional[str]
    languages: List[str]

class TranscriptResponse(BaseModel):
    segments: List[SubtitleSegment]
    language: str

class SummaryResponse(BaseModel):
    summary: str
    language: str = "en"

@router.get("/{video_id}")
async def get_video_details(
    video_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Get video details including available languages and summary"""
    try:
        # Check if video info exists
        info_path = f"uploads/{video_id}/info.json"
        if not os.path.exists(info_path):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Video not found"
            )

        # Read video info
        with open(info_path, "r") as f:
            info = json.load(f)

        # Get subtitle languages
        subtitles_dir = f"uploads/{video_id}/subtitles"
        available_languages = ["en"] # Default to English
        if os.path.exists(subtitles_dir):
            for file in os.listdir(subtitles_dir):
                if file.endswith('.json'):
                    lang_code = file.split('.')[0]
                    if lang_code not in available_languages:
                        available_languages.append(lang_code)

        # Check translations directory for additional languages
        translations_dir = f"uploads/{video_id}/translations"
        if os.path.exists(translations_dir):
            for file in os.listdir(translations_dir):
                if os.path.isdir(f"{translations_dir}/{file}"):
                    if file not in available_languages:
                        available_languages.append(file)

        # Get video URL
        video_url = f"/uploads/{video_id}/video.mp4"
        if not os.path.exists(f"uploads/{video_id}/video.mp4"):
            # For YouTube videos, we might not have a local video file
            video_url = None

        # Load result file for summary if available
        summary = None
        result_path = f"uploads/{video_id}/result.json"
        if os.path.exists(result_path):
            with open(result_path, "r") as f:
                result = json.load(f)
                summary = result.get("summary", {}).get("en")
        
        # Get original video URL if it exists (for YouTube videos)
        original_url = info.get("url", None)

        return {
            "video_id": video_id,
            "title": info.get("title", "Untitled"),
            "video_url": video_url,
            "url": original_url,
            "thumbnail": info.get("thumbnail"),
            "duration": info.get("duration"),
            "available_languages": available_languages,
            "summary": summary
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving video details: {str(e)}"
        )

@router.get("/{video_id}/subtitles/{language}")
async def get_video_subtitles(
    video_id: str,
    language: str,
    current_user: dict = Depends(get_current_user)
):
    """Get subtitles for a specific language"""
    try:
        subtitle_path = f"uploads/{video_id}/subtitles/{language}.json"
        if not os.path.exists(subtitle_path):
            # If the requested language doesn't exist, try to translate from English
            en_subtitle_path = f"uploads/{video_id}/subtitles/en.json"
            if not os.path.exists(en_subtitle_path):
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Subtitles not found for language: {language}"
                )
            
            # Check if we have a cached translation
            cache_key = f"subtitles_{video_id}_{language}"
            cached = get_cached_result(cache_key)
            if cached:
                return cached
            
            # Translate from English
            with open(en_subtitle_path, "r") as f:
                en_subtitles = json.load(f)
            
            # Create subtitles directory if it doesn't exist
            os.makedirs(os.path.dirname(subtitle_path), exist_ok=True)
            
            # Translate subtitle segments
            translated_segments = []
            for segment in en_subtitles["segments"]:
                translated_text = await translate_text(segment["text"], language)
                translated_segments.append({
                    "start": segment["start"],
                    "end": segment["end"],
                    "text": translated_text
                })
            
            translated_subtitles = {
                "segments": translated_segments,
                "language": language
            }
            
            # Save translated subtitles
            with open(subtitle_path, "w") as f:
                json.dump(translated_subtitles, f, ensure_ascii=False)
            
            # Cache the result
            cache_result(cache_key, translated_subtitles)
            
            return translated_subtitles
        
        with open(subtitle_path, "r") as f:
            subtitles = json.load(f)

        return subtitles

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving subtitles: {str(e)}"
        )

@router.get("/{video_id}/transcript/{language}")
async def get_video_transcript(
    video_id: str,
    language: str,
    current_user: dict = Depends(get_current_user)
):
    """Get transcript for a specific language"""
    try:
        # Check for dedicated transcript file first
        transcript_path = f"uploads/{video_id}/transcripts/{language}.json"
        
        if os.path.exists(transcript_path):
            with open(transcript_path, "r") as f:
                transcript = json.load(f)
            return transcript
        
        # Fall back to subtitles if no dedicated transcript exists
        return await get_video_subtitles(video_id, language, current_user)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving transcript: {str(e)}"
        )

@router.get("/{video_id}/summary/{language}")
async def get_video_summary(
    video_id: str,
    language: str,
    current_user: dict = Depends(get_current_user)
):
    """Get video summary in the specified language"""
    try:
        result_path = f"uploads/{video_id}/result.json"
        if not os.path.exists(result_path):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Summary not found"
            )

        with open(result_path, "r") as f:
            result = json.load(f)

        summary = None
        
        # Check if we already have this language's summary
        if language == "en":
            summary = result.get("summary", {}).get("en")
        else:
            # Check translations
            summary = result.get("translations", {}).get(language, {}).get("summary")
        
        # If we don't have the summary in this language, translate it
        if not summary and language != "en":
            # Get English summary
            en_summary = result.get("summary", {}).get("en")
            if not en_summary:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="English summary not available for translation"
                )
            
            # Check if we have a cached translation
            cache_key = f"summary_{video_id}_{language}"
            cached = get_cached_result(cache_key)
            if cached:
                return cached
            
            # Translate summary
            summary = await translate_text(en_summary, language)
            
            # Save translation to result file
            if not result.get("translations"):
                result["translations"] = {}
            if not result["translations"].get(language):
                result["translations"][language] = {}
            
            result["translations"][language]["summary"] = summary
            
            with open(result_path, "w") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            
            # Cache the result
            response = {"summary": summary, "language": language}
            cache_result(cache_key, response)
            return response

        if not summary:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Summary not found for language: {language}"
            )

        return {"summary": summary, "language": language}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving summary: {str(e)}"
        )

@router.get("/user/videos")
async def list_user_videos(
    current_user: dict = Depends(get_current_user),
    skip: int = Query(0, description="Skip the first N videos"),
    limit: int = Query(20, description="Limit the number of videos returned")
):
    """List all videos uploaded by the current user"""
    try:
        # Check cache first for faster response
        cache_key = f"user_videos:{current_user['id']}:{skip}:{limit}"
        cached_result = get_cached_result(cache_key)
        if cached_result:
            return cached_result
            
        # Get videos from database
        import sqlite3
        
        conn = sqlite3.connect("clipsummary.db")
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # Query videos for the current user, ordered by most recent first
        cursor.execute(
            "SELECT id, upload_id, title, filename, status, created_at FROM videos WHERE user_id = ? ORDER BY created_at DESC LIMIT ? OFFSET ?",
            (current_user["id"], limit, skip)
        )
        
        videos = []
        upload_ids = []
        for row in cursor.fetchall():
            video_data = dict(row)
            videos.append(video_data)
            upload_ids.append(video_data['upload_id'])
        
        conn.close()
        
        # Bulk load metadata for all videos at once (more efficient)
        thumbnail_data = {}
        for upload_id in upload_ids:
            info_path = f"uploads/{upload_id}/info.json"
            if os.path.exists(info_path):
                try:
                    with open(info_path, "r") as f:
                        info = json.load(f)
                    
                    thumbnail_data[upload_id] = {
                        "thumbnail": info.get("thumbnail"),
                        "duration": info.get("duration", 0)
                    }
                except Exception as e:
                    print(f"Error loading info for video {upload_id}: {str(e)}")
        
        # Add metadata to video objects
        for video in videos:
            if video['upload_id'] in thumbnail_data:
                video.update(thumbnail_data[video['upload_id']])
        
        result = {
            "videos": videos,
            "count": len(videos),
            "skip": skip,
            "limit": limit
        }
        
        # Cache the result for 5 minutes (adjust TTL as needed)
        cache_result(cache_key, result, ttl=300)
        
        return result
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving videos: {str(e)}"
        )