from fastapi import APIRouter, HTTPException, Depends, Path, Query, status
from typing import Dict, List, Optional
from pydantic import BaseModel
import os
import json
import time
from utils.helpers import get_video_metadata
from ai.translator import translate_text, translate_summary, translate_subtitle_segments, get_supported_languages
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
    current_user = Depends(get_current_user)  # Remove type hint to be flexible
):
    """Get video details including available languages and summary"""
    try:
        # Check if video info exists (for uploaded videos)
        info_path = f"uploads/{video_id}/info.json"
        result_path = f"uploads/{video_id}/result.json"
        
        # Initialize default values
        video_info = {}
        summary = None
        available_languages = ["en"]
        video_url = None
        original_url = None
        
        # Try to load from info.json first (uploaded videos)
        if os.path.exists(info_path):
            with open(info_path, "r") as f:
                video_info = json.load(f)
            
            # Check user ownership for uploaded videos
            video_user_id = video_info.get("user_id")
            current_user_id = current_user.id if hasattr(current_user, 'id') else current_user.get('id')
            
            # Convert both to strings for comparison to handle type mismatches
            if str(video_user_id) != str(current_user_id):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="You don't have permission to access this video"
                )
            
            # Check for local video file
            if os.path.exists(f"uploads/{video_id}/video.mp4"):
                video_url = f"/uploads/{video_id}/video.mp4"
        
        # Try to load from result.json (YouTube videos)
        elif os.path.exists(result_path):
            with open(result_path, "r") as f:
                result_data = json.load(f)
            
            # Check user ownership for YouTube videos
            video_user_id = result_data.get("user_id")
            current_user_id = current_user.id if hasattr(current_user, 'id') else current_user.get('id')
            
            # Convert both to strings for comparison to handle type mismatches
            if str(video_user_id) != str(current_user_id):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="You don't have permission to access this video"
                )
            
            # Extract video info from result data
            video_info = {
                "title": result_data.get("title", "Untitled"),
                "url": result_data.get("url"),
                "video_id": result_data.get("video_id"),
                "duration": 0  # YouTube videos don't have local duration
            }
            
            # Get summary from result data
            summary_data = result_data.get("summary", {})
            if isinstance(summary_data, dict):
                summary = summary_data.get("en")
            else:
                summary = summary_data
                
            # Get available languages from translations
            available_languages = ["en"]
            if "translations" in result_data:
                available_languages.extend(result_data["translations"].keys())
        
        else:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Video not found"
            )

        # If we still don't have summary, try loading from result.json for uploaded videos
        if not summary and os.path.exists(result_path):
            with open(result_path, "r") as f:
                result = json.load(f)
                summary_data = result.get("summary", {})
                if isinstance(summary_data, dict):
                    summary = summary_data.get("en")
                else:
                    summary = summary_data

        # Get subtitle languages from subtitles directory
        subtitles_dir = f"uploads/{video_id}/subtitles"
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

        # Get original video URL (for YouTube videos)
        original_url = video_info.get("url", None)

        return {
            "video_id": video_id,
            "title": video_info.get("title", "Untitled"),
            "video_url": video_url,
            "url": original_url,
            "thumbnail": video_info.get("thumbnail"),
            "duration": video_info.get("duration", 0),
            "available_languages": available_languages,
            "summary": summary,
            "status": "completed"  # If we can load the data, it's completed
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
        # Verify user ownership first
        info_path = f"uploads/{video_id}/info.json"
        result_path = f"uploads/{video_id}/result.json"
        
        user_verified = False
        current_user_id = current_user.id if hasattr(current_user, 'id') else current_user.get('id')
        
        if os.path.exists(info_path):
            with open(info_path, "r") as f:
                info_data = json.load(f)
            # Convert both to strings for comparison to handle type mismatches
            if str(info_data.get("user_id")) == str(current_user_id):
                user_verified = True
        elif os.path.exists(result_path):
            with open(result_path, "r") as f:
                result_data = json.load(f)
            # Convert both to strings for comparison to handle type mismatches
            if str(result_data.get("user_id")) == str(current_user_id):
                user_verified = True
        
        if not user_verified:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You don't have permission to access this video"
            )
        
        subtitle_path = f"uploads/{video_id}/subtitles/{language}.json"
        
        # First, check if we have dedicated subtitle files
        if os.path.exists(subtitle_path):
            with open(subtitle_path, "r") as f:
                subtitles = json.load(f)
            return subtitles
        
        # For YouTube videos, check if we have transcript data in result.json
        if os.path.exists(result_path):
            with open(result_path, "r") as f:
                result_data = json.load(f)
            
            # Check for English transcript first
            if language == "en" and "transcript" in result_data:
                transcript = result_data["transcript"]
                if "segments" in transcript:
                    return {
                        "segments": transcript["segments"],
                        "language": "en"
                    }
            
            # Check for translated transcripts
            elif language != "en" and "translations" in result_data:
                if language in result_data["translations"] and "transcript" in result_data["translations"][language]:
                    return {
                        "segments": result_data["translations"][language]["transcript"],
                        "language": language
                    }
        
        # If we don't have the requested language, try to translate from English
        # First check if we have English subtitles/transcript
        en_subtitle_path = f"uploads/{video_id}/subtitles/en.json"
        en_segments = None
        
        if os.path.exists(en_subtitle_path):
            with open(en_subtitle_path, "r") as f:
                en_data = json.load(f)
                en_segments = en_data.get("segments", [])
        elif os.path.exists(result_path):
            with open(result_path, "r") as f:
                result_data = json.load(f)
                if "transcript" in result_data and "segments" in result_data["transcript"]:
                    en_segments = result_data["transcript"]["segments"]
        
        if not en_segments:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No transcript data found for translation to {language}"
            )
        
        # Check if we have a cached translation
        cache_key = f"subtitles_{video_id}_{language}"
        cached = get_cached_result(cache_key)
        if cached:
            return cached
        
        # Create subtitles directory if it doesn't exist
        os.makedirs(os.path.dirname(subtitle_path), exist_ok=True)
        
        # Translate subtitle segments
        translated_segments = []
        for segment in en_segments:
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

        # Verify user ownership
        video_user_id = result.get("user_id")
        current_user_id = current_user.id if hasattr(current_user, 'id') else current_user.get('id')
        
        # Convert both to strings for comparison to handle type mismatches
        if str(video_user_id) != str(current_user_id):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You don't have permission to access this video"
            )

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

@router.get("/")
async def list_all_videos(
    current_user: dict = Depends(get_current_user),
    skip: int = Query(0, description="Skip the first N videos"),
    limit: int = Query(20, description="Limit the number of videos returned")
):
    """List all videos - this is what the frontend expects at /api/videos"""
    return await list_user_videos(current_user, skip, limit)

@router.get("/user/videos")
async def list_user_videos(
    current_user = Depends(get_current_user),  # Remove type hint to be flexible
    skip: int = Query(0, description="Skip the first N videos"),
    limit: int = Query(20, description="Limit the number of videos returned")
):
    """List all videos uploaded by the current user"""
    try:
        # Handle both dict and User object formats
        if hasattr(current_user, 'id'):
            user_id = current_user.id
        elif isinstance(current_user, dict):
            user_id = current_user['id']
        else:
            user_id = str(current_user)  # Fallback
            
        # Check cache first for faster response
        cache_key = f"user_videos:{user_id}:{skip}:{limit}"
        cached_result = get_cached_result(cache_key)
        if cached_result:
            return cached_result
            
        # Get videos from uploads directory (since you don't seem to be using the database)
        uploads_dir = "uploads"
        videos = []
        
        # Check if uploads directory exists
        if not os.path.exists(uploads_dir):
            os.makedirs(uploads_dir, exist_ok=True)
            return {"videos": [], "count": 0, "skip": skip, "limit": limit}
        
        # Scan uploads directory for video folders
        try:
            for item in os.listdir(uploads_dir):
                item_path = os.path.join(uploads_dir, item)
                if os.path.isdir(item_path):
                    # Check if this folder has video data
                    info_path = os.path.join(item_path, "info.json")
                    result_path = os.path.join(item_path, "result.json")
                    
                    video_data = None
                    
                    # Try to load from info.json (uploaded videos)
                    if os.path.exists(info_path):
                        try:
                            with open(info_path, "r") as f:
                                info = json.load(f)
                            
                            # Only include videos that belong to the current user
                            if info.get("user_id") == user_id:
                                video_data = {
                                    "id": item,
                                    "upload_id": item,
                                    "title": info.get("title", "Untitled"),
                                    "filename": info.get("filename", ""),
                                    "status": "completed",
                                    "created_at": info.get("created_at", time.time()),
                                    "thumbnail": info.get("thumbnail"),
                                    "duration": info.get("duration", 0),
                                    "url": None  # Local upload
                                }
                        except Exception as e:
                            print(f"Error loading info.json for {item}: {e}")
                            continue
                    
                    # Try to load from result.json (YouTube videos)
                    elif os.path.exists(result_path):
                        try:
                            with open(result_path, "r") as f:
                                result = json.load(f)
                            
                            # Only include videos that belong to the current user
                            if result.get("user_id") == user_id:
                                video_data = {
                                    "id": item,
                                    "upload_id": item,
                                    "title": result.get("title", "Untitled"),
                                    "filename": result.get("title", ""),
                                    "status": "completed",
                                    "created_at": result.get("created_at", time.time()),
                                    "thumbnail": result.get("thumbnail"),
                                    "duration": result.get("duration", 0),
                                    "url": result.get("url")  # YouTube URL
                                }
                        except Exception as e:
                            print(f"Error loading result.json for {item}: {e}")
                            continue
                    
                    if video_data:
                        videos.append(video_data)
        
        except PermissionError as e:
            print(f"Permission error accessing uploads directory: {e}")
            return {"videos": [], "count": 0, "skip": skip, "limit": limit, "error": "Permission denied"}
        
        # Sort by created_at (most recent first)
        videos.sort(key=lambda x: x.get("created_at", 0), reverse=True)
        
        # Apply pagination
        total_count = len(videos)
        videos = videos[skip:skip + limit]
        
        result = {
            "videos": videos,
            "count": len(videos),
            "total": total_count,
            "skip": skip,
            "limit": limit
        }
        
        # Cache the result for 5 minutes
        cache_result(cache_key, result, ttl=300)
        
        return result
        
    except Exception as e:
        print(f"Error in list_user_videos: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving videos: {str(e)}"
        )

@router.post("/{video_id}/generate-subtitles")
async def generate_all_subtitles(
    video_id: str,
    current_user: dict = Depends(get_current_user),
    request: Optional[Dict] = None
):
    """Generate subtitles in all supported languages or specified languages"""
    try:
        # Parse request body if provided
        if request:
            languages = request.get('languages', None)
        else:
            languages = None
        
        # Check if video exists
        result_path = f"uploads/{video_id}/result.json"
        info_path = f"uploads/{video_id}/info.json"
        
        if not os.path.exists(result_path) and not os.path.exists(info_path):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Video not found"
            )
        
        # Get transcript segments from the appropriate source
        original_segments = []
        
        # Try to load from result.json first (YouTube videos or completed processing)
        if os.path.exists(result_path):
            with open(result_path, "r") as f:
                result_data = json.load(f)
                
            if "transcript" in result_data and "segments" in result_data["transcript"]:
                original_segments = result_data["transcript"]["segments"]
        
        # If no segments found, try loading from subtitles/en.json
        if not original_segments:
            en_subtitle_path = f"uploads/{video_id}/subtitles/en.json"
            if os.path.exists(en_subtitle_path):
                with open(en_subtitle_path, "r") as f:
                    en_data = json.load(f)
                    original_segments = en_data.get("segments", [])
        
        if not original_segments:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No transcript data found for subtitle generation"
            )
        
        # Determine which languages to generate
        if languages is None:
            target_languages = list(get_supported_languages().keys())
        else:
            # Validate requested languages
            invalid_languages = [lang for lang in languages if lang not in get_supported_languages()]
            if invalid_languages:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Unsupported languages: {invalid_languages}"
                )
            target_languages = languages
        
        # Generate subtitles
        subtitle_results = {}
        for lang in target_languages:
            translated_segments = []
            for segment in original_segments:
                translated_text = await translate_text(segment["text"], lang)
                translated_segments.append({
                    "start": segment["start"],
                    "end": segment["end"],
                    "text": translated_text
                })
            
            subtitle_results[lang] = {
                "segments": translated_segments,
                "language": lang
            }
        
        # Save generated subtitles
        for lang, data in subtitle_results.items():
            subtitle_path = f"uploads/{video_id}/subtitles/{lang}.json"
            os.makedirs(os.path.dirname(subtitle_path), exist_ok=True)
            with open(subtitle_path, "w") as f:
                json.dump(data, f, ensure_ascii=False)
        
        return {
            "video_id": video_id,
            "generated_languages": list(subtitle_results.keys()),
            "total_generated": len(subtitle_results),
            "supported_languages": get_supported_languages(),
            "status": "completed"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error generating subtitles: {str(e)}"
        )

@router.get("/{video_id}/subtitle-stats")
async def get_subtitle_statistics(
    video_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Get statistics about available subtitles for a video"""
    try:
        # Get available subtitle languages directly instead of using multilingual.py
        subtitles_dir = f"uploads/{video_id}/subtitles"
        available_languages = []
        
        if os.path.exists(subtitles_dir):
            try:
                for file in os.listdir(subtitles_dir):
                    if file.endswith('.json'):
                        lang_code = file.replace('.json', '')
                        if lang_code in get_supported_languages():
                            available_languages.append(lang_code)
            except Exception as e:
                print(f"Error reading subtitles directory: {e}")
        
        return {
            "video_id": video_id,
            "available_languages": sorted(available_languages),
            "total_languages": len(available_languages),
            "supported_languages": get_supported_languages(),
            "missing_languages": [lang for lang in get_supported_languages().keys() if lang not in available_languages]
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving subtitle statistics: {str(e)}"
        )

@router.post("/{video_id}/translate")
async def translate_video_content(
    video_id: str,
    target_language: str,
    current_user: dict = Depends(get_current_user)
):
    """Translate both summary and subtitles to target language (Chinese or Korean)"""
    try:
        # Validate target language
        supported_langs = get_supported_languages()
        if target_language not in supported_langs:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unsupported language: {target_language}. Supported: {list(supported_langs.keys())}"
            )
        
        # Only allow Chinese and Korean for now
        if target_language not in ['zh', 'ko']:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Currently only Chinese (zh) and Korean (ko) translation are supported"
            )
        
        result_path = f"uploads/{video_id}/result.json"
        if not os.path.exists(result_path):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Video not found"
            )
        
        # Load video data
        with open(result_path, "r") as f:
            result_data = json.load(f)
        
        # Check if translation already exists
        if "translations" in result_data and target_language in result_data["translations"]:
            return {
                "video_id": video_id,
                "target_language": target_language,
                "status": "already_exists",
                "message": f"Translation to {supported_langs[target_language]} already exists"
            }
        
        # Initialize translations structure
        if "translations" not in result_data:
            result_data["translations"] = {}
        if target_language not in result_data["translations"]:
            result_data["translations"][target_language] = {}
        
        # Translate summary
        en_summary = result_data.get("summary", {}).get("en")
        if en_summary:
            translated_summary = translate_summary(en_summary, target_language, video_id)
            result_data["translations"][target_language]["summary"] = translated_summary
        
        # Translate subtitles
        if "transcript" in result_data and "segments" in result_data["transcript"]:
            en_segments = result_data["transcript"]["segments"]
            translated_segments = translate_subtitle_segments(en_segments, target_language, video_id)
            result_data["translations"][target_language]["transcript"] = translated_segments
        
        # Save updated result
        with open(result_path, "w") as f:
            json.dump(result_data, f, ensure_ascii=False, indent=2)
        
        # Also save individual subtitle file
        subtitle_path = f"uploads/{video_id}/subtitles/{target_language}.json"
        os.makedirs(os.path.dirname(subtitle_path), exist_ok=True)
        
        if target_language in result_data["translations"] and "transcript" in result_data["translations"][target_language]:
            subtitle_data = {
                "segments": result_data["translations"][target_language]["transcript"],
                "language": target_language
            }
            with open(subtitle_path, "w") as f:
                json.dump(subtitle_data, f, ensure_ascii=False, indent=2)
        
        return {
            "video_id": video_id,
            "target_language": target_language,
            "language_name": supported_langs[target_language],
            "status": "completed",
            "summary_translated": bool(en_summary),
            "subtitles_translated": "transcript" in result_data,
            "message": f"Successfully translated to {supported_langs[target_language]}"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Translation failed: {str(e)}"
        )

@router.get("/{video_id}/translations")
async def get_available_translations(
    video_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Get list of available translations for a video"""
    try:
        result_path = f"uploads/{video_id}/result.json"
        if not os.path.exists(result_path):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Video not found"
            )
        
        with open(result_path, "r") as f:
            result_data = json.load(f)
        
        supported_langs = get_supported_languages()
        available_translations = []
        
        # Always include English as original
        available_translations.append({
            "code": "en",
            "name": "English",
            "flag": "🇺🇸",
            "is_original": True
        })
        
        # Add available translations
        if "translations" in result_data:
            for lang_code in result_data["translations"].keys():
                if lang_code in supported_langs:
                    flag_map = {
                        'zh': '🇨🇳',
                        'ko': '🇰🇷',
                        'es': '🇪🇸',
                        'fr': '🇫🇷',
                        'de': '🇩🇪',
                        'ja': '🇯🇵',
                        'ru': '🇷🇺',
                        'ar': '🇸🇦',
                        'hi': '🇮🇳'
                    }
                    available_translations.append({
                        "code": lang_code,
                        "name": supported_langs[lang_code],
                        "flag": flag_map.get(lang_code, "🌐"),
                        "is_original": False
                    })
        
        return {
            "video_id": video_id,
            "available_translations": available_translations,
            "supported_for_new_translation": [
                {"code": "zh", "name": "Chinese", "flag": "🇨🇳"},
                {"code": "ko", "name": "Korean", "flag": "🇰🇷"}
            ]
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting translations: {str(e)}"
        )

@router.delete("/{video_id}")
async def delete_video(
    video_id: str,
    current_user = Depends(get_current_user)
):
    """Delete a video and all its associated files"""
    try:
        import shutil
        
        # Check if video directory exists
        video_dir = f"uploads/{video_id}"
        if not os.path.exists(video_dir):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Video not found"
            )
        
        # Verify this is a valid video directory by checking for info.json or result.json
        info_path = os.path.join(video_dir, "info.json")
        result_path = os.path.join(video_dir, "result.json")
        
        if not os.path.exists(info_path) and not os.path.exists(result_path):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Invalid video directory - no video data found"
            )
        
        # Delete the entire video directory
        try:
            shutil.rmtree(video_dir)
            
            # Clear any cached data for this video
            from utils.cache import clear_cache
            cache_patterns = [
                f"video_{video_id}_*",
                f"subtitles_{video_id}_*", 
                f"summary_{video_id}_*",
                f"user_videos:*"  # Clear user video list cache
            ]
            
            for pattern in cache_patterns:
                try:
                    clear_cache(pattern)
                except:
                    pass  # Ignore cache clearing errors
            
            return {
                "message": "Video deleted successfully",
                "video_id": video_id,
                "status": "deleted"
            }
            
        except PermissionError as e:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Permission denied: Unable to delete video files. {str(e)}"
            )
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error deleting video files: {str(e)}"
            )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error deleting video: {str(e)}"
        )

@router.get("/{video_id}/status")
async def get_video_status(
    video_id: str,
    current_user = Depends(get_current_user)
):
    """Get processing status of a video"""
    try:
        # Check if video directory exists
        video_dir = f"uploads/{video_id}"
        if not os.path.exists(video_dir):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Video not found"
            )
        
        # Check for result.json (completed processing)
        result_path = os.path.join(video_dir, "result.json")
        if os.path.exists(result_path):
            return {
                "status": "completed",
                "progress": 100,
                "message": "Video processing completed successfully",
                "video_id": video_id
            }
        
        # Check for info.json (video uploaded, processing may be in progress)
        info_path = os.path.join(video_dir, "info.json")
        if os.path.exists(info_path):
            # Check if there's a processing status file
            status_path = os.path.join(video_dir, "status.json")
            if os.path.exists(status_path):
                with open(status_path, "r") as f:
                    status_data = json.load(f)
                return status_data
            else:
                # If info.json exists but no status.json and no result.json,
                # assume processing is still in progress
                return {
                    "status": "processing",
                    "progress": 50,
                    "message": "Video processing in progress...",
                    "video_id": video_id
                }
        
        # If neither file exists, video not found
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Video not found"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error checking video status: {str(e)}"
        )