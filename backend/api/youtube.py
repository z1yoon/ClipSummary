from fastapi import APIRouter, HTTPException, status, BackgroundTasks
from pydantic import BaseModel, HttpUrl
import os
from typing import Optional
import yt_dlp
import re
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from dotenv import load_dotenv

# Change relative imports to absolute imports
from ai.whisperx import transcribe_audio
from ai.summarizer import generate_summary
from ai.translator import translate_text
from utils.cache import get_cached_result, cache_result

# Load environment variables
load_dotenv()
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")

router = APIRouter()

class YouTubeRequest(BaseModel):
    url: HttpUrl
    languages: list[str] = ["en"]
    summary_length: Optional[int] = 3  # Number of sentences

# Utility to extract YouTube video ID
def extract_video_id(url: str) -> str:
    patterns = [
        r'(?:youtube\.com\/watch\?v=|youtu\.be\/)([\w-]+)',
        r'youtube\.com\/embed\/([\w-]+)',
        r'youtube\.com\/v\/([\w-]+)',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    
    raise ValueError("Invalid YouTube URL")

@router.post("/process")
async def process_youtube_video(request: YouTubeRequest, background_tasks: BackgroundTasks):
    """Process a YouTube video to generate a transcript, summary, and translations."""
    try:
        # Extract video ID and generate a cache key
        video_id = extract_video_id(str(request.url))
        cache_key = f"youtube:{video_id}:{','.join(request.languages)}:{request.summary_length}"
        
        # Check if we have cached results
        cached = get_cached_result(cache_key)
        if cached:
            return cached
        
        # Fetch video information
        youtube = build('youtube', 'v3', developerKey=YOUTUBE_API_KEY)
        video_response = youtube.videos().list(
            part='snippet,contentDetails',
            id=video_id
        ).execute()
        
        if not video_response['items']:
            raise HTTPException(status_code=404, detail="YouTube video not found")
        
        video_info = video_response['items'][0]['snippet']
        
        # Download audio in background
        background_tasks.add_task(
            process_youtube_audio,
            video_id=video_id,
            title=video_info['title'],
            url=str(request.url),
            languages=request.languages,
            summary_length=request.summary_length,
            cache_key=cache_key
        )
        
        return {
            "status": "processing",
            "video_id": video_id,
            "title": video_info['title'],
            "thumbnail": video_info['thumbnails']['high']['url'],
            "message": "The video is being processed. Check status with the provided video_id."
        }
        
    except HttpError as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
                           detail=f"YouTube API error: {str(e)}")
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, 
                           detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
                           detail=f"An unexpected error occurred: {str(e)}")

@router.get("/status/{video_id}")
async def get_processing_status(video_id: str):
    """Check the status of video processing."""
    # This would typically check a database or cache for the status
    # For simplicity, we'll just check if the file exists
    result_file = f"uploads/{video_id}/result.json"
    if os.path.exists(result_file):
        return {
            "status": "completed",
            "video_id": video_id,
            "result_url": f"/api/youtube/result/{video_id}"
        }
    else:
        return {
            "status": "processing",
            "video_id": video_id,
            "message": "The video is still being processed."
        }

@router.get("/result/{video_id}")
async def get_processing_result(video_id: str):
    """Get the result of video processing."""
    # This would typically fetch from a database or cache
    # For simplicity, we'll just check if the file exists
    result_file = f"uploads/{video_id}/result.json"
    if os.path.exists(result_file):
        import json
        with open(result_file, 'r') as f:
            result = json.load(f)
        return result
    else:
        raise HTTPException(status_code=404, detail="Result not found. The video may still be processing.")

async def process_youtube_audio(video_id: str, title: str, url: str, 
                               languages: list[str], summary_length: int, cache_key: str):
    """Background task to process YouTube audio"""
    try:
        # Create directory for this video
        os.makedirs(f"uploads/{video_id}", exist_ok=True)
        
        # Download audio using yt-dlp
        ydl_opts = {
            'format': 'bestaudio/best',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'wav',
                'preferredquality': '192',
            }],
            'outtmpl': f'uploads/{video_id}/audio.%(ext)s',
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        
        # Path to the downloaded audio
        audio_path = f"uploads/{video_id}/audio.wav"
        
        # Transcribe with WhisperX
        transcript = transcribe_audio(audio_path)
        
        # Generate summary (English)
        summary = generate_summary(' '.join([segment['text'] for segment in transcript['segments']]), 
                                  max_sentences=summary_length)
        
        # Prepare results with the original language (English)
        result = {
            "video_id": video_id,
            "title": title,
            "url": url,
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
        with open(f"uploads/{video_id}/result.json", 'w') as f:
            json.dump(result, f)
        
        # Cache the result
        cache_result(cache_key, result)
        
    except Exception as e:
        # Log the error
        with open(f"uploads/{video_id}/error.log", 'w') as f:
            f.write(str(e))
        
        # You might want to update a status in the database here