from fastapi import APIRouter, HTTPException, status, BackgroundTasks, Depends
from pydantic import BaseModel, HttpUrl
import os
import json
import time
import uuid
from typing import Optional, Dict, Any
import yt_dlp
import re
import subprocess
import shutil
import random
import requests
from urllib.parse import parse_qs, urlparse
from pathlib import Path

# Change relative imports to absolute imports
from ai.whisperx import transcribe_audio
from ai.summarizer import generate_summary
from ai.translator import translate_text
from utils.cache import get_cached_result, cache_result
from api.auth import get_current_user

# Setup router
router = APIRouter()

class YouTubeRequest(BaseModel):
    url: HttpUrl
    languages: list[str] = ["en"]
    summary_length: Optional[int] = 3  # Number of sentences

# Utility to extract YouTube video ID
def extract_video_id(url: str) -> str:
    # Enhanced pattern matching for YouTube URLs
    patterns = [
        r'(?:youtube\.com\/watch\?v=|youtu\.be\/)([\w-]+)',
        r'youtube\.com\/embed\/([\w-]+)',
        r'youtube\.com\/v\/([\w-]+)',
        r'youtube\.com\/shorts\/([\w-]+)',
    ]
    
    # Try standard patterns first
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    
    # Try parsing URL parameters as fallback
    try:
        query = parse_qs(urlparse(url).query)
        if 'v' in query:
            return query['v'][0]
    except Exception as e:
        print(f"URL parsing error: {str(e)}")
    
    raise ValueError("Invalid YouTube URL")

# Get a random user agent for rotation
def get_random_user_agent():
    user_agents = [
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15',
        'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/121.0',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:109.0) Gecko/20100101 Firefox/121.0',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Edge/120.0.0.0 Safari/537.36',
    ]
    return random.choice(user_agents)

# Utility to get video info using yt-dlp instead of YouTube API
def get_youtube_video_info(url: str) -> Dict[str, Any]:
    """Get video information using yt-dlp instead of YouTube API"""
    ydl_opts = {
        'skip_download': True,
        'format': 'best',
        'quiet': True,
        'ignoreerrors': False,
        'no_warnings': False,
        'nocheckcertificate': True,
        'geo_bypass': True,
        'extractor_retries': 3,
        'socket_timeout': 30,
        'http_headers': {
            'User-Agent': get_random_user_agent(),
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Charset': 'ISO-8859-1,utf-8;q=0.7,*;q=0.7',
            'Connection': 'keep-alive',
        }
    }
    
    try:
        # First, make sure yt-dlp is up to date
        try:
            subprocess.run(["yt-dlp", "--update"], capture_output=True, check=False)
            print("yt-dlp has been updated to the latest version")
        except Exception as e:
            print(f"Error updating yt-dlp: {e}")
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            print(f"Attempting to extract info from: {url}")
            info = ydl.extract_info(url, download=False)
            
        if not info:
            raise ValueError("Could not retrieve video information")
            
        return {
            'id': info.get('id'),
            'title': info.get('title'),
            'description': info.get('description'),
            'thumbnail': info.get('thumbnail', info.get('thumbnails', [{'url': ''}])[0].get('url') if info.get('thumbnails') else ''),
            'duration': info.get('duration')
        }
    except Exception as e:
        print(f"YouTube info extraction error: {str(e)}")
        # Second attempt with different options if first attempt failed
        try:
            print("Retrying with alternative options...")
            ydl_opts['extractor_args'] = {'youtube': {'player_client': ['web', 'android']}}
            ydl_opts['http_headers']['User-Agent'] = get_random_user_agent()
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=False)
                
            if info:
                return {
                    'id': info.get('id'),
                    'title': info.get('title'),
                    'description': info.get('description'),
                    'thumbnail': info.get('thumbnail', info.get('thumbnails', [{'url': ''}])[0].get('url') if info.get('thumbnails') else ''),
                    'duration': info.get('duration')
                }
        except Exception as retry_error:
            print(f"Retry also failed: {str(retry_error)}")
        
        # If all info extraction attempts fail, return basic information from the URL
        video_id = extract_video_id(url)
        
        # Try fetching basic info from oEmbed API as a last resort
        try:
            oembed_url = f"https://www.youtube.com/oembed?url=https://www.youtube.com/watch?v={video_id}&format=json"
            response = requests.get(oembed_url, timeout=10)
            if response.status_code == 200:
                oembed_data = response.json()
                return {
                    'id': video_id,
                    'title': oembed_data.get('title', f"YouTube Video {video_id}"),
                    'description': "Description unavailable",
                    'thumbnail': oembed_data.get('thumbnail_url', f"https://i.ytimg.com/vi/{video_id}/hqdefault.jpg"),
                    'duration': 0
                }
        except Exception as oembed_error:
            print(f"oEmbed API error: {str(oembed_error)}")
            
        # Fallback to basic information
        return {
            'id': video_id,
            'title': f"YouTube Video {video_id}",
            'description': "Description unavailable",
            'thumbnail': f"https://i.ytimg.com/vi/{video_id}/hqdefault.jpg",
            'duration': 0
        }

@router.post("/process")
async def process_youtube_video(
    request: YouTubeRequest, 
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(get_current_user)
):
    """Process a YouTube video to generate a transcript, summary, and translations."""
    try:
        # Extract YouTube video ID for info and caching
        youtube_video_id = extract_video_id(str(request.url))
        
        # Generate cache key using YouTube video ID (for caching across requests)
        cache_key = f"youtube:{youtube_video_id}:{','.join(request.languages)}:{request.summary_length}"
        
        # Check if we have cached results
        cached = get_cached_result(cache_key)
        if cached:
            print(f"💾 Found cached result for YouTube ID: {youtube_video_id}")
            # Generate a new processing ID for this request
            processing_id = str(uuid.uuid4())
            
            # Create directory and save the cached result immediately
            os.makedirs(f"uploads/{processing_id}", exist_ok=True)
            
            # Save the cached result as a completed result
            cached_copy = cached.copy()
            cached_copy["upload_id"] = processing_id
            cached_copy["video_id"] = processing_id
            
            # Save to result.json immediately
            with open(f"uploads/{processing_id}/result.json", 'w') as f:
                json.dump(cached_copy, f)
            
            # Save info.json
            with open(f"uploads/{processing_id}/info.json", 'w') as f:
                json.dump({
                    "processing_id": processing_id,
                    "youtube_video_id": youtube_video_id,
                    "url": str(request.url),
                    "title": cached_copy.get('title'),
                    "thumbnail": cached_copy.get('thumbnail'),
                    "duration": cached_copy.get('duration'),
                    "languages_requested": request.languages,
                    "summary_length": request.summary_length,
                    "upload_time": time.time(),
                    "user_id": current_user.id,
                    "user_name": current_user.username,
                    "is_youtube": True,
                    "cached_result": True
                }, f)
            
            # Set status to completed immediately
            from utils.cache import update_processing_status
            update_processing_status(
                upload_id=processing_id,
                status="completed",
                progress=100,
                message="Processing completed (from cache)."
            )
            
            # Return with completed status - this tells frontend to go directly to result page
            return {
                "status": "completed",
                "upload_id": processing_id,
                "video_id": processing_id,
                "title": cached_copy.get('title'),
                "thumbnail": cached_copy.get('thumbnail'),
                "message": "Processing completed (cached result).",
                "result_url": f"/api/upload/result/{processing_id}",
                "cached": True  # Flag to indicate this is from cache
            }
        
        # Generate UNIQUE processing ID for this request (not the YouTube video ID)
        processing_id = str(uuid.uuid4())
        
        print(f"🆔 New processing request - Processing ID: {processing_id}, YouTube ID: {youtube_video_id}")
        
        # Get video information using yt-dlp
        video_info = get_youtube_video_info(str(request.url))
        
        # Create directory for this processing request (using unique processing ID)
        os.makedirs(f"uploads/{processing_id}", exist_ok=True)
        
        # Save basic video info with user information
        with open(f"uploads/{processing_id}/info.json", 'w') as f:
            json.dump({
                "processing_id": processing_id,
                "youtube_video_id": youtube_video_id,
                "url": str(request.url),
                "title": video_info.get('title'),
                "thumbnail": video_info.get('thumbnail'),
                "duration": video_info.get('duration'),
                "languages_requested": request.languages,
                "summary_length": request.summary_length,
                "upload_time": time.time(),
                "user_id": current_user.id,
                "user_name": current_user.username,
                "is_youtube": True
            }, f)
        
        print(f"📝 Saved video info to uploads/{processing_id}/info.json")
        
        # Download audio in background using the unique processing ID
        print(f"⚙️ Adding background task for processing ID: {processing_id}")
        background_tasks.add_task(
            process_youtube_audio,
            processing_id=processing_id,
            youtube_video_id=youtube_video_id,
            title=video_info.get('title'),
            url=str(request.url),
            languages=request.languages,
            summary_length=request.summary_length,
            cache_key=cache_key
        )
        
        return {
            "status": "processing",
            "upload_id": processing_id,  # Return processing ID, not YouTube video ID
            "video_id": processing_id,   # For compatibility
            "title": video_info.get('title'),
            "thumbnail": video_info.get('thumbnail'),
            "message": "The video is being processed. Check status with the provided upload_id."
        }
        
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, 
                           detail=str(e))
    except Exception as e:
        print(f"💥 Error in process_youtube_video: {str(e)}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
                           detail=f"An unexpected error occurred: {str(e)}")

@router.get("/status/{video_id}")
async def get_processing_status(video_id: str):
    """Check the status of video processing."""
    # First check for error log
    error_file = f"uploads/{video_id}/error.log"
    if os.path.exists(error_file):
        with open(error_file, 'r') as f:
            error_message = f.read()
        return {
            "status": "failed",
            "video_id": video_id,
            "message": f"Processing failed: {error_message}",
        }
    
    # Check if result file exists
    result_file = f"uploads/{video_id}/result.json"
    if os.path.exists(result_file):
        return {
            "status": "completed",
            "video_id": video_id,
            "result_url": f"/api/youtube/result/{video_id}"
        }
    
    # Check if info file exists to confirm processing started
    info_file = f"uploads/{video_id}/info.json"
    if os.path.exists(info_file):
        return {
            "status": "processing",
            "video_id": video_id,
            "message": "The video is still being processed."
        }
    
    # If no files exist, the video ID is invalid
    raise HTTPException(status_code=404, detail="Video not found. Invalid video ID.")

async def process_youtube_audio(processing_id: str, youtube_video_id: str, title: str, url: str, 
                              languages: list[str], summary_length: int, cache_key: str):
    """Background task to process YouTube audio"""
    # Import at the top to avoid conflicts
    import traceback
    from utils.cache import cache_result, update_processing_status
    from ai.whisperx import transcribe_audio
    from ai.summarizer import generate_summary
    from ai.translator import translate_text
    
    try:
        print(f"🚀 Starting YouTube processing for processing ID: {processing_id} (YouTube ID: {youtube_video_id})")
        
        # Create the uploads directory if it doesn't exist
        os.makedirs(f"uploads/{processing_id}", exist_ok=True)
        audio_path = f"uploads/{processing_id}/audio.wav"
        
        # Initialize status tracking using processing_id
        update_processing_status(
            upload_id=processing_id,
            status="processing",
            progress=5,
            message="Starting YouTube video download..."
        )
        
        print(f"📁 Created directory: uploads/{processing_id}")
        print(f"🎵 Target audio path: {audio_path}")
        
        # Download strategy 1: Direct audio extraction with yt-dlp
        ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': f'uploads/{processing_id}/%(title)s.%(ext)s',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'wav',
                'preferredquality': '192',
            }],
            'quiet': False,
            'no_warnings': False,
            'extractaudio': True,
            'audioformat': 'wav',
        }
        
        update_processing_status(
            upload_id=processing_id,
            status="processing",
            progress=10,
            message="Downloading YouTube video audio..."
        )
        
        print(f"🔽 Starting download with yt-dlp for: {url}")
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            try:
                # Download and extract audio
                ydl.download([url])
                print("✅ yt-dlp download completed")
                
                # Find the downloaded audio file
                upload_dir = Path(f"uploads/{processing_id}")
                audio_files = list(upload_dir.glob("*.wav"))
                
                if audio_files:
                    downloaded_file = audio_files[0]
                    # Move to expected location
                    downloaded_file.rename(audio_path)
                    print(f"📁 Moved audio file to: {audio_path}")
                else:
                    raise Exception("No audio file found after download")
                    
            except Exception as e:
                print(f"❌ yt-dlp strategy failed: {str(e)}")
                raise Exception(f"Audio download failed: {str(e)}")
        
        # Verify the audio file exists and has content
        if not os.path.exists(audio_path):
            raise Exception("Audio file was not created")
            
        audio_size = os.path.getsize(audio_path)
        print(f"📊 Audio file size: {audio_size/1024/1024:.2f} MB")
        
        # Check if the audio file is valid
        if audio_size < 10000:  # Less than 10KB is suspicious
            raise Exception("Downloaded audio file is too small and likely invalid")
        
        # Transcribe with WhisperX
        update_processing_status(
            upload_id=processing_id,
            status="processing",
            progress=40,
            message="Audio downloaded successfully. Starting transcription..."
        )
        
        print(f"🎤 Starting transcription for {processing_id}")
        transcript = transcribe_audio(audio_path, upload_id=processing_id)
        
        if not transcript or "segments" not in transcript or not transcript["segments"]:
            raise Exception("Transcription failed: No transcription segments were generated")
        
        print(f"✅ Transcription completed with {len(transcript['segments'])} segments")
        
        # Generate summary (English)
        update_processing_status(
            upload_id=processing_id,
            status="processing",
            progress=70,
            message="Generating summary..."
        )
        
        summary = generate_summary(' '.join([segment['text'] for segment in transcript['segments']]), 
                                  max_sentences=summary_length, upload_id=processing_id)
        
        print(f"📝 Summary generated, length: {len(summary)} characters")
        
        # Prepare results
        result = {
            "upload_id": processing_id,
            "video_id": processing_id,
            "youtube_video_id": youtube_video_id,
            "title": title,
            "url": url,
            "transcript": transcript,
            "summary": {
                "en": summary
            },
            "translations": {},
            "source": "audio_download"
        }
        
        # Translate to requested languages
        if len([lang for lang in languages if lang != "en"]) > 0:
            update_processing_status(
                upload_id=processing_id,
                status="processing",
                progress=80,
                message="Translating content..."
            )
            
            for lang in languages:
                if lang != "en":  # Skip English as it's already done
                    print(f"🌐 Translating to {lang}")
                    result["translations"][lang] = {
                        "summary": translate_text(summary, target_lang=lang, upload_id=processing_id),
                        "transcript": [
                            {
                                "start": segment["start"],
                                "end": segment["end"],
                                "text": translate_text(segment["text"], target_lang=lang, upload_id=processing_id)
                            }
                            for segment in transcript["segments"]
                        ]
                    }
                    print(f"✅ Translation to {lang} completed")
        
        # Save results
        with open(f"uploads/{processing_id}/result.json", 'w') as f:
            json.dump(result, f)
        
        # Cache the result using YouTube video ID
        cache_result(cache_key, result)
        
        update_processing_status(
            upload_id=processing_id,
            status="completed",
            progress=100,
            message="Processing completed successfully."
        )
        
        print(f"🎉 YouTube processing completed successfully for {processing_id}")
        
    except Exception as e:
        error_message = f"Processing failed: {str(e)}"
        error_traceback = traceback.format_exc()
        print(f"💥 Error processing YouTube video {processing_id}: {error_message}")
        print(f"💥 Full traceback:\n{error_traceback}")
        
        # Update status to failed - use the cache version to avoid conflicts
        try:
            from utils.cache import update_processing_status
            update_processing_status(
                upload_id=processing_id,
                status="failed",
                progress=0,
                message=error_message,
                error=error_message
            )
        except Exception as status_error:
            print(f"💥 Failed to update status: {str(status_error)}")
        
        # Log the error
        try:
            with open(f"uploads/{processing_id}/error.log", 'w') as f:
                f.write(f"{error_message}\n\nFull traceback:\n{error_traceback}")
        except Exception as log_error:
            print(f"💥 Failed to write error log: {str(log_error)}")