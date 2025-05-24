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
        
        # Generate UNIQUE processing ID for this request (not the YouTube video ID)
        processing_id = str(uuid.uuid4())
        
        # Generate cache key using YouTube video ID (for caching across requests)
        cache_key = f"youtube:{youtube_video_id}:{','.join(request.languages)}:{request.summary_length}"
        
        # Check if we have cached results
        cached = get_cached_result(cache_key)
        if cached:
            # Return cached results with new processing ID
            cached_copy = cached.copy()
            cached_copy["upload_id"] = processing_id
            cached_copy["video_id"] = processing_id
            return cached_copy
        
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
                "user_id": current_user.get('id'),
                "user_name": current_user.get('username'),
                "is_youtube": True
            }, f)
        
        # Download audio in background using the unique processing ID
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
    from utils.cache import cache_result
    from api.upload import update_processing_status
    
    try:
        print(f"Starting YouTube processing for processing ID: {processing_id} (YouTube ID: {youtube_video_id})")
        audio_path = f"uploads/{processing_id}/audio.wav"
        
        # Initialize status tracking using processing_id
        update_processing_status(
            upload_id=processing_id,
            status="processing",
            progress=5,
            message="Starting YouTube video download..."
        )
            
        # First update yt-dlp to latest version
        try:
            subprocess.run(["yt-dlp", "--update"], capture_output=True, check=False)
            print("yt-dlp has been updated to the latest version")
        except Exception as e:
            print(f"Error updating yt-dlp: {e}")
        
        download_success = False
        error_messages = []
        
        # Strategy 1: New Enhanced - Try the most reliable approach first with various client types
        if not download_success:
            try:
                print("Strategy 1: Using enhanced reliable approach with multiple client types")
                update_processing_status(
                    upload_id=processing_id,
                    status="processing",
                    progress=15,
                    message="Downloading audio from YouTube..."
                )
                
                # Try different client types
                client_types = ["android", "web", "android,web", "tv_embedded"]
                
                for client in client_types:
                    print(f"Trying with client type: {client}")
                    cmd = [
                        "yt-dlp",
                        "--no-playlist",
                        "--force-ipv4",
                        "--no-warnings",
                        "--geo-bypass", 
                        "--no-check-certificate",
                        "--extractor-args", f"youtube:player_client={client}",
                        "--extractor-retries", "10",
                        "--fragment-retries", "10",
                        "--retry-sleep", "2",
                        "--throttled-rate", "100K",
                        "-x", "--audio-format", "wav",
                        "-o", f"uploads/{processing_id}/audio.%(ext)s",
                        "--user-agent", get_random_user_agent(),
                    ]
                    
                    cmd.append(url)
                    
                    result = subprocess.run(cmd, capture_output=True, text=True)
                    
                    if os.path.exists(audio_path) and os.path.getsize(audio_path) > 10000:
                        download_success = True
                        print(f"Strategy 1 successful with client: {client}")
                        break
                    else:
                        print(f"Failed with client {client}: {result.stderr}")
                
                if not download_success:
                    error_messages.append(f"Strategy 1 failed with all client types")
            except Exception as e:
                error_msg = f"Strategy 1 failed: {str(e)}"
                error_messages.append(error_msg)
                print(error_msg)
        
        # Strategy 2: Try using a specific format with fallbacks
        if not download_success:
            try:
                print("Strategy 2: Using specific format with fallbacks")
                update_processing_status(
                    upload_id=processing_id,
                    status="processing",
                    progress=25,
                    message="Trying alternative download methods..."
                )
                
                # Try different format specifications
                format_specs = [
                    "bestaudio[ext=m4a]",
                    "bestaudio[ext=webm]",
                    "bestaudio",
                    "140",
                    "251",
                    "250",
                    "249"
                ]
                
                for fmt in format_specs:
                    print(f"Trying format: {fmt}")
                    cmd = [
                        "yt-dlp",
                        "-f", fmt,
                        "--force-ipv4",
                        "--no-warnings",
                        "--no-playlist",
                        "--extractor-args", "youtube:player_client=web", 
                        "--throttled-rate", "100K",
                        "-x", "--audio-format", "wav",
                        "-o", f"uploads/{processing_id}/audio.%(ext)s",
                        "--user-agent", get_random_user_agent(),
                    ]
                    
                    cmd.append(url)
                    
                    result = subprocess.run(cmd, capture_output=True, text=True)
                    
                    if os.path.exists(audio_path) and os.path.getsize(audio_path) > 10000:
                        download_success = True
                        print(f"Strategy 2 successful with format {fmt}")
                        break
                    print(f"Format {fmt} download failed or file too small")
            except Exception as e:
                error_msg = f"Strategy 2 failed: {str(e)}"
                error_messages.append(error_msg)
                print(error_msg)
        
        # Strategy 3: Try transcript extraction fallback
        if not download_success:
            try:
                print("Strategy 3: Using transcript API as fallback")
                update_processing_status(
                    upload_id=processing_id,
                    status="processing",
                    progress=30,
                    message="Audio download failed, trying transcript extraction..."
                )
                
                # Try to install youtube_transcript_api if not already installed
                try:
                    import importlib
                    importlib.import_module('youtube_transcript_api')
                except ImportError:
                    print("Installing youtube_transcript_api...")
                    subprocess.run(["pip", "install", "youtube_transcript_api"], check=False)
                
                from youtube_transcript_api import YouTubeTranscriptApi
                
                # Extract pure video ID
                pure_video_id = extract_video_id(url)
                
                transcript_list = YouTubeTranscriptApi.get_transcript(pure_video_id)
                
                if transcript_list:
                    print("Successfully retrieved transcript as fallback")
                    update_processing_status(
                        upload_id=processing_id,
                        status="processing",
                        progress=50,
                        message="Processing transcript data..."
                    )
                    
                    # Create a transcript structure
                    mock_transcript = {
                        "segments": [
                            {
                                "start": item["start"],
                                "end": item["start"] + item.get("duration", 5),
                                "text": item["text"]
                            } for item in transcript_list
                        ],
                        "language": "en"
                    }
                    
                    # Generate summary from transcript directly
                    update_processing_status(
                        upload_id=processing_id,
                        status="processing",
                        progress=70,
                        message="Generating summary from transcript..."
                    )
                    
                    transcript_text = ' '.join([segment['text'] for segment in mock_transcript["segments"]])
                    summary = generate_summary(transcript_text, max_sentences=summary_length, upload_id=processing_id)
                    
                    # Prepare results with the original language (English)
                    result = {
                        "video_id": youtube_video_id,
                        "title": title,
                        "url": url,
                        "transcript": mock_transcript,
                        "summary": {
                            "en": summary
                        },
                        "translations": {},
                        "source": "transcript_api"
                    }
                    
                    # Translate to requested languages
                    update_processing_status(
                        upload_id=processing_id,
                        status="processing",
                        progress=80,
                        message="Translating content..."
                    )
                    
                    for lang in languages:
                        if lang != "en":  # Skip English as it's already done
                            print(f"Translating to {lang}")
                            result["translations"][lang] = {
                                "summary": translate_text(summary, target_lang=lang, upload_id=processing_id),
                                "transcript": [
                                    {
                                        "start": segment["start"],
                                        "end": segment["end"],
                                        "text": translate_text(segment["text"], target_lang=lang, upload_id=processing_id)
                                    }
                                    for segment in mock_transcript["segments"]
                                ]
                            }
                    
                    # Save results
                    with open(f"uploads/{processing_id}/result.json", 'w') as f:
                        json.dump(result, f)
                    
                    # Cache the result
                    cache_result(cache_key, result)
                    
                    update_processing_status(
                        upload_id=processing_id,
                        status="completed",
                        progress=100,
                        message="Processing completed successfully."
                    )
                    
                    print(f"YouTube processing completed via transcript API for {processing_id}")
                    return  # Early return since we're done
                
            except Exception as e:
                error_msg = f"Transcript extraction failed: {str(e)}"
                error_messages.append(error_msg)
                print(error_msg)
        
        # If all strategies failed, raise exception
        if not download_success:
            all_errors = "\n".join(error_messages)
            raise Exception(f"Failed to download or extract audio/transcript from YouTube video. Errors:\n{all_errors}")
        
        # If we reach here, we successfully downloaded the audio
        audio_size = os.path.getsize(audio_path)
        print(f"Audio file size: {audio_size/1024/1024:.2f} MB")
        
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
        
        print(f"Starting transcription for {processing_id}")
        transcript = transcribe_audio(audio_path, upload_id=processing_id)
        
        if not transcript or "segments" not in transcript or not transcript["segments"]:
            raise Exception("Transcription failed: No transcription segments were generated")
        
        print(f"Transcription completed with {len(transcript['segments'])} segments")
        
        # Generate summary (English)
        update_processing_status(
            upload_id=processing_id,
            status="processing",
            progress=70,
            message="Generating summary..."
        )
        
        summary = generate_summary(' '.join([segment['text'] for segment in transcript['segments']]), 
                                  max_sentences=summary_length, upload_id=processing_id)
        
        print(f"Summary generated, length: {len(summary)} characters")
        
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
                    print(f"Translating to {lang}")
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
                    print(f"Translation to {lang} completed")
        
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
        
        print(f"YouTube processing completed successfully for {processing_id}")
        
    except Exception as e:
        error_message = f"Processing failed: {str(e)}"
        print(f"Error processing YouTube video {processing_id}: {error_message}")
        
        # Update status to failed
        update_processing_status(
            upload_id=processing_id,
            status="failed",
            progress=0,
            message=error_message
        )
        
        # Log the error
        with open(f"uploads/{processing_id}/error.log", 'w') as f:
            f.write(error_message)