from fastapi import APIRouter, HTTPException, status, BackgroundTasks, Depends
from pydantic import BaseModel, HttpUrl
import os
import json
import time
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
        # Extract video ID and generate a cache key
        video_id = extract_video_id(str(request.url))
        cache_key = f"youtube:{video_id}:{','.join(request.languages)}:{request.summary_length}"
        
        # Check if we have cached results
        cached = get_cached_result(cache_key)
        if cached:
            return cached
        
        # Get video information using yt-dlp instead of YouTube API
        video_info = get_youtube_video_info(str(request.url))
        
        # Create directory for this video
        os.makedirs(f"uploads/{video_id}", exist_ok=True)
        
        # Save basic video info with user information
        with open(f"uploads/{video_id}/info.json", 'w') as f:
            json.dump({
                "video_id": video_id,
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
        
        # Download audio in background
        background_tasks.add_task(
            process_youtube_audio,
            video_id=video_id,
            title=video_info.get('title'),
            url=str(request.url),
            languages=request.languages,
            summary_length=request.summary_length,
            cache_key=cache_key
        )
        
        return {
            "status": "processing",
            "video_id": video_id,
            "title": video_info.get('title'),
            "thumbnail": video_info.get('thumbnail'),
            "message": "The video is being processed. Check status with the provided video_id."
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

async def process_youtube_audio(video_id: str, title: str, url: str, 
                              languages: list[str], summary_length: int, cache_key: str):
    """Background task to process YouTube audio"""
    try:
        print(f"Starting YouTube processing for video ID: {video_id}")
        audio_path = f"uploads/{video_id}/audio.wav"
        status_path = f"uploads/{video_id}/status.json"
        
        # Update status
        with open(status_path, 'w') as f:
            json.dump({"status": "downloading", "progress": 0}, f)
            
        # First update yt-dlp to latest version
        try:
            subprocess.run(["yt-dlp", "--update"], capture_output=True, check=False)
            print("yt-dlp has been updated to the latest version")
        except Exception as e:
            print(f"Error updating yt-dlp: {e}")
        
        download_success = False
        error_messages = []
        
        # Choose a different IP address for each request if proxy is available
        proxy_options = []
        try:
            if os.environ.get('USE_PROXIES', 'false').lower() == 'true':
                # This would normally read from a proper proxy configuration
                # Replace with your actual proxy setup
                proxy_options = ["--proxy", os.environ.get('HTTP_PROXY', '')]
        except:
            proxy_options = []
        
        # Strategy 1: New Enhanced - Try the most reliable approach first with various client types
        if not download_success:
            try:
                print("Strategy 1: Using enhanced reliable approach with multiple client types")
                with open(status_path, 'w') as f:
                    json.dump({"status": "downloading", "strategy": "enhanced-reliable", "progress": 15}, f)
                
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
                        "-o", f"uploads/{video_id}/audio.%(ext)s",
                        "--user-agent", get_random_user_agent(),
                    ]
                    
                    # Add proxy if available
                    if proxy_options:
                        cmd.extend(proxy_options)
                        
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
                with open(status_path, 'w') as f:
                    json.dump({"status": "downloading", "strategy": "specific-formats", "progress": 25}, f)
                
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
                        "-o", f"uploads/{video_id}/audio.%(ext)s",
                        "--user-agent", get_random_user_agent(),
                    ]
                    
                    # Add proxy if available
                    if proxy_options:
                        cmd.extend(proxy_options)
                        
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
        
        # Strategy 3: Try bypassing age restriction
        if not download_success:
            try:
                print("Strategy 3: Attempting to bypass age restriction")
                with open(status_path, 'w') as f:
                    json.dump({"status": "downloading", "strategy": "age-bypass", "progress": 30}, f)
                
                cmd = [
                    "yt-dlp",
                    "--no-playlist",
                    "--age-limit", "21",
                    "--cookies-from-browser", "chrome",
                    "--user-agent", get_random_user_agent(),
                    "-x", "--audio-format", "wav",
                    "-o", f"uploads/{video_id}/audio.%(ext)s",
                    url
                ]
                
                result = subprocess.run(cmd, capture_output=True, text=True)
                
                if os.path.exists(audio_path) and os.path.getsize(audio_path) > 10000:
                    download_success = True
                    print("Strategy 3 successful: Age restriction bypass worked")
                else:
                    error_messages.append(f"Strategy 3 failed: {result.stderr}")
                    print(f"Strategy 3 failed: {result.stderr}")
            except Exception as e:
                error_msg = f"Strategy 3 failed: {str(e)}"
                error_messages.append(error_msg)
                print(error_msg)
                
        # Strategy 4: Use youtube-dl as fallback since it might have different request patterns
        if not download_success:
            try:
                print("Strategy 4: Using youtube-dl as fallback")
                with open(status_path, 'w') as f:
                    json.dump({"status": "downloading", "strategy": "youtube-dl-fallback", "progress": 35}, f)
                
                # Check if youtube-dl is installed
                try:
                    subprocess.run(["youtube-dl", "--version"], capture_output=True, check=True)
                except:
                    print("youtube-dl not found, installing it...")
                    subprocess.run(["pip", "install", "--upgrade", "youtube-dl"], check=False)
                
                cmd = [
                    "youtube-dl",
                    "-f", "bestaudio",
                    "--extract-audio",
                    "--audio-format", "wav",
                    "--no-warnings",
                    "--no-check-certificate",
                    "-o", f"uploads/{video_id}/audio.%(ext)s",
                    "--user-agent", get_random_user_agent(),
                ]
                
                # Add proxy if available
                if proxy_options:
                    cmd.extend([proxy_options[0].replace("--proxy", "--proxy-url"), proxy_options[1]])
                    
                cmd.append(url)
                
                result = subprocess.run(cmd, capture_output=True, text=True)
                
                if os.path.exists(audio_path) and os.path.getsize(audio_path) > 10000:
                    download_success = True
                    print("Strategy 4 successful: youtube-dl fallback")
                else:
                    error_messages.append(f"Strategy 4 failed: {result.stderr}")
                    print(f"Strategy 4 failed: {result.stderr}")
            except Exception as e:
                error_msg = f"Strategy 4 failed: {str(e)}"
                error_messages.append(error_msg)
                print(error_msg)
                
        # Strategy 5: Transcript extraction fallback
        if not download_success:
            try:
                print("Strategy 5: Using transcript API as fallback")
                with open(status_path, 'w') as f:
                    json.dump({"status": "using_transcript", "progress": 50}, f)
                
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
                    
                    # Save as JSON for processing
                    with open(f"uploads/{video_id}/transcript.json", "w") as f:
                        json.dump(mock_transcript, f)
                    
                    # Generate summary from transcript directly
                    with open(status_path, 'w') as f:
                        json.dump({"status": "summarizing", "progress": 70}, f)
                    
                    transcript_text = ' '.join([segment['text'] for segment in mock_transcript["segments"]])
                    summary = generate_summary(transcript_text, max_sentences=summary_length)
                    
                    # Prepare results with the original language (English)
                    result = {
                        "video_id": video_id,
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
                    with open(status_path, 'w') as f:
                        json.dump({"status": "translating", "progress": 80}, f)
                    
                    for lang in languages:
                        if lang != "en":  # Skip English as it's already done
                            print(f"Translating to {lang}")
                            result["translations"][lang] = {
                                "summary": translate_text(summary, target_lang=lang),
                                "transcript": [
                                    {
                                        "start": segment["start"],
                                        "end": segment["end"],
                                        "text": translate_text(segment["text"], target_lang=lang)
                                    }
                                    for segment in mock_transcript["segments"]
                                ]
                            }
                    
                    # Save results
                    with open(f"uploads/{video_id}/result.json", 'w') as f:
                        json.dump(result, f)
                    
                    # Cache the result
                    cache_result(cache_key, result)
                    
                    with open(status_path, 'w') as f:
                        json.dump({"status": "completed", "progress": 100}, f)
                    
                    print(f"YouTube processing completed via transcript API for {video_id}")
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
            
        # Enhanced audio validation
        if audio_size > 10000:
            try:
                # Try to read the first few bytes of the file to verify it's valid
                with open(audio_path, 'rb') as f:
                    header = f.read(12)
                if not header or len(header) < 12:
                    raise Exception("Audio file appears to be corrupted (header too small)")
            except Exception as e:
                print(f"Audio validation error: {str(e)}")
                # We'll continue anyway since the file might still be usable
        
        # Transcribe with WhisperX
        with open(status_path, 'w') as f:
            json.dump({"status": "transcribing", "progress": 60}, f)
        
        print(f"Starting transcription for {video_id}")
        transcript = transcribe_audio(audio_path)
        
        if not transcript or "segments" not in transcript or not transcript["segments"]:
            raise Exception("Transcription failed: No transcription segments were generated")
        
        print(f"Transcription completed with {len(transcript['segments'])} segments")
        
        # Generate summary (English)
        with open(status_path, 'w') as f:
            json.dump({"status": "summarizing", "progress": 80}, f)
        
        summary = generate_summary(' '.join([segment['text'] for segment in transcript['segments']]), 
                                  max_sentences=summary_length)
        
        print(f"Summary generated, length: {len(summary)} characters")
        
        # Prepare results with the original language (English)
        result = {
            "video_id": video_id,
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
        with open(status_path, 'w') as f:
            json.dump({"status": "translating", "progress": 90}, f)
        
        for lang in languages:
            if lang != "en":  # Skip English as it's already done
                print(f"Translating to {lang}")
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
                print(f"Translation to {lang} completed")
        
        # Save results
        with open(f"uploads/{video_id}/result.json", 'w') as f:
            json.dump(result, f)
        
        # Cache the result
        cache_result(cache_key, result)
        
        with open(status_path, 'w') as f:
            json.dump({"status": "completed", "progress": 100}, f)
        
        print(f"YouTube processing completed successfully for {video_id}")
        
    except Exception as e:
        error_message = f"Processing failed: {str(e)}"
        print(f"Error processing YouTube video {video_id}: {error_message}")
        # Log the error
        with open(f"uploads/{video_id}/error.log", 'w') as f:
            f.write(error_message)
        
        # Update status to failed
        try:
            with open(status_path, 'w') as f:
                json.dump({"status": "failed", "error": str(e)}, f)
        except:
            pass

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