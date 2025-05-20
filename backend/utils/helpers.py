import os
import ffmpeg
import subprocess
from typing import Dict, Any, Tuple, Optional
import shutil
import tempfile

def extract_audio_from_video(video_path: str, output_path: Optional[str] = None, 
                            sample_rate: int = 16000) -> str:
    """
    Extract audio from a video file using ffmpeg
    
    Args:
        video_path: Path to the video file
        output_path: Path to save the extracted audio (optional)
        sample_rate: Audio sample rate (default: 16kHz for speech recognition)
        
    Returns:
        Path to the extracted audio file
    """
    if output_path is None:
        # Create output path based on input path
        dirname = os.path.dirname(video_path)
        basename = os.path.splitext(os.path.basename(video_path))[0]
        output_path = os.path.join(dirname, f"{basename}_audio.wav")
    
    try:
        # Extract audio using ffmpeg
        (
            ffmpeg
            .input(video_path)
            .output(
                output_path,
                acodec='pcm_s16le',  # 16-bit PCM WAV format
                ac=1,                # Mono channel
                ar=sample_rate       # Sample rate (16kHz ideal for speech)
            )
            .run(quiet=True, overwrite_output=True)
        )
        
        return output_path
    
    except ffmpeg.Error as e:
        print(f"Error extracting audio: {e.stderr.decode() if e.stderr else str(e)}")
        raise

def get_video_metadata(video_path: str) -> Dict[str, Any]:
    """
    Get metadata from a video file using ffprobe
    
    Args:
        video_path: Path to the video file
        
    Returns:
        Dictionary containing video metadata
    """
    try:
        probe = ffmpeg.probe(video_path)
        
        # Extract video and audio stream info
        video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
        audio_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'audio'), None)
        
        # Format duration
        duration = float(probe['format']['duration'])
        mins, secs = divmod(duration, 60)
        hours, mins = divmod(mins, 60)
        
        # Build metadata object
        metadata = {
            "filename": os.path.basename(video_path),
            "duration": duration,
            "duration_formatted": f"{int(hours):02d}:{int(mins):02d}:{int(secs):02d}",
            "format": probe['format']['format_name'],
            "size_bytes": int(probe['format']['size']),
            "size_mb": round(int(probe['format']['size']) / (1024 * 1024), 2)
        }
        
        # Add video stream info if available
        if video_stream:
            metadata["video"] = {
                "codec": video_stream['codec_name'],
                "resolution": f"{video_stream.get('width', 'N/A')}x{video_stream.get('height', 'N/A')}",
                "fps": eval(video_stream.get('avg_frame_rate', '0/1')) if '/' in video_stream.get('avg_frame_rate', '0/1') else 0
            }
        
        # Add audio stream info if available
        if audio_stream:
            metadata["audio"] = {
                "codec": audio_stream['codec_name'],
                "channels": audio_stream.get('channels', 'N/A'),
                "sample_rate": audio_stream.get('sample_rate', 'N/A')
            }
        
        return metadata
        
    except ffmpeg.Error as e:
        print(f"Error getting video metadata: {e.stderr.decode() if e.stderr else str(e)}")
        raise

def process_youtube_url(url: str, output_dir: Optional[str] = None, 
                       format: str = "bestaudio/best") -> Tuple[str, Dict[str, Any]]:
    """
    Download audio from a YouTube URL using yt-dlp
    
    Args:
        url: YouTube URL
        output_dir: Directory to save the downloaded audio (optional)
        format: yt-dlp format string
        
    Returns:
        Tuple of (file_path, metadata)
    """
    if output_dir is None:
        output_dir = tempfile.mkdtemp()
    else:
        os.makedirs(output_dir, exist_ok=True)
    
    output_template = os.path.join(output_dir, "%(id)s.%(ext)s")
    
    try:
        # Use subprocess to call yt-dlp
        cmd = [
            "yt-dlp",
            "-f", format,
            "-o", output_template,
            "--print", "filename",  # Print the output filename
            "--no-playlist",
            url
        ]
        
        # Add audio extraction if downloading video
        if "audio" in format:
            cmd.extend([
                "--extract-audio",
                "--audio-format", "wav",
                "--audio-quality", "0"
            ])
        
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        file_path = result.stdout.strip()
        
        # Get metadata about the downloaded file
        metadata = get_video_metadata(file_path) if os.path.exists(file_path) else {}
        
        return file_path, metadata
        
    except subprocess.CalledProcessError as e:
        print(f"Error downloading YouTube video: {e.stderr}")
        raise
    except Exception as e:
        print(f"Unexpected error processing YouTube URL: {str(e)}")
        raise

def clean_temporary_files(directory: str, exclude_patterns: list = None) -> bool:
    """
    Clean up temporary files in a directory
    
    Args:
        directory: Directory containing files to clean
        exclude_patterns: List of filename patterns to exclude from cleanup
        
    Returns:
        True if successful, False otherwise
    """
    try:
        if not os.path.exists(directory):
            return True
        
        if exclude_patterns is None:
            exclude_patterns = []
        
        import fnmatch
        
        for filename in os.listdir(directory):
            filepath = os.path.join(directory, filename)
            
            # Skip directories and excluded files
            if os.path.isdir(filepath):
                continue
                
            if any(fnmatch.fnmatch(filename, pattern) for pattern in exclude_patterns):
                continue
            
            # Remove the file
            os.remove(filepath)
        
        return True
        
    except Exception as e:
        print(f"Error cleaning temporary files: {str(e)}")
        return False