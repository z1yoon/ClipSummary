import asyncio
from typing import List, Dict, Any
import os
import json
from ai.translator import translate_text
from utils.cache import cache_result, get_cached_result
import logging
import time

logger = logging.getLogger(__name__)

# Supported languages with their codes and names
SUPPORTED_LANGUAGES = {
    "en": "English",
    "ko": "Korean", 
    "zh": "Chinese",
    "ja": "Japanese",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "it": "Italian",
    "pt": "Portuguese",
    "ru": "Russian",
    "ar": "Arabic",
    "hi": "Hindi",
    "th": "Thai",
    "vi": "Vietnamese"
}

async def generate_multilingual_subtitles(video_id: str, original_segments: List[Dict], source_language: str = "en") -> Dict[str, Any]:
    """
    Generate subtitles in multiple languages from the original transcript
    
    Args:
        video_id: The video identifier
        original_segments: List of transcript segments with start, end, text
        source_language: Source language code (default: "en")
    
    Returns:
        Dictionary with language codes as keys and subtitle data as values
    """
    results = {}
    video_dir = f"uploads/{video_id}"
    subtitles_dir = f"{video_dir}/subtitles"
    
    # Create subtitles directory if it doesn't exist
    os.makedirs(subtitles_dir, exist_ok=True)
    
    # Save original language subtitles
    original_subtitles = {
        "segments": original_segments,
        "language": source_language,
        "total_segments": len(original_segments)
    }
    
    original_path = f"{subtitles_dir}/{source_language}.json"
    with open(original_path, "w", encoding="utf-8") as f:
        json.dump(original_subtitles, f, ensure_ascii=False, indent=2)
    
    results[source_language] = original_subtitles
    logger.info(f"Saved original {SUPPORTED_LANGUAGES.get(source_language, source_language)} subtitles for video {video_id}")
    
    # Generate translations for all supported languages
    for lang_code, lang_name in SUPPORTED_LANGUAGES.items():
        if lang_code == source_language:
            continue
            
        subtitle_path = f"{subtitles_dir}/{lang_code}.json"
        
        # Check if translation already exists
        if os.path.exists(subtitle_path):
            try:
                with open(subtitle_path, "r", encoding="utf-8") as f:
                    existing_subtitles = json.load(f)
                results[lang_code] = existing_subtitles
                logger.info(f"Using existing {lang_name} subtitles for video {video_id}")
                continue
            except Exception as e:
                logger.warning(f"Error loading existing {lang_name} subtitles: {e}")
        
        # Check cache first
        cache_key = f"subtitles_{video_id}_{lang_code}"
        cached_result = get_cached_result(cache_key)
        if cached_result:
            results[lang_code] = cached_result
            # Save cached result to file
            with open(subtitle_path, "w", encoding="utf-8") as f:
                json.dump(cached_result, f, ensure_ascii=False, indent=2)
            logger.info(f"Using cached {lang_name} subtitles for video {video_id}")
            continue
        
        try:
            logger.info(f"Generating {lang_name} subtitles for video {video_id}...")
            
            # Translate segments in batches for better performance
            translated_segments = []
            batch_size = 5  # Smaller batch size to avoid overwhelming translation service
            
            for i in range(0, len(original_segments), batch_size):
                batch = original_segments[i:i + batch_size]
                
                # Translate batch
                for segment in batch:
                    try:
                        translated_text = await translate_text(segment["text"], lang_code)
                        translated_segments.append({
                            "start": segment["start"],
                            "end": segment["end"], 
                            "text": translated_text,
                            "speaker": segment.get("speaker")  # Preserve speaker info if available
                        })
                    except Exception as e:
                        logger.warning(f"Error translating segment to {lang_name}: {e}")
                        # Use original text as fallback
                        translated_segments.append({
                            "start": segment["start"],
                            "end": segment["end"],
                            "text": segment["text"],
                            "speaker": segment.get("speaker")
                        })
                
                # Small delay to avoid overwhelming the translation service
                await asyncio.sleep(0.2)
                
                # Log progress
                progress = min(len(translated_segments), len(original_segments))
                logger.info(f"Translated {progress}/{len(original_segments)} segments for {lang_name}")
            
            translated_subtitles = {
                "segments": translated_segments,
                "language": lang_code,
                "total_segments": len(translated_segments),
                "source_language": source_language,
                "generated_at": time.time()
            }
            
            # Save to file
            with open(subtitle_path, "w", encoding="utf-8") as f:
                json.dump(translated_subtitles, f, ensure_ascii=False, indent=2)
            
            # Cache the result
            cache_result(cache_key, translated_subtitles, ttl=86400)  # Cache for 24 hours
            
            results[lang_code] = translated_subtitles
            logger.info(f"Generated {lang_name} subtitles: {len(translated_segments)} segments")
            
        except Exception as e:
            logger.error(f"Failed to generate {lang_name} subtitles: {e}")
            # Continue with other languages even if one fails
            continue
    
    logger.info(f"Multi-language subtitle generation completed. Generated {len(results)} languages: {list(results.keys())}")
    return results

async def generate_multilingual_summary(video_id: str, original_summary: str, source_language: str = "en") -> Dict[str, str]:
    """
    Generate summary in multiple languages
    
    Args:
        video_id: The video identifier
        original_summary: Original summary text
        source_language: Source language code (default: "en")
    
    Returns:
        Dictionary with language codes as keys and translated summaries as values
    """
    results = {source_language: original_summary}
    
    for lang_code, lang_name in SUPPORTED_LANGUAGES.items():
        if lang_code == source_language:
            continue
            
        # Check cache first
        cache_key = f"summary_{video_id}_{lang_code}"
        cached_result = get_cached_result(cache_key)
        if cached_result:
            results[lang_code] = cached_result
            logger.info(f"Using cached {lang_name} summary for video {video_id}")
            continue
        
        try:
            logger.info(f"Generating {lang_name} summary for video {video_id}...")
            translated_summary = await translate_text(original_summary, lang_code)
            
            results[lang_code] = translated_summary
            
            # Cache the result
            cache_result(cache_key, translated_summary, ttl=86400)  # Cache for 24 hours
            
            logger.info(f"Generated {lang_name} summary")
            
        except Exception as e:
            logger.error(f"Failed to generate {lang_name} summary: {e}")
            continue
    
    return results

def get_available_subtitle_languages(video_id: str) -> List[str]:
    """Get list of available subtitle languages for a video"""
    subtitles_dir = f"uploads/{video_id}/subtitles"
    
    if not os.path.exists(subtitles_dir):
        return []
    
    languages = []
    try:
        for file in os.listdir(subtitles_dir):
            if file.endswith('.json'):
                lang_code = file.replace('.json', '')
                if lang_code in SUPPORTED_LANGUAGES:
                    languages.append(lang_code)
    except Exception as e:
        logger.error(f"Error reading subtitles directory: {e}")
    
    return sorted(languages)

def get_subtitle_stats(video_id: str) -> Dict[str, Any]:
    """Get statistics about available subtitles"""
    available_languages = get_available_subtitle_languages(video_id)
    
    return {
        "video_id": video_id,
        "available_languages": available_languages,
        "total_languages": len(available_languages),
        "supported_languages": SUPPORTED_LANGUAGES,
        "missing_languages": [lang for lang in SUPPORTED_LANGUAGES.keys() if lang not in available_languages]
    }