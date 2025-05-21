import os
import json
import time
from typing import Any, Dict, Optional, Union
import redis
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get Redis connection details from environment
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
CACHE_TTL = int(os.getenv("CACHE_TTL", "86400"))  # 24 hours by default

# Redis client instance
redis_client = None

def get_redis_client():
    """Get or initialize Redis client"""
    global redis_client
    
    if redis_client is None:
        try:
            redis_client = redis.from_url(REDIS_URL)
            redis_client.ping()  # Check connection
        except redis.ConnectionError:
            print(f"Warning: Could not connect to Redis at {REDIS_URL}. Using fallback file cache.")
            redis_client = None
    
    return redis_client

def get_cached_result(key: str) -> Optional[Dict[str, Any]]:
    """
    Get a cached result if it exists
    
    Args:
        key: Cache key to look up
        
    Returns:
        Cached data if found, None otherwise
    """
    # Try Redis cache first
    client = get_redis_client()
    if client:
        try:
            data = client.get(key)
            if data:
                return json.loads(data)
        except Exception as e:
            print(f"Redis cache error: {str(e)}")
    
    # Fallback to file cache
    cache_dir = "cache"
    os.makedirs(cache_dir, exist_ok=True)
    
    cache_file = os.path.join(cache_dir, f"{key.replace(':', '_')}.json")
    
    if os.path.exists(cache_file):
        try:
            # Check if cache is still valid
            file_time = os.path.getmtime(cache_file)
            if time.time() - file_time < CACHE_TTL:
                with open(cache_file, 'r') as f:
                    return json.load(f)
        except Exception as e:
            print(f"File cache error: {str(e)}")
    
    return None

def cache_result(key: str, data: Dict[str, Any], ttl: int = None) -> bool:
    """
    Store data in cache
    
    Args:
        key: Cache key
        data: Data to cache
        ttl: Time to live in seconds
        
    Returns:
        True if successful, False otherwise
    """
    if ttl is None:
        ttl = CACHE_TTL
    
    # Try Redis cache first
    client = get_redis_client()
    if client:
        try:
            serialized_data = json.dumps(data)
            client.setex(key, ttl, serialized_data)
            return True
        except Exception as e:
            print(f"Redis cache error: {str(e)}")
    
    # Fallback to file cache
    try:
        cache_dir = "cache"
        os.makedirs(cache_dir, exist_ok=True)
        
        cache_file = os.path.join(cache_dir, f"{key.replace(':', '_')}.json")
        
        with open(cache_file, 'w') as f:
            json.dump(data, f)
        return True
    except Exception as e:
        print(f"File cache error: {str(e)}")
        return False

def clear_cache(key_pattern: str = None) -> bool:
    """
    Clear cache entries
    
    Args:
        key_pattern: Optional pattern to match keys to delete
        
    Returns:
        True if successful, False otherwise
    """
    success = True
    
    # Try Redis cache first
    client = get_redis_client()
    if client:
        try:
            if key_pattern:
                keys = client.keys(key_pattern)
                if keys:
                    client.delete(*keys)
            else:
                client.flushdb()
        except Exception as e:
            print(f"Redis cache error: {str(e)}")
            success = False
    
    # Fallback to file cache
    try:
        cache_dir = "cache"
        if os.path.exists(cache_dir):
            if key_pattern:
                import fnmatch
                file_pattern = f"{key_pattern.replace(':', '_').replace('*', '_*')}.json"
                for file in os.listdir(cache_dir):
                    if fnmatch.fnmatch(file, file_pattern):
                        os.remove(os.path.join(cache_dir, file))
            else:
                import shutil
                shutil.rmtree(cache_dir)
                os.makedirs(cache_dir, exist_ok=True)
    except Exception as e:
        print(f"File cache error: {str(e)}")
        success = False
    
    return success

def update_processing_status(
    upload_id: str,
    status: str,
    progress: Union[int, float] = 0,
    message: str = "",
    error: str = None
) -> bool:
    """
    Update the processing status for a video upload
    
    Args:
        upload_id: Unique identifier for the upload
        status: Current status (uploading, processing, completed, failed, error)
        progress: Progress percentage (0-100)
        message: Status message to display
        error: Error message if status is 'error' or 'failed'
        
    Returns:
        True if status was updated successfully, False otherwise
    """
    try:
        status_data = {
            "status": status,
            "progress": float(progress),
            "message": message,
            "timestamp": time.time(),
            "error": error
        }
        
        # Save to status file in uploads directory
        upload_dir = os.path.join("uploads", upload_id)
        os.makedirs(upload_dir, exist_ok=True)
        status_file = os.path.join(upload_dir, "status.json")
        
        with open(status_file, 'w') as f:
            json.dump(status_data, f)
        
        # Also cache in Redis for faster access
        cache_key = f"processing_status:{upload_id}"
        cache_result(cache_key, status_data, ttl=3600)  # Cache for 1 hour
        
        return True
        
    except Exception as e:
        print(f"Error updating processing status: {str(e)}")
        return False

def get_processing_status(upload_id: str) -> Optional[Dict[str, Any]]:
    """
    Get the current processing status for a video upload
    
    Args:
        upload_id: Unique identifier for the upload
        
    Returns:
        Status data if found, None otherwise
    """
    try:
        # Try Redis cache first
        cache_key = f"processing_status:{upload_id}"
        cached_status = get_cached_result(cache_key)
        if cached_status:
            return cached_status
        
        # Fallback to status file
        status_file = os.path.join("uploads", upload_id, "status.json")
        if os.path.exists(status_file):
            with open(status_file, 'r') as f:
                return json.load(f)
                
    except Exception as e:
        print(f"Error getting processing status: {str(e)}")
    
    return None