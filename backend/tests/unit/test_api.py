import pytest
from unittest.mock import patch, MagicMock
import asyncio
import json

class TestAPI:
    """Unit tests for API endpoints."""
    
    def test_health_check(self, client):
        """Test the health check endpoint."""
        response = client.get("/api/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}
    
    @patch('api.youtube.process_youtube_video')
    @patch('utils.cache.update_processing_status')
    def test_youtube_process(self, mock_update_status, mock_process, authenticated_client):
        """Test the YouTube process endpoint."""
        # Mock the background processing function
        mock_upload_id = "test-123456"
        
        # Create a mock for the background task
        def mock_background_task(background_tasks, func, *args, **kwargs):
            # Simulate adding the task but don't actually run it
            pass
        
        with patch('fastapi.BackgroundTasks.add_task', side_effect=mock_background_task):
            # Test request
            response = authenticated_client.post(
                "/api/youtube/process",
                json={"url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ"}
            )
            
            # Assertions - should return 200 with upload_id
            assert response.status_code == 200
            data = response.json()
            assert "upload_id" in data
            assert data["status"] == "processing"
    
    @patch('api.upload.process_uploaded_video')
    @patch('utils.cache.update_processing_status')
    def test_upload_video(self, mock_update_status, mock_process, authenticated_client):
        """Test the video upload endpoint."""
        # Create a mock file for testing
        test_file_content = b"test video content"
        
        # Mock the background task
        def mock_background_task(background_tasks, func, *args, **kwargs):
            # Simulate adding the task but don't actually run it
            pass
        
        with patch('fastapi.BackgroundTasks.add_task', side_effect=mock_background_task):
            # Make the request
            response = authenticated_client.post(
                "/api/upload/video",
                files={"file": ("test.mp4", test_file_content, "video/mp4")},
                data={"languages": "en,ko", "summary_length": "3"}
            )
            
            # Assertions
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "processing"
            assert "upload_id" in data
    
    def test_get_upload_status(self, client, mock_redis):
        """Test the upload status endpoint."""
        # Mock cached status
        mock_status = {
            "status": "completed",
            "upload_id": "test-123456",
            "progress": 100,
            "message": "Processing completed",
            "updated_at": 1234567890,
            "result_url": "/api/upload/result/test-123456"
        }
        mock_redis.get.return_value = json.dumps(mock_status)
        
        # Test request
        response = client.get("/api/upload/status/test-123456")
        
        # Assertions
        assert response.status_code == 200
        assert response.json() == mock_status
    
    def test_get_upload_result(self, client, mock_redis):
        """Test the upload result endpoint."""
        # Mock cached result
        mock_result = {
            "upload_id": "test-123456",
            "filename": "test.mp4",
            "transcript": {"segments": []},
            "summary": {"en": "Test summary"},
            "translations": {}
        }
        mock_redis.get.return_value = json.dumps(mock_result)
        
        # Test request
        response = client.get("/api/upload/result/test-123456")
        
        # Assertions
        assert response.status_code == 200
        assert response.json() == mock_result

class TestCacheModule:
    """Unit tests for the cache module."""
    
    @patch('utils.cache.get_redis_client')
    def test_cache_result(self, mock_get_redis):
        """Test caching functionality."""
        from utils.cache import cache_result
        
        # Mock Redis client
        mock_redis = MagicMock()
        mock_redis.setex.return_value = True
        mock_get_redis.return_value = mock_redis
        
        # Test data
        test_data = {"test": "data"}
        
        # Call the function
        result = cache_result("test_key", test_data, ttl=3600)
        
        # Assertions
        assert result is True
        mock_redis.setex.assert_called_once()
    
    @patch('utils.cache.get_redis_client')
    def test_get_cached_result(self, mock_get_redis):
        """Test retrieving cached results."""
        from utils.cache import get_cached_result
        
        # Mock Redis client
        mock_redis = MagicMock()
        mock_redis.get.return_value = '{"test": "data"}'
        mock_get_redis.return_value = mock_redis
        
        # Call the function
        result = get_cached_result("test_key")
        
        # Assertions
        assert result == {"test": "data"}
        mock_redis.get.assert_called_once_with("test_key")
    
    @patch('utils.cache.get_redis_client')
    def test_update_processing_status(self, mock_get_redis, mock_file_operations):
        """Test updating processing status."""
        from utils.cache import update_processing_status
        
        # Mock Redis client
        mock_redis = MagicMock()
        mock_redis.setex.return_value = True
        mock_get_redis.return_value = mock_redis
        
        # Call the function
        result = update_processing_status(
            upload_id="test-123456",
            status="processing",
            progress=50,
            message="Test message"
        )
        
        # Assertions
        assert result is True
        mock_file_operations['makedirs'].assert_called_once()
        mock_file_operations['open'].assert_called_once()
        mock_redis.setex.assert_called_once()