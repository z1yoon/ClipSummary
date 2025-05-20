import pytest
from unittest.mock import patch, MagicMock

class TestAPI:
    """Unit tests for API endpoints."""
    
    def test_health_check(self, client):
        """Test the health check endpoint."""
        response = client.get("/api/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}
    
    @patch('api.youtube.download_youtube_video')
    def test_youtube_download(self, mock_download, client):
        """Test the YouTube download endpoint."""
        # Mock the download function
        mock_download.return_value = "/tmp/test_video.mp4"
        
        # Test request
        response = client.post(
            "/api/youtube/download",
            json={"url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ", "languages": ["en", "ko"]}
        )
        
        # Assertions
        assert response.status_code == 200
        assert "upload_id" in response.json()
        assert response.json()["status"] == "processing"
        
        # Verify mock was called with correct parameters
        mock_download.assert_called_once_with("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
    
    @patch('api.upload.process_video')
    def test_upload_video(self, mock_process, client):
        """Test the video upload endpoint."""
        # Mock the processing function to avoid actual processing
        mock_upload_id = "test-123456"
        mock_process.return_value = mock_upload_id
        
        # Create a mock file for testing
        test_file_content = b"test video content"
        
        # Make the request
        response = client.post(
            "/api/upload/video",
            files={"file": ("test.mp4", test_file_content, "video/mp4")},
            data={"languages": "en,ko", "summary_length": "3"}
        )
        
        # Assertions
        assert response.status_code == 200
        assert response.json()["status"] == "processing"
        assert response.json()["upload_id"] == mock_upload_id