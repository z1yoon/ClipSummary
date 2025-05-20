import pytest
import os
import tempfile
from unittest.mock import patch, MagicMock
from pathlib import Path
import json

class TestIntegrationFlow:
    """Integration tests for the full processing pipeline."""
    
    @pytest.fixture
    def temp_video_file(self):
        """Create a temporary video file for testing."""
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            # Create an empty file - we're mocking actual processing
            tmp.write(b"test video content")
            tmp_path = tmp.name
        
        yield tmp_path
        # Cleanup after test
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
    
    @patch('utils.helpers.extract_audio_from_video')
    @patch('ai.whisperx.transcribe_audio')
    @patch('ai.summarizer.generate_summary')
    @patch('ai.translator.translate_text')
    def test_video_processing_flow(self, mock_translate, mock_summarize, 
                                  mock_transcribe, mock_extract, 
                                  client, temp_video_file):
        """Test the complete video processing flow."""
        # Setup mocks
        audio_path = temp_video_file.replace('.mp4', '_audio.wav')
        mock_extract.return_value = audio_path
        
        # Mock transcription results
        mock_transcribe.return_value = {
            "segments": [
                {
                    "start": 0,
                    "end": 5,
                    "text": "This is a test transcription.",
                    "words": [
                        {"word": "This", "start": 0, "end": 1},
                        {"word": "is", "start": 1.1, "end": 1.5},
                        {"word": "a", "start": 1.6, "end": 1.9},
                        {"word": "test", "start": 2, "end": 3},
                        {"word": "transcription", "start": 3.2, "end": 5}
                    ]
                }
            ]
        }
        
        # Mock summarization
        mock_summarize.return_value = "Test summary of the video."
        
        # Mock translation
        mock_translate.return_value = "번역된 요약."
        
        # Create a temporary form data for the file upload
        test_filename = os.path.basename(temp_video_file)
        
        with open(temp_video_file, 'rb') as f:
            response = client.post(
                "/api/upload/video",
                files={"file": (test_filename, f, "video/mp4")},
                data={"languages": "en,ko", "summary_length": "3"}
            )
        
        # Assertions
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "processing"
        assert "upload_id" in data
        
        # Assuming we would poll for status, mock that response as well
        with patch('api.upload.get_cached_result') as mock_cache:
            # Mock successful processing result
            mock_cache.return_value = {
                "status": "completed",
                "transcript": {
                    "segments": mock_transcribe.return_value["segments"]
                },
                "summary": {
                    "en": mock_summarize.return_value
                },
                "translations": {
                    "ko": {
                        "summary": mock_translate.return_value,
                        "transcript": mock_transcribe.return_value["segments"]
                    }
                }
            }
            
            # Check status endpoint
            status_response = client.get(f"/api/upload/status/{data['upload_id']}")
            assert status_response.status_code == 200
            assert status_response.json()["status"] == "completed"
            
            # Check result endpoint
            result_response = client.get(f"/api/upload/result/{data['upload_id']}")
            assert result_response.status_code == 200
            result_data = result_response.json()
            assert "transcript" in result_data
            assert "summary" in result_data
            assert "translations" in result_data
            assert "ko" in result_data["translations"]
            assert result_data["summary"]["en"] == "Test summary of the video."
            assert result_data["translations"]["ko"]["summary"] == "번역된 요약."

class TestVideoProcessing:
    """Integration tests for the end-to-end video processing workflow."""
    
    @patch('api.upload.process_video')
    @patch('ai.whisperx.transcribe_audio')
    @patch('ai.summarizer.generate_summary')
    @patch('ai.translator.translate_text')
    @patch('utils.cache.save_result')
    def test_complete_video_processing_flow(
        self, 
        mock_save_result,
        mock_translate, 
        mock_summarize, 
        mock_transcribe, 
        mock_process_video,
        client
    ):
        """Test the complete video processing flow from upload to final result."""
        # Set up mock return values
        mock_upload_id = "test-integration-123"
        mock_process_video.return_value = mock_upload_id
        
        # Mock transcription result
        mock_transcription = {
            "segments": [
                {
                    "start": 0.0,
                    "end": 5.0,
                    "text": "This is a test integration for the complete pipeline.",
                    "words": [
                        {"word": "This", "start": 0.0, "end": 0.5},
                        {"word": "is", "start": 0.6, "end": 0.9},
                        {"word": "a", "start": 1.0, "end": 1.2},
                        {"word": "test", "start": 1.3, "end": 1.8},
                        {"word": "integration", "start": 1.9, "end": 3.0},
                        {"word": "for", "start": 3.1, "end": 3.3},
                        {"word": "the", "start": 3.4, "end": 3.6},
                        {"word": "complete", "start": 3.7, "end": 4.2},
                        {"word": "pipeline", "start": 4.3, "end": 5.0}
                    ]
                }
            ]
        }
        mock_transcribe.return_value = mock_transcription
        
        # Mock summary
        mock_summary = "Integration test for pipeline."
        mock_summarize.return_value = mock_summary
        
        # Mock translation
        mock_translation = "파이프라인에 대한 통합 테스트."
        mock_translate.return_value = mock_translation
        
        # Step 1: Upload a test video
        test_file_content = b"test video content"
        response = client.post(
            "/api/upload/video",
            files={"file": ("integration_test.mp4", test_file_content, "video/mp4")},
            data={"languages": "en,ko", "summary_length": "3"}
        )
        
        # Verify upload response
        assert response.status_code == 200
        assert response.json()["upload_id"] == mock_upload_id
        assert response.json()["status"] == "processing"
        
        # Step 2: Mock the processing
        # In a real integration test, we would wait for processing to complete
        # Here we'll just ensure the mocks were called with expected parameters
        
        # Step 3: Simulate getting results
        with patch('utils.cache.get_result') as mock_get_result:
            # Create mock result data
            result_data = {
                "transcription": mock_transcription,
                "summary": mock_summary,
                "translations": {
                    "ko": mock_translation
                },
                "video_path": "/tmp/videos/integration_test.mp4",
                "status": "completed"
            }
            mock_get_result.return_value = result_data
            
            # Request results
            results_response = client.get(f"/api/results/{mock_upload_id}")
            
            # Verify results
            assert results_response.status_code == 200
            assert results_response.json()["status"] == "completed"
            assert results_response.json()["summary"] == mock_summary
            assert results_response.json()["translations"]["ko"] == mock_translation
            
    @patch('api.youtube.download_youtube_video')
    @patch('api.upload.process_video')  
    def test_youtube_download_integration(self, mock_process, mock_download, client):
        """Test the integration between YouTube download and video processing."""
        # Set up mocks
        mock_download.return_value = "/tmp/videos/youtube_test.mp4"
        mock_upload_id = "yt-integration-123"
        mock_process.return_value = mock_upload_id
        
        # Request YouTube download
        response = client.post(
            "/api/youtube/download",
            json={"url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ", "languages": ["en", "es"]}
        )
        
        # Verify response
        assert response.status_code == 200
        assert response.json()["upload_id"] == mock_upload_id
        assert response.json()["status"] == "processing"
        
        # Verify download was called
        mock_download.assert_called_once_with("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
        
        # Verify process_video was called with the downloaded file
        mock_process.assert_called_once()
        assert mock_process.call_args[0][0] == "/tmp/videos/youtube_test.mp4"