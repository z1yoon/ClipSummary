import pytest
import os
import tempfile
from unittest.mock import patch, MagicMock
from pathlib import Path
import json
import asyncio

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
    
    @patch('api.upload.process_uploaded_video')
    @patch('utils.cache.update_processing_status')
    def test_video_processing_flow(self, mock_update_status, mock_process, 
                                  authenticated_client, temp_video_file, mock_redis):
        """Test the complete video processing flow."""
        # Mock the background task
        def mock_background_task(background_tasks, func, *args, **kwargs):
            # Simulate adding the task but don't actually run it
            pass
        
        with patch('fastapi.BackgroundTasks.add_task', side_effect=mock_background_task):
            # Create a temporary form data for the file upload
            test_filename = os.path.basename(temp_video_file)
            
            with open(temp_video_file, 'rb') as f:
                response = authenticated_client.post(
                    "/api/upload/video",
                    files={"file": (test_filename, f, "video/mp4")},
                    data={"languages": "en,ko", "summary_length": "3"}
                )
            
            # Assertions
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "processing"
            assert "upload_id" in data
            
            # Mock successful processing result for status check
            mock_status = {
                "status": "completed",
                "upload_id": data["upload_id"],
                "progress": 100,
                "message": "Processing completed successfully.",
                "updated_at": 1234567890
            }
            mock_redis.get.return_value = json.dumps(mock_status)
            
            # Check status endpoint
            status_response = authenticated_client.get(f"/api/upload/status/{data['upload_id']}")
            assert status_response.status_code == 200
            assert status_response.json()["status"] == "completed"
            
            # Mock result data
            mock_result = {
                "upload_id": data["upload_id"],
                "filename": test_filename,
                "transcript": {
                    "segments": [
                        {
                            "start": 0,
                            "end": 5,
                            "text": "This is a test transcription."
                        }
                    ]
                },
                "summary": {
                    "en": "Test summary of the video."
                },
                "translations": {
                    "ko": {
                        "summary": "번역된 요약.",
                        "transcript": [
                            {
                                "start": 0,
                                "end": 5,
                                "text": "번역된 대본입니다."
                            }
                        ]
                    }
                }
            }
            
            # Mock the Redis response for result endpoint
            mock_redis.get.return_value = json.dumps(mock_result)
            
            # Check result endpoint
            result_response = authenticated_client.get(f"/api/upload/result/{data['upload_id']}")
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
    
    @patch('api.upload.process_uploaded_video')
    @patch('ai.whisperx.transcribe_audio')
    @patch('ai.summarizer.generate_summary')
    @patch('ai.translator.translate_text')
    @patch('utils.cache.cache_result')
    def test_complete_video_processing_flow(
        self, 
        mock_cache_result,
        mock_translate, 
        mock_summarize, 
        mock_transcribe, 
        mock_process_video,
        authenticated_client,
        mock_redis
    ):
        """Test the complete video processing flow from upload to final result."""
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
                        {"word": "integration", "start": 1.9, "end": 3.0}
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
        
        # Mock the background task
        def mock_background_task(background_tasks, func, *args, **kwargs):
            # Simulate adding the task but don't actually run it
            pass
        
        with patch('fastapi.BackgroundTasks.add_task', side_effect=mock_background_task):
            # Step 1: Upload a test video
            test_file_content = b"test video content"
            response = authenticated_client.post(
                "/api/upload/video",
                files={"file": ("integration_test.mp4", test_file_content, "video/mp4")},
                data={"languages": "en,ko", "summary_length": "3"}
            )
            
            # Verify upload response
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "processing"
            assert "upload_id" in data
            
            # Step 2: Simulate getting results
            # Create mock result data
            result_data = {
                "upload_id": data["upload_id"],
                "filename": "integration_test.mp4",
                "transcript": mock_transcription,
                "summary": {"en": mock_summary},
                "translations": {
                    "ko": {
                        "summary": mock_translation,
                        "transcript": mock_transcription["segments"]
                    }
                },
                "status": "completed"
            }
            mock_redis.get.return_value = json.dumps(result_data)
            
            # Request results
            results_response = authenticated_client.get(f"/api/upload/result/{data['upload_id']}")
            
            # Verify results
            assert results_response.status_code == 200
            result_json = results_response.json()
            assert result_json["status"] == "completed"
            assert result_json["summary"]["en"] == mock_summary
            assert result_json["translations"]["ko"]["summary"] == mock_translation
            
    @patch('api.youtube.process_youtube_video')
    @patch('utils.cache.update_processing_status')
    def test_youtube_processing_integration(self, mock_update_status, mock_process, authenticated_client):
        """Test the integration of YouTube video processing."""
        # Mock the background task
        def mock_background_task(background_tasks, func, *args, **kwargs):
            # Simulate adding the task but don't actually run it
            pass
        
        with patch('fastapi.BackgroundTasks.add_task', side_effect=mock_background_task):
            # Request YouTube processing
            response = authenticated_client.post(
                "/api/youtube/process",
                json={"url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ"}
            )
            
            # Verify response
            assert response.status_code == 200
            data = response.json()
            assert "upload_id" in data
            assert data["status"] == "processing"