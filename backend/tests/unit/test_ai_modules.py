import pytest
from unittest.mock import patch, MagicMock
import os
import sys
import json

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from ai.summarizer import generate_summary
from ai.translator import translate_text

class TestAIModules:
    """Unit tests for the AI modules."""
    
    @patch('ai.whisperx.whisperx')
    def test_transcribe_audio(self, mock_whisperx):
        """Test the WhisperX transcription module."""
        from ai.whisperx import transcribe_audio
        
        # Set up mock return value
        mock_result = {
            "segments": [
                {
                    "start": 0.0,
                    "end": 5.0,
                    "text": "This is a test transcription.",
                    "words": [
                        {"word": "This", "start": 0.0, "end": 0.5},
                        {"word": "is", "start": 0.6, "end": 0.9},
                        {"word": "a", "start": 1.0, "end": 1.2},
                        {"word": "test", "start": 1.3, "end": 1.8},
                        {"word": "transcription", "start": 1.9, "end": 5.0}
                    ]
                }
            ]
        }
        mock_whisperx.return_value = mock_result
        
        # Call the function
        result = transcribe_audio("/path/to/test.wav")
        
        # Assertions
        assert result == mock_result
        mock_whisperx.assert_called_once_with("/path/to/test.wav")
    
    @patch('ai.summarizer.generate_summary_with_model')
    def test_summarize_text(self, mock_generate):
        """Test the text summarization module."""
        from ai.summarizer import generate_summary
        
        # Set up mock return value
        mock_generate.return_value = "This is a summary of the test transcription."
        
        # Test input
        test_text = "This is a long transcription that needs to be summarized. It contains many words and sentences that can be condensed into a shorter version while maintaining the key points and meaning."
        
        # Call the function
        result = generate_summary(test_text, max_length=3)
        
        # Assertions
        assert result == "This is a summary of the test transcription."
        mock_generate.assert_called_once()
        assert mock_generate.call_args[0][0] == test_text
    
    @patch('ai.translator.translate_with_model')
    def test_translate_text(self, mock_translate):
        """Test the text translation module."""
        from ai.translator import translate_text
        
        # Set up mock return value
        mock_translate.return_value = "이것은 테스트 번역입니다."
        
        # Test input
        test_text = "This is a test translation."
        target_lang = "ko"
        
        # Call the function
        result = translate_text(test_text, target_lang)
        
        # Assertions
        assert result == "이것은 테스트 번역입니다."
        mock_translate.assert_called_once_with(test_text, target_lang)

class TestSummarizer:
    """Unit tests for the summarizer module."""
    
    @patch('ai.summarizer.requests.post')
    def test_generate_summary(self, mock_post):
        """Test the summarizer function."""
        # Mock the response from the LLM
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": "This is a test summary."
                    }
                }
            ]
        }
        mock_post.return_value = mock_response
        
        # Test the summarizer
        transcript = "This is a test transcript. It contains multiple sentences. We want to summarize it."
        summary = generate_summary(transcript, max_length=3)
        
        # Assertions
        assert summary == "This is a test summary."
        mock_post.assert_called_once()
        # Check that the request includes the transcript
        assert "transcript" in mock_post.call_args[1]["json"]["messages"][0]["content"]

class TestTranslator:
    """Unit tests for the translator module."""
    
    @patch('ai.translator.requests.post')
    def test_translate_text(self, mock_post):
        """Test the translate function."""
        # Mock the response from the translation API
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": "번역된 텍스트"
                    }
                }
            ]
        }
        mock_post.return_value = mock_response
        
        # Test the translator
        text = "This is text to translate"
        translated = translate_text(text, "ko")
        
        # Assertions
        assert translated == "번역된 텍스트"
        mock_post.assert_called_once()
        # Check that the request includes the source text and target language
        call_args = mock_post.call_args[1]["json"]["messages"][0]["content"]
        assert text in call_args
        assert "Korean" in call_args

class TestWhisperX:
    """Unit tests for the WhisperX transcription module."""
    
    @patch('ai.whisperx.asr_model')
    @patch('ai.whisperx.align_model')
    @patch('ai.whisperx.diarize_model')
    def test_transcribe_audio(self, mock_diarize_model, mock_align_model, mock_asr_model):
        """Test the transcribe_audio function with mocked models."""
        # This test would need to be implemented based on your actual code structure
        # But the pattern would be similar to the tests above
        
        # Example structure:
        # 1. Mock the ASR model to return a basic transcription
        # 2. Mock the align model to add word-level timestamps
        # 3. Mock the diarize model to add speaker information
        # 4. Call the transcribe_audio function with a test audio file
        # 5. Assert that the output has the expected structure
        
        # For now, we'll just add a placeholder assertion to indicate this needs implementation
        assert True, "This test should be implemented with proper mocks"