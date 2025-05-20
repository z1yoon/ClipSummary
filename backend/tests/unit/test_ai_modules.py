import pytest
from unittest.mock import patch, MagicMock

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