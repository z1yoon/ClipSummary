import pytest
from unittest.mock import patch, MagicMock
import os
import sys

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

class TestSummarizer:
    """Simple unit tests for the summarizer module."""
    
    @patch('ai.summarizer.BartForConditionalGeneration.from_pretrained')
    @patch('ai.summarizer.BartTokenizer.from_pretrained')
    def test_generate_summary(self, mock_tokenizer, mock_model):
        """Test the summarizer function with simple mocks."""
        from ai.summarizer import generate_summary
        
        # Mock the tokenizer
        mock_tokenizer_instance = MagicMock()
        mock_tokenizer_instance.return_value = {
            "input_ids": MagicMock(),
            "attention_mask": MagicMock()
        }
        mock_tokenizer_instance.decode.return_value = "This is a test summary."
        mock_tokenizer.return_value = mock_tokenizer_instance
        
        # Mock the model
        mock_model_instance = MagicMock()
        mock_model_instance.generate.return_value = [MagicMock()]
        mock_model_instance.parameters.return_value = [MagicMock()]
        mock_model.return_value.to.return_value = mock_model_instance
        
        # Test the summarizer
        transcript = "This is a test transcript. It contains multiple sentences. We want to summarize it."
        summary = generate_summary(transcript, max_sentences=3)
        
        # Assertions
        assert isinstance(summary, str)
        assert len(summary) > 0

class TestTranslator:
    """Simple unit tests for the translator module."""
    
    def test_translate_text_simple(self):
        """Test translate function exists and returns a string."""
        from ai.translator import translate_text
        
        # Simple test that just checks the function exists and can handle basic input
        with patch('ai.translator.AutoModelForSeq2SeqLM.from_pretrained') as mock_model, \
             patch('ai.translator.AutoTokenizer.from_pretrained') as mock_tokenizer:
            
            # Mock tokenizer
            mock_tokenizer_instance = MagicMock()
            mock_tokenizer_instance.return_value = {"input_ids": MagicMock(), "attention_mask": MagicMock()}
            mock_tokenizer_instance.batch_decode.return_value = ["번역된 텍스트"]
            mock_tokenizer_instance.get_lang_id.return_value = 123
            mock_tokenizer.return_value = mock_tokenizer_instance
            
            # Mock model
            mock_model_instance = MagicMock()
            mock_model_instance.generate.return_value = [MagicMock()]
            mock_model.return_value.to.return_value = mock_model_instance
            
            # Test
            result = translate_text("Hello world", "ko")
            assert isinstance(result, str)
            assert len(result) > 0

class TestWhisperX:
    """Simple unit tests for the WhisperX module."""
    
    @patch('ai.whisperx.whisperx.load_model')
    @patch('ai.whisperx.whisperx.load_audio')
    @patch('ai.whisperx.whisperx.load_align_model')
    @patch('ai.whisperx.whisperx.align')
    def test_transcribe_audio_simple(self, mock_align, mock_load_align_model, 
                                   mock_load_audio, mock_load_model):
        """Test the transcribe_audio function with simple mocks."""
        from ai.whisperx import transcribe_audio
        
        # Mock the main model
        mock_model = MagicMock()
        mock_model.transcribe.return_value = {
            "segments": [{"start": 0.0, "end": 5.0, "text": "Test transcription."}],
            "language": "en"
        }
        mock_load_model.return_value = mock_model
        
        # Mock audio loading
        mock_load_audio.return_value = "mock_audio_data"
        
        # Mock alignment
        mock_load_align_model.return_value = (MagicMock(), MagicMock())
        mock_align.return_value = {
            "segments": [
                {
                    "start": 0.0,
                    "end": 5.0, 
                    "text": "Test transcription.",
                    "words": [{"word": "Test", "start": 0.0, "end": 2.0}]
                }
            ]
        }
        
        # Test
        result = transcribe_audio("/fake/path.wav", diarize=False)
        
        # Simple assertions
        assert "segments" in result
        assert len(result["segments"]) >= 1