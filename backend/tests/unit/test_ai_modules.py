import pytest
from unittest.mock import patch, MagicMock
import os
import sys

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

class TestSummarizer:
    """Simple unit tests for the summarizer module."""
    
    def test_summarizer_module_exists(self):
        """Test that summarizer module can be imported."""
        try:
            from ai import summarizer
            assert hasattr(summarizer, 'generate_summary')
        except ImportError:
            pytest.skip("Summarizer module not available")
    
    @patch('ai.summarizer.BartForConditionalGeneration')
    @patch('ai.summarizer.BartTokenizer')
    def test_generate_summary_mocked(self, mock_tokenizer_class, mock_model_class):
        """Test summarizer with full mocking."""
        from ai.summarizer import generate_summary
        
        # Mock tokenizer instance
        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = {"input_ids": MagicMock(), "attention_mask": MagicMock()}
        mock_tokenizer.decode.return_value = "This is a test summary."
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        
        # Mock model instance
        mock_model = MagicMock()
        mock_model.generate.return_value = [MagicMock()]
        mock_model_class.from_pretrained.return_value.to.return_value = mock_model
        
        # Test
        result = generate_summary("Test text", max_sentences=3)
        assert isinstance(result, str)
        assert len(result) > 0

class TestTranslator:
    """Simple unit tests for the translator module."""
    
    def test_translator_module_exists(self):
        """Test that translator module can be imported."""
        try:
            from ai import translator
            assert hasattr(translator, 'translate_text')
        except ImportError:
            pytest.skip("Translator module not available")
    
    @patch('ai.translator.get_model_path')
    @patch('ai.translator.AutoTokenizer')
    @patch('ai.translator.AutoModelForSeq2SeqLM')
    def test_translate_text_mocked(self, mock_model_class, mock_tokenizer_class, mock_get_path):
        """Test translator with full mocking to avoid model loading."""
        from ai.translator import translate_text
        
        # Mock model path
        mock_get_path.return_value = "/fake/model/path"
        
        # Mock tokenizer
        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = {"input_ids": MagicMock(), "attention_mask": MagicMock()}
        mock_tokenizer.batch_decode.return_value = ["번역된 텍스트"]
        mock_tokenizer.get_lang_id.return_value = 123
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        
        # Mock model
        mock_model = MagicMock()
        mock_model.generate.return_value = [MagicMock()]
        mock_model_class.from_pretrained.return_value.to.return_value = mock_model
        
        # Test
        result = translate_text("Hello", "ko")
        assert isinstance(result, str)
        assert len(result) > 0

class TestWhisperX:
    """Simple unit tests for the WhisperX module."""
    
    def test_whisperx_module_exists(self):
        """Test that WhisperX module can be imported."""
        try:
            from ai import whisperx
            assert hasattr(whisperx, 'transcribe_audio')
        except ImportError:
            pytest.skip("WhisperX module not available")
    
    @patch('ai.whisperx.whisperx')
    def test_transcribe_audio_mocked(self, mock_whisperx):
        """Test transcribe_audio with full mocking."""
        from ai.whisperx import transcribe_audio
        
        # Mock WhisperX components
        mock_model = MagicMock()
        mock_model.transcribe.return_value = {
            "segments": [{"start": 0.0, "end": 5.0, "text": "Test"}],
            "language": "en"
        }
        mock_whisperx.load_model.return_value = mock_model
        mock_whisperx.load_audio.return_value = "fake_audio"
        mock_whisperx.load_align_model.return_value = (MagicMock(), MagicMock())
        mock_whisperx.align.return_value = {
            "segments": [{"start": 0.0, "end": 5.0, "text": "Test", "words": []}]
        }
        
        # Test
        result = transcribe_audio("/fake/path.wav", diarize=False)
        assert "segments" in result
        assert len(result["segments"]) >= 1