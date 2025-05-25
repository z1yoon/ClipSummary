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
    
    @patch('ai.whisperx.whisperx.load_model')
    @patch('ai.whisperx.whisperx.load_audio')
    def test_transcribe_audio(self, mock_load_audio, mock_load_model):
        """Test the WhisperX transcription module."""
        from ai.whisperx import transcribe_audio
        
        # Mock the model and its methods
        mock_model = MagicMock()
        mock_model.transcribe.return_value = {
            "segments": [
                {
                    "start": 0.0,
                    "end": 5.0,
                    "text": "This is a test transcription.",
                }
            ],
            "language": "en"
        }
        mock_load_model.return_value = mock_model
        
        # Mock audio loading
        mock_load_audio.return_value = "mock_audio_data"
        
        # Mock alignment model
        with patch('ai.whisperx.whisperx.load_align_model') as mock_align_model, \
             patch('ai.whisperx.whisperx.align') as mock_align:
            
            mock_align_model.return_value = (MagicMock(), MagicMock())
            mock_align.return_value = {
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
            
            # Call the function
            result = transcribe_audio("/path/to/test.wav", diarize=False)
            
            # Assertions
            assert "segments" in result
            assert len(result["segments"]) == 1
            assert result["segments"][0]["text"] == "This is a test transcription."
            mock_load_model.assert_called_once()
            mock_load_audio.assert_called_once_with("/path/to/test.wav")

class TestSummarizer:
    """Unit tests for the summarizer module."""
    
    @patch('ai.summarizer.BartForConditionalGeneration.from_pretrained')
    @patch('ai.summarizer.BartTokenizer.from_pretrained')
    def test_generate_summary(self, mock_tokenizer, mock_model):
        """Test the summarizer function."""
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
        assert summary == "This is a test summary."
        mock_tokenizer.assert_called_once()
        mock_model.assert_called_once()

class TestTranslator:
    """Unit tests for the translator module."""
    
    @patch('ai.translator.AutoModelForSeq2SeqLM.from_pretrained')
    @patch('ai.translator.AutoTokenizer.from_pretrained')
    @patch('ai.translator.get_model_path')
    def test_translate_text(self, mock_get_path, mock_tokenizer, mock_model):
        """Test the translate function."""
        # Mock the model path
        mock_get_path.return_value = "/fake/model/path"
        
        # Mock the tokenizer
        mock_tokenizer_instance = MagicMock()
        mock_tokenizer_instance.return_value = {
            "input_ids": MagicMock(),
            "attention_mask": MagicMock()
        }
        mock_tokenizer_instance.batch_decode.return_value = ["번역된 텍스트"]
        mock_tokenizer_instance.get_lang_id.return_value = 123
        mock_tokenizer.return_value = mock_tokenizer_instance
        
        # Mock the model
        mock_model_instance = MagicMock()
        mock_model_instance.generate.return_value = [MagicMock()]
        mock_model_instance.parameters.return_value = [MagicMock()]
        mock_model.return_value.to.return_value = mock_model_instance
        
        # Test the translator
        text = "This is text to translate"
        translated = translate_text(text, "ko")
        
        # Assertions
        assert translated == "번역된 텍스트"
        mock_tokenizer.assert_called_once()
        mock_model.assert_called_once()

class TestWhisperX:
    """Unit tests for the WhisperX transcription module."""
    
    @patch('ai.whisperx.whisperx.load_model')
    @patch('ai.whisperx.whisperx.load_audio')
    @patch('ai.whisperx.whisperx.load_align_model')
    @patch('ai.whisperx.whisperx.align')
    def test_transcribe_audio_with_alignment(self, mock_align, mock_load_align_model, 
                                           mock_load_audio, mock_load_model):
        """Test the transcribe_audio function with proper mocks."""
        from ai.whisperx import transcribe_audio
        
        # Mock the main model
        mock_model = MagicMock()
        mock_model.transcribe.return_value = {
            "segments": [
                {"start": 0.0, "end": 5.0, "text": "Test transcription."}
            ],
            "language": "en"
        }
        mock_load_model.return_value = mock_model
        
        # Mock audio loading
        mock_load_audio.return_value = "mock_audio_data"
        
        # Mock alignment model
        mock_align_model = MagicMock()
        mock_metadata = MagicMock()
        mock_load_align_model.return_value = (mock_align_model, mock_metadata)
        
        # Mock alignment result
        mock_align.return_value = {
            "segments": [
                {
                    "start": 0.0,
                    "end": 5.0, 
                    "text": "Test transcription.",
                    "words": [
                        {"word": "Test", "start": 0.0, "end": 2.0},
                        {"word": "transcription.", "start": 2.1, "end": 5.0}
                    ]
                }
            ]
        }
        
        # Call the function without diarization
        result = transcribe_audio("/path/to/test.wav", diarize=False)
        
        # Assertions
        assert "segments" in result
        assert len(result["segments"]) == 1
        assert result["segments"][0]["text"] == "Test transcription."
        assert "words" in result["segments"][0]
        
        # Verify mocks were called
        mock_load_model.assert_called_once()
        mock_load_audio.assert_called_once()
        mock_load_align_model.assert_called_once()
        mock_align.assert_called_once()