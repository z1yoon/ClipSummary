from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch
from typing import Dict, Optional

# Cache the model and tokenizers to avoid reloading
model = None
tokenizer = None

# Map of language codes to NLLB language codes
LANGUAGE_CODE_MAP = {
    "en": "eng_Latn",
    "ko": "kor_Latn",
    "zh": "zho_Hans",
    "fr": "fra_Latn",
    "es": "spa_Latn",
    "de": "deu_Latn",
    "it": "ita_Latn",
    "ja": "jpn_Jpan",
    "ru": "rus_Cyrl",
    "pt": "por_Latn",
    "ar": "ara_Arab"
    # Add more languages as needed
}

def load_nllb_model():
    """Load the NLLB-200 translation model if not already loaded"""
    global model, tokenizer
    
    if model is None or tokenizer is None:
        # We use the distilled 600M parameter model which is faster and more memory-efficient
        model_name = "facebook/nllb-200-distilled-600M"
        
        # Check if CUDA is available
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
    
    return model, tokenizer

def translate_text(text: str, source_lang: str = "en", target_lang: str = "ko") -> str:
    """
    Translate text using NLLB-200 model
    
    Args:
        text: Text to translate
        source_lang: Source language code (ISO 639-1)
        target_lang: Target language code (ISO 639-1)
        
    Returns:
        Translated text string
    """
    try:
        # Map language codes to NLLB format
        source_nllb = LANGUAGE_CODE_MAP.get(source_lang, "eng_Latn")
        target_nllb = LANGUAGE_CODE_MAP.get(target_lang, "eng_Latn")
        
        # Return original text if source and target are the same
        if source_lang == target_lang:
            return text
        
        # Load model and tokenizer
        model, tokenizer = load_nllb_model()
        
        # Truncate text if it's too long
        max_input_length = 512
        if len(text.split()) > max_input_length:
            text = ' '.join(text.split()[:max_input_length])
        
        # Tokenize input text
        inputs = tokenizer(text, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        # Set the language token
        inputs["forced_bos_token_id"] = tokenizer.lang_code_to_id[target_nllb]
        
        # Generate translation
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=512,
                num_beams=5,
                early_stopping=True
            )
        
        # Decode and return translation
        translation = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return translation
        
    except Exception as e:
        print(f"Error in translation: {str(e)}")
        return f"Translation failed: {str(e)}"