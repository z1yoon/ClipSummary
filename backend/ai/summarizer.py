from typing import List, Optional
from transformers import pipeline
import torch

# Cache summarization pipeline
summarizer = None

def load_summarizer():
    """Load the summarization model if not already loaded"""
    global summarizer
    
    if summarizer is None:
        # Check if CUDA is available
        device = 0 if torch.cuda.is_available() else -1
        
        # Initialize summarization pipeline with DistilBART CNN model
        # This is a lighter, distilled version of BART fine-tuned on CNN articles
        summarizer = pipeline(
            "summarization", 
            model="sshleifer/distilbart-cnn-12-6",
            device=device
        )
    
    return summarizer

def generate_summary(text: str, max_length: int = 150, min_length: int = 40, 
                     max_sentences: Optional[int] = None) -> str:
    """
    Generate a summary of the input text using DistilBART
    
    Args:
        text: The text to summarize
        max_length: Maximum length of the summary in tokens
        min_length: Minimum length of the summary in tokens
        max_sentences: Maximum number of sentences in the summary
        
    Returns:
        A string containing the generated summary
    """
    try:
        # Load the summarizer model
        summarizer = load_summarizer()
        
        # Truncate text if it's too long (model limit is around 1024 tokens)
        # This is a simple truncation; production systems might use a more
        # sophisticated approach like chunking and combining summaries
        max_input_length = 1024
        if len(text.split()) > max_input_length:
            text = ' '.join(text.split()[:max_input_length])
        
        # Generate summary
        summary = summarizer(
            text,
            max_length=max_length,
            min_length=min_length,
            do_sample=False  # Set to True for more varied outputs
        )
        
        # Extract the summary text
        summary_text = summary[0]['summary_text']
        
        # If max_sentences is specified, limit the number of sentences
        if max_sentences:
            import re
            sentences = re.split(r'(?<=[.!?])\s+', summary_text)
            summary_text = ' '.join(sentences[:max_sentences])
        
        return summary_text
        
    except Exception as e:
        print(f"Error in summarization: {str(e)}")
        return f"Summarization failed: {str(e)}"