from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime

class TranscriptionBase(BaseModel):
    language: str
    content: str
    summary: Optional[str] = None

class TranscriptionCreate(TranscriptionBase):
    video_id: int

class TranscriptionUpdate(BaseModel):
    language: Optional[str] = None
    content: Optional[str] = None
    summary: Optional[str] = None

class TranscriptionResponse(TranscriptionBase):
    id: int
    video_id: int
    created_at: datetime
    
    class Config:
        orm_mode = True

class TranscriptionSegment(BaseModel):
    start: float
    end: float
    text: str
    words: Optional[List[Dict[str, Any]]] = None

class WhisperXTranscription(BaseModel):
    segments: List[TranscriptionSegment]
    language: str