from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime

class VideoBase(BaseModel):
    title: str
    description: Optional[str] = None

class VideoCreate(VideoBase):
    filename: str
    file_path: str
    is_youtube: bool = False
    youtube_url: Optional[str] = None

class VideoUpdate(BaseModel):
    title: Optional[str] = None
    description: Optional[str] = None
    status: Optional[str] = None

class VideoResponse(VideoBase):
    id: int
    upload_id: str
    filename: str
    duration: Optional[float] = None
    thumbnail_path: Optional[str] = None
    status: str
    error_message: Optional[str] = None
    created_at: datetime
    updated_at: Optional[datetime] = None
    user_id: int
    
    class Config:
        orm_mode = True

class VideoBriefResponse(BaseModel):
    id: int
    upload_id: str
    title: str
    thumbnail_path: Optional[str] = None
    duration: Optional[float] = None
    status: str
    created_at: datetime
    
    class Config:
        orm_mode = True

class VideoProcessingStatus(BaseModel):
    status: str
    progress: float = 0
    message: str = ""
    error: Optional[str] = None
    upload_id: str