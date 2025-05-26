from sqlalchemy import Column, Integer, String, ForeignKey, DateTime, Boolean, Text, Float
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from .database import Base
import uuid

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True)
    username = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    # Relationship
    videos = relationship("Video", back_populates="user")

class Video(Base):
    __tablename__ = "videos"

    id = Column(Integer, primary_key=True, index=True)
    upload_id = Column(String, unique=True, index=True, default=lambda: str(uuid.uuid4()))
    title = Column(String)
    filename = Column(String)
    description = Column(Text, nullable=True)
    duration = Column(Float, nullable=True)
    file_path = Column(String)
    thumbnail_path = Column(String, nullable=True)
    
    # Video source
    is_youtube = Column(Boolean, default=False)
    youtube_url = Column(String, nullable=True)

    # Processing status
    status = Column(String) # 'processing', 'completed', 'failed'
    error_message = Column(Text, nullable=True)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Foreign key
    user_id = Column(Integer, ForeignKey("users.id"))
    
    # Relationships
    user = relationship("User", back_populates="videos")
    transcriptions = relationship("Transcription", back_populates="video")

class Transcription(Base):
    __tablename__ = "transcriptions"

    id = Column(Integer, primary_key=True, index=True)
    language = Column(String)
    content = Column(Text)
    summary = Column(Text, nullable=True)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Foreign key
    video_id = Column(Integer, ForeignKey("videos.id"))
    
    # Relationship
    video = relationship("Video", back_populates="transcriptions")