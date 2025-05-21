from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List

from db.database import get_db
from db.models import User, Video
from schemas.users import UserResponse
from security.auth import get_current_user

router = APIRouter()

@router.get("/me", response_model=UserResponse)
async def get_user_profile(current_user: User = Depends(get_current_user)):
    """
    Get current user profile.
    """
    return current_user

@router.get("/me/videos")
async def get_user_videos(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get all videos uploaded by the current user.
    """
    videos = db.query(Video).filter(Video.user_id == current_user.id).all()
    
    # Format the response
    video_list = []
    for video in videos:
        video_data = {
            "id": video.id,
            "upload_id": video.upload_id,
            "title": video.title,
            "filename": video.filename,
            "status": video.status,
            "created_at": video.created_at,
            "thumbnail_path": video.thumbnail_path,
            "duration": video.duration,
            "description": video.description,
            "is_youtube": video.is_youtube
        }
        video_list.append(video_data)
    
    return {"videos": video_list}

@router.delete("/me/videos/{video_id}")
async def delete_user_video(
    video_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Delete a user's video.
    """
    # Check if the video exists and belongs to the current user
    video = db.query(Video).filter(Video.id == video_id, Video.user_id == current_user.id).first()
    
    if not video:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Video not found or you do not have permission to delete it"
        )
    
    # Delete the video
    db.delete(video)
    db.commit()
    
    return {"message": "Video deleted successfully"}