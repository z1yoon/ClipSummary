from fastapi import APIRouter, Depends, HTTPException, status, Form
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
from passlib.context import CryptContext
from datetime import datetime, timedelta
from typing import Optional
from sqlalchemy.orm import Session
from pydantic import BaseModel
import os
import json
import uuid

from db.database import get_db
from db.models import User, Video
from schemas.users import UserCreate, TokenData

# Constants - Load from environment variables with fallback values
JWT_SECRET = os.getenv("JWT_SECRET")
if not JWT_SECRET:
    print("WARNING: JWT_SECRET not set in environment variables. Using fallback value for development only.")
    JWT_SECRET = os.getenv("SECRET_KEY", "jwtsecretkey")  # Fallback to SECRET_KEY for backward compatibility

ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "10080"))  # 7 days default
REFRESH_TOKEN_EXPIRE_DAYS = 30  # 30 days for refresh token

# Setup password hashing and OAuth2
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="api/auth/login")

# Router setup
router = APIRouter()

# Helper functions
def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, JWT_SECRET, algorithm=ALGORITHM)
    return encoded_jwt

def get_user(db: Session, username: str):
    """Get user by username using SQLAlchemy"""
    return db.query(User).filter(User.username == username).first()

def check_email_exists(db: Session, email: str) -> bool:
    """Check if email exists using SQLAlchemy"""
    return db.query(User).filter(User.email == email).first() is not None

def check_username_exists(db: Session, username: str) -> bool:
    """Check if username exists using SQLAlchemy"""
    return db.query(User).filter(User.username == username).first() is not None

def create_user(db: Session, username: str, email: str, password: str):
    """Create a new user using SQLAlchemy"""
    hashed_password = get_password_hash(password)
    
    new_user = User(
        username=username,
        email=email,
        hashed_password=hashed_password
    )
    
    try:
        db.add(new_user)
        db.commit()
        db.refresh(new_user)
        return new_user
    except Exception as e:
        db.rollback()
        print(f"Error creating user: {e}")
        return None

async def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    """Get the current user from JWT token using SQLAlchemy"""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        
        token_data = TokenData(username=username)
    except JWTError:
        raise credentials_exception
    
    user = get_user(db, username)
    if user is None:
        raise credentials_exception
    
    return user

# Add this model for signup request
class SignupRequest(BaseModel):
    username: str
    email: str
    password: str

# Add this model for token refresh
class TokenRefresh(BaseModel):
    token: str

# API Endpoints
@router.post("/signup")
async def signup(request: SignupRequest, db: Session = Depends(get_db)):
    if len(request.password) < 8:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Password must be at least 8 characters long"
        )
    
    # Check email first
    if check_email_exists(db, request.email):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email address is already registered"
        )
    
    # Then check username
    if check_username_exists(db, request.username):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username is already taken"
        )
    
    user = create_user(db, request.username, request.email, request.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Error creating user account"
        )
    
    return {"message": "User created successfully", "username": request.username}

@router.post("/login")
async def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = get_user(db, form_data.username)
    
    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Generate access token
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username, "id": user.id}, expires_delta=access_token_expires
    )
    
    return {"access_token": access_token, "token_type": "bearer"}

@router.post("/refresh-token")
async def refresh_token(request: TokenRefresh, db: Session = Depends(get_db)):
    """Refresh an authentication token before it expires"""
    try:
        # Verify the current token is valid
        payload = jwt.decode(request.token, JWT_SECRET, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token"
            )
        
        # Get user to ensure they still exist
        user = get_user(db, username)
        if user is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User not found"
            )
            
        # Generate a new token with fresh expiration
        access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = create_access_token(
            data={"sub": username, "id": user.id}, expires_delta=access_token_expires
        )
        
        return {"access_token": access_token, "token_type": "bearer"}
        
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token",
            headers={"WWW-Authenticate": "Bearer"},
        )

@router.get("/me")
async def get_me(current_user: User = Depends(get_current_user)):
    return {
        "id": current_user.id,
        "username": current_user.username,
        "email": current_user.email,
        "is_active": current_user.is_active
    }

@router.get("/my-videos")
async def get_my_videos(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    # Use SQLAlchemy to get user videos
    videos_query = db.query(Video).filter(Video.user_id == current_user.id).order_by(Video.created_at.desc())
    videos = videos_query.all()
    
    video_list = []
    for video in videos:
        video_data = {
            "id": video.id,
            "upload_id": video.upload_id,
            "title": video.title,
            "filename": video.filename,
            "status": video.status,
            "created_at": video.created_at,
            "is_youtube": video.is_youtube,
            "thumbnail_path": video.thumbnail_path,
            "duration": video.duration,
            "description": video.description
        }
        
        # Add additional metadata if available
        info_path = f"uploads/{video.upload_id}/info.json"
        if os.path.exists(info_path):
            try:
                with open(info_path, "r") as f:
                    info = json.load(f)
                
                if not video_data.get("thumbnail_path") and info.get("thumbnail"):
                    video_data["thumbnail"] = info.get("thumbnail")
                
                if not video_data.get("duration") and info.get("duration"):
                    video_data["duration"] = info.get("duration", 0)
                
                # If title is empty in DB but exists in info, use that
                if (not video_data["title"] or video_data["title"] == "Untitled") and info.get("title"):
                    video_data["title"] = info["title"]
                
                # Check status from status.json if available
                status_path = f"uploads/{video.upload_id}/status.json"
                if os.path.exists(status_path):
                    try:
                        with open(status_path, "r") as f:
                            status_data = json.load(f)
                            # Update status if it's more current than what's in the DB
                            if status_data.get("status") == "completed" and video_data["status"] != "completed":
                                video_data["status"] = "completed"
                            elif status_data.get("status") == "failed" and video_data["status"] != "failed":
                                video_data["status"] = "failed"
                                video_data["error_message"] = status_data.get("message", "")
                    except:
                        pass  # Ignore errors reading status file
            except Exception as e:
                print(f"Error loading info for video {video.upload_id}: {str(e)}")
        
        video_list.append(video_data)
    
    return {"videos": video_list}

@router.delete("/me/videos/{video_id}")
async def delete_video(video_id: int, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    # Check if the video belongs to the user
    video = db.query(Video).filter(Video.id == video_id, Video.user_id == current_user.id).first()
    
    if not video:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Video not found or does not belong to you"
        )
    
    # Delete the video record
    db.delete(video)
    db.commit()
    
    # In a real app, you would also delete the associated files
    
    return {"message": "Video deleted successfully"}