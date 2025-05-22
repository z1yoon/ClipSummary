from fastapi import APIRouter, Depends, HTTPException, status, Form, Request, Body
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
from passlib.context import CryptContext
from datetime import datetime, timedelta
from typing import Optional
import sqlite3
import uuid
from pydantic import BaseModel
import os
import json

# Constants
SECRET_KEY = "0d6913c5c949a24c8da877cc80eaeff0f0bf1428e4294194a9c898e376ceb2f1" # In a real app, store this in an env var
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24 * 7  # Extended to 7 days for better user experience
REFRESH_TOKEN_EXPIRE_DAYS = 30  # 30 days for refresh token

# Setup password hashing and OAuth2
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="api/auth/login")

# Initialize database
def init_db():
    conn = sqlite3.connect("clipsummary.db")
    cursor = conn.cursor()
    
    # Create users table if it doesn't exist
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id TEXT PRIMARY KEY,
            username TEXT UNIQUE,
            email TEXT UNIQUE,
            password TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Create videos table if it doesn't exist
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS videos (
            id TEXT PRIMARY KEY,
            user_id TEXT,
            title TEXT,
            filename TEXT,
            upload_id TEXT UNIQUE,
            status TEXT,
            is_youtube BOOLEAN,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    """)
    
    conn.commit()
    conn.close()

# Initialize DB on import
init_db()

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
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def get_user(username: str):
    conn = sqlite3.connect("clipsummary.db")
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    cursor.execute("SELECT * FROM users WHERE username = ?", (username,))
    user = cursor.fetchone()
    
    conn.close()
    
    if user:
        return dict(user)
    return None

def check_email_exists(email: str) -> bool:
    conn = sqlite3.connect("clipsummary.db")
    cursor = conn.cursor()
    
    cursor.execute("SELECT email FROM users WHERE email = ?", (email,))
    exists = cursor.fetchone() is not None
    
    conn.close()
    return exists

def check_username_exists(username: str) -> bool:
    conn = sqlite3.connect("clipsummary.db")
    cursor = conn.cursor()
    
    cursor.execute("SELECT username FROM users WHERE username = ?", (username,))
    exists = cursor.fetchone() is not None
    
    conn.close()
    return exists

def create_user(username: str, email: str, password: str):
    conn = sqlite3.connect("clipsummary.db")
    cursor = conn.cursor()
    
    hashed_password = get_password_hash(password)
    user_id = str(uuid.uuid4())
    
    try:
        cursor.execute(
            "INSERT INTO users (id, username, email, password) VALUES (?, ?, ?, ?)",
            (user_id, username, email, hashed_password)
        )
        conn.commit()
        
        # Create an empty user record
        user = {
            "id": user_id,
            "username": username,
            "email": email
        }
        
        return user
    except sqlite3.IntegrityError:
        conn.close()
        return None
    finally:
        conn.close()

async def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    
    user = get_user(username)
    if user is None:
        raise credentials_exception
    
    # Remove the password field for security
    user.pop("password", None)
    
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
async def signup(request: SignupRequest):
    if len(request.password) < 8:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Password must be at least 8 characters long"
        )
    
    # Check email first
    if check_email_exists(request.email):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email address is already registered"
        )
    
    # Then check username
    if check_username_exists(request.username):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username is already taken"
        )
    
    user = create_user(request.username, request.email, request.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Error creating user account"
        )
    
    return {"message": "User created successfully", "username": request.username}

@router.post("/login")
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    user = get_user(form_data.username)
    
    if not user or not verify_password(form_data.password, user["password"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Generate access token
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user["username"]}, expires_delta=access_token_expires
    )
    
    return {"access_token": access_token, "token_type": "bearer"}

@router.post("/refresh-token")
async def refresh_token(request: TokenRefresh):
    """Refresh an authentication token before it expires"""
    try:
        # Verify the current token is valid
        payload = jwt.decode(request.token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token"
            )
        
        # Get user to ensure they still exist
        user = get_user(username)
        if user is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User not found"
            )
            
        # Generate a new token with fresh expiration
        access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = create_access_token(
            data={"sub": username}, expires_delta=access_token_expires
        )
        
        return {"access_token": access_token, "token_type": "bearer"}
        
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token",
            headers={"WWW-Authenticate": "Bearer"},
        )

@router.get("/me")
async def get_me(current_user: dict = Depends(get_current_user)):
    return current_user

@router.get("/my-videos")
async def get_my_videos(current_user: dict = Depends(get_current_user)):
    conn = sqlite3.connect("clipsummary.db")
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    cursor.execute(
        "SELECT * FROM videos WHERE user_id = ? ORDER BY created_at DESC",
        (current_user["id"],)
    )
    
    videos = []
    for row in cursor.fetchall():
        video_data = dict(row)
        
        # Add thumbnail and additional metadata if available
        info_path = f"uploads/{video_data['upload_id']}/info.json"
        if os.path.exists(info_path):
            try:
                with open(info_path, "r") as f:
                    info = json.load(f)
                
                video_data["thumbnail"] = info.get("thumbnail")
                video_data["duration"] = info.get("duration", 0)
                
                # If title is empty in DB but exists in info, use that
                if (not video_data["title"] or video_data["title"] == "Untitled") and info.get("title"):
                    video_data["title"] = info["title"]
                
                # Check status from status.json if available
                status_path = f"uploads/{video_data['upload_id']}/status.json"
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
                print(f"Error loading info for video {video_data['upload_id']}: {str(e)}")
        
        videos.append(video_data)
    
    conn.close()
    
    return {"videos": videos}

@router.delete("/me/videos/{video_id}")
async def delete_video(video_id: str, current_user: dict = Depends(get_current_user)):
    conn = sqlite3.connect("clipsummary.db")
    cursor = conn.cursor()
    
    # First check if the video belongs to the user
    cursor.execute(
        "SELECT id FROM videos WHERE id = ? AND user_id = ?",
        (video_id, current_user["id"])
    )
    
    if cursor.fetchone() is None:
        conn.close()
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Video not found or does not belong to you"
        )
    
    # Delete the video record
    cursor.execute("DELETE FROM videos WHERE id = ?", (video_id,))
    conn.commit()
    conn.close()
    
    # In a real app, you would also delete the associated files
    
    return {"message": "Video deleted successfully"}