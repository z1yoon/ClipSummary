import os
import sqlite3
import uuid
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import json
from datetime import datetime

# Import models and database connection
from .database import Base, get_db
from .models import User, Video, Transcription

def init_postgres():
    """Initialize PostgreSQL database with tables"""
    from .database import engine
    
    # Create all tables defined in the models
    Base.metadata.create_all(bind=engine)
    print("PostgreSQL tables created successfully")

def sqlite_to_postgres():
    """Migrate data from SQLite to PostgreSQL"""
    # SQLite connection
    sqlite_db_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "clipsummary.db")
    if not os.path.exists(sqlite_db_path):
        print(f"SQLite database not found at {sqlite_db_path}. Skipping migration.")
        return
    
    sqlite_conn = sqlite3.connect(sqlite_db_path)
    sqlite_conn.row_factory = sqlite3.Row
    
    # Get SQLAlchemy session for PostgreSQL
    from .database import SessionLocal
    db = SessionLocal()
    
    try:
        # Migrate users
        print("Migrating users...")
        sqlite_cursor = sqlite_conn.cursor()
        sqlite_cursor.execute("SELECT * FROM users")
        users = sqlite_cursor.fetchall()
        
        for sqlite_user in users:
            user_dict = dict(sqlite_user)
            
            # Check if user already exists in PostgreSQL
            existing_user = db.query(User).filter_by(username=user_dict['username']).first()
            if existing_user:
                print(f"User {user_dict['username']} already exists. Skipping.")
                continue
            
            # Create new user with SQLAlchemy
            new_user = User(
                username=user_dict['username'],
                email=user_dict['email'],
                hashed_password=user_dict['password'],  # Field name might be different
                is_active=True
            )
            
            db.add(new_user)
            db.flush()  # Get the ID without committing
            
            # Store mapping between SQLite ID and PostgreSQL ID
            user_id_map = {user_dict['id']: new_user.id}
            
            print(f"Migrated user: {new_user.username}")
        
        # Commit users to get their IDs
        db.commit()
        
        # Migrate videos
        print("Migrating videos...")
        sqlite_cursor.execute("SELECT * FROM videos")
        videos = sqlite_cursor.fetchall()
        
        for sqlite_video in videos:
            video_dict = dict(sqlite_video)
            
            # Check if video already exists in PostgreSQL by upload_id
            existing_video = db.query(Video).filter_by(upload_id=video_dict['upload_id']).first()
            if existing_video:
                print(f"Video with upload_id {video_dict['upload_id']} already exists. Skipping.")
                continue
            
            # Find the corresponding PostgreSQL user_id
            user = db.query(User).filter_by(username=video_dict['user_id']).first()
            if not user:
                print(f"User {video_dict['user_id']} not found for video {video_dict['id']}. Skipping.")
                continue
            
            # Create new video with SQLAlchemy
            new_video = Video(
                upload_id=video_dict['upload_id'],
                title=video_dict['title'],
                filename=video_dict['filename'],
                status=video_dict['status'],
                is_youtube=bool(video_dict['is_youtube']),
                user_id=user.id
            )
            
            # Add additional fields if they exist
            for field in ['description', 'duration', 'thumbnail_path', 'youtube_url']:
                if field in video_dict and video_dict[field]:
                    setattr(new_video, field, video_dict[field])
            
            db.add(new_video)
            print(f"Migrated video: {new_video.title or 'Untitled'} (ID: {video_dict['id']})")
        
        # Commit all changes
        db.commit()
        print("Migration completed successfully!")
        
    except Exception as e:
        db.rollback()
        print(f"Error during migration: {str(e)}")
    finally:
        db.close()
        sqlite_conn.close()

def run_migration():
    """Run the full migration process"""
    print("Starting database migration from SQLite to PostgreSQL")
    
    # Initialize PostgreSQL tables
    init_postgres()
    
    # Migrate data from SQLite to PostgreSQL
    sqlite_to_postgres()
    
    print("Migration process completed")

if __name__ == "__main__":
    run_migration()