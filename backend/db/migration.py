import os
import sqlite3
from sqlalchemy.orm import Session
import logging

# Import models and database connection
from .database import Base, get_db
from .models import User, Video

# Configure logging
logger = logging.getLogger(__name__)

def init_postgres():
    """Initialize PostgreSQL database with tables"""
    from .database import engine
    
    # Create all tables defined in the models
    Base.metadata.create_all(bind=engine)
    logger.info("PostgreSQL tables created successfully")

def create_default_user(db: Session):
    """Create a default user if no users exist
    
    Args:
        db: SQLAlchemy database session
    """
    # Check if user already exists
    user_count = db.query(User).count()
    
    # Default user credentials - these should be overridden in production
    default_username = os.environ.get("DEFAULT_USER", "admin")
    
    if user_count == 0:
        # Import here to avoid circular import
        from security.auth import get_password_hash
        
        # Create default user
        logger.info(f"Creating default user: {default_username}")
        default_password = os.environ.get("DEFAULT_PASSWORD", "password")
        default_email = os.environ.get("DEFAULT_EMAIL", "admin@example.com")
        
        # Create user with hashed password
        hashed_password = get_password_hash(default_password)
        default_user = User(
            username=default_username,
            email=default_email,
            hashed_password=hashed_password,
            is_active=True
        )
        
        # Add to database
        db.add(default_user)
        db.commit()
        logger.info("Default user created successfully")
    else:
        logger.info("Users already exist, skipping default user creation")

def run_migration():
    """Initialize database with default user"""
    logger.info("Initializing database and default user")
    
    # Initialize PostgreSQL tables
    init_postgres()
    
    # Create default user
    from .database import SessionLocal
    db = SessionLocal()
    try:
        create_default_user(db)
    finally:
        db.close()
    
    logger.info("Database initialization completed")

if __name__ == "__main__":
    # Configure logging if run directly
    logging.basicConfig(level=logging.INFO)
    run_migration()