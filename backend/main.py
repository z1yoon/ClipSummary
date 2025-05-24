from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
import os
from dotenv import load_dotenv
import threading
import logging
from contextlib import asynccontextmanager
import warnings

# Add this line to treat the directory as a package
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import API route modules first - DON'T import whisperx here to avoid startup loading
from api.routes import router as main_router
# Import database modules
from db.database import engine
from db.models import Base

# Suppress compatibility warnings at startup
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', message='.*pyannote.audio.*')
warnings.filterwarnings('ignore', message='.*torch.*')
warnings.filterwarnings('ignore', message='.*TensorFloat-32.*')
warnings.filterwarnings('ignore', message='.*pytorch_lightning.*')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Custom middleware to handle large file uploads
class LargeFileMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        if request.url.path.startswith("/api/upload"):
            # Set longer timeouts for upload endpoints
            request.scope["state"] = {
                "timeout": 7200  # 2 hour timeout for uploads
            }
        return await call_next(request)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle application startup and shutdown"""
    logger.info("Starting ClipSummary API server...")
    
    # Initialize database first
    init_database()
    
    # Start preloading in background after a delay
    def delayed_preload():
        # Wait for 10 seconds before starting to load the model
        # This gives time for the API to be fully ready to serve requests
        import time
        time.sleep(10)
        try:
            from ai.whisperx import load_models
            load_models()
        except Exception as e:
            logger.error(f"Failed to preload models: {e}")
    
    # Start preloading in background
    preload_thread = threading.Thread(target=delayed_preload)
    preload_thread.daemon = True
    preload_thread.start()
    logger.info("WhisperX pre-loading will start in 10 seconds")
    
    yield
    
    logger.info("Shutting down ClipSummary API server...")

# Initialize FastAPI app with increased limits for large files
app = FastAPI(
    title="ClipSummary API",
    description="API for video transcription, summarization, and translation",
    version="1.0.0",
    lifespan=lifespan,
    # Optimize for large file uploads
    max_request_size=10*1024*1024*1024,  # 10GB max request size
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For production, specify your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add large file handling middleware
app.add_middleware(LargeFileMiddleware)

# API routes
app.include_router(main_router, prefix="/api")

# Create uploads directory if it doesn't exist
os.makedirs("uploads", exist_ok=True)

# Ensure uploads directory has proper permissions
try:
    os.chmod("uploads", 0o755)
except Exception as e:
    print(f"Warning: Could not set permissions on uploads directory: {e}")

# Initialize database tables
def init_database():
    logger.info("Initializing database...")
    try:
        # Create tables
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables created successfully")
        # No migration or default user creation - users will be added through the UI
    except Exception as e:
        logger.error(f"Error initializing database: {str(e)}")

# Health check endpoint
@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {"status": "ok"}

# Welcome message
@app.get("/")
def read_root():
    """Welcome message"""
    return {
        "message": "Welcome to Clip Summary API. See /docs for API documentation.",
        "docsUrl": "/docs"
    }

# Add exception handler for better error responses
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "status": "error",
            "statusCode": exc.status_code,
            "message": exc.detail
        }
    )

# Add global exception handler to prevent crashes
@app.middleware("http")
async def catch_exceptions_middleware(request: Request, call_next):
    try:
        return await call_next(request)
    except Exception as e:
        logger.error(f"Unhandled exception: {str(e)}")
        logger.error(f"Request URL: {request.url}")
        logger.error(f"Request method: {request.method}")
        
        # Graceful error response to prevent backend crashes
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": "Internal server error occurred. Please try again.",
                "error_code": "BACKEND_ERROR",
                "endpoint": str(request.url)
            }
        )

# Mount static directories for serving uploaded files
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        timeout_keep_alive=7200,  # 2 hour keep-alive
        limit_concurrency=2,  # Limit concurrent uploads
        backlog=4  # Small backlog for large file uploads
    )