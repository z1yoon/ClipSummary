from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
import os
from dotenv import load_dotenv
import threading
import logging

# Add this line to treat the directory as a package
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import API route modules first - DON'T import whisperx here to avoid startup loading
from api.routes import router as main_router
# Import database modules
from db.database import engine
from db.models import Base

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

# Initialize FastAPI app with increased limits for large files
app = FastAPI(
    title="Clip Summary API",
    description="API for video transcription, summarization, and translation",
    version="1.0.0"
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

# Pre-load WhisperX model in background thread to avoid blocking startup
def preload_whisperx_model():
    logger.info("Pre-loading WhisperX model in background thread...")
    try:
        # Import AI modules only when needed
        from ai import whisperx
        
        # Set a limit on CPU usage if possible to prevent server from becoming unresponsive
        try:
            import psutil
            current_process = psutil.Process()
            if hasattr(current_process, 'cpu_affinity'):
                # Use only half of the available CPUs for model loading to keep system responsive
                available_cpus = len(current_process.cpu_affinity())
                cpus_to_use = max(1, available_cpus // 2)
                new_affinity = list(range(cpus_to_use))
                current_process.cpu_affinity(new_affinity)
                logger.info(f"Limited model loading to {cpus_to_use} CPU cores to maintain responsiveness")
        except (ImportError, Exception) as e:
            logger.warning(f"Could not set CPU affinity for model loading thread: {e}")
        
        # Load the model with improved thread synchronization
        whisperx.load_models()
        logger.info("WhisperX model pre-loaded successfully!")
    except Exception as e:
        logger.error(f"Error pre-loading WhisperX model: {str(e)}")
        logger.info("Will load model on first request instead")

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

# Start model pre-loading in background after startup
@app.on_event("startup")
async def startup_event():
    logger.info("Starting ClipSummary API server...")
    
    # Initialize database first
    init_database()
    
    # Start preloading in background after a delay
    def delayed_preload():
        # Wait for 10 seconds before starting to load the model
        # This gives time for the API to be fully ready to serve requests
        import time
        time.sleep(10)
        preload_whisperx_model()
    
    # Start preloading in background
    preload_thread = threading.Thread(target=delayed_preload)
    preload_thread.daemon = True
    preload_thread.start()
    logger.info("WhisperX pre-loading will start in 10 seconds")

# Mount static directories for serving uploaded files
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")

@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {"status": "ok"}

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

# Add generic exception handler to catch all errors
@app.middleware("http")
async def catch_exceptions_middleware(request: Request, call_next):
    try:
        return await call_next(request)
    except Exception as e:
        print(f"Unhandled exception: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": f"Internal server error: {str(e)}",
                "endpoint": str(request.url)
            }
        )

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