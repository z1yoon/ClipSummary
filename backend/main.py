from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
import os
from dotenv import load_dotenv

# Add this line to treat the directory as a package
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import API routes
from api.routes import router as api_router

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="ClipSummary API",
    description="API for processing and summarizing video content",
    version="1.0.0",
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(api_router, prefix="/api")

# Create upload directory if it doesn't exist
os.makedirs("uploads", exist_ok=True)

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy", "version": "1.0.0"}

# Root endpoint
@app.get("/")
async def root():
    return {
        "message": "Welcome to ClipSummary API",
        "docs_url": "/docs",
        "health_check": "/health",
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)