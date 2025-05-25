from fastapi import APIRouter
from fastapi.responses import Response
from api.youtube import router as youtube_router
from api.upload import router as upload_router
from api.videos import router as videos_router
from api.auth import router as auth_router

router = APIRouter()

# Network diagnostic endpoint for frontend latency testing
@router.head("/ping")
@router.get("/ping")
async def ping():
    """Simple endpoint to test network latency and connectivity"""
    return Response(status_code=200, headers={"Cache-Control": "no-cache"})

# Include sub-routers
router.include_router(youtube_router, prefix="/youtube", tags=["YouTube"])
router.include_router(upload_router, prefix="/upload", tags=["Video Upload"])
router.include_router(videos_router, prefix="/videos", tags=["Video Playback"])
router.include_router(auth_router, prefix="/auth", tags=["Authentication"])