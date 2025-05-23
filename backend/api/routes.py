from fastapi import APIRouter
from api.youtube import router as youtube_router
from api.upload import router as upload_router
from api.videos import router as videos_router
from api.auth import router as auth_router

router = APIRouter()

# Include sub-routers
router.include_router(youtube_router, prefix="/youtube", tags=["YouTube"])
router.include_router(upload_router, prefix="/upload", tags=["Video Upload"])
router.include_router(videos_router, prefix="/videos", tags=["Video Playback"])
router.include_router(auth_router, prefix="/auth", tags=["Authentication"])