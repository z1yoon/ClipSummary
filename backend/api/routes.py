from fastapi import APIRouter
from api.youtube import router as youtube_router
from api.upload import router as upload_router

router = APIRouter()

# Include sub-routers
router.include_router(youtube_router, prefix="/youtube", tags=["YouTube"])
router.include_router(upload_router, prefix="/upload", tags=["Video Upload"])