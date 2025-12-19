from fastapi import APIRouter
from app.api.routes import history, auth, user

api_router = APIRouter()
api_router.include_router(history.router, prefix="/history", tags=["history"])
api_router.include_router(auth.router, prefix="/auth", tags=["auth"])
api_router.include_router(user.router, prefix="/user", tags=["user"])
