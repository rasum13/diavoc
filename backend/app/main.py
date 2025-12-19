from time import sleep
from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.main import api_router
from app.core.config import settings
from app.api.deps import get_current_user
from app.schemas.user import UserOutput

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router)

@app.get("/protected")
def read_protected(user: UserOutput = Depends(get_current_user)):
    return {"data": user}


### FOR TESTING ###

@app.post("/analyze")
def sleep5():
    sleep(2)
    return {"message": "done"}
