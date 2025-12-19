from fastapi import APIRouter, Depends, HTTPException
from fastapi.encoders import jsonable_encoder
from sqlalchemy.orm import Session
from app.api.deps import get_current_user
from app.crud.history import create_history_item
from app.db.session import get_db
from app.db.models.screening_history import ScreeningHistory
from app.db.models.user import User
from app.schemas.history import HistoryItem
from app.schemas.user import UserOutput

router = APIRouter()

history_list = [
    {"date": "2025-09-10", "score": 0.25, "accuracy": 0.71},
    {"date": "2025-09-13", "score": 0.41, "accuracy": 0.74},
    {"date": "2025-09-17", "score": 0.45, "accuracy": 0.63},
    {"date": "2025-09-21", "score": 0.71, "accuracy": 0.78},
]


@router.get("/")
def get_history(limit: int | None = None, current_user: UserOutput = Depends(get_current_user), db: Session = Depends(get_db)):
    # history_list = db.query(ScreeningHistory).filter(ScreeningHistory.user_id == current_user.id)
    print(history_list)
    if limit:
        return history_list[:limit]
    return history_list


@router.post("/")
def add_history(history_item: HistoryItem, current_user: UserOutput = Depends(get_current_user), db: Session = Depends(get_db)):
    create_history_item(db, history_item, user_id=current_user.id)
    return jsonable_encoder(history_item)
