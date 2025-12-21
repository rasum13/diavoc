from fastapi import APIRouter, Depends, HTTPException
from fastapi.encoders import jsonable_encoder
from sqlalchemy.orm import Session
from app.api.deps import get_current_user
from app.crud.history import create_history_item
from app.db.session import get_db
from app.db.models.screening_history import ScreeningHistory
from app.db.models.user import User
from app.schemas.history import HistoryItem, HistoryItemCreate
from app.schemas.user import UserOutput

router = APIRouter()

# history_list = [
#     {"date": "2025-09-10", "score": 0.25, "accuracy": 0.71},
#     {"date": "2025-09-13", "score": 0.41, "accuracy": 0.74},
#     {"date": "2025-09-17", "score": 0.45, "accuracy": 0.63},
#     {"date": "2025-09-21", "score": 0.71, "accuracy": 0.78},
# ]
#

@router.get("/")
def get_history(limit: int | None = None, current_user: UserOutput = Depends(get_current_user), db: Session = Depends(get_db)):
    history_list = db.query(ScreeningHistory).filter(ScreeningHistory.user_id == current_user.id).all()
    # print(history_list)
    if limit:
        return history_list[-limit:]
    return history_list


@router.post("/add")
def add_history(history_item: HistoryItemCreate, current_user: UserOutput = Depends(get_current_user), db: Session = Depends(get_db)):
    new_history = create_history_item(db, history_item, user_id=current_user.id)
    return new_history
