from datetime import datetime
from sqlalchemy.orm import Session
from app.db.session import get_db
from app.schemas.history import HistoryItem, HistoryItemCreate
from app.schemas.user import UserWithToken
from app.db.models.screening_history import ScreeningHistory

def create_history_item(db: Session, history_item: HistoryItemCreate, user_id: int) -> ScreeningHistory:
    new_item = ScreeningHistory(user_id=user_id, score=history_item.score, date=datetime.now())
    db.add(new_item)
    db.commit()
    db.refresh(new_item)
    return new_item
