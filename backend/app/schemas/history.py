from pydantic import BaseModel
from datetime import date, datetime
from app.db.models.user import User

class HistoryItem(BaseModel):
    date: date
    score: float

