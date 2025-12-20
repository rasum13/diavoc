from pydantic import BaseModel
from datetime import date, datetime
from app.db.models.user import User
from typing import Optional

class HistoryItem(BaseModel):
    date: date
    score: float

class HistoryItemCreate(BaseModel):
    score: float

class ScreeningHistoryBase(BaseModel):
    score: float

class ScreeningHistoryCreate(ScreeningHistoryBase):
    user_id: int
    date: datetime
    waterfall_plot: Optional[bytes] = None
    force_plot: Optional[bytes] = None

class ScreeningHistoryOutput(ScreeningHistoryBase):
    id: int
    user_id: int
    date: datetime
    # Note: We don't include the binary plot data in the output
    # Instead, clients can fetch them via dedicated endpoints
    
    class Config:
        from_attributes = True
