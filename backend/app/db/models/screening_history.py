from datetime import datetime
from sqlalchemy import Column, Date, DateTime, Float, ForeignKey, Integer, LargeBinary, String, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Mapped, mapped_column
from app.db.session import Base
from app.db.models.user import User

class ScreeningHistory(Base):
    __tablename__ = "screening_history"
    
    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    user_id: Mapped[int] = mapped_column(Integer, ForeignKey(User.id), nullable=False, index=True)
    score: Mapped[float] = mapped_column(Float, nullable=False)
    date: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    
    # XAI plot images stored as binary data (NEW FIELDS)
    waterfall_plot: Mapped[bytes] = mapped_column(LargeBinary, nullable=True)
    force_plot: Mapped[bytes] = mapped_column(LargeBinary, nullable=True)
