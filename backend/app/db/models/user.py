from datetime import datetime
from sqlalchemy import CheckConstraint, Column, Date, DateTime, Float, ForeignKey, Integer, String, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.ext.hybrid import hybrid_property
from app.db.session import Base

class User(Base):
    __tablename__ = "user"

    id: Mapped[int] = mapped_column(primary_key=True, nullable=False, index=True)
    full_name: Mapped[str] = mapped_column(String, nullable=False)
    email: Mapped[str] = mapped_column(String, nullable=False)
    hashed_password: Mapped[str] = mapped_column(String, nullable=False)
    gender: Mapped[bool] = mapped_column(Boolean, nullable=False)
    age: Mapped[int] = mapped_column(Integer, CheckConstraint("age >= 0"), nullable=False)
    height_m: Mapped[int] = mapped_column(Float, CheckConstraint("height_m >= 0"), nullable=False)
    weight_kg: Mapped[int] = mapped_column(Float, CheckConstraint("weight_kg >= 0"), nullable=False)
    is_asian: Mapped[bool] = mapped_column(Boolean, nullable=False)

    @hybrid_property
    def bmi(self):
        return self.weight_kg / (self.height_m ** 2)
