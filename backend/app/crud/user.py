from typing import Annotated, cast
from pydantic import EmailStr
from sqlalchemy.orm import Session
from fastapi import Depends, HTTPException, Header, status

from app.core.security import decode_jwt, get_password_hash, verify_password
from app.core.config import settings
from app.db.session import get_db
from app.db.models.user import User
from app.schemas.user import (
    UserCreate,
    UserOutput,
    UserUpdateEmail,
    UserUpdateInfo,
    UserUpdateName,
    UserUpdatePassword,
)


def get_user(db: Session, user_id: int) -> User:
    return db.query(User).filter(User.id == user_id).first()


def get_user_by_email(db: Session, email: str) -> User | None:
    return db.query(User).filter(User.email == email).first()


def user_exists(db: Session, user_id: int) -> bool:
    user = db.query(User).filter(User.id == user_id).first()
    return bool(user)


def user_exists_by_email(db: Session, email: str) -> bool:
    user = db.query(User).filter(User.email == email).first()
    return bool(user)


def create_user(db: Session, obj_in: UserCreate) -> User:
    hashed_password = get_password_hash(obj_in.password)
    db_obj = User(
        email=obj_in.email,
        hashed_password=hashed_password,
        full_name=obj_in.full_name,
        gender=obj_in.gender,
        age=obj_in.age,
        height_m=obj_in.height_m,
        weight_kg=obj_in.weight_kg,
        is_asian=obj_in.is_asian,
    )
    db.add(db_obj)
    db.commit()
    db.refresh(db_obj)
    return db_obj


def update_user_name(db: Session, user_id: int, obj_in: UserUpdateName) -> User:
    user_to_update = db.query(User).filter(User.id == user_id).first()

    if not user_to_update:
        raise HTTPException(status_code=404, detail="User not found")

    if obj_in.full_name:
        setattr(user_to_update, "full_name", obj_in.full_name)

    db.commit()
    db.refresh(user_to_update)

    return user_to_update


def update_user_email(db: Session, user_id: int, obj_in: UserUpdateEmail) -> User:
    user_to_update = db.query(User).filter(User.id == user_id).first()

    if not user_to_update:
        raise HTTPException(status_code=404, detail="User not found")

    if obj_in.email:
        setattr(user_to_update, "email", obj_in.email)

    db.commit()
    db.refresh(user_to_update)

    return user_to_update


def update_user_info(db: Session, user_id: int, obj_in: UserUpdateInfo) -> User:
    user_to_update = db.query(User).filter(User.id == user_id).first()

    if not user_to_update:
        raise HTTPException(status_code=404, detail="User not found")

    setattr(user_to_update, "gender", obj_in.gender)
    setattr(user_to_update, "age", obj_in.age)
    setattr(user_to_update, "height_m", obj_in.height_m)
    setattr(user_to_update, "weight_kg", obj_in.weight_kg)
    print("updated info")

    db.commit()
    db.refresh(user_to_update)
    return user_to_update


def update_user_password(db: Session, user_id: int, obj_in: UserUpdatePassword) -> User:
    user_to_update = db.query(User).filter(User.id == user_id).first()

    if not user_to_update:
        raise HTTPException(status_code=404, detail="User not found")

    if not verify_password(
        obj_in.old_password, cast(str, user_to_update.hashed_password)
    ):
        raise HTTPException(status_code=400, detail="Incorrect password")

    if obj_in.new_password:
        hashed_password = get_password_hash(obj_in.new_password)
        setattr(user_to_update, "hashed_password", hashed_password)

    db.commit()
    db.refresh(user_to_update)

    return user_to_update
