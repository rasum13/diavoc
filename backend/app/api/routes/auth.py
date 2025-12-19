from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import cast

from app.api.deps import get_current_user
from app.schemas.user import UserCreate, UserLogin, UserOutput, UserUpdateName, UserWithToken
from app.db.models.user import User
from app.core.security import get_password_hash, verify_password, sign_jwt
from app.crud import user
from app.db.session import get_db

router = APIRouter()


@router.post("/login", status_code=200, response_model=UserWithToken)
def login(login_info: UserLogin, db: Session = Depends(get_db)):
    try:
        user_to_login = user.get_user_by_email(db, login_info.email)
        if not user_to_login:
            raise HTTPException(status_code=400, detail="Account does not exist")
        if not verify_password(
            login_info.password, cast(str, user_to_login.hashed_password)
        ):
            raise HTTPException(status_code=400, detail="Incorrect password")
        token = sign_jwt(cast(int, user_to_login.id), 120)
        return UserWithToken(token=token)
    except Exception as err:
        print(err)
        raise err


@router.post("/signup", status_code=201, response_model=UserOutput)
def signup(signup_info: UserCreate, db: Session = Depends(get_db)):
    try:
        if user.user_exists_by_email(db, signup_info.email):
            raise HTTPException(status_code=400, detail="Please Login")
        created_user = user.create_user(db, signup_info)
        return created_user
    except Exception as err:
        print(err)
        raise err

@router.get("/users")
def get_users(db: Session = Depends(get_db)):
    return db.query(User).all()
