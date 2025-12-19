from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.api.deps import get_current_user
from app.db.session import get_db
from app.schemas.user import UserInfo, UserOutput, UserUpdateEmail, UserUpdateInfo, UserUpdateName, UserUpdatePassword
from app.crud import user


router = APIRouter()

@router.post("/update/name")
def change_name(user_info: UserUpdateName, current_user: UserOutput = Depends(get_current_user), db: Session = Depends(get_db)):
    try:
        user.update_user_name(db, current_user.id, user_info)
        return user_info
    except Exception as err:
        print(err)
        raise err

@router.post("/update/email")
def change_email(user_info: UserUpdateEmail, current_user: UserOutput = Depends(get_current_user), db: Session = Depends(get_db)):
    try:
        user.update_user_email(db, current_user.id, user_info)
        return user_info
    except Exception as err:
        print(err)
        raise err

@router.post("/update/password")
def change_password(user_info: UserUpdatePassword, current_user: UserOutput = Depends(get_current_user), db: Session = Depends(get_db)):
    try:
        user.update_user_password(db, current_user.id, user_info)
        return user_info
    except Exception as err:
        print(err)
        raise err

@router.post("/update/info")
def update_info(user_info: UserUpdateInfo, current_user: UserOutput = Depends(get_current_user), db: Session = Depends(get_db)):
    try:
        user.update_user_info(db, current_user.id, user_info)
        return user_info
    except Exception as err:
        print(err)
        raise err

@router.get("/me")
def get_user_info(current_user: UserOutput = Depends(get_current_user), db: Session = Depends(get_db)):
    user_db = user.get_user(db, current_user.id)
    user_info = UserInfo(full_name=user_db.full_name, email=user_db.email, gender=user_db.gender, age=user_db.age, height_m=user_db.height_m, weight_kg=user_db.weight_kg, is_asian=user_db.is_asian)
    return user_info
