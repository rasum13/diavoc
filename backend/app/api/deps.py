from typing import Annotated, cast

from fastapi import Depends, HTTPException, Header, status
from pydantic import EmailStr
from sqlalchemy.orm import Session

from app.core.security import decode_jwt
from app.crud.user import get_user
from app.db.session import get_db
from app.schemas.user import UserOutput
from app.core.config import settings


def get_current_user(
    db: Session = Depends(get_db), Authorization: Annotated[str | None, Header()] = None
) -> UserOutput:
    auth_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid Authentication Credentials",
    )

    if not Authorization:
        print("no token")
        raise auth_exception

    if not Authorization.startswith(settings.AUTH_PREFIX):
        print("no prefix")
        raise auth_exception

    payload = decode_jwt(Authorization.removeprefix(settings.AUTH_PREFIX))

    if payload and payload["user_id"]:
        user = get_user(db, int(payload["user_id"]))

        if not user:
            raise auth_exception

        return UserOutput(
            id=cast(int, user.id),
            full_name=cast(str, user.full_name),
            email=cast(EmailStr, user.email),
        )
    raise auth_exception
