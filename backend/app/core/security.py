from passlib.context import CryptContext
from jose import jwt
from typing import Any
import time

from app.core.config import settings

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

BCRYPT_MAX_LENGTH = 72


def get_password_hash(password: str) -> str:
    safe_password = password[:BCRYPT_MAX_LENGTH]
    return pwd_context.hash(safe_password)


def verify_password(password: str, hashed: str) -> bool:
    safe_password = password[:BCRYPT_MAX_LENGTH]
    return pwd_context.verify(safe_password, hashed)


def sign_jwt(user_id: int, expires_mins: int) -> str:
    payload = {"user_id": user_id, "expires": time.time() + expires_mins * 60}
    token = jwt.encode(payload, settings.SECRET_KEY, settings.ALGORITHM)
    return token

def decode_jwt(token: str) -> dict | None:
    try:
        decoded_token = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
        return decoded_token if decoded_token["expires"] >= time.time() else None
    except:
        print("Unable to decode the token")
