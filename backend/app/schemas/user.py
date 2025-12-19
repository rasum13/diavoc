from pydantic import BaseModel, EmailStr

class UserCreate(BaseModel):
    full_name: str
    email: EmailStr
    password: str
    gender: bool
    age: int
    height_m: float
    weight_kg: float
    is_asian: bool


class UserUpdateName(BaseModel):
    full_name: str | None

class UserUpdateEmail(BaseModel):
    email: EmailStr

class UserUpdatePassword(BaseModel):
    old_password: str
    new_password: str

class UserUpdateInfo(BaseModel):
    gender: bool
    age: int
    height_m: float
    weight_kg: float

class UserInfo(BaseModel):
    full_name: str
    email: EmailStr
    gender: bool
    age: int
    height_m: float
    weight_kg: float
    is_asian: bool

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class UserWithToken(BaseModel):
    token: str

class UserOutput(BaseModel):
    id: int
    full_name: str
    email: EmailStr
