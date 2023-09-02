import json
import datetime
from typing import Annotated

import fastapi
import fastapi.security
import jose
import jose.jwt
import passlib.context
import pydantic

# to get a string like this run:
# openssl rand -hex 32
# python -c 'import secrets; print(secrets.token_urlsafe(32))'
# python -c 'import secrets; print(secrets.token_hex(32))'
SECRET_KEY = "09d25e094faa6ca2556c818166b7a9563b93f7099f6f0f4caa6cf63b88e8d3e7"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30


USER_DATABASE = {
    "johndoe": {
        "username": "johndoe",
        "full_name": "John Doe",
        "email": "johndoe@example.com",
        "hashed_password": "$2b$12$EixZaYVK1fsbw1ZfbX3OXePaWxn96p36WQoeG6Lruj3vjPGga31lW",
        "disabled": False,
    }
}
# pwd_context.hash(password)


class Token(pydantic.BaseModel):
    access_token: str
    token_type: str


class User(pydantic.BaseModel):
    username: str
    email: str | None = None
    full_name: str | None = None
    disabled: bool | None = None



class UserInDB(User):
    hashed_password: str


pwd_context = passlib.context.CryptContext(schemes=["bcrypt"], deprecated="auto")

oauth2_scheme = fastapi.security.OAuth2PasswordBearer(tokenUrl="token")

app = fastapi.FastAPI()


def get_user(username: str):
    if username in USER_DATABASE:
        user_dict = USER_DATABASE[username]
        return UserInDB(**user_dict)


def authenticate_user(username: str, password: str):
    user = get_user(username)
    if not user:
        return False
    if not pwd_context.verify(password, user.hashed_password):
        return False
    return user


async def get_current_user(token: Annotated[str, fastapi.Depends(oauth2_scheme)]):
    tmp0 = fastapi.HTTPException(
        status_code=fastapi.status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jose.jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        subject: str = payload.get("sub")
        if subject is None:
            raise tmp0
        else:
            data = json.loads(subject)
            username = data['username']
            expire = datetime.datetime.strptime(data['exp'], '%Y%m%d %H:%M:%S.%f')
            if expire < datetime.datetime.utcnow():
                raise tmp0
    except jose.JWTError:
        raise tmp0
    user = get_user(username=username)
    if user is None:
        raise tmp0
    return user


async def get_current_active_user(current_user: Annotated[User, fastapi.Depends(get_current_user)]):
    if current_user.disabled:
        raise fastapi.HTTPException(status_code=400, detail="Inactive user")
    return current_user


@app.post("/token", response_model=Token)
async def login_for_access_token(form_data: Annotated[fastapi.security.OAuth2PasswordRequestForm, fastapi.Depends()]):
    user = authenticate_user(form_data.username, form_data.password)
    if not user:
        raise fastapi.HTTPException(
            status_code=fastapi.status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    else:
        tmp0 = datetime.datetime.utcnow() + datetime.timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        tmp1 = dict(username=user.username, exp=datetime.datetime.strftime(tmp0, '%Y%m%d %H:%M:%S.%f'))
        token = jose.jwt.encode({'sub':json.dumps(tmp1)}, SECRET_KEY, algorithm=ALGORITHM)
        ret = dict(access_token=token, token_type="bearer")
        return ret


@app.get("/users/me/", response_model=User)
async def read_users_me(current_user: Annotated[User, fastapi.Depends(get_current_active_user)]):
    return current_user


@app.get("/users/me/items/")
async def read_own_items(current_user: Annotated[User, fastapi.Depends(get_current_active_user)]):
    return [{"item_id": "Foo", "owner": current_user.username}]

def demo_requests():
    import requests
    headers = {'Content-Type': 'application/x-www-form-urlencoded', 'accept':'application/json'}
    data = dict(username='johndoe', password='secret', grant_type='', client_id='', client_secret='', scope='')
    x0 = requests.post('http://localhost:8000/token', data=data, headers=headers)

    access_token = x0.json()['access_token']
    headers = {'Authorization': f'Bearer {access_token}', 'accept':'application/json'}
    x1 = requests.get('http://localhost:8000/users/me/', headers=headers)

# uvicorn draft_password_auth:app --port 8000
