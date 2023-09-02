import fastapi
import fastapi.security

api_keys = {
    "a0767a55b7feb2d0166f8aabc51a38c262f1ff0f1324009dd5134c352d700b37"
}
# python -c 'import secrets; print(secrets.token_urlsafe(32))'
# python -c 'import secrets; print(secrets.token_hex(32))'


oauth2_scheme = fastapi.security.OAuth2PasswordBearer(tokenUrl="token")


def api_key_auth(api_key: str = fastapi.Depends(oauth2_scheme)):
    if api_key not in api_keys:
        raise fastapi.HTTPException(status_code=fastapi.status.HTTP_401_UNAUTHORIZED, detail="Forbidden")


app = fastapi.FastAPI()


@app.get("/protected", dependencies=[fastapi.Depends(api_key_auth)])
def add_post() -> dict:
    return {"data": "valid API key"}

def demo_requests():
    import requests
    access_token = 'a0767a55b7feb2d0166f8aabc51a38c262f1ff0f1324009dd5134c352d700b37'
    headers = {'Authorization': f'Bearer {access_token}', 'accept':'application/json'}
    x0 = requests.get('http://localhost:8000/protected/', headers=headers)

# uvicorn draft_apikey_auth:app --port 8000
