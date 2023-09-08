from typing import Annotated
import fastapi

SIGNED_API_KEYS = {"a0767a55b7feb2d0166f8aabc51a38c262f1ff0f1324009dd5134c352d700b37"}
# python -c 'import secrets; print(secrets.token_urlsafe(32))'
# python -c 'import secrets; print(secrets.token_hex(32))'

app = fastapi.FastAPI()

@app.get("/")
async def root():
    return {"message": "Hello World"}


def verify_apikey(Authorization: Annotated[str, fastapi.Header()]):
    if str(Authorization) not in SIGNED_API_KEYS:
        raise fastapi.HTTPException(status_code=fastapi.status.HTTP_401_UNAUTHORIZED, detail="Not authorized")


@app.get("/private", dependencies=[fastapi.Depends(verify_apikey)])
async def private():
    return {"message": "Hello Private"}

def demo_requests():
    import requests
    # authorization not required
    x0 = requests.get('http://localhost:8000/')
    print(x0.json())

    # authorized
    headers = {'Authorization': "a0767a55b7feb2d0166f8aabc51a38c262f1ff0f1324009dd5134c352d700b37"}
    x0 = requests.get('http://localhost:8000/private', headers=headers)
    print(x0.json())

    # not authorized
    headers = {'Authorization': "fake-apikey"}
    x0 = requests.get('http://localhost:8000/private', headers=headers)
    print(x0.json())

# uvicorn draft_apikey_auth:app --port 8000 --reload

