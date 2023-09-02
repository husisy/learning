from enum import Enum
import typing
import fastapi
import pydantic

app = fastapi.FastAPI()


@app.get("/")
async def read_root():
    return {"Hello": "World"}
# import requests
# x0 = requests.get('http://localhost:8000/')
# x0.text
# x0.json()
#> curl http://localhost:8000
#> curl http://127.0.0.1:8000/


@app.get("/users")
async def read_users():
    return ["Rick", "Morty"]


@app.get("/userid/{user_id}")
async def read_user_id(user_id:str):
    return {'user_id': user_id}
# import requests
# x0 = requests.get('http://localhost:8000/userid/hello')
# x0.text
# x0.json()


@app.get("/appendname/{user_id}")
async def read_user_name(user_id:str, name:str|None=None):
    name = name or ''
    return {'user_id': user_id+name}
# import requests
# x0 = requests.get('http://localhost:8000/appendname/hel')
# x0 = requests.get('http://localhost:8000/appendname/hel?name=lo')
# x0.text
# x0.json()
#> curl http://127.0.0.1:8000/appendname/hel?name=lo

class EnumString(str, Enum):
    cat = 'cat'
    dog = 'dog'

@app.get("/enum/{pet}")
async def read_pet(pet: EnumString):
    return {'pet': pet}
# import requests
# x0 = requests.get('http://localhost:8000/enum/cat')
# x0 = requests.get('http://localhost:8000/enum/dog')
# x0 = requests.get('http://localhost:8000/enum/doge') #fail

# @app.get("/items/{item_id}")
# async def read_item(item_id: int, q: str| None = None):
#     q = q or "default"
#     return {"item_id": item_id, "q": q}
# import requests
# x0 = requests.get('http://localhost:8000/items/5')
# x0.text
# x0.json()


class Item(pydantic.BaseModel):
    name: str
    price: float
    is_offer: bool|None = None

@app.put("/items/{item_id}")
def update_item(item_id: int, item: Item):
    return {"item_name": item.name, "item_id": item_id}
# import json
# import requests
# data = {"name": "233", "price": 23, "is_offer": True}
# headers = {'Content-Type': 'application/json', 'accept':'application/json'}
# x0 = requests.put('http://localhost:8000/items/5', data=json.dumps(data), headers=headers)
#> curl -X 'PUT' 'http://localhost:8000/items/5' -H 'accept: application/json' -H 'Content-Type: application/json' -d '{"name": "233", "price": 23, "is_offer": true}'
