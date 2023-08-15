import typing
import fastapi
import pydantic

app = fastapi.FastAPI()


class Item(pydantic.BaseModel):
    name: str
    price: float
    is_offer: typing.Union[bool, None] = None

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get("/items/{item_id}")
def read_item(item_id: int, q: typing.Union[str, None] = None):
    return {"item_id": item_id, "q": q}

@app.put("/items/{item_id}")
def update_item(item_id: int, item: Item):
    return {"item_name": item.name, "item_id": item_id}

# http://localhost:8000/doc
# http://localhost:8000/redoc
# uvicorn draft00:app --reload
# curl http://localhost:8000
# curl http://127.0.0.1:8000/
# curl http://127.0.0.1:8000/items/5?q=somequery
# curl -X 'PUT' 'http://localhost:8000/items/5' -H 'accept: application/json' -H 'Content-Type: application/json' -d '{"name": "233", "price": 23, "is_offer": true}'

def demo_requests():
    import requests
    x0 = requests.get('http://localhost:8000/items/5')
    x0.text
    x0.json()

    import json
    data = {"name": "233", "price": 23, "is_offer": True}
    headers = {'Content-Type': 'application/json', 'accept':'application/json'}
    x0 = requests.put('http://localhost:8000/items/5', data=json.dumps(data), headers=headers)
