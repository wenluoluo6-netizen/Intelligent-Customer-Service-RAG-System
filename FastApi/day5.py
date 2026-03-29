from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()

class UserIn(BaseModel):
    username: str
    password: str
    email: str

class UserOut(BaseModel):
    password: str
    email: str

@app.post("/users", response_model=UserOut, status_code=201)
async def create_user(user: UserIn):
    return user

items = ["foo", "bar", "baz"]
@app.get("/items/{item_id}")
async def read_item(item_id: int):
    if 0 <= item_id <= 2:
        return items[item_id]
    else:
        raise HTTPException(status_code=404, detail="Item not found")

