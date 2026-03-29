from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class Item(BaseModel):
    name: str
    description: str | None = None
    price: float
    tax: float | None = None

@app.post("/items/")
async def create_item(item: Item):
    result_dict = item.model_dump()     # 将Item对象转换为字典
    if item.tax:
        price_with_tax = item.price + item.tax
        price_with_tax.update({"price_with_tax":price_with_tax})
    return result_dict


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='127.0.0.1', port=8000)

