from fastapi import FastAPI

app = FastAPI()

# 根据用户ID获取用户信息
@app.get("/users/{user_id}")

async def get_user(user_id: int):
    return {"user_id": user_id, "message": "这是用户信息"}

# 目标： 编写一个模拟“获取假数据列表”的接口，支持分页功能。
# 要求：
# URL 路径设计为 /fake-items/。
# 接收两个参数：skip (跳过多少条，默认 0) 和 limit (取多少条，默认 10)。
# 返回一个包含对应数量假数据的列表。
fake_items_db = [{"item_name": "Foo"}, {"item_name": "Bar"}, {"item_name": "Baz"}]
@app.get("/fake-items/")
async def get_fake_items(skip: int = 0, limit: int = 10):
    return fake_items_db[skip:skip+limit]

# 任务 3：混合使用 & 可选参数
# 目标： 编写一个复杂的接口，既有路径参数，又有查询参数，还有一个可选的查询参数。
# 要求：
# URL 设计为 /users/{user_id}/items/。
# 必填参数：user_id (路径参数)。
# 可选参数：q (查询参数，代表搜索关键字)。如果用户没传 q，则值为 None。
# 提示： 在 Python 中表示“可选且默认为空”，可以使用 q: str | None = None (Python 3.10+) 或 q: Optional[str] = None。
@app.get("/users/{user_id}/items/")
async def get_user_items(user_id: int, q: str | None = None):
    item = {"item_id": "item_001", "owner_id": user_id}
    if q:
        item.update({"q": q})
    return item

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='127.0.0.1', port=8000)

