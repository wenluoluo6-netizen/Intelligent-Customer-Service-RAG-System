from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel, ConfigDict

app = FastAPI()
class TodoBase(BaseModel):
    title: str
    description: str | None = None
    is_completed: bool = False

class TodoOutput(TodoBase):     # 继承 TodoBase 类，增加 id 字段
    id: int
    model_config = ConfigDict(from_attributes=True)     # 定义 model_config 类，用于配置序列化输出

class TodoInput(TodoBase):      # 继承 TodoBase 类
    pass

async def query_pagination(skip: int = 0, limit: int = 100):        # 定义分页函数
    return {"skip": skip, "limit": limit}

fake_db = {}
id_count = 1

# 创建待办 (Create - POST):
# 用户发送 JSON（包含 title 和 description）。
# 后端自动生成一个唯一的 id。
# 默认状态 is_completed 为 False。
# 返回创建成功的对象，状态码 201。
@app.post("/todos/", response_model=TodoOutput, status_code=201)
async def create_todo(todo_in: TodoInput):
    global id_counter
    new_todo = TodoOutput(id=id_counter, **todo_in.dict())
    fake_db[id_counter] = new_todo
    id_counter += 1
    return new_todo

# 查询单个待办 (Read - GET):
# 访问 /todos/{todo_id}。
# 如果 ID 存在，返回数据；如果不存在，抛出 404 错误。
@app.get("/todos/{todo_id}", response_model=TodoOutput)
async def read_todo(todo_id: int):
    if todo_id not in fake_db:
        raise HTTPException(status_code=404, detail="Todo not found")
    return fake_db[todo_id]

# 查询所有待办 (Read - GET):
# 访问 /todos/。
# 进阶要求： 加上分页功能（使用 Depends 复用 Day 6 的代码），支持 skip 和 limit。
@app.get("/todos/", response_model=list[TodoOutput])
async def read_todos(pagination: dict = Depends(query_pagination)):
    all_todos = list(fake_db.values())
    start = pagination["skip"]
    end = start + pagination["limit"]
    return all_todos[start:end]

# 修改待办 (Update - PUT):
# 访问 /todos/{todo_id}
# 用户发送新的 JSON，覆盖旧的数据。
# 如果 ID 不存在，抛出 404。
@app.put("/todos/{todo_id}", response_model=TodoOutput)
async def update_todo(todo_id: int, todo_in: TodoInput):
    if todo_id not in fake_db:
        raise HTTPException(status_code=404, detail="Todo not found")
    fake_db[todo_id] = TodoOutput(id=todo_id, **todo_in.dict())
    return fake_db[todo_id]

# 删除待办 (Delete - DELETE):
# 访问 /todos/{todo_id}。
# 从内存中删除该数据。
# 成功后返回 {"message": "删除成功"}。
@app.delete("/todos/{todo_id}")
async def delete_todo(todo_id: int):
    if todo_id not in fake_db:
        raise HTTPException(status_code=404, detail="Todo not found")
    del fake_db[todo_id]
    return {"message": "删除成功"}

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)