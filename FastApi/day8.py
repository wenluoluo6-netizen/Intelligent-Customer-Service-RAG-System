from contextlib import asynccontextmanager
from fastapi import FastAPI, Depends, HTTPException
from sqlmodel import SQLModel, Field, Session, select, create_engine


# 继承自SQLModel，并且table=True，表示这是一个数据库表
class Todo(SQLModel, table=True):
    __table_args__ = {"extend_existing": True}
    id: int = Field(default=None, primary_key=True)
    title: str
    description: str | None = Field(default=None)
    is_completed: bool = False

# 数据库连接配置, 这里使用 SQLite 作为数据库，数据库文件名，如果不存在则自动创建
sqlite_file_name = "todo.db"
sqlite_url = f"sqlite:///{sqlite_file_name}"    # 数据库地址

# check_same_thread=False 是 SQLite 在多线程环境下的特殊配置，它会关闭数据库连接池，使得每个线程都能独占一个连接。connect_args是进出规则
connect_args = {"check_same_thread": False}

# 创建数据库引擎，这行代码执行时，并不会真的去连数据库，它只是把配置记下来。
engine = create_engine(sqlite_url, connect_args=connect_args)

# 这是一个工具函数，用于在启动前创建所有表
def create_db_and_tables():
    SQLModel.metadata.create_all(engine)

# 生命周期管理：在 App 启动时运行 create_db_and_tables
@asynccontextmanager        # 这是一个上下文管理器，你可以把它理解为"开店"和"关店"的流程。
async def lifespan(app: FastAPI):
    create_db_and_tables()
    print("数据库连接成功，表已就绪！")
    yield           # --- 暂停键 --- 程序会卡在这里，FastAPI 开始正常接收用户的请求（处理 GET/POST...）。
    print("数据库连接关闭")    # 这里的代码会在服务器关闭前执行。

app = FastAPI(lifespan=lifespan)

# 定义获取Session的生成器函数，它的意思是：只要在 with 代码块里，session 就是打开的；一旦代码跑出去了，它会自动执行 session.close()，把窗口关掉。
def get_session():
    with Session(engine) as session:
        yield session

# CREATE
@app.post("/todo", response_model=Todo)
async def create_todo(todo: Todo, session: Session = Depends(get_session)):
    session.add(todo)       # 1. 放入会话
    session.commit()        # 2. 提交事务 (真正写入文件)
    session.refresh(todo)   # 3. 刷新对象 (为了拿到数据库自动生成的 id)
    return todo

# READ
@app.get("/todo/{todo_id}", response_model=Todo)
async def get_todo(todo_id: int, session: Session = Depends(get_session)):
    todo = session.get(Todo, todo_id)   # session.get 是最常用的根据主键查找的方法
    if not todo:
        raise HTTPException(status_code=404, detail="Todo not found")
    return todo

# READ ALL
@app.get("/todo", response_model=list[Todo])
async def get_todos(session: Session = Depends(get_session)):
    todos = session.exec(select(Todo)).all()
    return todos

# UPDATE
@app.put("/todo/{todo_id}", response_model=Todo)
async def update_todo(todo_id: int, todo: Todo, session: Session = Depends(get_session)):
    db_todo = session.get(Todo, todo_id)
    if not db_todo:
        raise HTTPException(status_code=404, detail="Todo not found")
    db_todo.title = todo.title
    db_todo.description = todo.description
    db_todo.is_completed = todo.is_completed
    session.add(db_todo)
    session.commit()
    session.refresh(db_todo)
    return db_todo

# DELETE
@app.delete("/todo/{todo_id}")
async def delete_todo(todo_id: int, session: Session = Depends(get_session)):
    db_todo = session.get(Todo, todo_id)
    if not db_todo:
        raise HTTPException(status_code=404, detail="Todo not found")
    session.delete(db_todo)    # 标记为删除
    session.commit()        # 提交事务 (真正删除文件)
    return {"message": "Todo deleted"}

if __name__ == '__main__':
    import uvicorn
    uvicorn.run("day8:app", host='127.0.0.1', port=8000, reload=True)