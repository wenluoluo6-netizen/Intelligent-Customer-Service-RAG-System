from contextlib import asynccontextmanager
from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy import create_engine
from sqlmodel import SQLModel, Session, select, Field


class Book(SQLModel, table=True):
    __table_args__ = {"extend_existing": True}
    id: int | None = Field(default=None, primary_key=True)
    title: str
    author: str
    price: float

# 数据库配置
sql_url = "sqlite:///book.db"
engine = create_engine(sql_url)

# 创建所有表
def create_db_and_tables():
    SQLModel.metadata.create_all(engine)

# 添加Session依赖
def get_session():
    with Session(engine) as session:
        yield session

# 生命周期
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("开始运行")
    create_db_and_tables()
    yield
    print("结束运行")

app = FastAPI(lifespan=lifespan)

# 添加图书接口
@app.post("/book/", response_model=Book)
async def create_book(book: Book, session: Session = Depends(get_session)):
    session.add(book)
    session.commit()
    session.refresh(book)
    return book

# 添加查询单个图书接口
@app.get("/books/{book_id}/", response_model=Book)
async def get_book(book_id: int, session: Session = Depends(get_session)):
    book = session.get(Book, book_id)
    if book is None:
        raise HTTPException(status_code=404, detail="Book not found")
    return book

# 添加查询所有图书接口
@app.get("/books/", response_model=list[Book])
async def get_books(session: Session = Depends(get_session)):
    statement = select(Book)
    result = session.exec(statement)
    return result.all()

# 添加删除图书接口
@app.delete("/books/{book_id}/", response_model=Book)
async def delete_book(book_id: int, session: Session = Depends(get_session)):
    book = session.get(Book, book_id)
    if book is None:
        raise HTTPException(status_code=404, detail="Book not found")
    session.delete(book)
    session.commit()
    return {"message": "Book deleted successfully"}

# 添加更新图书接口
@app.put("/books/{book_id}/", response_model=Book)
async def update_book(book_id: int, book: Book, session: Session = Depends(get_session)):
    db_book = session.get(Book, book_id)
    if db_book is None:
        raise HTTPException(status_code=404, detail="Book not found")
    update_data = book.dict(exclude_unset=True)
    for key, value in update_data.items():
        setattr(db_book, key, value)
    session.add(db_book)
    session.commit()
    session.refresh(db_book)
    return db_book

if __name__ == '__main__':
    import uvicorn
    uvicorn.run("day5:app", host='127.0.0.1', port=8000, reload=True)



