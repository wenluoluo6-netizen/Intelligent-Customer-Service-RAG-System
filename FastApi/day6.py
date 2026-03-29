from fastapi import FastAPI, Depends, Header, HTTPException

app = FastAPI()

# 任务 1：消除重复代码 (Shared Parameters)
# 场景： 假设你有 /items/ 和 /users/ 两个接口，它们都需要分页查询（skip, limit）。
# 要求：
# 定义一个依赖函数 query_pagination，接收 skip 和 limit。
# 在两个路由中都注入这个依赖。
# 接口直接返回分页参数字典，证明注入成功。

# 定义依赖函数
async def query_pagination(skip: int = 0, limit: int = 10):
    return {"skip": skip, "limit": limit}

@app.get("/items/")
async def get_items(pagination: dict = Depends(query_pagination)):
    return pagination

@app.get("/users/")
async def get_users(pagination: dict = Depends(query_pagination)):
    return pagination


# 任务 2：逻辑拦截与验证 (Security Dependency)
# 场景： 某些接口（如“删除数据”）非常敏感，需要检查用户请求头里有没有带 token。
# 要求：
# 定义一个依赖函数 verify_token。
# 逻辑：读取 Header 中的 x-token。如果不等于 "supersecret"，则 raise HTTPException（直接拦截，报错 400）。
# 创建一个 /admin/ 接口，注入这个依赖。
# 效果： 如果没带 Token，接口代码根本不会运行，直接被依赖函数挡回去。

# 定义依赖函数
async def verify_token(x_token: str = Header(...)):
    if x_token != "supersecret":
        raise HTTPException(status_code=400, detail="Invalid token")
    return x_token

@app.get("/admin/")
async def admin(token: str = Depends(verify_token)):
    return {"message": "Welcome to admin page"}

if __name__ == '__main__':
    import uvicorn
    uvicorn.run("day6:app", host="127.0.0.1", port=8000, reload=True)