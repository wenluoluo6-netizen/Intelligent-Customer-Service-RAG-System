import time
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# --- 1. 配置 CORS (跨域) ---
# 这是最常见的配置，专门解决 "Access-Control-Allow-Origin" 报错
origins = [
    "http://localhost",
    "http://localhost:3000",  # 假设你的 React/Vue 前端在这里
    "http://127.0.0.1:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # 允许的来源列表
    allow_credentials=True,  # 是否允许携带 Cookie
    allow_methods=["*"],  # 允许的方法 (GET, POST, DELETE...)，"*" 表示全部
    allow_headers=["*"],  # 允许的请求头
)


# --- 2. 自定义中间件 (统计耗时) ---
# @app.middleware("http") 装饰器表示这是一个拦截 HTTP 请求的中间件
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    # [进门前]: 记录开始时间
    start_time = time.time()

    # [放行]: call_next(request) 会去寻找并执行对应的 API 路由函数
    # await 等待它执行完毕，拿到响应 response
    response = await call_next(request)

    # [出门后]: 计算耗时
    process_time = time.time() - start_time

    # 将耗时写入响应头 Header
    # 注意：Header 的值必须是字符串
    response.headers["X-Process-Time"] = str(process_time)

    # 最终返回响应
    return response


# --- 测试接口 ---
@app.get("/")
async def read_root():
    # 模拟一个耗时操作，睡 0.1 秒
    time.sleep(0.1)
    return {"message": "Hello Middleware"}


if __name__ == '__main__':
    import uvicorn
    uvicorn.run("day12:app", host="127.0.0.1", port=8000, reload=True)