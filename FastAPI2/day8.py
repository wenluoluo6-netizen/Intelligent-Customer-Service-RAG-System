from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from passlib.context import CryptContext
from pydantic import BaseModel

app = FastAPI()

# --- 1. 安全配置 ---
# 创建一个密码加密器，算法使用 bcrypt
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# 这是一个传送带：告诉 FastAPI，如果要找 token，就去 /token 这个接口拿
# (这主要为了让 Swagger UI 知道去哪里登录)
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# --- 2. 模拟数据库 ---
# 我们先生成一个 "secret" 的哈希值，假装这是数据库里存的密码
# 运行 pwd_context.hash("secret") 可以得到下面这串乱码
fake_users_db = {
    "admin": {
        "username": "admin",
        "password_hash": "$2b$12$EixZaYVK1fsbw1ZfbX3OXePaWxn96p36WQoeG6Lruj3vjPGga31lW"
    }
}


# (注：上面这个 hash 对应的明文密码是 "secret")

# --- 工具函数 ---
def verify_password(plain_password, hashed_password):
    """比较明文密码和哈希值是否匹配"""
    return pwd_context.verify(plain_password, hashed_password)


# --- 3. 登录接口 (获取 Token) ---
# 注意：这里不用 Pydantic 模型接收，而是用 OAuth2PasswordRequestForm
@app.post("/token")
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    # 1. 查用户
    user_dict = fake_users_db.get(form_data.username)
    if not user_dict:
        raise HTTPException(status_code=400, detail="用户名或密码错误")

    # 2. 查密码
    # form_data.password 是用户输入的明文，user_dict['password_hash'] 是存的乱码
    if not verify_password(form_data.password, user_dict["password_hash"]):
        raise HTTPException(status_code=400, detail="用户名或密码错误")

    # 3. 发 Token
    # 今天我们先简单地把用户名当做 token 返回。明天我们会把这里升级成真正的 JWT。
    return {"access_token": user_dict["username"], "token_type": "bearer"}


# --- 4. 受保护的接口 ---
@app.get("/users/me")
async def read_users_me(token: str = Depends(oauth2_scheme)):
    # oauth2_scheme 会自动从 Header 提取 token。
    # 如果 Header 里没有 Authorization: Bearer xxx，这里直接报错，进不来。
    return {"token": token, "msg": "你能看到这句话，说明你通过了身份验证！"}


if __name__ == '__main__':
    import uvicorn
    uvicorn.run("day8:app", host="127.0.0.1", port=8000,reload=True)