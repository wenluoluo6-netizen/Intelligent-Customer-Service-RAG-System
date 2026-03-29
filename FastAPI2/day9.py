from datetime import datetime, timedelta
from typing import Union

from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel

app = FastAPI()

# --- 1. 配置区域 (真实项目中这些应该放在配置文件里) ---
SECRET_KEY = "your_super_secret_key_change_this_in_production"  # 私钥，千万别泄露
ALGORITHM = "HS256"  # 加密算法
ACCESS_TOKEN_EXPIRE_MINUTES = 30  # Token 有效期 30 分钟

# --- 2. 安全工具初始化 ---
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# --- 3. 模拟数据库  ---
fake_users_db = {
    "admin": {
        "username": "admin",
        # 这是 'secret' 对应的正确 bcrypt 哈希值
        "password_hash": "$2b$12$EixZaYVK1fsbw1ZfbX3OXePaWxn96p36WQoeG6Lruj3vjPGga31lW"
    }
}


# --- 4. 核心工具函数 ---

def verify_password(plain_password, hashed_password):
    """验证密码"""
    return pwd_context.verify(plain_password, hashed_password)


def create_access_token(data: dict, expires_delta: Union[timedelta, None] = None):
    """生成 JWT Token 的核心函数"""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)

    # 关键步骤：把过期时间写入字典
    to_encode.update({"exp": expire})
    # 使用私钥和算法进行签名
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


async def get_current_user(token: str = Depends(oauth2_scheme)):
    """
    安检员依赖：
    1. 自动从 Header 拿 Token
    2. 解析 Token
    3. 验证 Token 是否合法/过期
    4. 返回当前用户
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="无法验证凭证 (Token无效或已过期)",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        # 解码 JWT
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")  # 按照惯例，用户名放在 'sub' 字段
        if username is None:
            raise credentials_exception
    except JWTError:
        # 如果签名不对，或者 Token 格式不对，jwt 库会报错
        raise credentials_exception

    # (可选) 拿着解出来的用户名去数据库查一下还在不在
    user = fake_users_db.get(username)
    if user is None:
        raise credentials_exception
    return user


# --- 5. 接口定义 ---

@app.post("/token")
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    print(f"DEBUG: 接收到的用户名: {form_data.username}")
    print(f"DEBUG: 接收到的密码长度: {len(form_data.password)}")  # 看看长度是不是 6
    print(f"DEBUG: 接收到的密码内容: {form_data.password}")  # 看看内容到底是不是 secret
    # 1. 验证账号密码
    user = fake_users_db.get(form_data.username)
    if not user or not verify_password(form_data.password, user["password_hash"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="用户名或密码错误",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # 2. 生成 JWT
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user["username"]}, expires_delta=access_token_expires
    )

    # 3. 返回 JWT
    return {"access_token": access_token, "token_type": "bearer"}


@app.get("/users/me")
async def read_users_me(current_user: dict = Depends(get_current_user)):
    # 只要加上 Depends(get_current_user)，能进到这个函数，说明已经是登录用户了
    return {
        "msg": "验证成功！这是你的加密信息",
        "username": current_user["username"],
        "data": "这里只有登录用户能看到"
    }

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app='day9:app', host='127.0.0.1', port=8000, reload=True)