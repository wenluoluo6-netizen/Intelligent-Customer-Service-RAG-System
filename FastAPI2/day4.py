from fastapi import FastAPI, Header, HTTPException, Depends

app = FastAPI()

async def verify_key(x_secret_key: str = Header(...)):
    if x_secret_key != "supersecret":
        raise HTTPException(status_code=400, detail="秘钥错误")

async def page_info(page: int = 1, size: int = 10):
    return {"page": page, "size": size}

@app.get("/archives")
async def get_archives(key: str = Depends(verify_key), page: int = Depends(page_info)):
    return {"archives": "获取成功", "page": page}

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)