from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

# 定义输入数据的结构
class Request(BaseModel):
    text: str
    threshold: float = 0.5

# 定义输出数据的结构
class Response(BaseModel):
    label: str
    score: float
    is_trusted: bool

# 定义接口
@app.post("/predict", response_model=Response)
async def predict(request: Request):

    input = request.text
    threshold = request.threshold

    # 模拟AI计算逻辑
    calculated_score = min(len(input) / 20, 1.0)

    if calculated_score > threshold:
        label_result = "positive"
    else:
        label_result = "negative"

    return {
        "label": label_result,
        "score": calculated_score,
        "is_trusted": True
    }

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)