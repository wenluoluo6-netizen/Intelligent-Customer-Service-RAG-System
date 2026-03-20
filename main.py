# 现有导入与设定保持不变
from pathlib import Path
import asyncio
import os
import tempfile
import json
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from memory import chain
from langchain_core.runnables import RunnableConfig

# 你的向量上传等导入保持不变
from vector_ingest import ingest_file_to_vector_store

app = FastAPI(title="LangChain Agent UI")

origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 基础目录与静态资源目录
BASE_DIR = Path(__file__).resolve().parent
INDEX_FILE = BASE_DIR / "index.html"
STATIC_DIR = BASE_DIR / "static"
FRONTEND_DIR = BASE_DIR / "frontend"

# ✅ 【修复】先定义所有静态资源挂载（在路由之前）
# 优先使用 frontend/static，否则使用根目录的 static
if (FRONTEND_DIR / "static").exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR / "static")), name="static")
elif STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/", response_class=HTMLResponse)
async def index():
    # 优先寻找我们新做的二次元页面
    anime_file = BASE_DIR / "anime_theme.html"

    if anime_file.exists():
        html_path = anime_file
    # 如果没找到，再降级使用原来的逻辑
    elif (FRONTEND_DIR / "index.html").exists():
        html_path = FRONTEND_DIR / "index.html"
    else:
        html_path = INDEX_FILE

    if not html_path.exists():
        raise HTTPException(status_code=404, detail=f"{html_path} not found")

    # 读取 HTML 文件并作为响应返回
    return HTMLResponse(html_path.read_text(encoding="utf-8"))

# 用户提交聊天的现有接口
class ChatRequest(BaseModel):
    session_id: str
    input: str
    stream: bool = False

@app.post("/api/chat")
async def chat(req: ChatRequest):
    config = RunnableConfig(configurable={"session_id": req.session_id})
    try:
        # 可能返回 dict、str、对象等，统一处理成 string
        result = await asyncio.to_thread(lambda: chain.invoke({"input": req.input}, config))

        if isinstance(result, dict):
            # 优先取 output
            output = result.get("output")
            if output is None:
                # 兜底：尝试取其他字段或整个对象的字符串表示
                output = result.get("text") or json.dumps(result, default=str)
        else:
            output = str(result)

        return {"output": output}
    except Exception as e:
        # 记录详细错误，方便排错
        import traceback
        import logging
        logging.exception("Error in /api/chat handler: %s", str(e))
        return {"output": f"错误：{str(e)}", "trace": traceback.format_exc()}

# 新增：文件上传并导入向量库
@app.post("/api/upload_vector_file")
async def upload_vector_file(file: UploadFile = File(...), collection: str = "text_search_1"):
    # 文件类型检查同现有逻辑
    allowed_exts = {".txt", ".md", ".pdf"}
    suffix = Path(file.filename).suffix.lower()
    if suffix not in allowed_exts:
        raise HTTPException(status_code=400, detail="Unsupported file type")

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        # 调用导入逻辑
        result = ingest_file_to_vector_store(tmp_path, collection_name=collection)

        # 删除临时文件
        try:
            os.remove(tmp_path)
        except Exception:
            pass

        return {"status": "ok", "collection": collection, "import": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Import failed: {str(e)}")



if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000)



















# from memory import chain
# from langchain_core.runnables import RunnableConfig
#
#
# from dotenv import load_dotenv
# load_dotenv()
#
# config = RunnableConfig(configurable={"session_id": "user-002"})
# print("开始对话（输入 'quit' 退出）")
# while True:
#     question = input("\n输入问题：")
#     if question.lower() in ['quit', 'exit', 'q']:
#         break
#     response = chain.invoke({"input": question}, config)
#     print(f"\n🤖 AI回答: {response['output']}")
#     print("=" * 60)  # 画一条长长的分割线，彻底隔开下一轮的输入提示