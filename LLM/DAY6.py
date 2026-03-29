import os
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel


# 1. 加载配置
load_dotenv()
API_KEY = os.getenv("API_KEY")
BASE_URL = os.getenv("BASE_URL")

# 2. 定义请求的数据格式 (点单卡)
class QueryRequest(BaseModel):
    question: str

# --- 全局变量 ---
# 我们把 RAG 链条放在这里，等会儿启动的时候再填充它
rag_chain = None

# 3. 核心逻辑：初始化 RAG 系统
# 这个函数只在服务器启动时运行一次
def init_rag_system():
    global rag_chain
    print("正在初始化 RAG 系统...")

    # 定义聊天模型和向量模型
    llm = ChatOpenAI(model="glm-4", temperature=0.1, openai_api_key=API_KEY, openai_api_base=BASE_URL)
    embeddings = OpenAIEmbeddings(model="embedding-2", openai_api_key=API_KEY, openai_api_base=BASE_URL)

    # 加载/切割数据
    loader = TextLoader("./my_info.txt", encoding="utf-8")
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
    splits = splitter.split_documents(docs)

    # 构建向量数据库
    vectorstore = FAISS.from_documents(splits, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})   # 构建检索器
    print("✅ 知识库构建完成！")

    # 定义 Prompt 和 Chain
    template = """使用以下上下文来回答最后的问题。如果你不知道，就诚实地说不知道。
        上下文: {context}

        问题: {question}
        """
    prompt = ChatPromptTemplate.from_template(template)
    rag_chain = ({"context":retriever, "question":RunnablePassthrough()} | prompt | llm | StrOutputParser())
    print("✅ RAG 系统初始化完毕！")

# 4. 使用 lifespan
@asynccontextmanager
async def lifespan(app:FastAPI):
    print("🔥 系统启动中：正在加载 RAG 模型和 FAISS 数据库...")
    init_rag_system()

    yield
    print("🛑 系统关闭中：正在清理资源...")

app = FastAPI(title="我的 RAG 智能问答系统", lifespan=lifespan)

# 5. 定义API接口
@app.post("/chat")
async def chat(request: QueryRequest):
    if not rag_chain:
        raise HTTPException(status_code=500, detail="系统还在初始化，请稍后再试")
    print(f"收到请求: {request.question}")
    try:
        answer = rag_chain.invoke(request.question)
        return {"answer": answer}
    except Exception as e:
        return {"error": e}

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)