import os
from contextlib import asynccontextmanager
from typing import List

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# LangChain & Milvus 组件
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pymilvus import MilvusClient

# 1. 加载环境变量
load_dotenv()

# --- 全局变量 ---
milvus_client = None
embeddings = None
llm = None
bm25_retriever = None


# --- 2. 核心组件初始化函数 ---
def init_system():
    global milvus_client, embeddings, llm, bm25_retriever

    print("⚙️ 正在初始化 AI 系统...")

    # A. 初始化模型
    llm = ChatOpenAI(
        model="glm-4",
        api_key=os.getenv("API_KEY"),
        base_url=os.getenv("BASE_URL"),
        temperature=0.1
    )
    embeddings = OpenAIEmbeddings(
        model="embedding-2",
        api_key=os.getenv("API_KEY"),
        base_url=os.getenv("BASE_URL")
    )

    # B. 连接 Milvus 数据库
    milvus_client = MilvusClient(uri=os.getenv("MILVUS_URI"), token=os.getenv("MILVUS_TOKEN"))

    collection_name = "my_collections"
    # 如果表不存在，先建个空的
    if not milvus_client.has_collection(collection_name):
        print("⚠️ 警告：未发现知识库集合，请先运行 init_data.py！")
        # 即使为空也先建一个，防止报错
        milvus_client.create_collection(collection_name=collection_name, dimension=1024)

    # C. 构建 BM25 检索器 (关键词检索)
    print("📥 正在加载数据构建 BM25 索引...")
    # 从 Milvus 拉取所有文本
    results = milvus_client.query(collection_name=collection_name, filter="id >= 0", output_fields=["text"])

    if len(results) > 0:
        docs = [Document(page_content=res["text"]) for res in results]
        # 初始化 BM25
        bm25_retriever = BM25Retriever.from_documents(docs)
        bm25_retriever.k = 3
        print(f"✅ BM25 索引构建完成，共加载 {len(docs)} 条数据。")
    else:
        print("⚠️ 数据库为空，跳过 BM25 构建。")

    print("✅ 系统初始化完成！")


# --- 3. 定义 Rerank (重排序) 函数 ---
def rerank_documents(query: str, docs: List[Document]) -> List[Document]:
    if not docs:
        return []

    # 简化的 Rerank 提示词
    doc_txts = "\n".join([f"[{i}] {d.page_content}" for i, d in enumerate(docs)])
    prompt = f"请从下面选出最能回答问题'{query}'的文档编号，只返回编号数字。若无相关返回-1。\n{doc_txts}"

    try:
        res = llm.invoke(prompt).content.strip()
        idx = int(res)
        if 0 <= idx < len(docs):
            return [docs[idx]]
    except:
        pass
    return docs[:2]


# --- 4. FastAPI 生命周期 ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    init_system()
    yield
    print("🛑 系统已关闭")


app = FastAPI(title="终极 RAG 知识库 API", lifespan=lifespan)


class ChatRequest(BaseModel):
    question: str
    collection: str = "my_collections"


# --- 5. 定义核心问答接口 ---
@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    global milvus_client, bm25_retriever

    question = request.question
    print(f"📩 收到提问: {question}")

    candidate_docs = []

    # === A 路：向量检索 (Milvus) ===
    # 擅长模糊匹配，如 "上网密码" -> "Wi-Fi密码"
    try:
        query_vector = embeddings.embed_query(question)
        search_res = milvus_client.search(
            collection_name=request.collection,
            data=[query_vector],
            limit=3,
            output_fields=["text"]
        )
        vector_docs = [Document(page_content=hit['entity']['text']) for hit in search_res[0]]
        print(f"🔍 Milvus 召回: {len(vector_docs)} 条")
        candidate_docs.extend(vector_docs)
    except Exception as e:
        print(f"Milvus 检索出错: {e}")

    # === B 路：关键词检索 (BM25) ===
    # 擅长精确匹配，如 "wifi" -> "Wi-Fi"
    if bm25_retriever:
        keyword_docs = bm25_retriever.invoke(question)
        print(f"🔍 BM25 召回: {len(keyword_docs)} 条")
        candidate_docs.extend(keyword_docs)

    # === 合并去重 ===
    # 不同的路可能搜到同一条，通过内容去重
    unique_docs = {d.page_content: d for d in candidate_docs}.values()
    final_candidates = list(unique_docs)
    print(f"⚡ 混合检索去重后: {len(final_candidates)} 条")

    # === Rerank 精排 ===
    final_docs = rerank_documents(question, final_candidates)
    print(f"⚖️ Rerank 最终保留: {len(final_docs)} 条")

    # === 生成回答 ===
    context_str = "\n\n".join([d.page_content for d in final_docs])

    if not context_str:
        return {"answer": "抱歉，知识库里没有找到相关内容。"}

    system_prompt = """你是一个专业的企业知识库助手。
    请根据以下检索到的上下文（Context）回答用户问题。
    如果上下文没有答案，请直接说不知道。

    Context:
    {context}
    """

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{question}")
    ])

    chain = prompt | llm | StrOutputParser()
    answer = chain.invoke({"context": context_str, "question": question})

    return {
        "question": question,
        "answer": answer,
        "source": [d.page_content for d in final_docs]
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)