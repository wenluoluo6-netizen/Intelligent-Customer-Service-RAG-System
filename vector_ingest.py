
import os
import tempfile
from typing import List
import json
from dotenv import load_dotenv
load_dotenv()

from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    CSVLoader,
    TextLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pymilvus import MilvusClient
from model import embeddings  # 复用现有的 embeddings 实例


def _extract_text_from_file(file_path: str) -> List[str]:
    """
        提取文件文本，支持 PDF, Word, CSV, TXT, Markdown 等主流格式
    """
    ext = os.path.splitext(file_path)[1].lower()
    texts: List[str] = []

    # print(f"正在解析文件: {file_path} (格式: {ext})")

    try:
        # 1. 智能分发加载器
        if ext == ".pdf":
            loader = PyPDFLoader(file_path)
            docs = loader.load()

        elif ext in [".docx", ".doc"]:
            # 处理 Word 文档 (需提前 pip install docx2txt)
            loader = Docx2txtLoader(file_path)
            docs = loader.load()

        elif ext == ".csv":
            # 处理 CSV 表格
            loader = CSVLoader(file_path, encoding="utf-8")
            docs = loader.load()

        elif ext in [".txt", ".md"]:
            # 处理纯文本和 Markdown
            loader = TextLoader(file_path, encoding="utf-8")
            docs = loader.load()

        else:
            # 2. 兜底方案：如果是不认识的后缀，尝试当作普通文本强读
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
                if content.strip():
                    texts.append(content)
            return texts

        # 3. 统一提取 Document 对象中的文本内容
        # LangChain 的 loader 返回的都是 [Document(page_content="...")] 格式
        texts = [d.page_content for d in docs if getattr(d, "page_content", None)]

    except Exception as e:
        print(f"❌ 读取文件 {file_path} 时发生错误: {str(e)}")

    return texts

def ingest_file_to_vector_store(file_path: str, collection_name: str = "text_search_1") -> dict:
    """
    将指定文件的文本提取、分块、向量化并导入 Milvus。
    返回导入结果摘要，例如 {'count': N, 'vectors_inserted': N}
    """
    # 提取文本
    texts = _extract_text_from_file(file_path)
    if not texts:
        return {"count": 0, "vectors_inserted": 0, "error": "No text extracted"}

    # 将文本分块
    splitter = RecursiveCharacterTextSplitter(chunk_size=350, chunk_overlap=50)
    # 将每段文本合并成一个 Documents 列表以便分块
    # 在这里简单实现：将 texts 作为单一“文档流”分块
    docs = []
    for t in texts:
        # 简单包装为 page_content，模拟 LangChain 文档对象
        from langchain_core.documents import Document
        docs.extend(splitter.split_documents([Document(page_content=t)]))

    # 过滤掉空文本
    texts_to_embed = [d.page_content for d in docs if getattr(d, "page_content", None)]
    texts_to_embed = [t for t in texts_to_embed if t is not None]
    texts_to_embed = [" ".join(t.replace("\n", " ").split()) for t in texts_to_embed]

    if not texts_to_embed:
        return {"count": 0, "vectors_inserted": 0, "error": "No valid text after cleaning"}

    # 向量化
    vectors = embeddings.embed_documents(texts_to_embed)
    data = []
    for i, text in enumerate(texts_to_embed):
        data.append({"id": i, "vector": vectors[i], "text": text})      # vector是量化后的向量值，text是量化之前的原始数据

    # Milvus 插入
    milvus_client = MilvusClient(uri=os.getenv("MILVUS_URI"), token=os.getenv("MILVUS_TOKEN"))
    if milvus_client.has_collection(collection_name):
        # 直接使用现有集合
        pass
    else:
        # 假设 embedding 维度保持一致，这里用文本向量的长度作为维度
        dim = len(vectors[0]) if vectors else 1024
        milvus_client.create_collection(collection_name=collection_name, dimension=dim)

    res = milvus_client.insert(collection_name=collection_name, data=data)

    # 尝试将 Milvus 的返回结果转换为 JSON 友好形式
    milvus_serializable = None
    try:
        milvus_serializable = json.loads(json.dumps(res, default=lambda o: str(o)))
    except Exception:
        # 如果完全无法序列化，就降级成字符串描述
        milvus_serializable = str(res)

    return {
        "count": len(data),
        "vectors_inserted": len(data),
        "milvus": milvus_serializable
    }