import os
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import PydanticOutputParser, BaseOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field

# --- 1. 准备环境与数据 (复用之前的逻辑) ---
load_dotenv()
API_KEY = os.getenv('API_KEY')
BASE_URL = os.getenv('BASE_URL')

llm = ChatOpenAI(
    model="glm-4",
    temperature=0.1,
    openai_api_key=API_KEY,
    openai_api_base=BASE_URL
)
embeddings = OpenAIEmbeddings(
    model="embedding-2",
    openai_api_key=API_KEY,
    openai_api_base=BASE_URL,
)

# 模拟一些“刁钻”的数据，测试改写效果
text = """
1. 如果遇到 SSH 连接超时，请检查防火墙 22 端口是否开放，或者检查服务器负载。
2. 数据库连接失败通常是因为 max_connections 参数设置过小，或者密码错误。
3. 申请发票需要填写纳税人识别号，且金额需大于 100 元，通常 3 个工作日寄出。
"""
file_name = "day8.txt"
with open(file_name, "w", encoding="utf-8") as f:
    f.write(text)
loader = TextLoader(file_name, encoding="utf-8")    # 读取数据
docs = loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)   # 切分数据
splits = splitter.split_documents(docs)
vectorstore = FAISS.from_documents(splits, embeddings)   # 向量化并存入数据库
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})   # 构建检索器
print("✅ 知识库构建完成！")

# --- 2. 核心：定义查询改写器 (Query Transform) ---
# 定义输出格式：我们需要 AI 返回一个字符串列表
class LineList(BaseModel):
    lines: list[str] = Field(..., description="3 queries")
class LineListOutputParser(BaseOutputParser):
    def parse(self, text: str) -> LineList:
        # 这里的逻辑是你自己写的，专门处理换行符
        lines = [line.strip() for line in text.strip().split("\n") if line.strip()]
        return LineList(lines=lines)
output_parser = LineListOutputParser()

QUERY_PROMPT = ChatPromptTemplate.from_template(
    """你是一个 AI 语言模型助手。你的任务是为给定的用户问题生成 3 个不同版本的搜索查询，
以便从向量数据库中检索相关文档。
通过生成关于用户问题的多个视角，你的目标是帮助用户克服基于距离的相似性搜索的一些局限性。
请直接分行输出这 3 个查询，不要包含编号或其他废话。

原始问题: {question}
"""
)
# 组装改写链： 输入问题 -> LLM -> 3个新问题
llm_chain = QUERY_PROMPT | llm | output_parser

# --- 3. 定义检索逻辑 (Multi-Query Retrieval) ---
def retrieval_chain(original_question):
    print(f"🔄 正在改写问题: {original_question}")
    # 1. 生成 3 个新问题
    queries = llm_chain.invoke({"question": original_question})
    print(f"✅ 生成的查询变体: {queries.lines}")
    # 2. 并行检索 (分别拿这 3 个问题去搜)
    all_docs = []
    for q in queries.lines:
        docs = retriever.invoke(q)
        all_docs.extend(docs)
    # 3. 去重 (Deduplicate) 不同的问题可能会搜出同一段话，需要去重
    unique_doc = {}
    for doc in all_docs:
        unique_doc[doc.page_content] = doc
    return list(unique_doc.values())

# --- 4. 测试效果 ---
query = "连不上服务器怎么办？"

# 普通检索（作为对比）
print("\n--- 普通检索结果 ---")
normal_docs = retriever.invoke(query)
for doc in normal_docs:
    print(f"-{doc.page_content}")

# 多路改写检索
print("\n--- 多路改写检索结果 ---")
multi_docs = retrieval_chain(query)
for doc in multi_docs:
    print(f"-{doc.page_content}")