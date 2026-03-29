import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 加载环境变量
load_dotenv()
API_KEY = os.getenv("API_KEY")
BASE_URL = os.getenv("BASE_URL")

# 初始化模型
llm = ChatOpenAI(
    model="glm-4",
    temperature=0.1,
    openai_api_key=API_KEY,
    openai_api_base=BASE_URL
)
embeddings = OpenAIEmbeddings(
    model="embedding-2",
    api_key=API_KEY,
    base_url=BASE_URL,
)

# --- 2. 数据准备 (模拟加载和入库) ---
long_text = """
智谱AI开放平台（BigModel Open Platform）是基于千亿级参数大模型 GLM-4 构建的。
GLM-4 拥有强大的通用能力，支持长文本、多模态。
使用智谱 API 需要申请 API Key。
目前的收费标准是：GLM-4-Flash 是免费的，GLM-4 标准版是收费的。
调用 Embeddings 模型时，推荐使用 embedding-2。
注意：不要把 API Key 泄露给他人。
"""

# 把它存成临时文件模拟一下
file_name = "knowledge.txt"     # 定义文件名
with open(file_name, "w", encoding="utf-8") as f:
    f.write(long_text)

# 加载 & 切割
loader = TextLoader(file_name, encoding="utf-8")
docs = loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
splits = splitter.split_documents(docs)

# 向量化存入库中
vectorstore = FAISS.from_documents(splits, embeddings)
print("✅ 知识库构建完成！")

# --- 3. 构建 Retriever (检索器) ---      这一步把数据库变成了一个可调用的组件，search_kwargs={"k": 2} 表示每次只找最相关的 2 条
retriever = vectorstore.as_retriever(search_kwargs={"k":2})

# --- 4. 构建 RAG Chain (核心中的核心) ---      定义提示词模板，context 就是检索出来的文档，question 是用户的问题
template = """你是一个智能助手。请严格根据下面的上下文（Context）回答问题。
如果上下文中没有提到答案，就说你不知道，不要瞎编。

Context:
{context}

Question:
{question}
"""
prompt = ChatPromptTemplate.from_template(template)

# 定义一个辅助函数：把检索到的文档列表拼成一个长字符串
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# 组装流水线
# 1. RunnablePassthrough() 只是占位符，把用户的输入传给 question
# 2. context 这一行会去调用 retriever，搜出文档，然后用 format_docs 拼成字符串
rag_chain = ({"context": retriever | format_docs, "question": RunnablePassthrough()} | prompt | llm | StrOutputParser())

# --- 5. 运行测试 ---
query = "智谱的 embedding 模型推荐用哪个？"
print(f"\n用户提问: {query}")
print("正在思考...")

# invoke 的时候，只需要传字符串，因为上面的 Chain 会自动处理检索
result = rag_chain.invoke(query)
print(f"AI 回答: {result}")

# 测试一个不知道的问题
query2 = "今天晚上吃什么？"
print(f"\n用户提问: {query2}")
print(f"AI 回答: {rag_chain.invoke(query2)}")
