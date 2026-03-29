import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings   # 用来把字变成数字
from langchain_community.vectorstores import FAISS      # 向量数据库
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 1. 加载环境变量 (必须最先执行)
load_dotenv()
API_KEY = os.getenv("API_KEY")
BASE_URL = os.getenv("BASE_URL")

# 2. 加载语言模型
embeddings_model = OpenAIEmbeddings(
    model="embedding-2",
    api_key=API_KEY,
    base_url=BASE_URL
)
print("模型配置完成！准备开始加载数据...")

# 3.加载数据      
loader = TextLoader("./test.txt", encoding="utf-8")
docs = loader.load()
print("正在清洗数据...")
for doc in docs:
    raw_content = doc.page_content
    clean_content = "".join(raw_content.split())
    doc.page_content = clean_content

print("数据加载完成！开始生成对话...")

# 4.创建切分器
spliter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=40,
    separators=[
            "\n\n",
            "\n",
            "。",
            "；",  # 新增：分号
            "！",  # 新增：感叹号
            "，",  # 新增：逗号 (优先级最低，放在最后)
            " ",
            ""
        ]
)
result = spliter.split_documents(docs)      # 开始切分

# 创建向量数据库
print("正在将文档向量化并存入 FAISS 数据库...")
db = FAISS.from_documents(result, embeddings_model)
print("✅ 数据库建立完成！")

# 5. 开始搜索
query = "深度学习"
results = db.similarity_search_with_score(query, k=4)       # results 变成了一个列表，里面装的是元组： (Document对象, 分数)
print("\n🔍 检索结果:")
for doc, score in results:
    print(f"内容: {doc.page_content}")
    print(f"来源: {doc.metadata}")
    # FAISS 默认使用的是 L2 距离 (欧氏距离)
    # 越接近 0 表示越相似，分数越大表示差距越大
    print(f"距离分数: {score}")
    print("-" * 20)