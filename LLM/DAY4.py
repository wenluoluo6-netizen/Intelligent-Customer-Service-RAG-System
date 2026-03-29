import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings   # 用来把字变成数字
from langchain_community.vectorstores import FAISS      # 向量数据库
from langchain_core.documents import Document      # 构造测试数据用

# 1. 加载环境变量 (必须最先执行)
load_dotenv()
API_KEY = os.getenv("API_KEY")
BASE_URL = os.getenv("BASE_URL")

# 2. 初始化 Embedding 模型 (关键！)
embeddings_model = OpenAIEmbeddings(
    model="embedding-2",
    api_key=API_KEY,
    base_url=BASE_URL,
)

# 3. 准备数据 (假设这是 Day 3 切割好的 chunks)
documents = [
    Document(page_content="小明喜欢吃苹果，但他不喜欢吃香蕉。", metadata={"source": "user_profile"}),
    Document(page_content="特斯拉是一家生产电动汽车的公司。", metadata={"source": "tech_news"}),
    Document(page_content="Python 是一种非常流行的编程语言，适合做 AI 开发。", metadata={"source": "coding_tutorial"}),
    Document(page_content="今天天气真好，适合出去野餐。", metadata={"source": "daily_log"}),
]
print("正在将文档向量化并存入 FAISS 数据库...")

# 4. 创建向量数据库,这一步做了两件事：
# A. 调用智谱 API 把上面的中文全变成数字列表
# B. 建立索引，方便快速查找
db = FAISS.from_documents(documents, embeddings_model)
print("✅ 数据库建立完成！")

# 5. 模拟用户提问 (语义搜索测试)
query = "想学习人工智能该用什么语言？"
print(f"\n用户提问{query}")

# 6. 在数据库中搜索,k=1 表示只找最相似的那 1 条
result = db.similarity_search(query, k=1)

print("\n🔍 检索结果:")
for doc in result:
    print(f"内容: {doc.page_content}")
    print(f"来源: {doc.metadata}")

# 7. 保存这个数据库 (持久化),这样下次就不用重新花钱调 API 做向量化了
db.save_local("faiss_index")
print("\n💾 向量数据库已保存到本地文件夹 'faiss_index'")