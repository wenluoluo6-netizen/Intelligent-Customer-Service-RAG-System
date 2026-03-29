import os

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from pymilvus import MilvusClient

load_dotenv()

embeddings_model = OpenAIEmbeddings(
    model="embedding-2",
    api_key=os.getenv("API_KEY"),
    base_url=os.getenv("BASE_URL")
)


client = MilvusClient(uri=os.getenv("MILVUS_URI"), token=os.getenv("MILVUS_TOKEN"))
print("✅ 成功连接到 Milvus 数据库！")

collection_name = "my_rag_collection"

# 如果表已经存在，先删掉（方便我们反复测试）
if client.has_collection(collection_name=collection_name):
    client.drop_collection(collection_name=collection_name)

print("正在创建 Collection...")
client.create_collection(
    collection_name=collection_name,
    dimension=1024
)

# 真实场景下，这往往是经过 LangChain 切割后的 Document 列表
docs = [
    "产品 A-800 的最大功率是 2000W。",
    "产品 B-200 是一款家用设备，功率 500W。",
    "发生 502 错误通常是因为网关超时。"
]

print("正在调用智谱 API 将文本转为向量...")
# 将文本列表直接批量转成向量
vectors = embeddings_model.embed_documents(docs)

# 组装插入 Milvus 的数据格式
data_to_insert = []
for i in range(len(docs)):
    data_to_insert.append({
        "id": i,
        "vector": vectors[i],
        "text": docs[i]
    })

print("正在将数据插入 Milvus...")
client.insert(
    collection_name=collection_name,
    data=data_to_insert
)
print(f"✅ 成功插入 {len(data_to_insert)} 条数据！")

query = "A-800 需要多大功率？"
print(f"\n🔍 用户提问: {query}")
#先把问题变成向量
query_vector = embeddings_model.embed_query(query)

# 去 Milvus 里搜最相近的 2 个
search_results = client.search(
    collection_name=collection_name,
    data=[query_vector],
    limit=2,
    output_fields=["text"]
)

print("\n🎯 搜索结果:")
# search_results 返回的是一个复杂的列表嵌套字典，我们把它解开
for hit in search_results[0]:
    print(f"ID: {hit['id']}, 距离分数: {hit['distance']:.4f}")
    print(f"内容: {hit['entity']['text']}\n")