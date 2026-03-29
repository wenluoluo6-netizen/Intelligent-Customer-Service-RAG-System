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

collection_name = "my_rag_collection2"

if client.has_collection(collection_name=collection_name):
    client.drop_collection(collection_name=collection_name)

print("正在创建 Collection...")
client.create_collection(
    collection_name=collection_name,
    dimension=1024
)

docs = [
    "差旅报销每天补贴 200 元。",
    "打车报销需提供发票及行程单。",
    "员工离职需提前 30 天提交申请。",
    "报销离职路费请联系人力专员。"
]

print("正在调用智谱 API 将文本转为向量...")
vectors = embeddings_model.embed_documents(docs)

data_to_insert = []
for i in range(len(vectors)):
    if i <=1 :
        data_to_insert.append({"id":i, "vector":vectors[i], "text":docs[i], "department":"finance"})
    else:
        data_to_insert.append({"id":i, "vector":vectors[i], "text":docs[i], "department":"hr"})

print("正在将数据插入 Milvus...")
client.insert(collection_name=collection_name, data=data_to_insert)
print(f"✅ 成功插入 {len(data_to_insert)} 条数据！")

query = "怎么报销？"
print(f"\n🔍 用户提问: {query}")
query_vector = embeddings_model.embed_query(query)

search_results = client.search(collection_name=collection_name, data=[query_vector], filter="department=='finance'", output_fields=["text"])

print("\n🎯 搜索结果:")
# search_results 返回的是一个复杂的列表嵌套字典，我们把它解开
for hit in search_results[0]:
    print(f"ID: {hit['id']}, 距离分数: {hit['distance']:.4f}")
    print(f"内容: {hit['entity']['text']}\n")
