import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from dotenv import load_dotenv
from datasets import Dataset
from langchain_classic.retrievers import EnsembleRetriever
from ragas import evaluate
from ragas.metrics import ContextPrecision, ContextRecall
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

# --- 1. 准备数据与构建索引 ---
knowledge = """
1. 产品型号 A-800 的最大功率是 2000W，适用于工业场景。
2. 产品型号 B-200 是一款家用设备，功率 500W，静音效果好。
3. 工业设备的维护周期通常是 6 个月。
"""
with open("device_manual.txt", "w", encoding="utf-8") as f:
    f.write(knowledge)

loader = TextLoader("device_manual.txt", encoding="utf-8")
docs = loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=50, chunk_overlap=0)
splits = splitter.split_documents(docs)

embeddings = OpenAIEmbeddings(
    model="embedding-2",
    api_key=os.getenv("API_KEY"),
    base_url=os.getenv("BASE_URL")
)

# 构建两路索引
print("🏗️ 正在构建索引...")
vectorstore = FAISS.from_documents(splits, embeddings)
faiss_retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

bm25_retriever = BM25Retriever.from_documents(splits)
bm25_retriever.k = 2

# 混合检索器
ensemble_retriever = EnsembleRetriever(
    retrievers=[faiss_retriever, bm25_retriever],
    weights=[0.5, 0.5]
)

# --- 2. 准备评估数据 ---
test_questions = [
    "A-800 的功率是多少？",
    "家用设备的维护周期是多久？"
]

ground_truths = [
    "产品型号 A-800 的最大功率是 2000W。",
    "文档中未提及家用设备的维护周期，只提到了工业设备是 6 个月。"
]

print("🚀 正在运行检索 (Retrieval)...")
contexts = []
answers = []

# 【修改3】修复循环逻辑错误
for q in test_questions:
    # 关键：必须先调用检索器！
    docs = ensemble_retriever.invoke(q)
    # 提取内容
    retrieved_text = [doc.page_content for doc in docs]

    contexts.append(retrieved_text)
    answers.append("占位符")  # Ragas 需要这一列，虽然我们只测检索

# --- 3. 组装数据集 ---
data_samples = {
    'question': test_questions,
    'answer': answers,  # 【修改4】这里必须是单数 'answer'，不能是 'answers'
    'contexts': contexts,
    'ground_truth': ground_truths
}
dataset = Dataset.from_dict(data_samples)

# --- 4. 召唤 LLM 考官 ---
llm_judge = ChatOpenAI(
    model="glm-4",
    api_key=os.getenv("API_KEY"),
    base_url=os.getenv("BASE_URL"),
    temperature=0
)

print("⚖️ 开始 RAGAs 评估 (Context Precision & Recall)...")

# 【修改5】实例化指标对象 (ContextPrecision() 带括号)
metrics = [ContextPrecision(), ContextRecall()]

results = evaluate(
    dataset,
    metrics=metrics,
    llm=llm_judge,
    embeddings=embeddings
)

print("\n📊 检索能力体检报告:")
print(results)

# 打印详细 DataFrame
df = results.to_pandas()
# 防止列名不一致的保险写法
cols = ['question', 'context_recall', 'context_precision']
# 如果新版 Ragas 把 question 改名了，自动适配
if 'user_input' in df.columns:
    cols[0] = 'user_input'

print("\n📝 详细得分:")
print(df[cols])