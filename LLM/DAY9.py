import os
from dotenv import load_dotenv
from langchain.retrievers import EnsembleRetriever
from langchain_community.document_loaders import TextLoader
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

# --- 1. 准备数据 ---
# 模拟一个混合了“专业名词”和“一般描述”的知识库
knowledge = """
1. 产品型号 A-800 的最大功率是 2000W，适用于工业场景。
2. 产品型号 B-200 是一款家用设备，功率 500W，静音效果好。
3. 如果设备发出“滴滴”报警声，通常是因为电压不稳定。
4. 工业设备的维护周期通常是 6 个月，家用设备是 1 年。
5. A-800 需要使用专用的 380V 电源接口。
"""
with open("device_manual.txt", "w", encoding="utf-8") as f:
    f.write(knowledge)

loader = TextLoader("device_manual.txt", encoding="utf-8")
docs = loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=50, chunk_overlap=0)
splits = splitter.split_documents(docs)

# --- 2. 构建两路检索器 ---

# A. 向量检索 (Vector Search) - 懂语义
embeddings = OpenAIEmbeddings(
    model="embedding-2",
    openai_api_key=os.getenv("API_KEY"),
    openai_api_base=os.getenv("BASE_URL")
)
vectorstore = FAISS.from_documents(splits, embeddings)
faiss_retriever = vectorstore.as_retriever(search_kwargs={"k": 3})      # 向量检索器，每次检索 3 条

# B. 关键词检索 (Keyword Search) - 懂字眼
# BM25Retriever 会自动对文档进行分词并建立索引
bm25_retriever = BM25Retriever.from_documents(splits)
bm25_retriever.k = 3        # 关键词检索器，每次检索3条

# --- 3. 混合检索 (Hybrid Search) ---
# weights=[0.5, 0.5] 表示两者同等重要
ensemble_retriever = EnsembleRetriever(
    retrievers=[faiss_retriever, bm25_retriever],
    weights=[0.5, 0.5]
)

# --- 4. 重排序 (Rerank) - 使用 LLM ---
# 由于我们没有专门的 Rerank 模型，我们定义一个简单的函数，
# 让 LLM 帮我们从混合检索的候选集中挑出最好的。
llm = ChatOpenAI(
    model="glm-4",
    api_key=os.getenv("API_KEY"),
    base_url=os.getenv("BASE_URL"),
    temperature=0
)

def rerank_documents(query, docs):
    """
        让 LLM 对文档列表进行打分和排序
        """
    if not docs:
        return []

    # 把文档内容拼起来给 LLM 看
    doc_texts = "\n".join([f"[{i}] {doc.page_content}" for i, doc in enumerate(docs)])
    prompt = f"""
        你是一个文档相关性打分专家。
        请分析下面的候选文档与用户问题 "{query}" 的相关性。

        候选文档列表:
        {doc_texts}

        请从中选出最相关的 1 个文档。
        只返回该文档的编号（例如 0, 1, 2...），不要返回其他文字。
        如果都没有关系，返回 -1。
        """
    response = llm.invoke(prompt).content.strip()

    try:
        best_idx = int(response)
        if 0 <= best_idx < len(docs):
            print(f"🎯 LLM Rerank 选中了第 [{best_idx}] 个文档")
            return [docs[best_idx]]
        else:
            return docs[:1]  # 兜底：如果 LLM 瞎回，就默认取第一个
    except:
        return docs[:1]

# --- 5. 测试流程 ---
query = "A-800 电源要求"
print(f"🔍 用户提问: {query}")

# 步骤 A: 混合检索 (粗排)
# 这可能会返回 4-6 个文档（两路去重后的并集）
candidate_docs = ensemble_retriever.invoke(query)
print(f"\n⚡ 混合检索找到 {len(candidate_docs)} 个候选文档:")
for i, doc in enumerate(candidate_docs):
    print(f"[{i}]{doc.page_content}")

# 步骤 B: LLM 重排序 (精排)
print("\n⚖️ 正在进行 LLM 重排序...")
final_docs = rerank_documents(query, candidate_docs)
print(f"\n✅ 最终交给大模型的文档: {final_docs[0].page_content if final_docs else '无'}")