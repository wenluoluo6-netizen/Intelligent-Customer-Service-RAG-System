import os
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import PydanticOutputParser, BaseOutputParser, StrOutputParser
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
    temperature=0.7,        # 使用温度较高的 LLM，让它更有创造力地“瞎编”假设性文档
    openai_api_key=API_KEY,
    openai_api_base=BASE_URL
)
embeddings = OpenAIEmbeddings(
    model="embedding-2",
    openai_api_key=API_KEY,
    openai_api_base=BASE_URL,
)

# --- 2. 构造“刁钻”的知识库 ---
# 这里的难点在于：知识库里只有“不可抗力”，没有“台风”这个词
rule_text = """
《航空旅客运输服务管理规定》第十九条：
1. 对于遭遇不可抗力（包括但不限于天气因素、突发事件、空中交通管制）导致的航班延误，且延误时间超过 4 小时的，旅客可申请非自愿退票，全额退还票款。
2. 旅客因个人原因申请退票的，需按照对应舱位的退改签规则收取手续费。
"""
file_name = "day8_task.txt"
with open(file_name, "w", encoding="utf-8") as f:
    f.write(rule_text)
loader = TextLoader(file_name, encoding="utf-8")    # 读取数据
docs = loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=0)   # 切分数据
splits = splitter.split_documents(docs)
vectorstore = FAISS.from_documents(splits, embeddings)   # 向量化并存入数据库
retriever = vectorstore.as_retriever(search_kwargs={"k": 1})   # 构建检索器
print("✅ 知识库构建完成！")

# --- 3. 定义对比实验 ---
user_query = "台风天飞机没飞，能退钱吗？"
print(f"🔴 原始问题: {user_query}")

# === 实验 A: 直接检索 (Direct Retrieval) ===
direct_result = retriever.invoke(user_query)
if direct_result:
    print(f"🟠 直接检索结果: {direct_result[0].page_content}")
else:
    print("🟠 直接检索结果: 未检索到相关内容")

# === 实验 B: HyDE 策略 (Hypothetical Document Embeddings) ===
print("\n--- 实验 B: HyDE 策略检索 ---")

hyde_template = """请你扮演一个专业的航空客服专家。
针对用户的问题，请写一段**假设性**的、语气正式的回答段落。
这段回答不需要保证事实完全正确（你可以编造具体的条款号），但必须使用专业的行业术语。

用户问题: {question}
假设性回答:"""

hyde_prompt = ChatPromptTemplate.from_template(hyde_template)
hyde_chain = hyde_prompt | llm | StrOutputParser()
# B2. 生成假设性文档
print("🔄 正在生成假设性文档 (Hallucinating)...")
hypothetical_doc = hyde_chain.invoke({"question": user_query})
print(f"👻 生成的假答案: \n{hypothetical_doc}")
print("-" * 30)

# B3. 用这个“假答案”去搜“真答案”
# 因为假答案里大概率会包含“天气原因”、“不可抗力”、“全额退款”等词，
# 这些词跟知识库里的原文在向量空间里离得非常近。
hyde_results = retriever.invoke(hypothetical_doc)

if hyde_results:
    print(f"✅ HyDE 检索到的真答案: \n{hyde_results[0].page_content}")
else:
    print("❌ HyDE 也没搜到")