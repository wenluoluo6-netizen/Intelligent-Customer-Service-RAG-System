import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


# 加载环境变量
load_dotenv()
API_KEY = os.getenv("API_KEY")
BASE_URL = os.getenv("BASE_URL")

# 初始化模型
model = ChatOpenAI(
    model="glm-4",
    temperature=0.5,
    openai_api_key=API_KEY,
    openai_api_base=BASE_URL,
)
print("模型配置完成！准备开始加载数据...")

# 加载数据
loader = TextLoader("./test.txt", encoding="utf-8")
docs = loader.load()
print("正在清洗数据...")

# 遍历加载进来的每一个文档块 (Document Object)
for doc in docs:
    raw_content = doc.page_content
    clean_content = "".join(raw_content.split())
    doc.page_content = clean_content

print("数据加载完成！开始生成对话...")

# 查看加载结果
print(f"成功加载了{len(docs)}个文档")
print(f"文档内容预览（前100字）：{docs[0].page_content[:100]}")

# 创建切分器
test_splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=20,  # 重叠部分，让两块之间有过渡
    # 【关键修改】加入中文逗号、分号、感叹号、问号
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

# 执行切分
splits = test_splitter.split_documents(docs)

print(f"原文档被切成了 {len(splits)} 个小块")
print("--- 第一块内容 ---")
print(splits[0].page_content)
print("--- 第二块内容 ---")
print(splits[1].page_content)

