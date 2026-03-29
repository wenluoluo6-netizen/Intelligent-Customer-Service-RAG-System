import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 加载环境变量
load_dotenv()
API_KEY = os.getenv("API_KEY")
BASE_URL = os.getenv("BASE_URL")

# 初始化模型
model = ChatOpenAI(
    model="glm-4",
    temperature=0.7,
    openai_api_key=API_KEY,
    openai_api_base=BASE_URL
)

# 定义经理手中的“单子”
prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", "你是一个情商极高的情感专家，擅长帮人写幽默的道歉信。"),
        ("user", "我的{relation}生气了，因为我{reason}，请你帮我写一段道歉的话")
    ]
)

# 定义最后的切片机
output_parser = StrOutputParser()

# 核心知识点，LCEL组装流水线，这里的 | 就是管道： 提示词 -> 模型 -> 字符串解析器
chain = prompt_template | model | output_parser

# 运行流水线，invoke就是“启动”的意思
try:
    result = chain.invoke({"relation": "男朋友",
                            "reason": "因为打游戏入迷而没回消息"})
    print(f"{result}")
except Exception as e:
    print(f"出错了：{e}")
