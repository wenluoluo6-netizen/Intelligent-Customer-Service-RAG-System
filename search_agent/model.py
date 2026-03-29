import os
from dotenv import load_dotenv
load_dotenv()
from langchain.chat_models import init_chat_model
from langchain_openai import OpenAIEmbeddings


# 初始化模型
llm = init_chat_model(
    model="qwen-plus",
    model_provider="openai",
    api_key=os.getenv("API_KEY"),
    base_url=os.getenv("BASE_URL"),
    temperature=0.1
)

embeddings = OpenAIEmbeddings(
    model="text-embedding-v2",
    api_key=os.getenv("API_KEY"),
    base_url=os.getenv("BASE_URL")
)