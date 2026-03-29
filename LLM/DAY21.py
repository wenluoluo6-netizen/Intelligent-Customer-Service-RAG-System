import os

from langchain_classic.agents import create_tool_calling_agent, AgentExecutor
from langchain_community.chat_message_histories import RedisChatMessageHistory
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableWithMessageHistory, RunnableConfig
from langchain_core.tools import tool
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_openai import OpenAIEmbeddings
from pymilvus import MilvusClient
from tenacity import retry, stop_after_attempt, wait_fixed
import redis


load_dotenv()

# 初始化模型
llm = init_chat_model(
    model="glm-4",
    model_provider="openai",
    api_key=os.getenv("API_KEY"),
    base_url=os.getenv("BASE_URL"),
    temperature=0.1
)

embeddings = OpenAIEmbeddings(
    model="embedding-2",
    api_key=os.getenv("API_KEY"),
    base_url=os.getenv("BASE_URL")
)

# 连接知识库
milvus_client = MilvusClient(uri=os.getenv("MILVUS_URI"), token=os.getenv("MILVUS_TOKEN"))
collection_name = "my_collections"

# 创建两个工具(tool)
# 1.内部知识库检索工具
@tool
def search_internal_knowledge(query: str) -> str:
    """
    搜索公司内部知识库。
    当用户询问公司规章制度、内部产品参数、报销流程、员工福利等内部信息时，必须使用此工具。
    """
    print(f"\n[📚 查阅内网] 正在知识库中检索: '{query}'")
    try:
        query_vector = embeddings.embed_query(query)
        search_res = milvus_client.search(
            collection_name=collection_name,
            data=[query_vector],
            limit=3,
            output_fields=["text"]
        )
        docs = [hit['entity']['text'] for hit in search_res[0]]
        if not docs:
            return "内部知识库中未找到相关内容。"
        return f"内部知识库检索结果：{docs}"
    except Exception as e:
        return f"知识库检索失败，系统异常。报错：{str(e)}"

# 2.外部实时网络搜索工具
ddg_search = DuckDuckGoSearchRun()

@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
def robust_web_search(query: str) -> str:
    return ddg_search.run(query)

@tool
def search_internet(query: str) -> str:
    """
    进行外网实时搜索。
    当用户询问最新的新闻、公众人物动态、当前股市、汇率或常识性外部问题时，使用此工具。
    """
    print(f"\n[🌐 呼叫外网] 正在 Google/Bing 检索: '{query}'")
    try:
        result = robust_web_search(query)
        return f"外网检索结果：\n{result}"
    except Exception as e:
        print("   [🛡️ 工具降级] 网络搜索彻底挂了...")
        return "外部网络搜索当前不可用，请告知用户网络超时。"

tools = [search_internal_knowledge, search_internet]

# Prompt & Agent
system_prompt = """你是一个全能的首席研究助理。
你拥有强大的工具库，请根据用户的问题，聪明地选择调用内部知识库还是外部网络，甚至两者都用！
如果你使用了工具，请在回答时明确指出信息来源是“根据公司内部规定”还是“根据网络公开信息”。"""

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

# 创建agent
agent = create_tool_calling_agent(llm, tools, prompt)  # 支持 tool 并行处理
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True, max_iterations=3)

# 装配记忆模块，通过redis存到内存当中
REDIS_URL = "redis://localhost:26379"
redis_client = redis.Redis.from_url(REDIS_URL, decode_responses=True)   # 创建原生Redis客户端,decode_responses 控制 Redis 返回数据的类型：False 返字节串，True 返字符串

def get_session_history(session_id: str) -> RedisChatMessageHistory:
    """获取或创建会话历史（使用 Redis）"""
    # 创建 Redis 历史对象
    history = RedisChatMessageHistory(
        session_id=session_id,
        url=REDIS_URL
    )
    return history

# 创建带历史记录的链
chain = RunnableWithMessageHistory(
    agent_executor,
    get_session_history=get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history"
)

config = RunnableConfig(configurable={"session_id": "user-002"})
print("开始对话（输入 'quit' 退出）")
while True:
    question = input("\n输入问题：")
    if question.lower() in ['quit', 'exit', 'q']:
        break
    response = chain.invoke({"input": question}, config)
    print(f"\n🤖 AI回答: {response['output']}")
    print("=" * 60)  # 画一条长长的分割线，彻底隔开下一轮的输入提示