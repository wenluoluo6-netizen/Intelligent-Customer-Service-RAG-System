import os

from langchain_classic.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from pydantic import BaseModel, Field
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory

load_dotenv()

llm = init_chat_model(
    model="glm-4",
    model_provider="openai",
    api_key=os.getenv("API_KEY"),
    base_url=os.getenv("BASE_URL"),
    temperature=0
)

class GetweatherInout(BaseModel):
    """获取指定城市的天气信息，参数必须是城市名。"""
    city: str = Field(..., description="城市名，例如：北京，上海")


@tool(args_schema=GetweatherInout)
def get_weather(city: str) -> str:
    """获取指定城市的天气信息。参数必须是城市名。"""
    print(f"\n[🔧 工具执行中...] 正在查询 {city} 的天气...")
    mock_db = {"北京": "沙尘暴，不建议出门", "上海": "晴天，适合郊游", "深圳": "暴雨，请带伞"}
    return mock_db.get(city, f"查不到 {city} 的天气。")

tools = [get_weather]

prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个聪明、贴心的私人助理。你会记住用户的信息，并适时使用工具。"),
    MessagesPlaceholder(variable_name="chat_history"), # 核心：存放历史记忆
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"), # 核心：存放 Agent 思考和调用工具的记录
])

agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# --- 5. 记忆管理模块 (复习 Day 8) ---
store = {}
def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

# 将 Agent 包装上一层“记忆外壳”
agent_with_memory = RunnableWithMessageHistory(
    agent_executor,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
)

# --- 6. 见证奇迹的测试 ---

# 设定一个会话 ID，模拟同一个用户在连续聊天
config = {"configurable": {"session_id": "boss_001"}}

print("🤖 第一轮对话：提供背景信息")
res1 = agent_with_memory.invoke(
    {"input": "你好，我是老板。我下周要去北京出差。"},
    config=config
)
print(f"AI 回答: {res1['output']}\n")
print("="*50)

print("🤖 第二轮对话：考验记忆与工具联动")
# 注意：我这里完全没有提“北京”两个字！
res2 = agent_with_memory.invoke(
    {"input": "那我这次出差需要带几把伞？天气怎么样？"},
    config=config
)
print(f"AI 回答: {res2['output']}\n")