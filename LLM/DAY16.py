import os

from dotenv import load_dotenv
from langchain_classic.agents import create_react_agent, AgentExecutor
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

load_dotenv()

llm = ChatOpenAI(
    model="glm-4",
    api_key=os.getenv("API_KEY"),
    base_url=os.getenv("BASE_URL"),
    temperature=0
)

@tool
def get_weather(city: str) -> str:
    """
        获取指定城市的天气情况。
        当用户问及天气、温度、下雨、出门带不带伞等信息时，必须调用此工具。
        参数 city 必须是城市的中文名称，例如：北京、上海。
        """
    city = city.split('\n')[0].replace("Observ", "").strip()

    print(f"\n[后台执行] 🔧 工具被调用啦！正在查询 {city} 的天气...")

    mock_db = {
        "北京": "晴天，25度，紫外线较强",
        "上海": "雷阵雨，20度，风力4级",
        "深圳": "暴雨，28度，部分路段有积水"
    }

    # 返回给 Agent 的观察结果 (Observation)
    return mock_db.get(city, f"抱歉，系统里暂时查不到 {city} 的天气数据。")

@tool
def check_order_status(order_id: str) ->str:
    """
    获取指定订单号的信息。
    当用户问及订单号、快递等信息时，必须调用此工具。
    参数 order_id 必须是四位数的数字，例如：1001，1002.
    """

    mock_db = {
        "1001": "已发货，目前包裹正在前往北京的分拨中心。",
        "1002": "未发货，商家正在打包中。",
        "1003": "已签收，包裹已放在菜鸟驿站。"
    }

    order_id = order_id.split('\n')[0].replace("Observ", "").strip()

    return mock_db.get(order_id, f"抱歉，系统里暂时查不到 {order_id} 的订单数据。")


tools = [get_weather, check_order_status]

react_template = """Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought:{agent_scratchpad}"""

prompt = PromptTemplate.from_template(react_template)

agent = create_react_agent(llm, tools, prompt)  #大脑

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True, max_iterations=3)  #执行器

# --- 运行测试 ---
print("🤖 任务 1：")
query1 = "帮我查一下订单号 1001 的状态。如果发货了，帮我查一下北京今天的天气，因为那是我的收货地，我想知道包裹会不会被雨淋湿。"
agent_executor.invoke({"input": query1})



