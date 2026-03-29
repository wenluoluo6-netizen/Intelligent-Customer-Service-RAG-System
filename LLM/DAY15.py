import os

from dotenv import load_dotenv
from langchain_classic.agents import create_react_agent, AgentExecutor
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

# 1. 加载环境变量
load_dotenv()


# 2. 初始化大模型
llm = ChatOpenAI(
    model="glm-4",
    api_key=os.getenv("API_KEY"),
    base_url=os.getenv("BASE_URL"),
    temperature=0
)


# 3. 准备工具箱 (Tools)
# 这里我们只给它一把“锤子”：搜索引擎
search = DuckDuckGoSearchRun()

# 把工具包装成列表，以后可以放计算器、数据库查询等更多工具
tools = [search]


# 4. 手写 ReAct 专用的 Prompt 模板 (完美替代 hub.pull)
# 这就是云端 hwchase17/react 的真实内容，直接写在本地，永远不会因为网络或导包报错！
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

# 打印出来看看这个 Prompt 到底长啥样（这是理解 Agent 的关键！）
print("=== Agent 的系统指令 (System Prompt) ===")
print(prompt.template)
print("=======================================\n")

# 5. 构建 Agent (大脑)
# create_react_agent 负责把 LLM、工具描述、Prompt 组装在一起
agent = create_react_agent(llm, tools, prompt)

# 6. 构建 Executor (执行器/手脚)
# Agent 只负责思考，Executor 负责真正去调用函数、处理报错、循环执行
# verbose=True 是关键！它会把 Agent 的每一步思考过程打印在控制台
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# 7. 运行任务
print("🤖 Agent 启动中...")
query = "目前特斯拉(Tesla)的股价是多少美元？如果我买 100 股需要多少人民币？(假设汇率是 7.2)"
agent_executor.invoke({"input": query})