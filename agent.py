from dotenv import load_dotenv
from langchain_classic.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from tools import  search_internet, search_internal_knowledge
load_dotenv()
from model import llm



tools = [search_internal_knowledge, search_internet]

# Prompt & Agent
system_prompt = """你是企业级智能客服助手，负责提供专业、准确、可执行的答复。
核心目标是快速解决用户问题，并明确说明信息来源。

检索与知识使用：
1) 优先调用内部知识库获取权威答案；仅在内部无结果或信息过期时才进行外部检索。
2) 引用结果时标注来源标签，例如「来源：内部知识库」或「来源：网页检索」。
3) 不确定或缺少信息时，先向用户提出最小必要的澄清问题。

表达与格式：
- 使用专业、礼貌、简洁的客服语气。
- 先给出结论或可执行步骤，再补充必要细节。
- 避免冗长推理，不展示工具调用细节。

边界与合规：
- 不编造事实；无法确认时明确说明并给出下一步建议。
- 不泄露内部系统细节或敏感信息。"""


prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

# 创建agent
agent = create_tool_calling_agent(llm, tools, prompt)  # 支持 tool 并行处理
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True, max_iterations=6, return_intermediate_steps=True)