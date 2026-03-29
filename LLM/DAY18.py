import os
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

llm = ChatOpenAI(
    model="glm-4",
    api_key=os.getenv("API_KEY"),
    base_url=os.getenv("BASE_URL"),
    temperature=0
)


# 定义工具说明书
class GetWeather(BaseModel):
    """获取指定城市的天气信息"""
    # Field 用来详细描述这个参数到底是什么
    city: str = Field(..., description="城市的中文名称，例如：北京、广州")
    date: str = Field(..., description="查询的日期，例如：今天、明天")

class Calculator(BaseModel):
    """进行复杂的数学运算"""
    expression: str = Field(..., description="要计算的数学表达式，例如：25 * 4")

class TicketExtraction(BaseModel):
    """获取用户信息的姓名、电话、遇到的问题，并存入Excel"""
    customer_name: str = Field(..., description="客户姓名")
    phone_number: str = Field(..., description="11位手机号，如果没有则输出“未知")
    issue_type: str = Field(..., description="必须是“退货”、“维修”、“投诉”中的一种")
    is_urgent: bool = Field(..., description="客户是否生气、要求加急")

# 绑定工具
llm_with_tools = llm.bind_tools([GetWeather, Calculator, TicketExtraction])

print("🤖 测试 1：普通聊天")
# 模型发现不需要用工具，就会正常回答
res1 = llm_with_tools.invoke("你好，请问你是谁？")
print(f"返回类型: {type(res1)}")
print(f"返回内容: {res1.content}")

print("\n" + "="*50 + "\n")

print("🤖 测试 2：触发天气工具")
# 模型发现需要查询天气，它不会直接回答，而是返回工具调用请求 (tool_calls)
res2 = llm_with_tools.invoke("帮我查一下深圳明天的天气")
print(f"大模型说话了吗？(content): '{res2.content}'") # 这里通常是空的！
print(f"是否触发了工具？(tool_calls): {res2.tool_calls}")


print("\n" + "="*50 + "\n")
res3 = llm_with_tools.invoke("哎哟气死我了！我叫李建国，我昨天刚买的那个破音响，今天一插电就冒黑烟了！你们这什么破质量！赶紧给我派人来修，越快越好，听见没！我电话是 13812345678。")
print(f"大模型说话了吗？(content): '{res3.content}'") # 这里通常是空的！
print(f"是否触发了工具？(tool_calls): {res3.tool_calls}")