import os
import sqlite3

from dotenv import load_dotenv
from langchain_community.agent_toolkits import create_sql_agent
from langchain_community.utilities import SQLDatabase
from langchain_openai import ChatOpenAI

# 1. 加载环境变量
load_dotenv()


# --- 第一步：手动造一个模拟的电商数据库 ---
db_path = "ecommerce.db"
# 连接 SQLite（如果文件不存在会自动创建）
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# 建表：users (用户表) 和 orders (订单表)
cursor.execute("CREATE TABLE IF NOT EXISTS users (id INTEGER PRIMARY KEY, name TEXT, age INTEGER, city TEXT)")
cursor.execute("CREATE TABLE IF NOT EXISTS orders (id INTEGER PRIMARY KEY, user_id INTEGER, amount REAL, product TEXT)")

# 清空旧数据（防止你反复运行代码导致数据重复）
cursor.execute("DELETE FROM users")
cursor.execute("DELETE FROM orders")

# 插入一些假数据，顺便加点我们做深度学习经常买的硬件
cursor.executemany("INSERT INTO users (id, name, age, city) VALUES (?, ?, ?, ?)", [
    (1, '张三', 28, '北京'),
    (2, '李四', 35, '上海'),
    (3, '王五', 24, '深圳')
])
cursor.executemany("INSERT INTO orders (id, user_id, amount, product) VALUES (?, ?, ?, ?)", [
    (101, 1, 2000.0, 'RTX 3050 显卡'), # 给张三安排一张显卡跑跑深度学习
    (102, 1, 150.0, '机械键盘'),
    (103, 2, 8999.0, '苹果笔记本'),
    (104, 3, 299.0, '蓝牙耳机'),
    (105, 3, 50.0, '鼠标垫')
])
conn.commit()
conn.close()
print("✅ 本地测试数据库 (ecommerce.db) 准备完毕！\n")


# --- 第二步：见证魔法 (构建 SQL Agent) ---
# 1. 让 LangChain 连接到我们刚才建的数据库
# sqlite:/// 是 SQLAlchemy 连接 sqlite 的标准写法
db = SQLDatabase.from_uri(f"sqlite:///{db_path}")

# 2. 初始化大模型 (严谨的数据查询，temperature 必须是 0)
llm = ChatOpenAI(
    model="glm-4",
    api_key=os.getenv("API_KEY"),
    base_url=os.getenv("BASE_URL"),
    temperature=0
)

# 3. 创建 SQL Agent
print("⚙️ 正在唤醒 SQL 数据分析师...")
agent_executor = create_sql_agent(llm, db=db, agent_type="openai-tools", verbose=True)

# --- 第三步：开始压榨你的数据分析师 ---

print("\n🤖 测试 1：简单单表查询")
agent_executor.invoke({"input": "数据库里一共有几个用户？"})

print("\n" + "="*50 + "\n")

print("🤖 测试 2：跨表关联查询 (稍微上点难度)")
agent_executor.invoke({"input": "张三一共消费了多少钱？"})

print("\n" + "="*50 + "\n")
agent_executor.invoke({"input": "帮我查一下，买过显卡的人，他所在的城市是哪里"})