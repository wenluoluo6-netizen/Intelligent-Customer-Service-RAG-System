import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from datasets import Dataset
from ragas import evaluate
# 【修改1】修复警告：从 collections 导入指标，或者直接导入类
# 注意：新版 Ragas 建议显式导入 Metrics 类，但为了兼容你的写法，我们按警告修改路径
from ragas.metrics import Faithfulness, AnswerRelevancy

# 加载配置
load_dotenv()

# --- 1. 配置你的 LLM 和 Embedding ---
llm = ChatOpenAI(
    model="glm-4",
    api_key=os.getenv("API_KEY"),
    base_url=os.getenv("BASE_URL"),
    # 【修改2】调高温度，解决 "returned 1 generations" 问题
    # Ragas 需要 LLM 发散思维生成多个问题，温度太低会导致生成内容重复
    temperature=0.8
)

embeddings = OpenAIEmbeddings(
    model="embedding-2",
    api_key=os.getenv("API_KEY"),
    base_url=os.getenv("BASE_URL")
)

# --- 2. 准备测试数据 ---
data_samples = {
    'question': [
        'Python 中的列表是什么？',
        '谁是美国队长？'
    ],
    'answer': [
        'Python 列表是一种有序的可变集合，允许存储重复的元素。',
        '美国队长是史蒂夫·罗杰斯。'
    ],
    'contexts': [
        [
            'Python 列表 (List) 是最常用的数据类型。列表是可变的，元素是有序的。',
            'Python 字典是无序的键值对集合。'
        ],
        [
            '钢铁侠是托尼·斯塔克。',
            '雷神是索尔。'
        ]
    ]
}

dataset = Dataset.from_dict(data_samples)

# --- 3. 运行评估 ---
print("🚀 正在开始评估 (可能需要几十秒，请耐心等待)...")

# 实例化指标对象
faithfulness = Faithfulness()
answer_relevancy = AnswerRelevancy()

results = evaluate(
    dataset,
    metrics=[faithfulness, answer_relevancy],
    llm=llm,
    embeddings=embeddings
)

print("\n📊 评估报告:")
print(results)

# 打印详细的单项分数
df = results.to_pandas()

print("\n🔍 数据表的所有列名 (用于调试):", df.columns)

print("\n📝 详细数据:")
# 【修改3】修复 KeyError
# 新版 Ragas 将 'question' 重命名为 'user_input'
# 我们做一个简单的判断，防止版本差异再次报错
if 'user_input' in df.columns:
    print(df[['user_input', 'faithfulness', 'answer_relevancy']])
else:
    # 如果旧版本还是叫 question
    print(df[['question', 'faithfulness', 'answer_relevancy']])