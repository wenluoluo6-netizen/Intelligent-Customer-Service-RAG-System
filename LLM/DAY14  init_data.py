import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from pymilvus import MilvusClient

# 1. 加载环境变量
load_dotenv()

# --- 配置部分 ---
COLLECTION_NAME = "my_collections"

# 模拟的企业知识库数据 (30条)
knowledge_base = [
    # --- 人事 (HR) ---
    "公司实行弹性工作制，核心工作时间为上午 10:00 至 下午 4:00。",
    "员工每年享有 15 天带薪年假，试用期通过后即可申请。",
    "病假需提供二级甲等以上医院的诊断证明，每月前 2 天全薪。",
    "试用期通常为 3 个月，表现优秀者可申请提前转正。",
    "每季度进行一次绩效考核，考核结果直接影响年终奖金。",
    "内部推荐奖励：推荐 P6 及以上职级入职通过试用期，奖励 5000 元。",
    "结婚礼金：正式员工结婚可申请 1000 元致贺金。",
    "生育津贴：符合国家计划生育政策的员工，公司额外发放 2000 元慰问金。",

    # --- 行政 (Admin) ---
    "公司食堂位于 B1 层，早餐免费，午餐和晚餐刷工卡扣费。",
    "免费晚餐提供时间为晚上 18:30 至 20:00。",
    "晚上加班超过 22:00，可报销打车费，需使用企业滴滴支付。",
    "办公用品（笔、本子、鼠标等）每周二、周四下午 2 点在行政前台领取。",
    "访客来访需提前在飞书/钉钉系统上预约，生成访客二维码。",
    "健身房位于 5 楼，全天 24 小时对员工免费开放，需刷卡进入。",
    "会议室预定需提前 24 小时在 OA 系统申请，使用完毕后请带走垃圾。",

    # --- IT 技术 (Tech) ---
    "公司内部 Wi-Fi 名称为 'Staff-WiFi'，密码为工号+身份证后4位。",
    "VPN 连接地址为 vpn.company.com，需使用动态令牌登录。",
    "电脑密码每 90 天强制修改一次，新密码不能与旧密码相同。",
    "发生 502 错误通常是因为网关超时，请联系运维组检查 Nginx 配置。",
    "申请 Github 企业版权限，请发送邮件至 it-support@company.com。",
    "Jira 账号权限申请需经过部门 TL 审批。",
    "严禁在公司电脑上安装盗版软件或未授权的远程控制工具。",
    "显示器、键盘等外设申请，需填写《IT资产领用单》并由主管签字。",

    # --- 财务 (Finance) ---
    "每月的报销截止日期为 25 号，逾期将顺延至下月处理。",
    "差旅住宿标准：一线城市 600元/晚，二线城市 400元/晚。",
    "因公招待费需提前申请，单人标准不超过 150 元。",
    "发票抬头：未来科技有限公司，税号：91110108MA000000。",
    "工资发放日为每月 10 号，遇节假日顺延。",
    "年终奖通常在农历春节前一周发放。",
    "打车报销必须提供电子发票和行程单，缺一不可。"
]


def init_database():
    print(f"🚀 开始初始化知识库，共准备了 {len(knowledge_base)} 条数据...")

    # 1. 初始化 Embedding 模型 (用于生成向量)
    embeddings = OpenAIEmbeddings(
        model="embedding-2",
        api_key=os.getenv("API_KEY"),
        base_url=os.getenv("BASE_URL")
    )

    # 2. 连接 Milvus
    client = MilvusClient(uri=os.getenv("MILVUS_URI"), token=os.getenv("MILVUS_TOKEN"))

    # 3. 清理旧数据 (如果存在，先删了重建，保证干净)
    if client.has_collection(COLLECTION_NAME):
        print(f"⚠️ 发现旧集合 {COLLECTION_NAME}，正在删除...")
        client.drop_collection(COLLECTION_NAME)

    # 4. 创建新集合
    print("📦 正在创建新集合...")
    client.create_collection(
        collection_name=COLLECTION_NAME,
        dimension=1024  # 智谱 embedding-2 维度
    )

    # 5. 批量生成向量 (Vectorization)
    print("⚡ 正在调用 API 生成向量 (请稍候)...")
    vectors = embeddings.embed_documents(knowledge_base)

    # 6. 组装数据
    data = []
    for i, text in enumerate(knowledge_base):
        data.append({
            "id": i,
            "vector": vectors[i],
            "text": text
        })

    # 7. 插入 Milvus
    print(f"💾 正在写入 {len(data)} 条数据到 Milvus...")
    res = client.insert(collection_name=COLLECTION_NAME, data=data)

    print(f"✅ 初始化完成！成功插入数据。")
    print(f"📊 插入统计: {res}")


if __name__ == "__main__":
    init_database()