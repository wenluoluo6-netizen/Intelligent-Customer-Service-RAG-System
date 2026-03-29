import time
import cProfile
import pstats
import io


# --- 1. 模拟各种耗时的组件 ---

def simulate_llm_api_call():
    """模拟大模型 API 调用 (典型的 I/O 密集型，干等网络)"""
    # 假设调用 GPT-4 接口需要等待 2 秒
    time.sleep(2)
    return "这是大模型生成的回答"


def simulate_database_query():
    """模拟查询 Milvus 向量数据库 (也是 I/O 密集型)"""
    # 假设查数据库需要 0.5 秒
    time.sleep(0.5)
    return ["文档1", "文档2"]


def heavy_data_processing():
    """模拟复杂的文本处理/计算 (典型的 CPU 密集型)"""
    # 比如我们在对检索到的几万字做非常复杂的清洗
    result = 0
    for i in range(10_000_000):  # 一千万次循环，强行消耗 CPU
        result += i
    return result


# --- 2. 主流程 ---
def agent_workflow():
    print("开始执行 Agent 工作流...")

    docs = simulate_database_query()
    print("知识库查询完毕。")

    heavy_data_processing()
    print("数据清洗完毕。")

    answer = simulate_llm_api_call()
    print("大模型回答完毕。")

    print("工作流执行结束！")


# --- 3. 核心：使用 cProfile 给代码拍 CT 片 ---
if __name__ == "__main__":
    # 创建分析器对象
    profiler = cProfile.Profile()

    # 开始监控
    profiler.enable()

    # 执行我们的核心代码
    agent_workflow()

    # 停止监控
    profiler.disable()

    # --- 4. 打印分析报告 ---
    print("\n" + "=" * 50)
    print("📊 性能分析报告 (Profile Report)")
    print("=" * 50)

    # 将结果读入流中以便格式化打印
    s = io.StringIO()
    # 根据 'cumtime' (累计耗时) 进行排序，把最慢的排在最前面
    sortby = 'cumtime'
    ps = pstats.Stats(profiler, stream=s).sort_stats(sortby)

    # 打印前 10 行最耗时的函数调用
    ps.print_stats(10)
    print(s.getvalue())