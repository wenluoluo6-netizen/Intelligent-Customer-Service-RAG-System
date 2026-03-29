# 1. 选底座：我们要一个装好了 Python 3.10 的纯净系统
FROM python:3.10-slim

# 2. 设地盘：在容器里创建一个叫 /app 的文件夹，后面我们就在这里干活
WORKDIR /app

# 3. 搬清单：把你电脑上的 requirements.txt 复制进容器
COPY requirements.txt .

# 4. 装依赖：让 Docker 按照清单自动安装库（使用清华源加速，防止卡住）
RUN pip install --no-cache-dir -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 5. 搬代码：把你当前目录下所有的代码（server.py, .env, knowledge.txt等）都复制进去
COPY . .

# 6. 暴露端口：告诉 Docker 我这个程序要用 8000 端口
EXPOSE 8000

# 7. 启动命令：容器一启动，就自动运行这句话
# 重点：host 必须是 0.0.0.0，不能写 127.0.0.1
CMD ["uvicorn", "DAY6:app", "--host", "0.0.0.0", "--port", "8000"]