import pandas as pd
import pdfplumber


# 1. 定义文件路径
pdf_path = "test_table.pdf"

print("🔍 正在扫描 PDF 中的表格...\n")

# 2. 打开 PDF 文件
with pdfplumber.open(pdf_path) as pdf:
    # 遍历 PDF 的每一页
    for page_num, page in enumerate(pdf.pages):
        tables = page.extract_tables()      # extract_tables() 会返回一个三维列表：[表格[行[单元格]]]

        if not tables:
            print(f"第 {page_num + 1} 页：没有发现表格。")
            continue

        print(f"✅ 在第 {page_num + 1} 页发现了 {len(tables)} 个表格！")
        print(f"extract_tables()返回的三维列表内容：{tables}")

        # 3. 处理提取到的表格
        for table_idx, table in enumerate(tables):
            # 清理表格中的空行和 None 值（pdfplumber 提取时偶尔会有空数据）
            cleaned_table = []
            for row in table:
                cleaned_row = [str(cell).replace('\n', ' ') if cell else "" for cell in row]
                cleaned_table.append(cleaned_row)

            if len(cleaned_table) > 1:   # 至少得有表头和一行数据
                # 第一行作为表头 (columns)，后面的作为数据
                df = pd.DataFrame(cleaned_table[1:], columns=cleaned_table[0])

                print(f"\n--- 表格 {table_idx + 1} 的 Markdown 源码 ---")

                md_table = df.to_markdown(index=False)
                print(md_table)

                print("-" * 40)
