#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
使用 Gemini 从 WebQSP 数据集中提取实体
"""

import pandas as pd
from google import genai
import json
from tqdm import tqdm
import time
import os
from pathlib import Path

# ====== 配置部分 ======
# 设置您的 Gemini API Key（通过代理）
GEMINI_API_KEY = "sk-IQ8vi7XzSOgnTAW805DchQy2YVSOA8q6WYb7vUZRYOHKN6vN"
PROXY_BASE_URL = "https://api.openai-proxy.org/google"  # 代理地址
MODEL_NAME = "gemini-2.5-flash-lite-preview-09-2025"  # 或 "gemini-1.5-flash", "gemini-1.5-pro" 等

# 数据集路径
DATASET_DIR = "g:/小论文/第一章/新建文件夹/SubgraphRAG-main/datasets/webqsp"
OUTPUT_DIR = "g:/小论文/第一章/新建文件夹/SubgraphRAG-main/datasets/webqsp_processed"

# ====== 初始化 Gemini 客户端 ======
client = genai.Client(
    api_key=GEMINI_API_KEY,
    http_options={"base_url": PROXY_BASE_URL}
)

# ====== 实体提取提示词 ======
ENTITY_EXTRACTION_PROMPT = """You are an expert in named entity recognition. Your task is to extract all key entities from a given question.

Please identify and extract the following types of entities:
- Person names (e.g., "Barack Obama", "Michael Jordan")
- Location names (e.g., "New York", "Lejre Municipality", "United States")
- Organization names (e.g., "Google", "United Nations", "Harvard University")
- Product/brand names
- Date/time references that refer to specific entities
- Any other named entities that are crucial to understanding the question

Instructions:
1. Extract complete entity names (e.g., "New York City" not just "New York")
2. Include compound entities (e.g., "Lejre Municipality")
3. Do not extract common words or generic terms
4. Return entities in their original form from the question

Question: {question}

Return ONLY a JSON object with a single key "entities" containing a list of extracted entities.
Example format: {{"entities": ["Entity1", "Entity2", "Entity3"]}}

JSON Response:"""


def extract_entities_with_gemini(question: str, max_retries: int = 3) -> list:
    """
    使用 Gemini 从问题中提取实体

    Args:
        question: 问题文本
        max_retries: 最大重试次数

    Returns:
        提取出的实体列表
    """
    prompt = ENTITY_EXTRACTION_PROMPT.format(question=question)

    for attempt in range(max_retries):
        try:
            # 调用 Gemini API
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=prompt,
                config={
                    "temperature": 0.0,  # 使用确定性输出
                    "max_output_tokens": 1000,
                    "top_p": 0.95,
                    "top_k": 40,
                    "response_mime_type": "application/json",  # 强制返回 JSON
                }
            )

            # 解析响应
            result_text = response.text.strip()
            result_json = json.loads(result_text)
            entities = result_json.get("entities", [])

            # 确保返回的是列表
            if isinstance(entities, list):
                return entities
            else:
                return [entities] if entities else []

        except json.JSONDecodeError as e:
            print(f"JSON 解析错误 (尝试 {attempt + 1}/{max_retries}): {e}")
            if 'response' in locals():
                print(f"原始响应: {response.text[:200]}...")
            if attempt < max_retries - 1:
                time.sleep(1)
            else:
                return []

        except Exception as e:
            error_msg = str(e).lower()
            print(f"API 调用错误 (尝试 {attempt + 1}/{max_retries}): {e}")

            # 处理速率限制
            if "rate" in error_msg or "quota" in error_msg or "429" in error_msg:
                wait_time = 2 ** attempt
                print(f"  速率限制，等待 {wait_time} 秒...")
                time.sleep(wait_time)
            # 处理超时
            elif "timeout" in error_msg:
                print(f"  请求超时，等待后重试...")
                time.sleep(2)
            else:
                time.sleep(1)

            if attempt >= max_retries - 1:
                return []

    return []


def process_dataset_file(input_file: str, output_file: str, sample_size: int = None):
    """
    处理单个数据集文件

    Args:
        input_file: 输入的 parquet 文件路径
        output_file: 输出的 parquet 文件路径
        sample_size: 如果指定，只处理前 N 条数据（用于测试）
    """
    print(f"\n正在读取文件: {input_file}")
    df = pd.read_parquet(input_file)

    print(f"数据集大小: {len(df)} 条")
    print(f"数据集列名: {df.columns.tolist()}")
    print(f"\n前 3 条数据示例:")
    print(df.head(3))

    # 如果是测试，只处理部分数据
    if sample_size:
        df = df.head(sample_size)
        print(f"\n测试模式：只处理前 {sample_size} 条数据")

    # 检查是否存在问题列（可能是 'question' 或 'Q' 等）
    question_col = None
    for col in ['question', 'Question', 'Q', 'query', 'text']:
        if col in df.columns:
            question_col = col
            break

    if question_col is None:
        print(f"错误：未找到问题列！可用列: {df.columns.tolist()}")
        return

    print(f"\n使用列 '{question_col}' 作为问题文本")

    # 提取实体
    entities_list = []
    print(f"\n开始提取实体...")

    # 记录处理时间
    start_time = time.time()

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="提取实体"):
        question = row[question_col]

        # 调用 Gemini 提取实体
        entities = extract_entities_with_gemini(question)
        entities_list.append(entities)

        # 打印前几个结果作为示例
        if idx < 3:
            print(f"\n问题: {question}")
            print(f"提取的实体: {entities}")

        # 每处理10条显示一次进度
        if (idx + 1) % 10 == 0:
            elapsed_time = time.time() - start_time
            avg_time = elapsed_time / (idx + 1)
            remaining = (len(df) - idx - 1) * avg_time
            print(f"\n  已处理 {idx + 1}/{len(df)} 条 | 平均 {avg_time:.2f}秒/条 | 预计剩余 {remaining / 60:.1f}分钟")

        # 添加延迟以避免超过 API 速率限制
        time.sleep(0.5)

    # 添加新列
    df['q_entity_list'] = entities_list

    # 保存结果
    print(f"\n正在保存到: {output_file}")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df.to_parquet(output_file, index=False)

    print(f"完成！已保存 {len(df)} 条数据")

    # 显示统计信息
    total_entities = sum(len(e) for e in entities_list)
    avg_entities = total_entities / len(entities_list) if entities_list else 0
    total_time = time.time() - start_time

    print(f"\n统计信息:")
    print(f"  总实体数: {total_entities}")
    print(f"  平均每个问题的实体数: {avg_entities:.2f}")
    print(f"  总耗时: {total_time / 60:.1f} 分钟")
    print(f"  平均耗时: {total_time / len(df):.2f} 秒/条")


def load_progress(save_path):
    """加载已保存的进度（用于断点续传）"""
    if os.path.exists(save_path):
        try:
            df = pd.read_parquet(save_path)
            print(f"✓ 发现已保存的进度文件，已处理 {len(df)} 条数据")
            return df
        except Exception as e:
            print(f"加载进度文件失败: {e}")
    return None


def process_dataset_file_with_resume(input_file: str, output_file: str, sample_size: int = None):
    """
    处理单个数据集文件（支持断点续传）

    Args:
        input_file: 输入的 parquet 文件路径
        output_file: 输出的 parquet 文件路径
        sample_size: 如果指定，只处理前 N 条数据（用于测试）
    """
    print(f"\n正在读取文件: {input_file}")
    df = pd.read_parquet(input_file)

    print(f"数据集大小: {len(df)} 条")
    print(f"数据集列名: {df.columns.tolist()}")

    # 如果是测试，只处理部分数据
    if sample_size:
        df = df.head(sample_size)
        print(f"\n测试模式：只处理前 {sample_size} 条数据")

    # 检查是否存在问题列
    question_col = None
    for col in ['question', 'Question', 'Q', 'query', 'text']:
        if col in df.columns:
            question_col = col
            break

    if question_col is None:
        print(f"错误：未找到问题列！可用列: {df.columns.tolist()}")
        return

    print(f"\n使用列 '{question_col}' 作为问题文本")

    # 加载已有进度
    processed_df = load_progress(output_file)

    if processed_df is not None and len(processed_df) > 0:
        # 从已处理的位置继续
        start_idx = len(processed_df)
        print(f"从第 {start_idx + 1} 条开始处理...")
        entities_list = processed_df['q_entity_list'].tolist()
    else:
        start_idx = 0
        entities_list = []

    print(f"\n开始提取实体...")
    start_time = time.time()

    for idx in range(start_idx, len(df)):
        row = df.iloc[idx]
        question = row[question_col]

        print(f"\n[{idx + 1}/{len(df)}] 处理中...")

        # 调用 Gemini 提取实体
        entities = extract_entities_with_gemini(question)
        entities_list.append(entities)

        # 打印结果
        print(f"问题: {question[:80]}...")
        print(f"提取的实体: {entities}")

        # 每处理5条保存一次
        if (idx + 1) % 5 == 0:
            temp_df = df.iloc[:idx + 1].copy()
            temp_df['q_entity_list'] = entities_list

            elapsed_time = time.time() - start_time
            avg_time = elapsed_time / (idx + 1 - start_idx)
            remaining = (len(df) - idx - 1) * avg_time

            print(f"\n💾 保存进度... (已处理 {idx + 1} 条)")
            print(f"   平均耗时: {avg_time:.2f}秒/条")
            print(f"   预计剩余时间: {remaining / 60:.1f}分钟")

            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            temp_df.to_parquet(output_file, index=False)

        # 添加延迟
        time.sleep(0.5)

    # 最终保存
    df['q_entity_list'] = entities_list
    print(f"\n正在保存最终结果到: {output_file}")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df.to_parquet(output_file, index=False)

    print(f"完成！已保存 {len(df)} 条数据")

    # 显示统计信息
    total_entities = sum(len(e) for e in entities_list)
    avg_entities = total_entities / len(entities_list) if entities_list else 0
    total_time = time.time() - start_time

    print(f"\n📊 统计信息:")
    print(f"  总数据量: {len(df)}")
    print(f"  总实体数: {total_entities}")
    print(f"  平均每个问题的实体数: {avg_entities:.2f}")
    print(f"  总耗时: {total_time / 60:.1f} 分钟")
    print(f"  平均耗时: {total_time / len(df):.2f} 秒/条")


def process_all_datasets(test_mode: bool = False, use_resume: bool = True):
    """
    处理所有数据集文件

    Args:
        test_mode: 如果为 True，只处理每个文件的前 5 条数据
        use_resume: 是否启用断点续传
    """
    dataset_files = [
        "train-00000-of-00002.parquet",
        "train-00001-of-00002.parquet",
        "validation-00000-of-00001.parquet",
        "test-00000-of-00002.parquet",
        "test-00001-of-00002.parquet"
    ]

    for filename in dataset_files:
        input_path = os.path.join(DATASET_DIR, filename)
        output_path = os.path.join(OUTPUT_DIR, filename)

        if not os.path.exists(input_path):
            print(f"跳过不存在的文件: {input_path}")
            continue

        try:
            sample_size = 5 if test_mode else None

            if use_resume:
                process_dataset_file_with_resume(input_path, output_path, sample_size)
            else:
                process_dataset_file(input_path, output_path, sample_size)

        except KeyboardInterrupt:
            print(f"\n⏸️  用户中断，进度已保存")
            break
        except Exception as e:
            print(f"处理文件 {filename} 时出错: {e}")
            import traceback
            traceback.print_exc()
            continue


def main():
    """主函数"""
    print("=" * 80)
    print("WebQSP 数据集实体提取工具 (Gemini API)")
    print("=" * 80)
    print(f"模型: {MODEL_NAME}")
    print(f"代理: {PROXY_BASE_URL}")
    print("=" * 80)

    # 询问用户是否要测试模式
    mode = input(
        "\n选择模式：\n1. 测试模式（每个文件只处理 5 条数据）\n2. 完整模式（处理所有数据，支持断点续传）\n请输入 (1/2): ")

    test_mode = (mode.strip() == "1")

    if test_mode:
        print("\n⚠️  测试模式：每个文件只处理前 5 条数据")
    else:
        print("\n开始处理所有数据...")
        print("提示：支持断点续传，按 Ctrl+C 可随时中断，下次继续")
        confirm = input("这将调用大量 API 请求，确认继续？(yes/no): ")
        if confirm.lower() != "yes":
            print("已取消。")
            return

    # 处理数据集
    try:
        process_all_datasets(test_mode=test_mode, use_resume=True)
    except KeyboardInterrupt:
        print("\n⏸️  用户中断，进度已保存")
    except Exception as e:
        print(f"\n❌ 处理过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 80)
    print("处理完成！")
    print("=" * 80)


if __name__ == "__main__":
    main()