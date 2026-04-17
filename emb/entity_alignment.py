#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
WebQSP 数据集实体对齐到 FB15k237
输出简化版映射文件：entity_name -> entity_name_fb
"""

import pandas as pd
import json
from typing import List, Dict, Set
from difflib import SequenceMatcher
from tqdm import tqdm
import os

# 可选：使用语义相似度
try:
    from sentence_transformers import SentenceTransformer
    import torch

    SEMANTIC_AVAILABLE = True
except ImportError:
    SEMANTIC_AVAILABLE = False
    print("⚠️  sentence-transformers 未安装，将只使用字符串相似度")

# ====== 配置部分 ======
ENTITY2ID_PATH = "datasets/entity2id.txt"
MID2NAME_PATH = "datasets/FB15k_mid2name.txt"
DATASET_DIR = "datasets/webqsp_processed"
OUTPUT_MAPPING_FILE = "datasets/webqsp_processed/entity_mapping.txt"  # 输出的映射文件

# 相似度阈值
STRING_SIMILARITY_THRESHOLD = 0.75
SEMANTIC_SIMILARITY_THRESHOLD = 0.65

# 方法选择
USE_SEMANTIC = True and SEMANTIC_AVAILABLE
USE_STRING = False
SEMANTIC_MODEL = "/media/shu1004/pytorch/projects/lyx/subgraph/SubgraphRAG-main/all-mpnet-base-v2"


class EntityAligner:
    """实体对齐器"""

    def __init__(self):
        self.mid2name = {}
        self.name2mid = {}
        self.all_kb_entities = []
        self.semantic_model = None
        self.kb_embeddings = None

        self._load_fb15k237()

        if USE_SEMANTIC:
            self._load_semantic_model()

    def _load_fb15k237(self):
        """加载FB15k237实体数据"""
        print("\n正在加载 FB15k237 数据...")

        # 加载 entity2id.txt
        print(f"  读取: {ENTITY2ID_PATH}")
        valid_mids = set()
        with open(ENTITY2ID_PATH, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines[1:]:
                if line.strip():
                    parts = line.strip().split('\t')
                    if len(parts) == 2:
                        valid_mids.add(parts[0])
        print(f"    有效 mid 数量: {len(valid_mids)}")

        # 加载 mid2name.txt
        print(f"  读取: {MID2NAME_PATH}")
        with open(MID2NAME_PATH, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    parts = line.strip().split('\t')
                    if len(parts) == 2:
                        mid, name = parts
                        if mid in valid_mids:
                            self.mid2name[mid] = name
                            name_lower = name.lower().replace('_', ' ')
                            self.name2mid[name_lower] = mid

        print(f"    加载了 {len(self.mid2name)} 个实体")

        # 构建实体列表
        self.all_kb_entities = [
            {
                'mid': mid,
                'name': name,
                'name_normalized': self._normalize_entity(name)
            }
            for mid, name in self.mid2name.items()
        ]
        print(f"    共有 {len(self.all_kb_entities)} 个可用实体")

    def _load_semantic_model(self):
        """加载语义相似度模型"""
        if not SEMANTIC_AVAILABLE:
            return

        print(f"\n正在加载语义模型: {SEMANTIC_MODEL}")
        self.semantic_model = SentenceTransformer(SEMANTIC_MODEL)

        print("  预计算KB实体embeddings...")
        kb_texts = [e['name_normalized'] for e in self.all_kb_entities]
        self.kb_embeddings = self.semantic_model.encode(
            kb_texts,
            convert_to_tensor=True,
            show_progress_bar=True,
            batch_size=256
        )
        print(f"  完成！")

    def _normalize_entity(self, entity: str) -> str:
        """增强的实体标准化"""
        import re

        # 替换下划线为空格
        entity = entity.replace('_', ' ')

        # 移除括号内容 (例如: "Paris (city)" -> "Paris")
        entity = re.sub(r'\([^)]*\)', '', entity)

        # 移除引号
        entity = entity.replace('"', '').replace("'", '')

        # 处理常见缩写
        entity = entity.replace('U.S.', 'US')
        entity = entity.replace('U.K.', 'UK')

        # 统一 "the" 的处理
        entity = re.sub(r'^the\s+', '', entity, flags=re.IGNORECASE)

        # 移除多余空格和标点
        entity = re.sub(r'[,;:]', ' ', entity)
        entity = ' '.join(entity.split())

        # 转小写用于匹配
        return entity.strip()

    def _string_similarity(self, str1: str, str2: str) -> float:
        """计算字符串相似度"""
        return SequenceMatcher(None, str1.lower(), str2.lower()).ratio()

    def _semantic_similarity(self, query: str, return_top_k: int = 1) -> List:
        """使用语义相似度查找最相似的KB实体"""
        if not self.semantic_model or self.kb_embeddings is None:
            return []

        query_embedding = self.semantic_model.encode(query, convert_to_tensor=True)
        cos_scores = torch.nn.functional.cosine_similarity(
            query_embedding.unsqueeze(0),
            self.kb_embeddings
        )

        top_results = torch.topk(cos_scores, k=min(return_top_k, len(cos_scores)))

        results = []
        for score, idx in zip(top_results.values, top_results.indices):
            entity = self.all_kb_entities[idx.item()]
            results.append((entity['name'], score.item()))

        return results

    # 在 EntityAligner 初始化中添加
    def _build_alias_mapping(self):
        """构建常见别名映射"""
        self.alias_to_mid = {
            # 常见缩写
            'us': 'United_States',
            'usa': 'United_States',
            'uk': 'United_Kingdom',
            'nyc': 'New_York_City',
            'ny': 'New_York',
            'la': 'Los_Angeles',

            # 可以根据实际情况添加更多
        }

        # 在精确匹配前先查别名
        entity_lower = entity_normalized.lower()
        if entity_lower in self.alias_to_mid:
            alias_name = self.alias_to_mid[entity_lower]
            # 查找对应的mid
            for name_lower, mid in self.name2mid.items():
                if alias_name.lower() == name_lower:
                    return self.mid2name[mid]

    def align_entity(self, entity: str, method='hybrid') -> str:
        """改进的实体对齐"""
        entity_normalized = self._normalize_entity(entity)
        entity_lower = entity_normalized.lower()

        # 1. 精确匹配
        if entity_lower in self.name2mid:
            mid = self.name2mid[entity_lower]
            return self.mid2name[mid]

        # 2. 部分匹配（实体名包含关系）
        for kb_name_lower, mid in self.name2mid.items():
            # 如果查询实体是KB实体的子串或相反
            if (entity_lower in kb_name_lower or kb_name_lower in entity_lower):
                # 长度差异不能太大
                len_diff = abs(len(entity_lower) - len(kb_name_lower))
                if len_diff <= max(3, len(entity_lower) * 0.2):
                    return self.mid2name[mid]

        # 3. 去除常见后缀再匹配
        suffixes = [' (film)', ' (band)', ' (album)', ' (song)',
                    ' (city)', ' (country)', ' (company)']
        for suffix in suffixes:
            if entity_lower.endswith(suffix):
                base_name = entity_lower[:-len(suffix)]
                if base_name in self.name2mid:
                    mid = self.name2mid[base_name]
                    return self.mid2name[mid]

        # 4. 字符串相似度匹配
        best_match = None
        best_score = 0.0
        threshold = STRING_SIMILARITY_THRESHOLD

        if USE_STRING:
            for kb_entity in self.all_kb_entities:
                score = self._string_similarity(
                    entity_normalized,
                    kb_entity['name_normalized']
                )
                if score > best_score:
                    best_score = score
                    best_match = kb_entity['name']

        # 5. 语义相似度匹配
        if USE_SEMANTIC:
            semantic_results = self._semantic_similarity(entity_normalized)
            if semantic_results:
                name, score = semantic_results[0]
                if score > best_score:
                    best_score = score
                    best_match = name
                    threshold = SEMANTIC_SIMILARITY_THRESHOLD

        # 判断阈值
        if best_match and best_score >= threshold:
            return best_match
        else:
            return 'not_linked'

    def extract_entities_from_dataset(self, row: pd.Series) -> Set[str]:
        """从数据集一行中提取所有实体"""
        entities = set()

        # q_entity_list
        if 'q_entity_list' in row:
            try:
                val = row['q_entity_list']
                if not pd.isna(val) and val is not None:
                    if isinstance(val, list):
                        entities.update([str(e).strip() for e in val if e])
            except:
                pass

        # q_entity
        if 'q_entity' in row:
            try:
                val = row['q_entity']
                if not pd.isna(val) and val is not None:
                    if isinstance(val, list):
                        entities.update([str(e).strip() for e in val if e])
                    else:
                        entities.add(str(val).strip())
            except:
                pass

        # a_entity
        if 'a_entity' in row:
            try:
                val = row['a_entity']
                if not pd.isna(val) and val is not None:
                    if isinstance(val, list):
                        entities.update([str(e).strip() for e in val if e])
                    else:
                        entities.add(str(val).strip())
            except:
                pass

        # graph中的三元组
        if 'graph' in row:
            try:
                val = row['graph']
                if not pd.isna(val) and val is not None:
                    if isinstance(val, str):
                        graph = json.loads(val)
                    else:
                        graph = val

                    if isinstance(graph, (list, tuple)):
                        for triple in graph:
                            if isinstance(triple, (list, tuple)) and len(triple) >= 3:
                                entities.add(str(triple[0]).strip())
                                entities.add(str(triple[2]).strip())
            except Exception as e:
                pass  # 忽略解析错误

        # 过滤空字符串
        entities = {e for e in entities if e and len(e) > 0}
        return entities


def process_all_datasets(aligner: EntityAligner, output_file: str):
    """处理所有数据集文件并生成统一的映射文件"""

    dataset_files = [
        "train-00000-of-00002.parquet",
        "train-00001-of-00002.parquet",
        "validation-00000-of-00001.parquet",
        "test-00000-of-00002.parquet",
        "test-00001-of-00002.parquet"
    ]

    # 收集所有唯一实体
    print("\n收集所有数据集中的实体...")
    all_entities = set()

    for filename in dataset_files:
        file_path = os.path.join(DATASET_DIR, filename)

        if not os.path.exists(file_path):
            print(f"⚠️  文件不存在，跳过: {file_path}")
            continue

        print(f"  处理: {filename}")
        df = pd.read_parquet(file_path)

        for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"  提取 {filename}"):
            entities = aligner.extract_entities_from_dataset(row)
            all_entities.update(entities)

    print(f"\n共收集到 {len(all_entities)} 个唯一实体")

    # 对齐所有实体
    print("\n开始实体对齐...")
    entity_mapping = {}
    linked_count = 0

    for entity in tqdm(sorted(all_entities), desc="对齐实体"):
        fb_entity = aligner.align_entity(entity)
        entity_mapping[entity] = fb_entity
        if fb_entity != 'not_linked':
            linked_count += 1

    # 统计
    print(f"\n对齐结果:")
    print(f"  成功链接: {linked_count}/{len(all_entities)} ({linked_count / len(all_entities) * 100:.1f}%)")
    print(f"  未链接: {len(all_entities) - linked_count}")

    # 保存映射文件
    print(f"\n保存映射文件: {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        # 写入表头
        f.write("entity_name\tentity_name_fb\n")

        # 写入映射（按entity_name排序）
        for entity in sorted(entity_mapping.keys()):
            fb_entity = entity_mapping[entity]
            f.write(f"{entity}\t{fb_entity}\n")

    print(f"✅ 完成！映射已保存到: {output_file}")

    # 显示前10个示例
    print(f"\n前10个映射示例:")
    for i, entity in enumerate(sorted(entity_mapping.keys())[:10]):
        fb_entity = entity_mapping[entity]
        status = "✓" if fb_entity != 'not_linked' else "✗"
        print(f"  {status} {entity} -> {fb_entity}")


def main():
    """主函数"""
    print("=" * 80)
    print("WebQSP 实体对齐到 FB15k237")
    print("输出格式: entity_name\\tentity_name_fb")
    print("=" * 80)
    print(f"\n配置:")
    print(f"  FB15k237 数据: {ENTITY2ID_PATH}, {MID2NAME_PATH}")
    print(f"  数据集目录: {DATASET_DIR}")
    print(f"  输出文件: {OUTPUT_MAPPING_FILE}")
    print(f"  字符串相似度阈值: {STRING_SIMILARITY_THRESHOLD}")
    print(f"  语义相似度阈值: {SEMANTIC_SIMILARITY_THRESHOLD}")
    print(f"  使用字符串匹配: {USE_STRING}")
    print(f"  使用语义匹配: {USE_SEMANTIC}")
    if USE_SEMANTIC:
        print(f"  语义模型: {SEMANTIC_MODEL}")
    print("=" * 80)

    confirm = input("\n确认开始处理？(yes/no): ")
    if confirm.lower() != 'yes':
        print("已取消。")
        return

    # 初始化对齐器
    aligner = EntityAligner()

    # 处理所有数据集
    process_all_datasets(aligner, OUTPUT_MAPPING_FILE)

    print("\n" + "=" * 80)
    print("处理完成！")
    print(f"映射文件: {OUTPUT_MAPPING_FILE}")
    print("=" * 80)


if __name__ == "__main__":
    main()