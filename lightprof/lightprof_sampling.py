#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LightPROF - 关系链重构与推理子图采样（含 PoG Fuzzy Selection 增强）
基于 inference_hybrid.py + pog_indicator.py 的输出进行进一步处理

算法流程：
1.  关系链检索：从锚点实体出发，使用深度受限BFS收集关系链
1.5 模糊筛选（Fuzzy Selection，PoG 论文 Section 4.3）：
    使用 SentenceBERT 将 I_LLM 指标和所有候选链编码为稠密向量，
    通过余弦相似度保留 Top-W1 条语义相关链（默认80），快速过滤无关噪声
2a. 语义相关性精筛：使用LLM对模糊筛选后的链进行精确打分
2b. 推理图采样：实例化Top-K关系链为具体的推理路径，构建推理图 G_R
"""

import os
import torch
import json
import collections
from typing import List, Dict, Set, Tuple, Optional
from tqdm import tqdm
from argparse import ArgumentParser


class SimpleKG:
    """
    基于scored_triples构建的简单知识图谱
    支持正向和反向边的遍历
    """

    def __init__(self, triples: List[Tuple[str, str, str, float]], bidirectional=True):
        """
        Args:
            triples: [(head, relation, tail, score), ...]
            bidirectional: 是否构建双向边（用于无向图检索）
        """
        # 邻接表: {entity: [(relation, neighbor_entity), ...]}
        self.adj_list = collections.defaultdict(list)
        # 三元组字典: {(h, r, t): score}
        self.triple_scores = {}

        for h, r, t, score in triples:
            self.adj_list[h].append((r, t))
            self.triple_scores[(h, r, t)] = score

            if bidirectional:
                # 添加反向边（用于双向搜索）
                self.adj_list[t].append((f"INV_{r}", h))
                self.triple_scores[(t, f"INV_{r}", h)] = score

    def get_neighbors(self, entity: str) -> List[Tuple[str, str]]:
        """获取实体的所有邻居: [(relation, neighbor_entity), ...]"""
        return self.adj_list.get(entity, [])

    def get_score(self, h: str, r: str, t: str) -> float:
        """获取三元组的分数"""
        return self.triple_scores.get((h, r, t), 0.0)


class FuzzySelector:
    """
    模糊选择器（对应 PoG 论文 Section 4.3 Fuzzy Selection）
    使用 SentenceBERT 对关系链进行轻量语义粗筛

    原理：
      1. 将 I_LLM 指标文本和所有候选关系链编码为稠密向量 H_I 、H_Paths
      2. 计算余弦相似度，衡量路径与推理指标的语义匹配度
      3. 保留相似度最高的前 W1 条路径，其余直接剔除
    """

    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """
        Args:
            model_name: SentenceBERT 模型名称
                        推荐: 'all-MiniLM-L6-v2'（轻量快速，优先选择）
                             'all-mpnet-base-v2'（效果更好但更慢）
        """
        self.model_name = model_name
        self.model = None
        self._load_model()

    def _load_model(self):
        """加载 SentenceBERT 模型（延迟加载，首次使用时才初始化）"""
        try:
            from sentence_transformers import SentenceTransformer
            print(f"  ✓ 加载 FuzzySelector 模型: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
        except ImportError:
            print("  ⚠️  未安装 sentence-transformers，模糊筛选将被跳过")
            print("       安装命令: pip install sentence-transformers")
            self.model = None
        except Exception as e:
            print(f"  ⚠️  FuzzySelector 模型加载失败: {e}，模糊筛选将被跳过")
            self.model = None

    def _chain_to_text(self, chain: Tuple[str, ...]) -> str:
        """将关系链 tuple 转换为自然语言文本用于编码"""
        # 去除 INV_ 前缀（反向边标记），使语义更自然
        cleaned = [r.replace('INV_', '').replace('_', ' ') for r in chain]
        return ' -> '.join(cleaned)

    def select(
        self,
        indicator_text: str,
        chains: List[Tuple[str, ...]],
        w1: int = 80
    ) -> List[Tuple[str, ...]]:
        """
        模糊筛选：保留与指标语义最相关的 Top-W1 条关系链

        Args:
            indicator_text: I_LLM 推理指标文本
                            如 "Lou Seal - mascot for - team - won - World Series"
            chains:         候选关系链列表
            w1:             保留的最大链数量（论文默认 80）

        Returns:
            filtered_chains: 经过语义粗筛后的关系链列表（最多 w1 条）
        """
        # 模型未加载或链数量不超过 w1 时，直接返回
        if self.model is None or len(chains) <= w1:
            return list(chains)

        # 将关系链转换为文本
        chain_texts = [self._chain_to_text(c) for c in chains]

        try:
            import numpy as np

            # 编码 I_LLM 指标文本和所有候选链（normalize_embeddings=True 后点积即余弦相似度）
            h_i = self.model.encode(
                [indicator_text], normalize_embeddings=True
            )  # shape: (1, D)
            h_paths = self.model.encode(
                chain_texts, normalize_embeddings=True, show_progress_bar=False
            )  # shape: (N, D)

            # 计算余弦相似度
            similarities = (h_paths @ h_i.T).squeeze(-1)  # shape: (N,)

            # 取 Top-W1 的索引（降序）
            top_indices = np.argsort(similarities)[::-1][:w1]
            filtered = [chains[i] for i in top_indices]

            return filtered

        except Exception as e:
            print(f"  ⚠️  模糊筛选计算失败: {e}，返回原始链列表")
            return list(chains)


class LightPROF_Sampler:
    """
    LightPROF 推理图采样器
    实现关系链检索、PoG模糊筛选、LLM精筛和推理路径采样
    """

    def __init__(self, kg: SimpleKG, llm_scorer=None, fuzzy_selector=None, verbose=True):
        """
        Args:
            kg:              SimpleKG 实例
            llm_scorer:      LLM打分函数，输入(question, chains)，输出scores列表
            fuzzy_selector:  FuzzySelector 实例（可选），用于步骤1.5模糊粗筛
            verbose:         是否打印详细信息
        """
        self.kg = kg
        self.llm_scorer = llm_scorer if llm_scorer else self._mock_llm_scoring
        self.fuzzy_selector = fuzzy_selector
        self.verbose = verbose

    def _mock_llm_scoring(self, question: str, chains: List[Tuple[str, ...]]) -> List[float]:
        """
        模拟LLM打分（用于测试）
        实际使用时应该替换为真实的LLM API调用
        """
        scores = []
        for chain in chains:
            # 简单的启发式打分：链越短得分越高，包含关键词得分更高
            score = 1.0 / (len(chain) + 1)

            # 关键词匹配（可根据实际情况调整）
            chain_str = " ".join(chain).lower()
            if "championship" in chain_str or "winner" in chain_str:
                score += 0.3
            if "mascot" in chain_str or "team" in chain_str:
                score += 0.2

            scores.append(score)
        return scores

    def step1_extract_relation_chains(self,
                                      anchor_entities: List[str],
                                      max_hops: int = 2,
                                      max_chains: int = 1000) -> Set[Tuple[str, ...]]:
        """
        步骤1: 关系链检索
        从锚点实体出发，使用深度受限BFS收集所有关系链

        Args:
            anchor_entities: 锚点实体列表（q_entity_list）
            max_hops: 最大跳数 h_q
            max_chains: 最大关系链数量（防止爆炸）

        Returns:
            relation_chains: 关系链集合 {(r1,), (r1, r2), ...}
        """
        relation_chains = set()

        # BFS队列: (当前实体, 当前关系链, 已访问实体集合)
        queue = collections.deque()

        for anchor in anchor_entities:
            if anchor:  # 确保锚点实体非空
                queue.append((anchor, (), {anchor}))

        while queue and len(relation_chains) < max_chains:
            curr_entity, curr_chain, visited = queue.popleft()

            # 如果当前链不为空，加入结果集
            if len(curr_chain) > 0:
                relation_chains.add(curr_chain)

            # 如果达到最大跳数，停止探索
            if len(curr_chain) >= max_hops:
                continue

            # 扩展邻居
            neighbors = self.kg.get_neighbors(curr_entity)
            for relation, next_entity in neighbors:
                # 避免环路（可选：根据需求决定是否允许重复访问）
                if next_entity not in visited:
                    new_chain = curr_chain + (relation,)
                    new_visited = visited.copy()
                    new_visited.add(next_entity)
                    queue.append((next_entity, new_chain, new_visited))

        if self.verbose:
            print(f"  ✓ 提取了 {len(relation_chains)} 条唯一关系链")

        return relation_chains

    def step1b_fuzzy_select(
        self,
        indicator_text: str,
        all_chains: Set[Tuple[str, ...]],
        w1: int = 80
    ) -> List[Tuple[str, ...]]:
        """
        步骤1.5: 模糊选择（Fuzzy Selection）
        对应 PoG 论文 Section 4.3 第一步粗筛

        用 SBERT 对 I_LLM 指标和所有关系链进行向量编码，
        通过余弦相似度过滤出最相关的 Top-W1 条链，
        为后续 LLM 精筛（step2a）大幅减少输入量。

        Args:
            indicator_text: I_LLM 推理指标文本
                            如 "Lou Seal - mascot for - team - won - World Series"
                            为空时用问题文本替代
            all_chains:     步骤1提取的全量关系链集合
            w1:             模糊筛选后保留的最大链数量（论文默认 80）

        Returns:
            filtered_chains: 筛选后的关系链列表（最多 w1 条）
        """
        chain_list = list(all_chains)

        # 未配置 fuzzy_selector 时跳过
        if self.fuzzy_selector is None:
            if self.verbose:
                print(f"  ⏭️  未配置 FuzzySelector，跳过模糊筛选（保留全部 {len(chain_list)} 条）")
            return chain_list

        # I_LLM 为空时跳过
        if not indicator_text or not indicator_text.strip():
            if self.verbose:
                print(f"  ⏭️  I_LLM 为空，跳过模糊筛选（保留全部 {len(chain_list)} 条）")
            return chain_list

        # 链数量不超过 w1 时无需筛选
        if len(chain_list) <= w1:
            if self.verbose:
                print(f"  ✓ 链数量 ({len(chain_list)}) ≤ W1 ({w1})，跳过模糊筛选")
            return chain_list

        # 执行模糊筛选
        filtered = self.fuzzy_selector.select(indicator_text, chain_list, w1=w1)

        if self.verbose:
            reduction = (1 - len(filtered) / len(chain_list)) * 100
            print(f"  ✓ 模糊筛选: {len(chain_list)} → {len(filtered)} 条（过滤了 {reduction:.0f}%）")

        return filtered

    def step2a_rank_and_filter(self,
                               question: str,
                               all_chains: Set[Tuple[str, ...]],
                               top_k: int = 5) -> List[Tuple[str, ...]]:
        """
        步骤2a: 语义相关性排序与筛选
        使用LLM对关系链进行打分，选择Top-K最相关的链

        Args:
            question: 问题文本
            all_chains: 所有关系链
            top_k: 选择的关系链数量

        Returns:
            top_k_chains: Top-K关系链列表
        """
        if not all_chains:
            return []

        chain_list = list(all_chains)

        # 调用LLM打分
        if self.verbose:
            print(f"  调用LLM对 {len(chain_list)} 条关系链进行打分...")

        scores = self.llm_scorer(question, chain_list)

        # 组合并排序
        scored_chains = list(zip(scores, chain_list))
        scored_chains.sort(key=lambda x: x[0], reverse=True)

        # 取Top-K
        top_k_chains = [chain for score, chain in scored_chains[:top_k]]

        if self.verbose:
            print(f"  ✓ 选择了Top-{top_k}关系链:")
            for i, (score, chain) in enumerate(scored_chains[:top_k], 1):
                chain_str = " -> ".join(chain)
                print(f"    {i}. [{score:.4f}] {chain_str}")

        return top_k_chains

    def step2b_instantiate_paths(self,
                                 anchor_entities: List[str],
                                 selected_chains: List[Tuple[str, ...]],
                                 max_paths_per_chain: int = 10) -> List[List[Tuple[str, str, str]]]:
        """
        步骤2b: 推理路径采样与推理图构建
        将抽象的关系链实例化为具体的实体路径

        Args:
            anchor_entities: 锚点实体列表
            selected_chains: 选中的关系链列表
            max_paths_per_chain: 每条关系链最多采样的路径数

        Returns:
            final_paths: 推理路径列表 [[(h1,r1,t1), (h2,r2,t2), ...], ...]
        """
        final_paths = []

        for chain_idx, chain in enumerate(selected_chains):
            chain_paths = []

            # 对每个锚点实体尝试实例化该关系链
            for anchor in anchor_entities:
                if not anchor:
                    continue

                # 使用DFS寻找匹配该关系序列的路径
                self._dfs_match_chain(
                    curr_entity=anchor,
                    target_chain=chain,
                    depth=0,
                    current_path=[],
                    results=chain_paths,
                    visited={anchor}
                )

                # 限制每条链的路径数量
                if len(chain_paths) >= max_paths_per_chain:
                    break

            final_paths.extend(chain_paths[:max_paths_per_chain])

            if self.verbose and chain_paths:
                chain_str = " -> ".join(chain)
                print(f"    关系链 '{chain_str}': 找到 {len(chain_paths[:max_paths_per_chain])} 条路径")

        return final_paths

    def _dfs_match_chain(self,
                         curr_entity: str,
                         target_chain: Tuple[str, ...],
                         depth: int,
                         current_path: List[Tuple[str, str, str]],
                         results: List[List],
                         visited: Set[str]):
        """
        DFS递归寻找匹配target_chain的具体路径
        """
        # 递归终止：完全匹配了关系链
        if depth == len(target_chain):
            if current_path:
                results.append(list(current_path))
            return

        # 当前要匹配的关系
        target_relation = target_chain[depth]

        # 遍历邻居
        neighbors = self.kg.get_neighbors(curr_entity)
        for relation, next_entity in neighbors:
            if relation == target_relation and next_entity not in visited:
                # 找到匹配的边
                current_path.append((curr_entity, relation, next_entity))
                visited.add(next_entity)

                # 递归搜索
                self._dfs_match_chain(next_entity, target_chain, depth + 1,
                                      current_path, results, visited)

                # 回溯
                current_path.pop()
                visited.remove(next_entity)

    def run_pipeline(self,
                     question: str,
                     anchor_entities: List[str],
                     h_q: int = 2,
                     top_k_chains: int = 5,
                     max_paths_per_chain: int = 10,
                     i_llm: Optional[str] = None,
                     w1: int = 80) -> Tuple[List[Tuple[str, str, str]], Dict]:
        """
        运行完整的 LightPROF 采样流程（含 PoG Fuzzy Selection 增强）

        Args:
            question:            问题文本
            anchor_entities:     锚点实体列表（q_entity_list）
            h_q:                 预测的推理跳数
            top_k_chains:        LLM精筛后保留的 Top-K 关系链数量
            max_paths_per_chain: 每条关系链最多采样的路径数
            i_llm:               I_LLM 推理指标文本（来自 pog_indicator.py 的 I_LLM.R_IA 字段）
                                 如 "Lou Seal - mascot for - team - won - World Series"
                                 为 None 时用 question 替代，模糊筛选效果会稍弱
            w1:                  Fuzzy Selection 保留数量（论文默认 80）

        Returns:
            gr_triples: 推理图的三元组列表
            stats:      统计信息（含 num_chains_after_fuzzy）
        """
        if self.verbose:
            print(f"\n{'=' * 80}")
            print(f"🚀 LightPROF 推理图采样（含 PoG Fuzzy Selection）")
            print(f"{'=' * 80}")
            print(f"问题: {question}")
            print(f"锚点实体: {anchor_entities}")
            print(f"最大跳数 (h_q): {h_q}")
            print(f"Top-K关系链: {top_k_chains}")
            i_llm_display = (i_llm[:60] + '...') if i_llm and len(i_llm) > 60 else i_llm
            print(f"I_LLM: {i_llm_display}")
            print(f"模糊筛选 W1: {w1}")

        # ── 步骤1: 关系链检索（深度受限 BFS）─────────────────────────────
        if self.verbose:
            print(f"\n📍 步骤1: 关系链检索（深度受限BFS，最大{h_q}跳）")

        all_chains = self.step1_extract_relation_chains(anchor_entities, max_hops=h_q)

        if not all_chains:
            print(f"  ⚠️  未找到任何关系链")
            return [], {
                'num_chains_total': 0,
                'num_chains_after_fuzzy': 0,
                'num_chains_selected': 0,
                'num_paths': 0,
                'num_triples': 0
            }

        # ── 步骤1.5: 模糊筛选（Fuzzy Selection，PoG 论文 Section 4.3）──
        if self.verbose:
            print(f"\n🔍 步骤1.5: 模糊筛选（SBERT 语义粗筛，W1={w1}）")

        fuzzy_chains = self.step1b_fuzzy_select(
            indicator_text=i_llm if i_llm else question,
            all_chains=all_chains,
            w1=w1
        )

        # ── 步骤2a: LLM 精筛（基于模糊筛选结果）────────────────────────
        if self.verbose:
            print(f"\n🤖 步骤2a: LLM语义相关性排序与筛选（输入 {len(fuzzy_chains)} 条）")

        selected_chains = self.step2a_rank_and_filter(
            question, set(fuzzy_chains), top_k=top_k_chains
        )

        # ── 步骤2b: 路径实例化──────────────────────────────────────────
        if self.verbose:
            print(f"\n🔗 步骤2b: 推理路径采样与推理图构建")

        instantiated_paths = self.step2b_instantiate_paths(
            anchor_entities,
            selected_chains,
            max_paths_per_chain=max_paths_per_chain
        )

        # 收集所有三元组构建推理图 G_R
        gr_triples_set = set()
        for path in instantiated_paths:
            for triple in path:
                gr_triples_set.add(triple)

        gr_triples = list(gr_triples_set)

        # 统计信息（新增 num_chains_after_fuzzy）
        stats = {
            'num_chains_total': len(all_chains),
            'num_chains_after_fuzzy': len(fuzzy_chains),
            'num_chains_selected': len(selected_chains),
            'num_paths': len(instantiated_paths),
            'num_triples': len(gr_triples)
        }

        if self.verbose:
            print(f"\n✅ 推理图构建完成:")
            print(f"  原始关系链数:   {stats['num_chains_total']}")
            print(f"  模糊筛选后:     {stats['num_chains_after_fuzzy']}")
            print(f"  LLM精筛选中:    {stats['num_chains_selected']}")
            print(f"  实例化路径数:   {stats['num_paths']}")
            print(f"  推理图三元组数: {stats['num_triples']}")
            print(f"{'=' * 80}\n")

        return gr_triples, stats


class LLMScorer:
    """
    LLM关系链打分器
    支持多种LLM后端（OpenAI、DeepSeek等）
    """

    def __init__(self, model_name=None, api_key=None, base_url=None, use_mock=False):
        self.model_name = model_name or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self.use_mock = use_mock

        api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        base_url = base_url or os.getenv("OPENAI_BASE_URL", "")

        # 如果没有提供api_key，从配置中读取
        if not api_key:
            api_key = self._load_api_key_from_config()

        if not use_mock:
            try:
                from openai import OpenAI
                print(f"  LLM后端: OpenAI ({model_name})")
                print(f"api_key {api_key},base_url {base_url}")
                self.client = OpenAI(
                    api_key=api_key, 
                    base_url=base_url,
                    timeout=120.0,
                    max_retries=3
                )
                print(f"  ✓ 客户端初始化成功 (超时: 120秒, 重试: 3次)")
            except Exception as e:
                print(f"  ⚠️  LLM客户端初始化失败: {e}")
                print(f"  将使用模拟打分")
                self.use_mock = True

    def _load_api_key_from_config(self):
        """从配置文件或代码中加载API密钥"""
        import os

        # 方式1: 直接在代码中硬编码（最简单）
        HARDCODED_API_KEY = "sk-your-actual-key-here"  # 🔑 在这里直接填写你的API密钥
        if HARDCODED_API_KEY and HARDCODED_API_KEY != "sk-your-actual-key-here":
            return HARDCODED_API_KEY

        # 方式2: 从config.json读取（更安全）
        config_file = 'config.json'
        if os.path.exists(config_file):
            import json
            with open(config_file, 'r') as f:
                config = json.load(f)
                return config.get('openai_api_key', None)

        # 方式3: 从.env文件读取
        env_file = '.env'
        if os.path.exists(env_file):
            with open(env_file, 'r') as f:
                for line in f:
                    if line.startswith('OPENAI_API_KEY='):
                        return line.split('=', 1)[1].strip()

        # 方式4: 仍然尝试从环境变量读取（作为fallback）
        return os.environ.get('OPENAI_API_KEY', None)

    def score_chains(self, question: str, chains: List[Tuple[str, ...]]) -> List[float]:
        """
        对关系链进行语义相关性打分

        Args:
            question: 问题文本
            chains: 关系链列表

        Returns:
            scores: 每条链的分数（0-1之间）
        """
        if self.use_mock:
            return self._mock_scoring(question, chains)

        # 构建prompt
        chains_text = "\n".join([
            f"{i + 1}. {' -> '.join(chain)}"
            for i, chain in enumerate(chains)
        ])

        prompt = f"""Given the question and relation chains, rate the semantic relevance of each chain to answering the question.

Question: {question}

Relation Chains:
{chains_text}

Please rate each chain's relevance on a scale of 0.0 to 1.0, where:
- 1.0 = highly relevant to answering the question
- 0.5 = moderately relevant
- 0.0 = not relevant

Output format (one score per line):
1. 0.X
2. 0.X
..."""

        max_retries = 3
        retry_delay = 3  # 秒
        
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system",
                         "content": "You are a helpful assistant that evaluates the semantic relevance of relation chains to questions."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.0,
                    max_tokens=3000,
                    timeout=120.0
                )

                # 解析LLM输出
                content = response.choices[0].message.content
                scores = self._parse_scores(content, len(chains))

                return scores

            except Exception as e:
                error_msg = str(e)
                if attempt < max_retries - 1:
                    print(f"  ⚠️  尝试 {attempt+1}/{max_retries} 失败: {error_msg[:100]}")
                    print(f"  等待 {retry_delay} 秒后重试...")
                    import time
                    time.sleep(retry_delay)
                else:
                    print(f"  ❌ 所有重试失败: {error_msg[:100]}")
                    print(f"  使用模拟打分")
                    return self._mock_scoring(question, chains)

    def _parse_scores(self, llm_output: str, expected_count: int) -> List[float]:
        """解析LLM输出的分数"""
        import re

        scores = []
        lines = llm_output.strip().split('\n')

        for line in lines:
            # 匹配形如 "1. 0.85" 或 "0.85" 的分数
            match = re.search(r'(\d+\.)?\s*([0-1]\.\d+)', line)
            if match:
                score = float(match.group(2))
                scores.append(score)

        # 如果解析的分数数量不对，用默认值填充
        while len(scores) < expected_count:
            scores.append(0.1)

        return scores[:expected_count]

    def _mock_scoring(self, question: str, chains: List[Tuple[str, ...]]) -> List[float]:
        """模拟打分（基于简单启发式规则）"""
        scores = []
        question_lower = question.lower()

        for chain in chains:
            score = 0.3  # 基础分数

            # 链长度惩罚
            score += max(0, (1.0 - len(chain) * 0.1))

            # 关键词匹配
            chain_str = " ".join(chain).lower()

            # 根据问题类型调整
            if "when" in question_lower or "year" in question_lower:
                if "time" in chain_str or "date" in chain_str or "year" in chain_str:
                    score += 0.3

            if "who" in question_lower:
                if "person" in chain_str or "author" in chain_str or "creator" in chain_str:
                    score += 0.3

            if "where" in question_lower:
                if "location" in chain_str or "place" in chain_str or "located" in chain_str:
                    score += 0.3

            # 常见相关关系加分
            for keyword in ["champion", "winner", "mascot", "team", "member", "part_of"]:
                if keyword in chain_str:
                    score += 0.2
                    break

            scores.append(min(score, 1.0))

        return scores


def process_retrieval_results(input_file: str,
                              output_file: str,
                              h_q: int = 2,
                              top_k_chains: int = 5,
                              max_paths_per_chain: int = 10,
                              llm_api_key: Optional[str] = None,
                              llm_model: str = 'gpt-4o-mini',
                              use_mock_llm: bool = True,
                              bidirectional_graph: bool = True,
                              enable_fuzzy: bool = True,
                              w1: int = 80,
                              sbert_model: str = 'all-MiniLM-L6-v2'):
    """
    处理retrieval结果，应用LightPROF采样（含 PoG Fuzzy Selection）

    Args:
        input_file:         inference_hybrid.py 或 pog_indicator.py 的输出文件路径
        output_file:        输出文件路径
        h_q:                固定的推理跳数
        top_k_chains:       LLM精筛后保留的 Top-K 关系链数量
        max_paths_per_chain:每条关系链最多采样的路径数
        llm_api_key:        LLM API密钥
        llm_model:          LLM模型名称
        use_mock_llm:       是否使用模拟LLM（测试用）
        bidirectional_graph:是否构建双向图
        enable_fuzzy:       是否启用 Fuzzy Selection 模糊筛选（默认开启）
        w1:                 Fuzzy Selection 保留数量（论文默认 80）
        sbert_model:        FuzzySelector 使用的 SentenceBERT 模型名称
    """
    print(f"\n{'=' * 80}")
    print(f"LightPROF 推理图采样处理（含 PoG Fuzzy Selection）")
    print(f"{'=' * 80}")
    print(f"输入文件: {input_file}")
    print(f"输出文件: {output_file}")
    print(f"推理跳数 (h_q): {h_q}")
    print(f"Top-K关系链: {top_k_chains}")
    print(f"每链最大路径数: {max_paths_per_chain}")
    print(f"LLM模型: {llm_model if not use_mock_llm else 'Mock (模拟)'}")
    print(f"图类型: {'双向' if bidirectional_graph else '单向'}")
    print(f"模糊筛选 (Fuzzy): {'开启 W1=' + str(w1) if enable_fuzzy else '关闭'}")
    print(f"SBERT 模型: {sbert_model if enable_fuzzy else 'N/A'}")
    print(f"{'=' * 80}\n")

    # 加载retrieval结果
    print(f"加载检索结果: {input_file}")
    retrieval_data = torch.load(input_file, map_location='cpu')
    print(f"✓ 加载完成，共 {len(retrieval_data)} 个样本\n")

    # 初始化LLM打分器
    if use_mock_llm:
        llm_scorer = None  # 使用默认的mock scorer
        print("✓ 使用模拟LLM打分器\n")
    else:
        scorer_obj = LLMScorer(model_name=llm_model, api_key=llm_api_key, use_mock=False)
        llm_scorer = scorer_obj.score_chains

    # 初始化 FuzzySelector（模糊筛选器）
    fuzzy_selector = None
    if enable_fuzzy:
        print(f"✓ 初始化 FuzzySelector (model={sbert_model}, W1={w1})")
        fuzzy_selector = FuzzySelector(model_name=sbert_model)
        print()
    else:
        print("✓ 模糊筛选已禁用\n")

    # 处理每个样本
    output_data = {}
    overall_stats = {
        'total_samples': len(retrieval_data),
        'processed_samples': 0,
        'avg_chains': 0,
        'avg_chains_after_fuzzy': 0,
        'avg_paths': 0,
        'avg_gr_triples': 0
    }

    print("开始处理样本...")
    failed_samples = []
    
    for sample_id, sample in tqdm(retrieval_data.items(), desc="处理进度"):
        try:
            # 1. 准备数据
            question = sample['question']
            anchor_entities = sample.get('q_entity_list', sample.get('q_entity', []))

            # 确保anchor_entities是列表
            if not isinstance(anchor_entities, list):
                anchor_entities = [anchor_entities]

            # 过滤空实体
            anchor_entities = [e for e in anchor_entities if e]

            if not anchor_entities:
                # 如果没有锚点实体，跳过
                output_data[sample_id] = {
                    **sample,
                    'lightprof_gr_triples': [],
                    'lightprof_stats': {'num_chains_total': 0, 'num_chains_selected': 0,
                                        'num_paths': 0, 'num_triples': 0}
                }
                continue

            # 2. 构建KG
            scored_triples = sample['scored_triples']
            if not scored_triples:
                output_data[sample_id] = {
                    **sample,
                    'lightprof_gr_triples': [],
                    'lightprof_stats': {'num_chains_total': 0, 'num_chains_selected': 0,
                                        'num_paths': 0, 'num_triples': 0}
                }
                continue

            # 转换为(h, r, t, score)格式（确保有4个元素）
            triples_for_kg = []
            for triple in scored_triples:
                if len(triple) == 4:
                    triples_for_kg.append(triple)
                elif len(triple) == 3:
                    triples_for_kg.append((*triple, 0.5))  # 默认分数

            kg = SimpleKG(triples_for_kg, bidirectional=bidirectional_graph)

            # 3. 初始化采样器（传入 fuzzy_selector）
            sampler = LightPROF_Sampler(
                kg,
                llm_scorer=llm_scorer,
                fuzzy_selector=fuzzy_selector,
                verbose=False
            )

            # 4. 运行采样流程（传入 I_LLM 和 W1）
            i_llm_text = sample.get('I_LLM', {}).get('R_IA', '') if isinstance(sample.get('I_LLM'), dict) else ''
            gr_triples, stats = sampler.run_pipeline(
                question=question,
                anchor_entities=anchor_entities,
                h_q=h_q,
                top_k_chains=top_k_chains,
                max_paths_per_chain=max_paths_per_chain,
                i_llm=i_llm_text,
                w1=w1
            )

            # 5. 保存结果
            output_data[sample_id] = {
                **sample,  # 保留原有字段
                'lightprof_gr_triples': gr_triples,  # 新增：推理图三元组
                'lightprof_stats': stats  # 新增：统计信息
            }

            # 更新统计
            overall_stats['processed_samples'] += 1
            overall_stats['avg_chains'] += stats['num_chains_total']
            overall_stats['avg_chains_after_fuzzy'] += stats['num_chains_after_fuzzy']
            overall_stats['avg_paths'] += stats['num_paths']
            overall_stats['avg_gr_triples'] += stats['num_triples']
            
        except Exception as e:
            # 捕获单个样本的处理错误，不中断整体流程
            error_msg = str(e)
            print(f"\n  ❌ 样本 {sample_id} 处理失败: {error_msg[:200]}")
            failed_samples.append((sample_id, error_msg))
            
            # 保存失败样本的空结果
            output_data[sample_id] = {
                **sample,
                'lightprof_gr_triples': [],
                'lightprof_stats': {'num_chains_total': 0, 'num_chains_selected': 0,
                                    'num_paths': 0, 'num_triples': 0},
                'error': error_msg
            }
            continue

    # 计算平均值
    if overall_stats['processed_samples'] > 0:
        overall_stats['avg_chains'] /= overall_stats['processed_samples']
        overall_stats['avg_chains_after_fuzzy'] /= overall_stats['processed_samples']
        overall_stats['avg_paths'] /= overall_stats['processed_samples']
        overall_stats['avg_gr_triples'] /= overall_stats['processed_samples']

    # 保存结果
    print(f"\n保存结果到: {output_file}")
    torch.save(output_data, output_file)

    # 打印总结
    print(f"\n{'=' * 80}")
    print(f"✅ 处理完成！")
    print(f"{'=' * 80}")
    print(f"总样本数: {overall_stats['total_samples']}")
    print(f"成功处理: {overall_stats['processed_samples']}")
    
    if failed_samples:
        print(f"失败样本数: {len(failed_samples)}")
        print(f"\n前5个失败样本:")
        for i, (sid, err) in enumerate(failed_samples[:5], 1):
            print(f"  {i}. ID={sid}: {err[:80]}...")
    
    if overall_stats['processed_samples'] > 0:
        print(f"\n平均关系链数:       {overall_stats['avg_chains']:.1f}")
        print(f"模糊筛选后平均链数: {overall_stats['avg_chains_after_fuzzy']:.1f}")
        print(f"平均路径数:         {overall_stats['avg_paths']:.1f}")
        print(f"平均推理图三元组数: {overall_stats['avg_gr_triples']:.1f}")
    print(f"{'=' * 80}\n")

    # 保存统计信息
    overall_stats['failed_samples'] = len(failed_samples)
    overall_stats['failed_sample_ids'] = [sid for sid, _ in failed_samples[:20]]  # 保存前20个
    
    stats_file = output_file.replace('.pth', '_stats.json')
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(overall_stats, f, indent=2, ensure_ascii=False)
    print(f"✓ 统计信息已保存: {stats_file}")

    return output_data, overall_stats


def display_sample_results(output_data: Dict, num_samples: int = 2):
    """展示几个样本的处理结果"""
    import random

    sample_ids = list(output_data.keys())
    selected_ids = random.sample(sample_ids, min(num_samples, len(sample_ids)))

    print(f"\n{'=' * 80}")
    print(f"📋 样本结果展示（随机抽取 {num_samples} 条）")
    print(f"{'=' * 80}")

    for idx, sample_id in enumerate(selected_ids, 1):
        sample = output_data[sample_id]
        stats = sample['lightprof_stats']

        print(f"\n{'─' * 80}")
        print(f"样本 {idx}/{num_samples} | ID: {sample_id}")
        print(f"{'─' * 80}")

        print(f"\n❓ 问题: {sample['question']}")
        print(f"\n🎯 锚点实体: {sample.get('q_entity_list', sample.get('q_entity', []))}")

        print(f"\n📊 采样统计:")
        print(f"  原始关系链数:   {stats['num_chains_total']}")
        print(f"  模糊筛选后:     {stats.get('num_chains_after_fuzzy', 'N/A')}")
        print(f"  LLM精筛选中:    {stats['num_chains_selected']}")
        print(f"  实例化路径数:   {stats['num_paths']}")
        print(f"  推理图三元组数: {stats['num_triples']}")

        # 展示推理图三元组（前10个）
        gr_triples = sample['lightprof_gr_triples']
        if gr_triples:
            print(f"\n🔍 推理图三元组 (前10/{len(gr_triples)}):")
            for i, (h, r, t) in enumerate(gr_triples[:10], 1):
                print(f"  {i:2d}. ({h}, {r}, {t})")

            if len(gr_triples) > 10:
                print(f"  ... 还有 {len(gr_triples) - 10} 个三元组")
        else:
            print(f"\n⚠️  未生成推理图三元组")

        # 对比原始检索结果
        original_triples = sample['scored_triples']
        compression_ratio = len(gr_triples) / len(original_triples) if original_triples else 0
        print(f"\n📉 压缩率: {len(gr_triples)}/{len(original_triples)} = {compression_ratio:.2%}")

    print(f"\n{'=' * 80}\n")


def main():
    parser = ArgumentParser(description='LightPROF - 关系链重构与推理子图采样')

    # 输入输出
    parser.add_argument('-i', '--input', type=str,default='./test_retrieval_result.pth',
                        help='输入文件路径（inference_hybrid.py的输出，如: test_retrieval_result.pth）')
    parser.add_argument('-o', '--output', type=str, default=None,
                        help='输出文件路径（默认: 在输入文件名后添加_lightprof）')

    # 采样参数
    parser.add_argument('--h_q', type=int, default=2,
                        help='推理跳数（默认: 2）')
    parser.add_argument('--top_k_chains', type=int, default=5,
                        help='选择的Top-K关系链数量（默认: 5）')
    parser.add_argument('--max_paths_per_chain', type=int, default=10,
                        help='每条关系链最多采样的路径数（默认: 10）')
    parser.add_argument('--bidirectional', action='store_true',
                        help='使用双向图（默认: 单向）')

    # 模糊筛选参数（PoG Fuzzy Selection）
    parser.add_argument('--no_fuzzy', action='store_true',
                        help='禁用 PoG Fuzzy Selection 模糊筛选（默认: 开启）')
    parser.add_argument('--w1', type=int, default=80,
                        help='Fuzzy Selection 保留的最大链数量，对应论文 W1（默认: 80）')
    parser.add_argument('--sbert_model', type=str, default='all-MiniLM-L6-v2',
                        help='FuzzySelector 使用的 SentenceBERT 模型（默认: all-MiniLM-L6-v2）')

    # LLM配置
    parser.add_argument('--use_mock_llm', action='store_true',
                        help='使用模拟LLM打分（测试用）')
    parser.add_argument('--llm_model', type=str, default='gpt-4.1-mini',
                        help='LLM模型名称（默认: gpt-4.1-mini）')
    parser.add_argument('--llm_api_key', type=str, default=os.getenv("OPENAI_API_KEY", ""),
                        help='LLM API密钥（从环境变量OPENAI_API_KEY读取）')
    parser.add_argument('--llm_base_url', type=str, default=os.getenv("OPENAI_BASE_URL", ""),
                        help='LLM API基础URL（可选）')

    # 展示选项
    parser.add_argument('--num_display_samples', type=int, default=2,
                        help='展示的样本数量（默认: 2）')

    args = parser.parse_args()

    # 处理输出文件名
    if args.output is None:
        base_name = args.input.replace('.pth', '')
        args.output = f"{base_name}_lightprof.pth"

    # 获取API密钥
    def load_api_key():
        """从多个来源加载API密钥"""
        import os

        HARDCODED_KEY = "sk-your-key-here"  # 🔑 填写你的密钥
        if HARDCODED_KEY and HARDCODED_KEY != "sk-your-key-here":
            return HARDCODED_KEY

        if os.path.exists('config.json'):
            import json
            with open('config.json', 'r') as f:
                return json.load(f).get('openai_api_key')

        return os.environ.get('OPENAI_API_KEY')

    # ⭐ 再调用
    if args.llm_api_key is None and not args.use_mock_llm:
        args.llm_api_key = load_api_key()  # ✅ 现在可以了

        if args.llm_api_key is None:
            print("⚠️  未提供LLM API密钥，将使用模拟打分")
            args.use_mock_llm = True

    # 初始化LLM打分器
    if args.use_mock_llm:
        llm_scorer = None
    else:
        scorer = LLMScorer(
            model_name=args.llm_model,
            api_key=args.llm_api_key,
            base_url=args.llm_base_url,
            use_mock=args.use_mock_llm
        )
        llm_scorer = scorer.score_chains

    # 处理数据
    output_data, stats = process_retrieval_results(
        input_file=args.input,
        output_file=args.output,
        h_q=args.h_q,
        top_k_chains=args.top_k_chains,
        max_paths_per_chain=args.max_paths_per_chain,
        llm_api_key=args.llm_api_key,
        llm_model=args.llm_model,
        use_mock_llm=args.use_mock_llm,
        bidirectional_graph=args.bidirectional,
        enable_fuzzy=not args.no_fuzzy,
        w1=args.w1,
        sbert_model=args.sbert_model
    )

    # 展示样本
    if args.num_display_samples > 0:
        display_sample_results(output_data, num_samples=args.num_display_samples)

    print(f"✅ 全部完成！")


if __name__ == '__main__':
    main()
