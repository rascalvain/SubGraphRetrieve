#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PoG (Paths-over-Graph) LLM-Indicator 构造器
基于 inference_hybrid.py 的输出，为每个样本生成：
  - Split_q: 拆分子问题列表
  - I_LLM: {List_T, R_IA, D_predict}

参考论文: Paths-over-Graph: Knowledge Graph Empowered Large Language Model Reasoning (WWW'25)
"""

import json
import re
import os
import time
import torch
from openai import OpenAI
from typing import Dict, List, Any, Tuple, Optional
from tqdm import tqdm
from argparse import ArgumentParser


# ==========================================
# Prompt 模板（对齐论文 Appendix E）
# ==========================================
PROMPT_TEMPLATE = """You will receive a multi-hop question, which is composed of several interconnected queries, along with a list of topic entities that serve as the main keywords for the question. Your task is to:
1. Break the question into simpler split questions, using each topic entity once (each split question explores the connection between one topic entity and the answer/other entities);
2. Generate a Chain of Thought (CoT) thinking indicator (I_LLM), which must include all topic entities and sort them according to the reasoning logic of the question, and clearly describe the semantic relationship between the topic entities and the potential answer;
3. Predict the maximum hop distance (D_predict) between the potential answer and each topic entity. D_predict is a single integer representing the maximum distance (1 <= D_predict <= 3 is recommended).

--- Example 1 ---
Q: What American state is the birthplace of the actor who played Iron Man?
Topic Entity: Iron Man, American state
A:
Split Question:
1. Who is the actor who played Iron Man?
2. What American state is the birthplace of this actor?
Thinking Indicator (I_LLM): Iron Man - played by - actor - birthplace in - American state
D_predict: 2

--- Example 2 ---
Q: What European Union country sharing borders with Germany contains the Lejre Municipality?
Topic Entity: Germany, Lejre Municipality
A:
Split Question:
1. What European Union country shares borders with Germany?
2. What country contains the Lejre Municipality?
Thinking Indicator (I_LLM): Lejre Municipality - contained by - European Union country (answer) - shares borders with - Germany
D_predict: 2

--- Example 3 ---
Q: Lou Seal is the mascot for the team that last won the World Series when?
Topic Entity: Lou Seal, World Series
A:
Split Question:
1. What team is Lou Seal the mascot for?
2. When did that team last win the World Series?
Thinking Indicator (I_LLM): Lou Seal - mascot for - team - last won - World Series - when
D_predict: 2

--- Real Task ---
Q: {question}
Topic Entity: {entities}
A:"""


# ==========================================
# 核心处理类
# ==========================================
class PoGIndicatorGenerator:
    """
    PoG LLM-Indicator 生成器
    对应论文 Section 4.1 Question Analysis 阶段
    """

    def __init__(
        self,
        model: str = None,
        api_key: str = None,
        base_url: str = None,
        max_retries: int = 3,
        verbose: bool = False
    ):
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self.max_retries = max_retries
        self.verbose = verbose

        api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        base_url = base_url or os.getenv("OPENAI_BASE_URL", "")

        # 客户端在类内初始化，不依赖外部全局变量
        try:
            self.client = OpenAI(
                api_key=api_key,
                base_url=base_url if base_url else None,
                timeout=90.0,
            )
            if self.verbose:
                print(f"  ✓ LLM客户端初始化成功 (model={model}, base_url={base_url})")
        except Exception as e:
            print(f"  ❌ LLM客户端初始化失败: {e}")
            raise

    def call_llm(self, prompt: str) -> str:
        """
        调用 LLM，参数对齐论文：temperature=0.4, max_tokens=256
        带指数退避重试机制
        """
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You are a logical reasoning assistant for Knowledge Graph "
                                "Question Answering. Follow the output format exactly."
                            )
                        },
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.4,   # 论文指定：提升生成多样性
                    max_tokens=256,    # 论文指定：限制长度
                )
                return response.choices[0].message.content.strip()

            except Exception as e:
                wait_time = 2 ** attempt  # 指数退避: 1s, 2s, 4s
                if attempt < self.max_retries - 1:
                    if self.verbose:
                        print(f"  ⚠️  LLM调用失败 (尝试{attempt+1}/{self.max_retries}): "
                              f"{str(e)[:80]} → 等待{wait_time}s后重试")
                    time.sleep(wait_time)
                else:
                    print(f"  ❌ LLM调用全部失败: {str(e)[:100]}")

        return ""

    def parse_output(
        self,
        llm_output: str,
        original_entities: List[str]
    ) -> Tuple[List[str], Dict[str, Any]]:
        """
        解析 LLM 输出，提取 Split_q 和 I_LLM
        采用鲁棒的正则匹配，处理 LLM 输出格式变体
        """
        split_q = []
        i_llm = {
            "List_T": [],
            "R_IA": "",
            "D_predict": -1
        }

        if not llm_output:
            return split_q, i_llm

        # ── 1. 提取 Split Questions ──────────────────────────────────────
        # 匹配 "Split Question:" 到 "Thinking Indicator" 之间的内容
        sq_match = re.search(
            r'Split\s+Question[s]?\s*:(.*?)(?:Thinking\s+Indicator|I_LLM\s*[:\(])',
            llm_output,
            re.IGNORECASE | re.DOTALL
        )
        if sq_match:
            raw_sq = sq_match.group(1).strip()
            lines = raw_sq.split('\n')
            for line in lines:
                # 去除行首数字编号（如 "1." "1)" "1:"）
                cleaned = re.sub(r'^\s*\d+[\.\)\:]\s*', '', line).strip()
                if cleaned and len(cleaned) > 5:  # 过滤太短的行
                    split_q.append(cleaned)
        else:
            # 备用：直接用数字列表模式提取
            numbered = re.findall(r'^\s*\d+[\.\)]\s*(.+)$', llm_output, re.MULTILINE)
            split_q = [q.strip() for q in numbered if len(q.strip()) > 5]

        # ── 2. 提取 Thinking Indicator (R_IA) ────────────────────────────
        # 支持多种格式：
        # "Thinking Indicator (I_LLM): ..."
        # "Thinking Indicator: ..."
        # "I_LLM: ..."
        ti_patterns = [
            r'Thinking\s+Indicator\s*\(I_LLM\)\s*:\s*(.+?)(?:\n|$)',
            r'Thinking\s+Indicator\s*:\s*(.+?)(?:\n|$)',
            r'I_LLM\s*:\s*(.+?)(?:\n|$)',
        ]
        for pattern in ti_patterns:
            ti_match = re.search(pattern, llm_output, re.IGNORECASE)
            if ti_match:
                i_llm["R_IA"] = ti_match.group(1).strip()
                # 去除末尾的 D_predict 内容（有时 LLM 会连在一起输出）
                i_llm["R_IA"] = re.sub(r'\s*D_predict.*$', '', i_llm["R_IA"]).strip()
                break

        # ── 3. 提取 D_predict ────────────────────────────────────────────
        d_match = re.search(r'D_predict\s*:\s*(\d+)', llm_output, re.IGNORECASE)
        if d_match:
            val = int(d_match.group(1))
            i_llm["D_predict"] = max(1, min(val, 3))  # 夹紧到 [1, 3]
        else:
            i_llm["D_predict"] = 2  # 论文中默认值为 2

        # ── 4. 构建有序主题实体链 (List_T) ─────────────────────────────
        # 根据实体在 R_IA 中出现的位置排序
        # 注意：跳过 Freebase ID 格式（m.xxxxx），它们不会出现在自然语言输出中
        r_ia_text = i_llm["R_IA"]
        entity_positions = []

        for ent in original_entities:
            # 跳过 Freebase ID
            if re.match(r'^m\.\w+$', ent):
                continue
            # 使用正则模糊匹配（不区分大小写）
            try:
                pattern = re.escape(ent)
                match = re.search(pattern, r_ia_text, re.IGNORECASE)
                if match:
                    entity_positions.append((match.start(), ent))
            except re.error:
                pass

        # 按出现位置升序排序
        entity_positions.sort(key=lambda x: x[0])
        i_llm["List_T"] = [ent for _, ent in entity_positions]

        # 如果 List_T 为空（全是 Freebase ID 或匹配失败），
        # 保留原始实体列表（过滤掉 ID）
        if not i_llm["List_T"]:
            i_llm["List_T"] = [
                e for e in original_entities
                if not re.match(r'^m\.\w+$', e)
            ] or original_entities  # 如果过滤后为空则保留全部

        return split_q, i_llm

    def validate_output(
        self,
        split_q: List[str],
        i_llm: Dict[str, Any],
        original_entities: List[str]
    ) -> bool:
        """
        验证 LLM 输出的有效性（4个硬指标）
        """
        # 1. 深度合理性
        if not (1 <= i_llm["D_predict"] <= 3):
            return False

        # 2. 必须有拆分问题和关系描述
        if not split_q or not i_llm["R_IA"]:
            return False

        # 3. 拆分问题长度合理（至少有一个有意义的问题）
        valid_questions = [q for q in split_q if len(q) > 10 and '?' in q]
        if not valid_questions:
            return False

        # 4. 实体完整性（仅对非 Freebase ID 实体进行检查）
        natural_entities = [e for e in original_entities if not re.match(r'^m\.\w+$', e)]
        if natural_entities:
            # 要求至少 50% 的自然语言实体出现在 R_IA 中
            matched = sum(
                1 for e in natural_entities
                if re.search(re.escape(e), i_llm["R_IA"], re.IGNORECASE)
            )
            if matched < len(natural_entities) * 0.5:
                return False

        return True

    def generate(self, question: str, entities: List[str]) -> Dict[str, Any]:
        """
        主控函数：带重试机制的生成流程
        对应论文 Section 4.1 Question Analysis
        """
        if not entities:
            return self._make_fallback(question, entities, reason="no_entities")

        prompt = PROMPT_TEMPLATE.format(
            question=question,
            entities=", ".join(entities)
        )

        for attempt in range(self.max_retries):
            llm_output = self.call_llm(prompt)
            if not llm_output:
                continue

            split_q, i_llm = self.parse_output(llm_output, entities)

            if self.validate_output(split_q, i_llm, entities):
                return {
                    "Split_q": split_q,
                    "I_LLM": i_llm,
                    "Raw_LLM_Output": llm_output,
                    "is_fallback": False
                }

            if self.verbose:
                print(f"  ⚠️  验证失败 (尝试 {attempt+1}/{self.max_retries}): "
                      f"split_q={len(split_q)}, R_IA='{i_llm['R_IA'][:30]}', "
                      f"D={i_llm['D_predict']}")

        # 多次重试均失败，返回降级结果
        return self._make_fallback(question, entities, reason="validation_failed")

    def _make_fallback(
        self,
        question: str,
        entities: List[str],
        reason: str = "unknown"
    ) -> Dict[str, Any]:
        """生成降级兜底结果"""
        natural_entities = [e for e in entities if not re.match(r'^m\.\w+$', e)]
        return {
            "Split_q": [question],
            "I_LLM": {
                "List_T": natural_entities or entities,
                "R_IA": " - ".join(natural_entities or entities),
                "D_predict": 2
            },
            "Raw_LLM_Output": f"FALLBACK:{reason}",
            "is_fallback": True
        }


# ==========================================
# 数据处理函数
# ==========================================
def process_retrieval_file(
    input_file: str,
    output_file: str,
    generator: PoGIndicatorGenerator,
    entity_field: str = "q_entity_list",
    num_display: int = 2
) -> Dict:
    """
    处理 inference_hybrid.py 输出的 .pth 文件
    为每个样本生成 PoG Indicator 并注入新字段

    Args:
        input_file:   inference_hybrid.py 输出的 .pth 文件路径
        output_file:  输出 .pth 文件路径
        generator:    PoGIndicatorGenerator 实例
        entity_field: 使用哪个字段作为主题实体
                      可选: 'q_entity_list' | 'q_entity' | 'q_entity_in_graph'
        num_display:  处理完成后展示的样本数量
    """
    print(f"\n{'='*70}")
    print(f"PoG LLM-Indicator 构造")
    print(f"{'='*70}")
    print(f"输入文件: {input_file}")
    print(f"输出文件: {output_file}")
    print(f"实体来源字段: {entity_field}")
    print(f"{'='*70}\n")

    # 加载数据
    print(f"加载检索结果...")
    data = torch.load(input_file, map_location='cpu')
    print(f"✓ 共 {len(data)} 个样本\n")

    # 统计信息
    stats = {
        'total': len(data),
        'success': 0,
        'fallback': 0,
        'skipped': 0
    }

    print("开始生成 PoG Indicator...")
    for qid, sample in tqdm(data.items(), desc="处理进度"):
        try:
            question = sample['question']

            # 获取主题实体（优先级: 指定字段 → q_entity_in_graph → q_entity）
            entities = (
                sample.get(entity_field, [])
                or sample.get('q_entity_in_graph', [])
                or sample.get('q_entity', [])
            )

            # 确保是列表
            if not isinstance(entities, list):
                entities = [entities] if entities else []

            # 过滤空值
            entities = [e for e in entities if e]

            if not entities:
                stats['skipped'] += 1
                sample['Split_q'] = [question]
                sample['I_LLM'] = {'List_T': [], 'R_IA': '', 'D_predict': 2}
                sample['pog_is_fallback'] = True
                continue

            # 生成 Indicator
            result = generator.generate(question, entities)

            # 注入新字段（保留原有所有字段）
            sample['Split_q'] = result['Split_q']
            sample['I_LLM'] = result['I_LLM']
            sample['pog_is_fallback'] = result['is_fallback']

            if result['is_fallback']:
                stats['fallback'] += 1
            else:
                stats['success'] += 1

        except Exception as e:
            print(f"\n  ❌ 样本 {qid} 处理异常: {str(e)[:100]}")
            stats['fallback'] += 1
            sample['Split_q'] = [sample.get('question', '')]
            sample['I_LLM'] = {'List_T': [], 'R_IA': '', 'D_predict': 2}
            sample['pog_is_fallback'] = True

    # 保存结果
    print(f"\n保存结果到: {output_file}")
    torch.save(data, output_file)

    # 同时保存为 JSON 方便查看（仅保存新增字段，轻量版）
    json_output = output_file.replace('.pth', '_indicators.json')
    json_data = {}
    for qid, sample in data.items():
        json_data[qid] = {
            'question': sample['question'],
            'entities_used': (
                sample.get(entity_field, [])
                or sample.get('q_entity_in_graph', [])
            ),
            'Split_q': sample.get('Split_q', []),
            'I_LLM': sample.get('I_LLM', {}),
            'is_fallback': sample.get('pog_is_fallback', False)
        }
    with open(json_output, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)

    # 打印统计
    print(f"\n{'='*70}")
    print(f"✅ 处理完成！")
    print(f"{'='*70}")
    print(f"总样本数:     {stats['total']}")
    print(f"成功生成:     {stats['success']}  ({stats['success']/stats['total']*100:.1f}%)")
    print(f"降级结果:     {stats['fallback']}  ({stats['fallback']/stats['total']*100:.1f}%)")
    print(f"跳过(无实体): {stats['skipped']}  ({stats['skipped']/stats['total']*100:.1f}%)")
    print(f"\n结果文件:    {output_file}")
    print(f"指标JSON文件: {json_output}")
    print(f"{'='*70}\n")

    # 展示样本
    if num_display > 0:
        display_samples(data, entity_field=entity_field, num_samples=num_display)

    return data


def display_samples(data: Dict, entity_field: str = 'q_entity_list', num_samples: int = 2):
    """展示处理结果样本"""
    import random

    sample_ids = list(data.keys())
    selected = random.sample(sample_ids, min(num_samples, len(sample_ids)))

    print(f"\n{'='*70}")
    print(f"📋 样本展示（随机 {num_samples} 条）")
    print(f"{'='*70}")

    for idx, qid in enumerate(selected, 1):
        sample = data[qid]
        i_llm = sample.get('I_LLM', {})

        print(f"\n{'─'*70}")
        print(f"样本 {idx} | ID: {qid}")
        print(f"{'─'*70}")
        print(f"❓ 问题:   {sample['question']}")
        print(f"🎯 实体:   {sample.get(entity_field, sample.get('q_entity', []))}")
        print(f"\n📝 Split Questions:")
        for i, q in enumerate(sample.get('Split_q', []), 1):
            print(f"  {i}. {q}")
        print(f"\n🔗 I_LLM:")
        print(f"  R_IA:      {i_llm.get('R_IA', '')}")
        print(f"  List_T:    {i_llm.get('List_T', [])}")
        print(f"  D_predict: {i_llm.get('D_predict', -1)}")
        print(f"  降级:      {sample.get('pog_is_fallback', False)}")

    print(f"\n{'='*70}\n")


# ==========================================
# 命令行入口
# ==========================================
def main():
    parser = ArgumentParser(description='PoG LLM-Indicator 生成器')

    # 输入输出
    parser.add_argument(
        '-i', '--input', type=str,
        default='./test_retrieval_result.pth',
        help='输入 .pth 文件（inference_hybrid.py 的输出）'
    )
    parser.add_argument(
        '-o', '--output', type=str, default=None,
        help='输出 .pth 文件路径（默认: 在输入文件名后添加 _pog）'
    )

    # LLM 配置
    parser.add_argument(
        '--model', type=str, default='gpt-4o-mini',
        help='LLM 模型名称（默认: gpt-4o-mini）'
    )
    parser.add_argument(
        '--api_key', type=str,
        default=os.getenv("OPENAI_API_KEY", ""),
        help='OpenAI API 密钥'
    )
    parser.add_argument(
        '--base_url', type=str,
        default=os.getenv("OPENAI_BASE_URL", ""),
        help='API Base URL（使用中转时填写，默认官方接口可留空）'
    )

    # 实体字段
    parser.add_argument(
        '--entity_field', type=str,
        default='q_entity_list',
        choices=['q_entity_list', 'q_entity', 'q_entity_in_graph'],
        help=(
            '使用哪个字段作为主题实体（默认: q_entity_list）\n'
            '  q_entity_list:    LLM提取+sentence-transformer映射的扩展实体（推荐）\n'
            '  q_entity_in_graph: 在图中确认存在的自然语言实体\n'
            '  q_entity:         原始图匹配实体'
        )
    )

    # 其他
    parser.add_argument(
        '--max_retries', type=int, default=3,
        help='LLM 调用最大重试次数（默认: 3）'
    )
    parser.add_argument(
        '--num_display', type=int, default=2,
        help='处理完成后展示的样本数量（默认: 2，0表示不展示）'
    )
    parser.add_argument(
        '--verbose', action='store_true',
        help='打印详细日志'
    )

    args = parser.parse_args()

    # 处理输出文件名
    if args.output is None:
        args.output = args.input.replace('.pth', '_pog.pth')

    # 初始化生成器
    generator = PoGIndicatorGenerator(
        model=args.model,
        api_key=args.api_key,
        base_url=args.base_url if args.base_url else None,
        max_retries=args.max_retries,
        verbose=args.verbose
    )

    # 处理数据
    process_retrieval_file(
        input_file=args.input,
        output_file=args.output,
        generator=generator,
        entity_field=args.entity_field,
        num_display=args.num_display
    )

    print("✅ 全部完成！")


# ==========================================
# 用于快速测试的 mock 入口（不依赖真实 .pth 文件）
# ==========================================
def run_mock_test():
    """使用模拟数据快速验证代码逻辑"""
    print("=== Mock 测试模式 ===\n")

    mock_data = {
        'WebQTest-123': {
            'question': "What year did the team with mascot named Lou Seal win the World Series?",
            'q_entity_list': ['Lou Seal', 'World Series'],
            'q_entity': ['Lou Seal'],
            'q_entity_in_graph': ['Lou Seal', 'World Series'],
            'scored_triples': [],
            'a_entity': ['2014']
        },
        'WebQTest-124': {
            'question': "Which country bordering France contains an airport that serves Nijmegen?",
            'q_entity_list': ['France', 'Nijmegen'],
            'q_entity': ['France', 'Nijmegen'],
            'q_entity_in_graph': ['France', 'Nijmegen'],
            'scored_triples': [],
            'a_entity': ['Germany']
        },
        'WebQTest-125': {
            'question': "What Central Time Zone continent do the Falkland Islands belong to?",
            'q_entity_list': ['Falkland Islands', 'Central Time Zone'],
            'q_entity': ['Falkland Islands', 'Central Time Zone'],
            'q_entity_in_graph': ['Falkland Islands', 'Central Time Zone'],
            'scored_triples': [],
            'a_entity': ['Americas']
        }
    }

    generator = PoGIndicatorGenerator(
        model=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
        verbose=True
    )

    print("开始构造 PoG 推理指示器...\n" + "=" * 40)

    for qid, data in mock_data.items():
        print(f"\nProcessing {qid}...")
        entities = data['q_entity_list']
        result = generator.generate(data['question'], entities)

        data['Split_q'] = result['Split_q']
        data['I_LLM'] = result['I_LLM']

        print(f"  [{'FALLBACK' if result['is_fallback'] else 'SUCCESS'}]")
        print(f"  R_IA:      {data['I_LLM']['R_IA']}")
        print(f"  List_T:    {data['I_LLM']['List_T']}")
        print(f"  D_predict: {data['I_LLM']['D_predict']}")
        print(f"  Split_q:")
        for i, q in enumerate(data['Split_q'], 1):
            print(f"    {i}. {q}")

    print("\n" + "=" * 40 + "\n最终数据结构（新增字段）:")
    for qid, data in mock_data.items():
        print(f"\n{qid}:")
        print(json.dumps({
            'Split_q': data['Split_q'],
            'I_LLM': data['I_LLM']
        }, indent=4, ensure_ascii=False))


if __name__ == '__main__':
    import sys
    if '--mock' in sys.argv:
        run_mock_test()
    else:
        main()