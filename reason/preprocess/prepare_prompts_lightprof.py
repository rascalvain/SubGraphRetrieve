#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LightPROF + PoG 增强版 Prompt 构建器
支持使用 lightprof_gr_triples 作为输入证据，
并可选择性地将 PoG 的 I_LLM 和 Split_q 注入 prompt

新增的 prompt_mode：
  lightprof_100          : 使用 lightprof_gr_triples，最多 100 条三元组，无 PoG 增强
  lightprof_pog_100      : 使用 lightprof_gr_triples + PoG Split_q/I_LLM 注入
  lightprof_fallback_100 : 使用 lightprof_gr_triples，若为空则回退到 scored_triplets
"""

from .prepare_prompts import triplet_to_str, unique_preserve_order


def sort_triplets_by_relevance(
    triplets: list,
    anchor_entities: list,
    scored: bool = False
) -> list:
    """
    对三元组按相关性排序，使模型在 prompt 中优先看到最重要的证据

    排序策略（优先级从高到低）：
      1. 头实体是锚点实体（1-hop 直连，最直接的证据）
      2. 尾实体是锚点实体（反向 1-hop）
      3. 其他（2-hop 及更远）

    对于 fallback_scored 模式（scored=True），在每个层内进一步按分数降序排列。

    Args:
        triplets:         三元组列表，lightprof 模式为 (h, r, t)，
                          scored 模式为 (h, r, t, score)
        anchor_entities:  锚点实体列表（q_entity_list），可为空
        scored:           是否包含分数字段（第4个元素）

    Returns:
        排序后的三元组列表（格式与输入相同）
    """
    if not triplets:
        return triplets

    anchor_set = set(e.lower() for e in anchor_entities if e)

    def sort_key(triplet):
        h = str(triplet[0]).lower()
        t = str(triplet[2]).lower()
        score = float(triplet[3]) if scored and len(triplet) >= 4 else 0.0

        if h in anchor_set:
            tier = 0        # 头实体是锚点 → 最高优先级
        elif t in anchor_set:
            tier = 1        # 尾实体是锚点
        else:
            tier = 2        # 其他

        return (tier, -score)   # tier 越小越前，分数越高越前

    return sorted(triplets, key=sort_key)


# ==========================================
# PoG 增强 Prompt 模板
# ==========================================

POG_ENHANCED_SYS_PROMPT = (
    "Based on the triplets retrieved from a knowledge graph, please answer the question. "
    "A reasoning chain and sub-questions are provided to guide your thinking. "
    'Please return formatted answers as a list, each prefixed with "ans:".'
)

POG_ENHANCED_SYS_PROMPT_STRICT = (
    "Based on the triplets retrieved from a knowledge graph, please answer the question. "
    "Follow the reasoning chain and sub-questions to derive the answer step by step. "
    "Only use information from the provided triplets. "
    'Please return formatted answers as a list, each prefixed with "ans:".'
)


def build_pog_guidance_block(each_qa: dict) -> str:
    """
    构建 PoG 推理指导文本块
    包含 I_LLM 思维链和 Split_q 子问题

    Returns:
        str: 格式化的指导文本，如果无 PoG 信息则返回空字符串
    """
    i_llm = each_qa.get('I_LLM', {})
    split_q = each_qa.get('Split_q', [])

    if not i_llm or not split_q:
        return ''

    r_ia = i_llm.get('R_IA', '').strip() if isinstance(i_llm, dict) else ''
    d_predict = i_llm.get('D_predict', -1) if isinstance(i_llm, dict) else -1

    lines = ['Reasoning Guidance:']

    if r_ia:
        lines.append(f'Thinking Chain: {r_ia}')

    if isinstance(d_predict, int) and d_predict > 0:
        lines.append(f'Predicted Reasoning Depth: {d_predict}')

    # 过滤有效的子问题（长度合理）
    valid_split_q = [q for q in split_q if isinstance(q, str) and len(q.strip()) > 5]
    if valid_split_q:
        lines.append('Sub-questions to answer in order:')
        for i, q in enumerate(valid_split_q, 1):
            lines.append(f'  {i}. {q}')

    if len(lines) <= 1:  # 只有标题行，没有内容
        return ''

    return '\n'.join(lines)


def get_prompts_lightprof(
    each_qa: dict,
    mode: str,
    sys_prompt: str,
    cot_prompt: str,
    thres: float,
    use_pog_prompt: bool = False,
    fallback_to_scored: bool = True,
    sort_by_relevance: bool = True,
    seed: int = 0
) -> dict:
    """
    LightPROF 增强版 prompt 构建函数

    Args:
        each_qa:             单个 QA 样本（来自 lightprof_sampling.py 输出）
        mode:                prompt 模式字符串，如 'lightprof_100'、'lightprof_pog_100'
        sys_prompt:          系统提示词
        cot_prompt:          CoT 提示词
        thres:               三元组分数阈值（对 lightprof 模式无效，保持接口兼容）
        use_pog_prompt:      是否将 I_LLM/Split_q 注入 prompt（方案2增强）
        fallback_to_scored:  lightprof_gr_triples 为空时是否回退到 scored_triplets
        sort_by_relevance:   是否对三元组按锚点相关性排序（靶向提升 Hit@1，默认开启）
        seed:                随机种子（保持接口兼容）

    Returns:
        each_qa: 注入了 sys_query/user_query/all_query/cot_query 的样本
    """
    # ── 解析 mode 中的三元组数量 ────────────────────────────────────
    # 支持格式: 'lightprof_100', 'lightprof_pog_100', 'lightprof_fallback_100'
    num_sampled_triplets = 100  # 默认值
    for part in mode.split('_'):
        if part.isdigit():
            num_sampled_triplets = int(part)
            break

    # ── 判断是否实际使用 PoG 增强（mode 中含 'pog' 或显式开启）──────
    use_pog = use_pog_prompt or ('pog' in mode)

    # ── 构建问题文本 ────────────────────────────────────────────────
    question_prompt = "Question:\n" + each_qa['question']
    if question_prompt[-1] != '?':
        question_prompt += '?'

    # ── 获取三元组证据 ───────────────────────────────────────────────
    gr_triples = each_qa.get('lightprof_gr_triples', [])
    anchor_entities = each_qa.get('q_entity_list', each_qa.get('q_entity', []))
    if not isinstance(anchor_entities, list):
        anchor_entities = [anchor_entities]

    if gr_triples:
        # 使用 LightPROF 精炼的推理子图（已是 (h, r, t) 格式，无 score）
        input_triplets = [(h, r, t) for h, r, t in gr_triples]
        # 按锚点相关性排序：1-hop 直连优先，确保模型先看到最关键的证据
        if sort_by_relevance:
            input_triplets = sort_triplets_by_relevance(
                input_triplets, anchor_entities, scored=False
            )

        # 论文 Fig.4：Llama 8B 在三元组不足时可用高分原始三元组补全至 K 条
        # 补全策略：LightPROF 路径优先，剩余槽位用 scored_triplets 中评分最高的填充
        if fallback_to_scored or ('fallback' in mode):
            raw_scored = each_qa.get('scored_triplets', [])
            if raw_scored and len(input_triplets) < num_sampled_triplets:
                existing = set((h, r, t) for h, r, t in input_triplets)
                if sort_by_relevance:
                    raw_scored = sort_triplets_by_relevance(
                        raw_scored, anchor_entities, scored=True
                    )
                for triple in raw_scored:
                    h, r, t = triple[0], triple[1], triple[2]
                    if (h, r, t) not in existing:
                        input_triplets.append((h, r, t))
                        existing.add((h, r, t))
                    if len(input_triplets) >= num_sampled_triplets:
                        break
                evidence_source = 'lightprof+scored'
            else:
                evidence_source = 'lightprof'
        else:
            evidence_source = 'lightprof'

    elif fallback_to_scored or ('fallback' in mode):
        # lightprof_gr_triples 完全为空时才完全回退到原始 scored_triplets
        raw_scored = each_qa.get('scored_triplets', [])
        if sort_by_relevance:
            raw_scored = sort_triplets_by_relevance(
                raw_scored, anchor_entities, scored=True
            )
        input_triplets = [(t[0], t[1], t[2]) for t in raw_scored]
        evidence_source = 'fallback_scored'
    else:
        input_triplets = []
        evidence_source = 'empty'

    # 去重、截断
    input_triplets = unique_preserve_order(input_triplets)
    input_triplets = input_triplets[:num_sampled_triplets]
    input_triplets_str = [triplet_to_str(t) for t in input_triplets]

    triplet_prompt = "Triplets:\n" + "\n".join(input_triplets_str)

    # ── 构建 PoG 推理指导块（可选）───────────────────────────────────
    pog_guidance_block = ''
    if use_pog:
        pog_guidance_block = build_pog_guidance_block(each_qa)

    # ── 拼接最终 prompt ───────────────────────────────────────────────
    # 带 PoG 指导的 prompt 结构：
    #   [sys_prompt]
    #   [pog_guidance_block]   ← 可选
    #   [triplet_prompt]
    #   [question_prompt]
    prompt_parts = [sys_prompt]
    if pog_guidance_block:
        prompt_parts.append(pog_guidance_block)
    prompt_parts.append(triplet_prompt)
    prompt_parts.append(question_prompt)

    all_query = "\n\n".join(prompt_parts)

    # user_query 不含 sys_prompt
    user_parts = prompt_parts[1:]
    user_query = "\n\n".join(user_parts)

    # ── 写回 each_qa ──────────────────────────────────────────────────
    each_qa['sys_query'] = sys_prompt
    each_qa['user_query'] = user_query
    each_qa['all_query'] = all_query
    each_qa['cot_query'] = cot_prompt
    each_qa['_evidence_source'] = evidence_source
    each_qa['_num_triplets_used'] = len(input_triplets)

    return each_qa


def get_prompts_for_data_lightprof(
    data: list,
    mode: str,
    sys_prompt: str,
    cot_prompt: str,
    thres: float,
    use_pog_prompt: bool = False,
    fallback_to_scored: bool = True,
    sort_by_relevance: bool = True
) -> list:
    """
    批量处理所有样本的 prompt 构建

    Args:
        data:               样本列表
        mode:               prompt 模式
        sys_prompt:         系统提示词
        cot_prompt:         CoT 提示词
        thres:              分数阈值
        use_pog_prompt:     是否注入 PoG 推理指导
        fallback_to_scored: 空时是否回退到 scored_triplets
        sort_by_relevance:  是否对三元组按锚点相关性排序（提升 Hit@1，默认开启）

    Returns:
        处理后的样本列表
    """
    lightprof_count = 0
    fallback_count = 0
    empty_count = 0

    new_data = []
    for each_qa in data:
        processed = get_prompts_lightprof(
            each_qa, mode, sys_prompt, cot_prompt, thres,
            use_pog_prompt=use_pog_prompt,
            fallback_to_scored=fallback_to_scored,
            sort_by_relevance=sort_by_relevance
        )
        source = processed.get('_evidence_source', 'unknown')
        if source == 'lightprof':
            lightprof_count += 1
        elif source == 'fallback_scored':
            fallback_count += 1
        else:
            empty_count += 1
        new_data.append(processed)

    # 打印统计
    total = len(new_data)
    if total > 0:
        print(f"\n[LightPROF Prompt 统计]")
        print(f"  使用 lightprof_gr_triples: {lightprof_count}/{total} ({lightprof_count/total*100:.1f}%)")
        if fallback_count > 0:
            print(f"  回退到 scored_triplets:    {fallback_count}/{total} ({fallback_count/total*100:.1f}%)")
        if empty_count > 0:
            print(f"  空证据（无三元组）:        {empty_count}/{total} ({empty_count/total*100:.1f}%)")
        pog_enabled = use_pog_prompt or ('pog' in mode)
        print(f"  PoG 推理指导注入:          {'开启' if pog_enabled else '关闭'}")

    return new_data
