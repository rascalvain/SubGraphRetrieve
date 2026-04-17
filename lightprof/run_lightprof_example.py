#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LightPROF 使用示例
展示如何从头到尾使用LightPROF采样
"""

import os
import torch
from lightprof_sampling import SimpleKG, LightPROF_Sampler, LLMScorer


def example_1_basic_usage():
    """示例1: 基础用法（使用模拟数据）"""
    print("\n" + "=" * 80)
    print("示例1: 基础用法")
    print("=" * 80)
    
    # 模拟retrieval结果
    sample = {
        'question': "What year did the team with mascot named Lou Seal win the World Series?",
        'q_entity_list': ['Lou Seal'],
        'scored_triples': [
            ("Lou Seal", "sports.mascot.team", "San Francisco Giants", 0.98),
            ("San Francisco Giants", "sports.sports_team.championships", "2012 World Series", 0.95),
            ("San Francisco Giants", "sports.sports_team.championships", "2014 World Series", 0.93),
            ("San Francisco Giants", "sports.sports_team.championships", "2010 World Series", 0.91),
            ("2012 World Series", "time.event.start_date", "2012-10-24", 0.87),
            ("2014 World Series", "time.event.start_date", "2014-10-21", 0.85),
            ("2010 World Series", "time.event.start_date", "2010-10-27", 0.83),
            ("San Francisco Giants", "sports.sports_team.location", "San Francisco", 0.80),
        ]
    }
    
    # 构建KG
    kg = SimpleKG(sample['scored_triples'], bidirectional=False)
    
    # 初始化采样器
    sampler = LightPROF_Sampler(kg, verbose=True)
    
    # 运行采样
    gr_triples, stats = sampler.run_pipeline(
        question=sample['question'],
        anchor_entities=sample['q_entity_list'],
        h_q=2,
        top_k_chains=3,
        max_paths_per_chain=10
    )
    
    print(f"\n✅ 结果:")
    print(f"  原始三元组: {len(sample['scored_triples'])}")
    print(f"  推理图三元组: {len(gr_triples)}")
    print(f"  压缩率: {len(gr_triples)/len(sample['scored_triples']):.1%}")


def example_2_process_file():
    """示例2: 处理实际的retrieval结果文件"""
    print("\n" + "=" * 80)
    print("示例2: 处理实际文件")
    print("=" * 80)
    
    # 检查是否存在测试文件
    test_files = [
        'webqsp_hybrid_llm_kg_Feb05-09_40_35/test_retrieval_result.pth',
        'test_retrieval_result.pth'
    ]
    
    input_file = None
    for f in test_files:
        if os.path.exists(f):
            input_file = f
            break
    
    if input_file is None:
        print("⚠️  未找到测试文件，跳过此示例")
        print("提示: 先运行 inference_hybrid.py 生成检索结果")
        return
    
    print(f"找到输入文件: {input_file}")
    
    # 加载数据
    data = torch.load(input_file, map_location='cpu')
    print(f"✓ 加载了 {len(data)} 个样本")
    
    # 随机选择一个样本进行演示
    import random
    sample_id = random.choice(list(data.keys()))
    sample = data[sample_id]
    
    print(f"\n处理样本: {sample_id}")
    print(f"问题: {sample['question']}")
    
    # 构建KG并采样
    kg = SimpleKG(sample['scored_triples'], bidirectional=False)
    sampler = LightPROF_Sampler(kg, verbose=True)
    
    gr_triples, stats = sampler.run_pipeline(
        question=sample['question'],
        anchor_entities=sample.get('q_entity_list', sample.get('q_entity', [])),
        h_q=2,
        top_k_chains=5,
        max_paths_per_chain=10
    )
    
    print(f"\n✅ 采样完成")


def example_3_compare_parameters():
    """示例3: 对比不同参数设置"""
    print("\n" + "=" * 80)
    print("示例3: 对比不同参数设置")
    print("=" * 80)
    
    # 模拟数据
    sample = {
        'question': "Who is the president?",
        'q_entity_list': ['President'],
        'scored_triples': [
            ("President", "is_a", "Position", 0.9),
            ("Position", "held_by", "Obama", 0.8),
            ("Obama", "born_in", "Hawaii", 0.7),
            ("President", "part_of", "Government", 0.6),
            ("Government", "located_in", "DC", 0.5),
        ]
    }
    
    kg = SimpleKG(sample['scored_triples'], bidirectional=False)
    
    # 测试不同的top_k设置
    for top_k in [1, 2, 3]:
        print(f"\n{'─' * 40}")
        print(f"Top-K = {top_k}")
        print(f"{'─' * 40}")
        
        sampler = LightPROF_Sampler(kg, verbose=False)
        gr_triples, stats = sampler.run_pipeline(
            question=sample['question'],
            anchor_entities=sample['q_entity_list'],
            h_q=2,
            top_k_chains=top_k,
            max_paths_per_chain=5
        )
        
        print(f"推理图三元组数: {len(gr_triples)}")
        print(f"选中关系链数: {stats['num_chains_selected']}")


def example_4_llm_integration():
    """示例4: 集成真实LLM（需要API密钥）"""
    print("\n" + "=" * 80)
    print("示例4: LLM集成示例")
    print("=" * 80)
    
    api_key = os.environ.get('OPENAI_API_KEY')
    
    if api_key is None:
        print("⚠️  未设置OPENAI_API_KEY环境变量")
        print("提示: export OPENAI_API_KEY='your_key_here'")
        print("跳过此示例")
        return
    
    print("✓ 检测到API密钥")
    
    # 初始化LLM打分器
    scorer = LLMScorer(model_name='gpt-4o-mini', api_key=api_key, use_mock=False)
    
    # 测试打分
    question = "What year did the team win?"
    chains = [
        ('mascot_of', 'winner_of'),
        ('located_in', 'part_of'),
        ('type', 'category')
    ]
    
    print(f"\n测试LLM打分:")
    print(f"问题: {question}")
    print(f"关系链: {len(chains)} 条")
    
    scores = scorer.score_chains(question, chains)
    
    print(f"\n打分结果:")
    for i, (chain, score) in enumerate(zip(chains, scores), 1):
        chain_str = ' -> '.join(chain)
        print(f"  {i}. [{score:.4f}] {chain_str}")


def main():
    """运行所有示例"""
    print("\n" + "=" * 100)
    print("LightPROF 使用示例集合")
    print("=" * 100)
    
    # 运行示例
    example_1_basic_usage()
    
    example_2_process_file()
    
    example_3_compare_parameters()
    
    # example_4_llm_integration()  # 需要API密钥，默认注释
    
    print("\n" + "=" * 100)
    print("示例演示完成！")
    print("=" * 100)
    print("\n下一步:")
    print("1. 运行 inference_hybrid.py 生成检索结果")
    print("2. 运行 lightprof_sampling.py 对结果进行采样")
    print("3. 在 reason/ 目录使用采样后的结果进行推理")
    print()


if __name__ == '__main__':
    main()
