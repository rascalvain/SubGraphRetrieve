#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LightPROF 功能测试脚本
使用模拟数据验证采样逻辑
"""

import torch
from lightprof_sampling import SimpleKG, LightPROF_Sampler

def create_mock_data():
    """创建模拟的retrieval结果"""
    mock_data = {
        'WebQTest-001': {
            'question': "What year did the team with mascot named Lou Seal win the World Series?",
            'q_entity': ['Lou Seal'],
            'q_entity_list': ['Lou Seal'],
            'a_entity': ['2010 World Series', '2012 World Series', '2014 World Series'],
            'scored_triples': [
                ("Lou Seal", "sports.mascot.team", "San Francisco Giants", 0.98),
                ("San Francisco Giants", "sports.sports_team.championships", "2012 World Series", 0.95),
                ("San Francisco Giants", "sports.sports_team.championships", "2014 World Series", 0.93),
                ("San Francisco Giants", "sports.sports_team.championships", "2010 World Series", 0.91),
                ("2012 World Series", "time.event.start_date", "2012-10-24", 0.87),
                ("2014 World Series", "time.event.start_date", "2014-10-21", 0.85),
                ("2010 World Series", "time.event.start_date", "2010-10-27", 0.83),
                ("San Francisco Giants", "sports.sports_team.location", "San Francisco", 0.80),
                ("San Francisco Giants", "sports.sports_team.sport", "Baseball", 0.78),
                ("Lou Seal", "fictional_character.appears_in", "Baseball Games", 0.75),
                ("San Francisco", "location.location.contains", "AT&T Park", 0.70),
                ("AT&T Park", "sports.sports_facility.home_team", "San Francisco Giants", 0.68),
            ],
            'max_path_length': 2,
            'target_relevant_triples': []
        }
    }
    return mock_data


def test_basic_workflow():
    """测试基本工作流程"""
    print("\n" + "=" * 80)
    print("测试 LightPROF 基本工作流程")
    print("=" * 80)
    
    # 1. 创建模拟数据
    mock_data = create_mock_data()
    sample = mock_data['WebQTest-001']
    
    print(f"\n问题: {sample['question']}")
    print(f"锚点实体: {sample['q_entity_list']}")
    print(f"原始三元组数: {len(sample['scored_triples'])}")
    
    # 2. 构建KG
    kg = SimpleKG(sample['scored_triples'], bidirectional=False)
    
    # 3. 初始化采样器
    sampler = LightPROF_Sampler(kg, verbose=True)
    
    # 4. 运行采样
    gr_triples, stats = sampler.run_pipeline(
        question=sample['question'],
        anchor_entities=sample['q_entity_list'],
        h_q=2,
        top_k_chains=3,
        max_paths_per_chain=5
    )
    
    # 5. 展示结果
    print(f"\n" + "=" * 80)
    print(f"采样结果")
    print(f"=" * 80)
    print(f"推理图 G_R 包含 {len(gr_triples)} 个三元组:")
    for i, (h, r, t) in enumerate(gr_triples, 1):
        print(f"  {i}. ({h}, {r}, {t})")
    
    print(f"\n压缩率: {len(gr_triples)}/{len(sample['scored_triples'])} = {len(gr_triples)/len(sample['scored_triples']):.2%}")
    
    print(f"\n统计信息:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    return True


def test_relation_chain_extraction():
    """测试关系链提取"""
    print("\n" + "=" * 80)
    print("测试关系链提取功能")
    print("=" * 80)
    
    # 创建简单的KG
    triples = [
        ("A", "r1", "B", 0.9),
        ("B", "r2", "C", 0.8),
        ("C", "r3", "D", 0.7),
        ("A", "r4", "E", 0.6),
        ("E", "r5", "F", 0.5),
    ]
    
    kg = SimpleKG(triples, bidirectional=False)
    sampler = LightPROF_Sampler(kg, verbose=False)
    
    # 测试不同跳数
    for max_hops in [1, 2, 3]:
        chains = sampler.step1_extract_relation_chains(['A'], max_hops=max_hops)
        print(f"\n最大跳数 = {max_hops}, 关系链数 = {len(chains)}")
        for chain in sorted(chains, key=lambda x: len(x)):
            print(f"  {' -> '.join(chain)}")
    
    return True


def test_bidirectional_graph():
    """测试双向图"""
    print("\n" + "=" * 80)
    print("测试双向图功能")
    print("=" * 80)
    
    triples = [
        ("A", "r1", "B", 0.9),
        ("B", "r2", "C", 0.8),
    ]
    
    # 单向图
    kg_uni = SimpleKG(triples, bidirectional=False)
    sampler_uni = LightPROF_Sampler(kg_uni, verbose=False)
    chains_uni = sampler_uni.step1_extract_relation_chains(['C'], max_hops=2)
    
    print(f"\n单向图（从C出发）: {len(chains_uni)} 条链")
    for chain in chains_uni:
        print(f"  {' -> '.join(chain)}")
    
    # 双向图
    kg_bi = SimpleKG(triples, bidirectional=True)
    sampler_bi = LightPROF_Sampler(kg_bi, verbose=False)
    chains_bi = sampler_bi.step1_extract_relation_chains(['C'], max_hops=2)
    
    print(f"\n双向图（从C出发）: {len(chains_bi)} 条链")
    for chain in chains_bi:
        print(f"  {' -> '.join(chain)}")
    
    return True


if __name__ == '__main__':
    print("\n" + "=" * 80)
    print("LightPROF 功能测试")
    print("=" * 80)
    
    try:
        # 测试1: 基本工作流程
        success1 = test_basic_workflow()
        
        # 测试2: 关系链提取
        success2 = test_relation_chain_extraction()
        
        # 测试3: 双向图
        success3 = test_bidirectional_graph()
        
        print("\n" + "=" * 80)
        print("✅ 所有测试通过！")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
