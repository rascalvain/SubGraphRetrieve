#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
只生成弱监督信号，不训练模型
"""
import torch
import sys

sys.path.append('.')

from src.config.retriever import load_yaml
from src.dataset.retriever_hybrid import RetrieverDataset


def main():
    # 加载配置
    config_file = 'configs/retriever/webqsp.yaml'
    config = load_yaml(config_file)

    print("=" * 80)
    print("生成弱监督信号")
    print("=" * 80)

    # 设置参数
    use_llm_entities = True  # 是否使用LLM提取的实体
    mapping_file = './src/data/entity_mapping_with_mid.txt'

    print(f"\n配置:")
    print(f"  数据集: {config['dataset']['name']}")
    print(f"  使用LLM实体: {use_llm_entities}")
    print(f"  映射文件: {mapping_file}")
    print("")

    # 生成训练集的弱监督信号
    print("\n[1/3] 处理训练集...")
    train_set = RetrieverDataset(
        config=config,
        split='train',
        skip_no_path=False,  # 不跳过，查看所有样本
        use_llm_extracted_entities=use_llm_entities,
        entity_mapping_file=mapping_file,
        emb_dir_suffix='_kg_enhanced'
    )
    print(f"✓ 训练集处理完成，共 {len(train_set)} 个样本")

    # 生成验证集的弱监督信号
    print("\n[2/3] 处理验证集...")
    val_set = RetrieverDataset(
        config=config,
        split='val',
        skip_no_path=False,
        use_llm_extracted_entities=use_llm_entities,
        entity_mapping_file=mapping_file,
        emb_dir_suffix='_kg_enhanced'
    )
    print(f"✓ 验证集处理完成，共 {len(val_set)} 个样本")

    # 生成测试集的弱监督信号
    print("\n[3/3] 处理测试集...")
    test_set = RetrieverDataset(
        config=config,
        split='test',
        skip_no_path=False,
        use_llm_extracted_entities=use_llm_entities,
        entity_mapping_file=mapping_file,
        emb_dir_suffix='_kg_enhanced'
    )
    print(f"✓ 测试集处理完成，共 {len(test_set)} 个样本")

    print("\n" + "=" * 80)
    print("✓ 弱监督信号生成完成！")
    print("=" * 80)
    print("\n生成的文件位置:")
    print(f"  - data_files/{config['dataset']['name']}/triple_scores/train_with_llm_entities.pth")
    print(f"  - data_files/{config['dataset']['name']}/triple_scores/val_with_llm_entities.pth")
    print(f"  - data_files/{config['dataset']['name']}/triple_scores/test_with_llm_entities.pth")
    print("\n这些文件包含:")
    print("  - triple_scores: 每个三元组的得分（0或1）")
    print("  - max_path_length: 最长路径的长度")
    print("\n下一步: 运行 train.py 训练检索器")

    # 显示一些统计信息
    print("\n" + "=" * 80)
    print("样本示例分析")
    print("=" * 80)
    sample = train_set[0]
    print(f"\n样本ID: {sample['id']}")
    print(f"问题: {sample['question']}")
    print(f"原始问题实体: {sample.get('q_entity', [])}")
    if 'q_entity_list' in sample:
        print(f"LLM提取的实体: {sample['q_entity_list']}")
    print(f"答案实体: {sample['a_entity']}")
    print(f"图中三元组数: {len(sample['h_id_list'])}")
    print(f"相关三元组数: {sample['target_triple_probs'].sum().item():.0f}")
    print(f"最大路径长度: {sample['max_path_length']}")


if __name__ == '__main__':
    main()