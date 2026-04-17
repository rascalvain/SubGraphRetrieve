#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
合并多个LightPROF采样结果文件
用于并行处理后的结果合并
"""

import torch
import argparse
from pathlib import Path


def merge_lightprof_results(input_files, output_file):
    """
    合并多个LightPROF结果文件
    
    Args:
        input_files: 输入文件列表
        output_file: 输出文件路径
    """
    print(f"\n{'=' * 80}")
    print(f"合并 LightPROF 结果")
    print(f"{'=' * 80}")
    print(f"输入文件数: {len(input_files)}")
    print(f"输出文件: {output_file}")
    print(f"{'=' * 80}\n")
    
    merged_data = {}
    total_samples = 0
    
    for i, input_file in enumerate(input_files, 1):
        print(f"[{i}/{len(input_files)}] 加载: {input_file}")
        
        try:
            data = torch.load(input_file, map_location='cpu')
            print(f"  ✓ 加载了 {len(data)} 个样本")
            
            # 检查重复的sample_id
            duplicates = set(merged_data.keys()) & set(data.keys())
            if duplicates:
                print(f"  ⚠️  发现 {len(duplicates)} 个重复样本，将覆盖")
            
            merged_data.update(data)
            total_samples += len(data)
            
        except Exception as e:
            print(f"  ❌ 加载失败: {e}")
            continue
    
    print(f"\n{'─' * 80}")
    print(f"合并统计:")
    print(f"  输入文件: {len(input_files)}")
    print(f"  总样本数: {total_samples}")
    print(f"  唯一样本数: {len(merged_data)}")
    print(f"  重复样本数: {total_samples - len(merged_data)}")
    print(f"{'─' * 80}\n")
    
    # 保存合并结果
    print(f"保存到: {output_file}")
    torch.save(merged_data, output_file)
    print(f"✓ 保存成功！")
    
    # 验证
    print(f"\n验证合并结果...")
    loaded = torch.load(output_file, map_location='cpu')
    assert len(loaded) == len(merged_data), "验证失败：样本数不匹配"
    
    # 检查数据完整性
    sample = list(loaded.values())[0]
    required_fields = ['question', 'scored_triples', 'lightprof_gr_triples', 'lightprof_stats']
    for field in required_fields:
        if field not in sample:
            print(f"  ⚠️  缺少字段: {field}")
    
    print(f"✓ 验证通过！")
    
    # 统计信息
    total_gr_triples = sum(len(s.get('lightprof_gr_triples', [])) for s in loaded.values())
    avg_gr_triples = total_gr_triples / len(loaded) if loaded else 0
    
    print(f"\n{'=' * 80}")
    print(f"最终统计:")
    print(f"  样本数: {len(loaded)}")
    print(f"  平均推理图三元组数: {avg_gr_triples:.1f}")
    print(f"{'=' * 80}\n")


def main():
    parser = argparse.ArgumentParser(description='合并LightPROF结果文件')
    parser.add_argument('-i', '--input', type=str, nargs='+', required=True,
                       help='输入文件列表（支持通配符）')
    parser.add_argument('-o', '--output', type=str, required=True,
                       help='输出文件路径')
    
    args = parser.parse_args()
    
    # 展开通配符
    from glob import glob
    input_files = []
    for pattern in args.input:
        matches = glob(pattern)
        if matches:
            input_files.extend(matches)
        else:
            print(f"⚠️  未找到匹配文件: {pattern}")
    
    if not input_files:
        print("错误: 未找到任何输入文件")
        return
    
    merge_lightprof_results(input_files, args.output)


if __name__ == '__main__':
    main()
