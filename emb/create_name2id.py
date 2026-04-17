#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
将 entity2id.txt 和 FB15k_mid2name.txt 两个文件匹配，生成 name2id.txt 文件
"""

def create_name2id(entity2id_path, mid2name_path, output_path):
    """
    匹配 mid，创建 name2id 文件
    
    Args:
        entity2id_path: entity2id.txt 文件路径
        mid2name_path: FB15k_mid2name.txt 文件路径
        output_path: 输出的 name2id.txt 文件路径
    """
    # 步骤1：读取 entity2id.txt，建立 mid -> id 的映射
    print("正在读取 entity2id.txt...")
    mid2id = {}
    with open(entity2id_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        total_entities = int(lines[0].strip())  # 第一行是总数
        print(f"实体总数: {total_entities}")
        
        for line in lines[1:]:  # 跳过第一行
            line = line.strip()
            if line:
                parts = line.split('\t')
                if len(parts) == 2:
                    mid, entity_id = parts
                    mid2id[mid] = entity_id
    
    print(f"成功读取 {len(mid2id)} 个 mid -> id 映射")
    
    # 步骤2：读取 FB15k_mid2name.txt，建立 mid -> name 的映射
    print("\n正在读取 FB15k_mid2name.txt...")
    mid2name = {}
    with open(mid2name_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                parts = line.split('\t')
                if len(parts) == 2:
                    mid, name = parts
                    mid2name[mid] = name
    
    print(f"成功读取 {len(mid2name)} 个 mid -> name 映射")
    
    # 步骤3：匹配 mid，创建 name -> id 的映射
    print("\n正在匹配 mid 并创建 name2id 映射...")
    name2id = {}
    matched_count = 0
    unmatched_mids = []
    
    for mid, name in mid2name.items():
        if mid in mid2id:
            name2id[name] = mid2id[mid]
            matched_count += 1
        else:
            unmatched_mids.append(mid)
    
    print(f"成功匹配 {matched_count} 个实体")
    if unmatched_mids:
        print(f"未匹配的 mid 数量: {len(unmatched_mids)}")
        print(f"前5个未匹配的 mid: {unmatched_mids[:5]}")
    
    # 步骤4：将 name2id 写入文件
    print(f"\n正在写入 {output_path}...")
    with open(output_path, 'w', encoding='utf-8') as f:
        # 写入总数
        f.write(f"{len(name2id)}\n")
        
        # 写入 name -> id 映射
        for name, entity_id in sorted(name2id.items(), key=lambda x: int(x[1])):
            f.write(f"{name}\t{entity_id}\n")
    
    print(f"成功！已将 {len(name2id)} 个 name2id 映射写入 {output_path}")
    
    # 打印统计信息
    print("\n" + "="*60)
    print("统计信息:")
    print(f"  entity2id.txt 中的实体数: {len(mid2id)}")
    print(f"  FB15k_mid2name.txt 中的实体数: {len(mid2name)}")
    print(f"  成功匹配的实体数: {matched_count}")
    print(f"  未匹配的实体数: {len(unmatched_mids)}")
    print(f"  name2id.txt 中的实体数: {len(name2id)}")
    print("="*60)


if __name__ == "__main__":
    # 设置文件路径
    entity2id_path = "entity2id.txt"
    mid2name_path = "FB15k_mid2name.txt"
    output_path = "name2id.txt"
    
    # 执行匹配和生成
    create_name2id(entity2id_path, mid2name_path, output_path)
