import os
import torch
import pandas as pd

from datasets import Dataset, load_dataset
from tqdm import tqdm

from src.config.emb import load_yaml
from src.dataset.emb_hybrid import EmbInferDataset_Hybrid


def load_local_parquet_dataset(parquet_dir: str, split: str) -> Dataset:
    """
    从本地 parquet 文件加载数据集

    Parameters
    ----------
    parquet_dir : str
        parquet 文件所在目录
    split : str
        数据集分割: 'train', 'validation', 'test'

    Returns
    -------
    Dataset
        HuggingFace Dataset 对象
    """
    # 定义分割对应的文件模式
    split_patterns = {
        'train': ['train-*.parquet'],
        'validation': ['validation-*.parquet'],
        'test': ['test-*.parquet']
    }

    if split not in split_patterns:
        raise ValueError(f"Invalid split: {split}. Must be one of {list(split_patterns.keys())}")

    # 查找匹配的文件
    import glob
    parquet_files = []
    for pattern in split_patterns[split]:
        files = glob.glob(os.path.join(parquet_dir, pattern))
        parquet_files.extend(files)

    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found for split '{split}' in {parquet_dir}")

    print(f"  找到 {len(parquet_files)} 个文件: {[os.path.basename(f) for f in parquet_files]}")

    # 读取并合并所有文件
    dfs = []
    for file in sorted(parquet_files):
        df = pd.read_parquet(file)
        dfs.append(df)

    # 合并数据
    merged_df = pd.concat(dfs, ignore_index=True)
    print(f"  加载了 {len(merged_df)} 条数据")

    # 转换为 HuggingFace Dataset
    dataset = Dataset.from_pandas(merged_df)

    return dataset


def get_emb(subset, text_encoder, save_file, save_link_info=True):
    """
    计算嵌入并保存（支持问题主题实体）
    """
    emb_dict = dict()
    all_link_info = dict()

    for i in tqdm(range(len(subset)), desc="计算嵌入"):
        # 获取数据（现在包含问题主题实体）
        sample = subset[i]
        id = sample['id']
        q_text = sample['question']
        text_entity_list = sample['text_entity_list']
        relation_list = sample['relation_list']
        q_entity_list = sample.get('q_entity_list', [])  # 新增

        # 调用编码器
        result = text_encoder(
            q_text,
            text_entity_list,
            relation_list,
            q_entity_list  # 新增参数
        )

        # 检查返回值数量
        if len(result) == 5:
            # KG增强版：(q_emb, entity_embs, relation_embs, q_topic_embs, link_info)
            q_emb, entity_embs, relation_embs, q_topic_embs, link_info = result
            all_link_info[id] = link_info
        elif len(result) == 4:
            # 普通版但有问题主题实体：(q_emb, entity_embs, relation_embs, q_topic_embs)
            q_emb, entity_embs, relation_embs, q_topic_embs = result
        elif len(result) == 3:
            # 旧版本：(q_emb, entity_embs, relation_embs)
            q_emb, entity_embs, relation_embs = result
            q_topic_embs = torch.zeros(0, q_emb.shape[-1])  # 空张量
        else:
            raise ValueError(f"编码器返回了意外的值数量: {len(result)}")

        emb_dict_i = {
            'q_emb': q_emb,
            'entity_embs': entity_embs,
            'relation_embs': relation_embs,
            'q_topic_embs': q_topic_embs  # 新增
        }
        emb_dict[id] = emb_dict_i

    # 保存嵌入
    torch.save(emb_dict, save_file)
    print(f"✓ 嵌入已保存: {save_file}")

    # 保存链接信息
    if save_link_info and all_link_info:
        link_info_file = save_file.replace('.pth', '_link_info.pth')
        torch.save(all_link_info, link_info_file)
        print(f"✓ 链接信息已保存: {link_info_file}")


def main(args):
    # 加载配置
    config_file = f'configs/emb/gte-large-en-v1.5/{args.dataset}.yaml'
    config = load_yaml(config_file)

    torch.set_num_threads(config['env']['num_threads'])

    # ========== 修改部分：支持本地数据集 ==========
    if args.use_local_dataset:
        print(f"\n从本地加载数据集: {args.local_dataset_path}")

        print("\n加载训练集...")
        train_set = load_local_parquet_dataset(args.local_dataset_path, 'train')

        print("\n加载验证集...")
        val_set = load_local_parquet_dataset(args.local_dataset_path, 'validation')

        print("\n加载测试集...")
        test_set = load_local_parquet_dataset(args.local_dataset_path, 'test')
    else:
        # 原有方式：从 HuggingFace 加载
        print("\n从 HuggingFace 加载数据集...")
        if args.dataset == 'cwq':
            input_file = os.path.join('rmanluo', 'RoG-cwq')
        else:
            input_file = os.path.join('ml1996', 'webqsp')

        train_set = load_dataset(input_file, split='train')
        val_set = load_dataset(input_file, split='validation')
        test_set = load_dataset(input_file, split='test')

    # 🔑 注意：这里取消缩进，因为已经退出 if-else 块
    # 加载实体标识符
    entity_identifiers = []
    with open(config['entity_identifier_file'], 'r') as f:
        for line in f:
            entity_identifiers.append(line.strip())
    entity_identifiers = set(entity_identifiers)

    # 🔑 根据参数确定输出目录
    if args.output_dir:
        # 使用脚本传入的输出目录
        base_output_dir = args.output_dir
        print(f"\n✓ 使用自定义输出目录: {base_output_dir}")
    else:
        # 使用默认目录
        base_output_dir = f'data_files/{args.dataset}'
        print(f"\n✓ 使用默认输出目录: {base_output_dir}")

    # 创建子目录
    save_dir = os.path.join(base_output_dir, 'processed')
    os.makedirs(save_dir, exist_ok=True)

    # 处理数据集
    train_set = EmbInferDataset_Hybrid(
        train_set, entity_identifiers,
        os.path.join(save_dir, 'train.pkl'),
        use_q_topic_entities=args.use_local_dataset
    )
    val_set = EmbInferDataset_Hybrid(
        val_set, entity_identifiers,
        os.path.join(save_dir, 'val.pkl'),
        use_q_topic_entities=args.use_local_dataset
    )
    test_set = EmbInferDataset_Hybrid(
        test_set, entity_identifiers,
        os.path.join(save_dir, 'test.pkl'),
        skip_no_topic=False, skip_no_ans=False,
        use_q_topic_entities=args.use_local_dataset
    )

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 初始化编码器
    if args.use_kg_enhancement:
        print("\n" + "=" * 60)
        print("使用 KG 增强模式")
        print("=" * 60)

        from src.model.text_encoders import GTEKGEnhanced

        text_encoder = GTEKGEnhanced(
            device=device,
            kg_emb_path=args.kg_emb_path,
            entity_mapping_path=args.entity_mapping_path,
            fusion_method=args.fusion_method,
            text_weight=args.text_weight,
            kg_weight=args.kg_weight,
            normalize=True
        )

        # 使用自定义目录或默认目录
        emb_save_dir = os.path.join(base_output_dir, 'emb', 'gte-large-en-v1.5_kg_enhanced')
    else:
        print("\n使用普通文本编码器")
        from src.model.text_encoders import GTELargeEN
        text_encoder = GTELargeEN(device)

        # 使用自定义目录或默认目录
        emb_save_dir = os.path.join(base_output_dir, 'emb', 'gte-large-en-v1.5')

    os.makedirs(emb_save_dir, exist_ok=True)
    print(f"✓ 嵌入保存目录: {emb_save_dir}")

    # 计算嵌入
    print("\n" + "=" * 60)
    print("开始计算嵌入")
    print("=" * 60)

    print("\n[1/3] 训练集")
    get_emb(train_set, text_encoder, os.path.join(emb_save_dir, 'train.pth'))

    print("\n[2/3] 验证集")
    get_emb(val_set, text_encoder, os.path.join(emb_save_dir, 'val.pth'))

    print("\n[3/3] 测试集")
    get_emb(test_set, text_encoder, os.path.join(emb_save_dir, 'test.pth'))

    # 打印统计
    if args.use_kg_enhancement:
        text_encoder.print_stats()

    print("\n" + "=" * 60)
    print("✓ 全部完成！")
    print("=" * 60)
    print(f"输出根目录: {base_output_dir}")
    print(f"  - 处理数据: {save_dir}")
    print(f"  - 嵌入文件: {emb_save_dir}")
    print(f"  - 训练集: {os.path.join(emb_save_dir, 'train.pth')}")
    print(f"  - 验证集: {os.path.join(emb_save_dir, 'val.pth')}")
    print(f"  - 测试集: {os.path.join(emb_save_dir, 'test.pth')}")
    print("=" * 60)


if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser('Text Embedding Pre-Computation for Retrieval')
    parser.add_argument('-d', '--dataset', type=str, required=True,
                        choices=['webqsp', 'cwq'], help='Dataset name')

    # ========== 新增参数 ==========
    parser.add_argument('--use_local_dataset', action='store_true',
                        help='从本地 parquet 文件加载数据集')
    parser.add_argument('--local_dataset_path', type=str,
                        default='../datasets/webqsp_processed',
                        help='本地数据集路径')
    # 🔑 添加输出目录参数
    parser.add_argument('--output_dir', type=str,
                        default=None,
                        help='输出根目录（不指定则使用默认 data_files/ 目录）')
    # KG增强参数
    parser.add_argument('--use_kg_enhancement', action='store_true',
                        help='使用KG嵌入增强')
    parser.add_argument('--kg_emb_path', type=str,
                        default='../emb/entity_embeddings.pkl',
                        help='KG嵌入文件路径')
    parser.add_argument('--entity_mapping_path', type=str,
                        default='../emb/datasets/webqsp_processed/entity_mapping.txt',
                        help='实体映射文件路径')
    parser.add_argument('--fusion_method', type=str,
                        default='weighted',
                        choices=['weighted', 'concat', 'gated'],
                        help='融合方法')
    parser.add_argument('--text_weight', type=float,
                        default=0.7,
                        help='文本嵌入权重')
    parser.add_argument('--kg_weight', type=float,
                        default=0.3,
                        help='KG嵌入权重')

    args = parser.parse_args()

    main(args)