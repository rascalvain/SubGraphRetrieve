import os
import torch
import numpy as np

from tqdm import tqdm
from collections import defaultdict

from src.dataset.retriever_hybrid import RetrieverDataset, collate_retriever
from src.model.retriever import Retriever
from src.setup import set_seed, prepare_sample

try:
    from sentence_transformers import SentenceTransformer
    import torch.nn.functional as F
    SENTENCE_TRANSFORMER_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMER_AVAILABLE = False
    print("⚠️  sentence-transformers 未安装，将跳过LLM实体映射功能")


class EntityMapper:
    """使用sentence-transformer将LLM提取的实体映射到图中的实体"""
    
    def __init__(self, model_name='all-MiniLM-L6-v2', similarity_threshold=0.6):
        """
        Args:
            model_name: sentence-transformer模型名称
            similarity_threshold: 相似度阈值，低于此值则忽略
        """
        self.similarity_threshold = similarity_threshold
        self.model = None
        
        if SENTENCE_TRANSFORMER_AVAILABLE:
            try:
                print(f"  加载实体映射模型: {model_name}")
                self.model = SentenceTransformer(model_name)
                print(f"  ✓ 实体映射模型加载成功")
            except Exception as e:
                print(f"  ⚠️  加载实体映射模型失败: {e}")
                self.model = None
    
    def _normalize_entity(self, entity):
        """标准化实体名称"""
        import re
        entity = str(entity).strip()
        entity = entity.replace('_', ' ')
        entity = re.sub(r'\([^)]*\)', '', entity)
        entity = re.sub(r'[,;:]', ' ', entity)
        entity = ' '.join(entity.split())
        return entity.lower()
    
    def map_entities_to_graph(self, llm_entities, graph_entities, raw_q_entity):
        """
        将LLM提取的实体映射到图中的实体节点
        
        Args:
            llm_entities: LLM提取的实体列表（来自raw_sample.get('q_entity_list', [])）
            graph_entities: 图中的所有实体列表
            raw_q_entity: 原始的q_entity列表
        
        Returns:
            expanded_q_entity_list: 扩展后的问题实体列表
        """
        # 1. 先复制q_entity中的所有实体
        expanded_list = list(raw_q_entity) if isinstance(raw_q_entity, list) else [raw_q_entity]
        
        # 如果没有LLM实体或模型未加载，直接返回
        if not llm_entities or not self.model or not graph_entities:
            return expanded_list
        
        # 2. 标准化图中的实体
        graph_entities_normalized = [self._normalize_entity(e) for e in graph_entities]
        
        # 3. 预计算图实体的嵌入（批量计算，提高效率）
        try:
            graph_embeddings = self.model.encode(
                graph_entities_normalized, 
                convert_to_tensor=True,
                show_progress_bar=False
            )
        except Exception as e:
            print(f"  ⚠️  计算图实体嵌入失败: {e}")
            return expanded_list
        
        # 4. 对每个LLM实体进行映射
        mapped_entities = set(expanded_list)  # 用集合去重
        
        for llm_entity in llm_entities:
            llm_entity = str(llm_entity).strip()
            if not llm_entity or llm_entity in mapped_entities:
                continue
            
            # 标准化LLM实体
            llm_entity_normalized = self._normalize_entity(llm_entity)
            
            # 精确匹配（快速路径）
            if llm_entity in graph_entities or llm_entity_normalized in graph_entities_normalized:
                if llm_entity in graph_entities:
                    mapped_entities.add(llm_entity)
                else:
                    idx = graph_entities_normalized.index(llm_entity_normalized)
                    mapped_entities.add(graph_entities[idx])
                continue
            
            # 语义相似度匹配
            try:
                llm_embedding = self.model.encode(
                    llm_entity_normalized, 
                    convert_to_tensor=True,
                    show_progress_bar=False
                )
                
                # 计算余弦相似度
                similarities = F.cosine_similarity(
                    llm_embedding.unsqueeze(0),
                    graph_embeddings,
                    dim=1
                )
                
                # 找到最相似的实体
                max_sim, max_idx = torch.max(similarities, dim=0)
                
                # 如果相似度超过阈值，则映射
                if max_sim.item() >= self.similarity_threshold:
                    mapped_entity = graph_entities[max_idx.item()]
                    mapped_entities.add(mapped_entity)
                    
            except Exception as e:
                # 静默忽略错误，继续处理下一个实体
                continue
        
        return list(mapped_entities)


def evaluate_split(split_name, dataset, model, device, args, entity_mapper=None):
    """
    在指定数据集上进行推理和评估
    
    Args:
        split_name: 数据集名称 ('val' 或 'test')
        dataset: RetrieverDataset 实例
        model: 训练好的模型
        device: 设备
        args: 命令行参数
    
    Returns:
        avg_metrics: 平均指标字典
        pred_dict: 预测结果字典
    """
    metric_dict = defaultdict(list)
    pred_dict = dict()
    
    print(f"\n开始在 {split_name} 集上推理和评估...")
    
    for i in tqdm(range(len(dataset)), desc=f"{split_name} 集推理进度"):
        raw_sample = dataset[i]
        sample = collate_retriever([raw_sample])

        h_id_tensor, r_id_tensor, t_id_tensor, q_emb, entity_embs, \
            num_non_text_entities, relation_embs, topic_entity_one_hot, \
            target_triple_probs, a_entity_id_list = prepare_sample(device, sample)

        entity_list = raw_sample['text_entity_list'] + raw_sample['non_text_entity_list']
        relation_list = raw_sample['relation_list']
        top_K_triples = []
        target_relevant_triples = []

        if len(h_id_tensor) != 0:
            # 前向传播
            pred_triple_logits = model(
                h_id_tensor, r_id_tensor, t_id_tensor, q_emb, entity_embs,
                num_non_text_entities, relation_embs, topic_entity_one_hot)
            pred_triple_scores = torch.sigmoid(pred_triple_logits).reshape(-1)

            # 计算评估指标
            sorted_triple_ids_pred = torch.argsort(
                pred_triple_scores, descending=True).cpu()
            triple_ranks_pred = torch.empty_like(sorted_triple_ids_pred)
            triple_ranks_pred[sorted_triple_ids_pred] = torch.arange(
                len(triple_ranks_pred))

            target_triple_ids = target_triple_probs.nonzero().squeeze(-1)
            num_target_triples = len(target_triple_ids)

            if num_target_triples > 0:
                num_total_entities = len(entity_embs) + num_non_text_entities

                for k in args.eval_k_list:
                    # Triple Recall@K
                    recall_k_sample = (
                            triple_ranks_pred[target_triple_ids] < k).sum().item()
                    metric_dict[f'triple_recall@{k}'].append(
                        recall_k_sample / num_target_triples)

                    # Answer Recall@K
                    triple_mask_k = triple_ranks_pred < k
                    entity_mask_k = torch.zeros(num_total_entities)
                    entity_mask_k[h_id_tensor[triple_mask_k]] = 1.
                    entity_mask_k[t_id_tensor[triple_mask_k]] = 1.
                    recall_k_sample_ans = entity_mask_k[a_entity_id_list].sum().item()
                    metric_dict[f'ans_recall@{k}'].append(
                        recall_k_sample_ans / len(a_entity_id_list))

            # Top-K 检索（用于保存结果）
            top_K = min(args.max_K, len(pred_triple_scores))
            top_K_results = torch.topk(pred_triple_scores, top_K)
            top_K_scores = top_K_results.values.cpu().tolist()
            top_K_triple_IDs = top_K_results.indices.cpu().tolist()

            # 转换为文本形式
            for j, triple_id in enumerate(top_K_triple_IDs):
                top_K_triples.append((
                    entity_list[h_id_tensor[triple_id].item()],
                    relation_list[r_id_tensor[triple_id].item()],
                    entity_list[t_id_tensor[triple_id].item()],
                    top_K_scores[j]
                ))

            # 目标相关三元组
            target_relevant_triple_ids = target_triple_probs.nonzero().reshape(-1).tolist()
            for triple_id in target_relevant_triple_ids:
                target_relevant_triples.append((
                    entity_list[h_id_tensor[triple_id].item()],
                    relation_list[r_id_tensor[triple_id].item()],
                    entity_list[t_id_tensor[triple_id].item()],
                ))

        # 构建扩展的问题实体列表
        if entity_mapper is not None:
            llm_entities = raw_sample.get('q_entity_list', [])
            expanded_q_entity_list = entity_mapper.map_entities_to_graph(
                llm_entities=llm_entities,
                graph_entities=entity_list,
                raw_q_entity=raw_sample['q_entity']
            )
        else:
            expanded_q_entity_list = raw_sample.get('q_entity_list', [])
        
        # 保存样本结果
        sample_dict = {
            'question': raw_sample['question'],
            'scored_triples': top_K_triples,
            'q_entity': raw_sample['q_entity'],
            'q_entity_list': expanded_q_entity_list,
            'q_entity_in_graph': [entity_list[e_id] for e_id in raw_sample['q_entity_id_list']],
            'a_entity': raw_sample['a_entity'],
            'a_entity_in_graph': [entity_list[e_id] for e_id in raw_sample['a_entity_id_list']],
            'max_path_length': raw_sample['max_path_length'],
            'target_relevant_triples': target_relevant_triples
        }

        pred_dict[raw_sample['id']] = sample_dict

    # 计算平均指标
    avg_metrics = {}
    for key, val in metric_dict.items():
        avg_metrics[key] = np.mean(val)

    return avg_metrics, pred_dict


def display_sample_results(split_name, pred_dict, num_samples=3, num_triples_per_sample=10):
    """
    在控制台展示几条结果样本
    
    Args:
        split_name: 数据集名称 ('val' 或 'test')
        pred_dict: 预测结果字典
        num_samples: 展示的样本数量
        num_triples_per_sample: 每个样本展示的三元组数量
    """
    import random
    
    print("\n" + "=" * 80)
    print(f"📋 {split_name} 集结果样本展示（随机抽取 {num_samples} 条）")
    print("=" * 80)
    
    # 随机选择样本
    sample_ids = list(pred_dict.keys())
    selected_ids = random.sample(sample_ids, min(num_samples, len(sample_ids)))
    
    for idx, sample_id in enumerate(selected_ids, 1):
        sample = pred_dict[sample_id]
        
        print(f"\n{'─' * 80}")
        print(f"样本 {idx}/{num_samples} | ID: {sample_id}")
        print(f"{'─' * 80}")
        
        # 问题
        print(f"\n❓ 问题:")
        print(f"   {sample['question']}")
        
        # 问题实体信息
        print(f"\n🎯 问题实体信息:")
        print(f"   原始标注 (q_entity): {sample['q_entity']}")
        if sample.get('q_entity_list'):
            q_entity_set = set(sample['q_entity']) if isinstance(sample['q_entity'], list) else {sample['q_entity']}
            q_entity_list_set = set(sample['q_entity_list'])
            new_entities = q_entity_list_set - q_entity_set
            
            print(f"   扩展实体列表 (q_entity_list): {sample['q_entity_list']}")
            if new_entities:
                print(f"   └─ 新增实体（LLM映射）: {list(new_entities)}")
        print(f"   图中匹配 (q_entity_in_graph): {sample['q_entity_in_graph']}")
        
        # 答案实体信息
        print(f"\n✅ 答案实体信息:")
        print(f"   标准答案 (a_entity): {sample['a_entity']}")
        print(f"   图中匹配 (a_entity_in_graph): {sample['a_entity_in_graph']}")
        
        # 路径长度
        print(f"\n📏 最大路径长度: {sample['max_path_length']}")
        
        # Top-N 检索到的三元组
        scored_triples = sample['scored_triples']
        print(f"\n🔍 检索到的三元组 (Top-{num_triples_per_sample}/{len(scored_triples)}):")
        for i, (h, r, t, score) in enumerate(scored_triples[:num_triples_per_sample], 1):
            print(f"   {i:2d}. [{score:.4f}] ({h}, {r}, {t})")
        
        if len(scored_triples) > num_triples_per_sample:
            print(f"   ... 还有 {len(scored_triples) - num_triples_per_sample} 个三元组未显示")
        
        # Ground truth 相关三元组
        target_triples = sample['target_relevant_triples']
        if target_triples:
            print(f"\n📌 标注的相关三元组 (Ground Truth, 共{len(target_triples)}个):")
            for i, (h, r, t) in enumerate(target_triples[:5], 1):
                print(f"   {i}. ({h}, {r}, {t})")
            if len(target_triples) > 5:
                print(f"   ... 还有 {len(target_triples) - 5} 个相关三元组未显示")
    
    print(f"\n{'=' * 80}\n")


def save_results(split_name, pred_dict, avg_metrics, args, config, root_path):
    """
    保存指定数据集的评估结果
    
    Args:
        split_name: 数据集名称 ('val' 或 'test')
        pred_dict: 预测结果字典
        avg_metrics: 平均指标字典
        args: 命令行参数
        config: 配置字典
        root_path: 保存目录
    """
    use_llm_entities = config['dataset'].get('use_llm_extracted_entities', False)
    use_kg_enhanced = config['dataset'].get('use_kg_enhanced', False)
    
    # 按指标名称排序
    sorted_metrics = sorted(avg_metrics.items())

    # 打印指标
    print("\n" + "=" * 60)
    print(f"{split_name} 集评估结果:")
    print("=" * 60)

    print("\n三元组召回率 (Triple Recall):")
    print("-" * 60)
    for metric_name, metric_value in sorted_metrics:
        if 'triple_recall' in metric_name:
            k_value = metric_name.split('@')[1]
            print(f"  Recall@{k_value:>4s}: {metric_value:.4f} ({metric_value * 100:.2f}%)")

    print("\n答案召回率 (Answer Recall):")
    print("-" * 60)
    for metric_name, metric_value in sorted_metrics:
        if 'ans_recall' in metric_name:
            k_value = metric_name.split('@')[1]
            print(f"  Recall@{k_value:>4s}: {metric_value:.4f} ({metric_value * 100:.2f}%)")

    print("=" * 60 + "\n")

    # 保存检索结果
    output_file = os.path.join(root_path, f'{split_name}_retrieval_result.pth')
    torch.save(pred_dict, output_file)

    # 保存评估指标（文本格式）
    metrics_file = os.path.join(root_path, f'{split_name}_metrics.txt')
    with open(metrics_file, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write(f"{split_name} 集评估结果\n")
        f.write("=" * 60 + "\n")
        f.write(f"模型检查点: {args.path}\n")
        f.write(f"样本数: {len(pred_dict)}\n")
        f.write(f"使用 LLM 实体: {use_llm_entities}\n")
        f.write(f"使用 KG 增强: {use_kg_enhanced}\n")
        f.write("=" * 60 + "\n\n")

        f.write("三元组召回率 (Triple Recall):\n")
        f.write("-" * 60 + "\n")
        for metric_name, metric_value in sorted_metrics:
            if 'triple_recall' in metric_name:
                k_value = metric_name.split('@')[1]
                f.write(f"  Recall@{k_value:>4s}: {metric_value:.4f} ({metric_value * 100:.2f}%)\n")

        f.write("\n答案召回率 (Answer Recall):\n")
        f.write("-" * 60 + "\n")
        for metric_name, metric_value in sorted_metrics:
            if 'ans_recall' in metric_name:
                k_value = metric_name.split('@')[1]
                f.write(f"  Recall@{k_value:>4s}: {metric_value:.4f} ({metric_value * 100:.2f}%)\n")

        f.write("\n" + "=" * 60 + "\n")

    # 保存详细指标（JSON格式）
    import json
    metrics_json_file = os.path.join(root_path, f'{split_name}_metrics.json')
    with open(metrics_json_file, 'w', encoding='utf-8') as f:
        metrics_data = {
            'checkpoint_path': args.path,
            'num_samples': len(pred_dict),
            'use_llm_entities': use_llm_entities,
            'use_kg_enhanced': use_kg_enhanced,
            'metrics': avg_metrics
        }
        json.dump(metrics_data, f, indent=2, ensure_ascii=False)

    print(f"✓ {split_name} 集推理完成！")
    print(f"  处理样本数: {len(pred_dict)}")
    print(f"  检索结果已保存: {output_file}")
    print(f"  评估指标已保存: {metrics_file}")
    print(f"  评估指标(JSON)已保存: {metrics_json_file}")

    # 统计信息
    total_triples = sum(len(v['scored_triples']) for v in pred_dict.values())
    avg_triples = total_triples / len(pred_dict) if pred_dict else 0
    print(f"  平均每个样本检索到: {avg_triples:.1f} 个三元组\n")


@torch.no_grad()
def main(args):
    device = torch.device(f'cuda:0' if torch.cuda.is_available() else 'cpu')

    # 加载检查点
    print(f"加载模型检查点: {args.path}")
    cpt = torch.load(args.path, map_location='cpu')
    config = cpt['config']

    set_seed(config['env']['seed'])
    torch.set_num_threads(config['env']['num_threads'])

    # 读取增强功能配置
    use_llm_entities = config['dataset'].get('use_llm_extracted_entities', False)
    use_kg_enhanced = config['dataset'].get('use_kg_enhanced', False)
    entity_mapping_file = config['dataset'].get('entity_mapping_file', None)
    data_base_dir = config['dataset'].get('data_base_dir', 'data_files')

    # 打印配置
    print("\n" + "=" * 60)
    print("推理配置:")
    print("=" * 60)
    print(f"数据集: {config['dataset']['name']}")
    print(f"设备: {device}")
    print(f"使用 LLM 实体: {use_llm_entities}")
    print(f"使用 KG 增强: {use_kg_enhanced}")
    if entity_mapping_file:
        print(f"实体映射文件: {entity_mapping_file}")
    print(f"Top-K (保存): {args.max_K}")
    print(f"评估 K 值: {args.eval_k_list}")
    print(f"评估数据集: {', '.join(args.splits)}")
    print("=" * 60 + "\n")

    # 创建模型（先用测试集获取嵌入维度）
    temp_dataset = RetrieverDataset(
        config=config,
        split='test',
        skip_no_path=False,
        use_llm_extracted_entities=use_llm_entities,
        entity_mapping_file=entity_mapping_file if use_llm_entities else None,
        emb_dir_suffix='_kg_enhanced' if use_kg_enhanced else None
    )
    
    emb_size = temp_dataset[0]['q_emb'].shape[-1]
    model = Retriever(emb_size, **config['retriever']).to(device)
    model.load_state_dict(cpt['model_state_dict'])
    model.eval()

    print(f"✓ 模型加载完成")
    print(f"  嵌入维度: {emb_size}")
    print(f"  参数数量: {sum(p.numel() for p in model.parameters()):,}\n")

    # 初始化实体映射器（如果启用）
    entity_mapper = None
    if args.enable_entity_mapping:
        print("初始化实体映射器...")
        entity_mapper = EntityMapper(
            model_name=args.entity_mapping_model,
            similarity_threshold=args.entity_mapping_threshold
        )
        if entity_mapper.model is None:
            print("  ⚠️  实体映射器初始化失败，将跳过实体映射\n")
            entity_mapper = None
        else:
            print(f"  相似度阈值: {args.entity_mapping_threshold}")
            print()

    root_path = os.path.dirname(args.path)
    all_metrics = {}

    # 对每个数据集进行评估
    for split in args.splits:
        print(f"\n{'=' * 60}")
        print(f"处理 {split} 集")
        print(f"{'=' * 60}")
        
        # 创建数据集
        dataset = RetrieverDataset(
            config=config,
            split=split,
            skip_no_path=False,
            use_llm_extracted_entities=use_llm_entities,
            entity_mapping_file=entity_mapping_file if use_llm_entities else None,
            emb_dir_suffix='_kg_enhanced' if use_kg_enhanced else None
        )
        print(f"✓ {split} 集加载完成: {len(dataset)} 个样本")

        # 评估
        avg_metrics, pred_dict = evaluate_split(split, dataset, model, device, args, entity_mapper)
        
        # 保存结果
        save_results(split, pred_dict, avg_metrics, args, config, root_path)
        
        # 展示样本结果
        display_sample_results(split, pred_dict, num_samples=args.num_display_samples, 
                              num_triples_per_sample=args.num_display_triples)
        
        all_metrics[split] = avg_metrics

    # 打印总结
    print("\n" + "=" * 60)
    print("所有数据集评估完成！")
    print("=" * 60)
    for split in args.splits:
        print(f"\n{split} 集结果:")
        sorted_metrics = sorted(all_metrics[split].items())
        for metric_name, metric_value in sorted_metrics:
            print(f"  {metric_name}: {metric_value:.4f} ({metric_value * 100:.2f}%)")
    print("\n" + "=" * 60 + "\n")

    return all_metrics


if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser('Hybrid Retriever Inference with Evaluation')
    parser.add_argument('-p', '--path', type=str, required=True,
                        help='模型检查点路径 (例如: webqsp_llm_kg_Nov08-01:14:47/cpt.pth)')
    parser.add_argument('--max_K', type=int, default=500,
                        help='Top-K 三元组数量（用于保存到 retrieval_result.pth）')
    parser.add_argument('--eval_k_list', type=int, nargs='+',
                        default=[50, 100, 200, 500],
                        help='评估的 K 值列表（例如: --eval_k_list 50 100 200 500）')
    parser.add_argument('--splits', type=str, nargs='+',
                        default=['test'],
                        choices=['val', 'test'],
                        help='要评估的数据集（例如: --splits val test）')
    parser.add_argument('--num_display_samples', type=int, default=3,
                        help='在控制台展示的结果样本数量（默认: 3）')
    parser.add_argument('--num_display_triples', type=int, default=10,
                        help='每个样本展示的三元组数量（默认: 10）')
    parser.add_argument('--enable_entity_mapping', action='store_true',
                        help='启用LLM实体到图实体的映射（需要安装sentence-transformers）')
    parser.add_argument('--entity_mapping_model', type=str, default='all-MiniLM-L6-v2',
                        help='用于实体映射的sentence-transformer模型（默认: all-MiniLM-L6-v2）')
    parser.add_argument('--entity_mapping_threshold', type=float, default=0.6,
                        help='实体映射的相似度阈值（默认: 0.6）')

    args = parser.parse_args()

    main(args)