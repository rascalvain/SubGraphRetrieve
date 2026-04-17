import numpy as np
import os
import pandas as pd
import time
import torch
import torch.nn.functional as F
import wandb

from collections import defaultdict
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config.retriever import load_yaml
# from src.dataset.retriever import RetrieverDataset, collate_retriever
from src.dataset.retriever_hybrid import RetrieverDataset, collate_retriever
from src.model.retriever import Retriever
from src.setup import set_seed, prepare_sample


@torch.no_grad()
def eval_epoch(config, device, data_loader, model):
    model.eval()

    metric_dict = defaultdict(list)

    for sample in tqdm(data_loader):
        h_id_tensor, r_id_tensor, t_id_tensor, q_emb, entity_embs, \
            num_non_text_entities, relation_embs, topic_entity_one_hot, \
            target_triple_probs, a_entity_id_list = prepare_sample(device, sample)

        pred_triple_logits = model(
            h_id_tensor, r_id_tensor, t_id_tensor, q_emb, entity_embs,
            num_non_text_entities, relation_embs, topic_entity_one_hot).reshape(-1)

        # Triple ranking
        sorted_triple_ids_pred = torch.argsort(
            pred_triple_logits, descending=True).cpu()
        triple_ranks_pred = torch.empty_like(sorted_triple_ids_pred)
        triple_ranks_pred[sorted_triple_ids_pred] = torch.arange(
            len(triple_ranks_pred))

        target_triple_ids = target_triple_probs.nonzero().squeeze(-1)
        num_target_triples = len(target_triple_ids)

        if num_target_triples == 0:
            continue

        num_total_entities = len(entity_embs) + num_non_text_entities
        for k in config['eval']['k_list']:
            recall_k_sample = (
                    triple_ranks_pred[target_triple_ids] < k).sum().item()
            metric_dict[f'triple_recall@{k}'].append(
                recall_k_sample / num_target_triples)

            triple_mask_k = triple_ranks_pred < k
            entity_mask_k = torch.zeros(num_total_entities)
            entity_mask_k[h_id_tensor[triple_mask_k]] = 1.
            entity_mask_k[t_id_tensor[triple_mask_k]] = 1.
            recall_k_sample_ans = entity_mask_k[a_entity_id_list].sum().item()
            metric_dict[f'ans_recall@{k}'].append(
                recall_k_sample_ans / len(a_entity_id_list))

    for key, val in metric_dict.items():
        metric_dict[key] = np.mean(val)

    return metric_dict


def train_epoch(device, train_loader, model, optimizer):
    model.train()
    epoch_loss = 0
    for sample in tqdm(train_loader):
        h_id_tensor, r_id_tensor, t_id_tensor, q_emb, entity_embs, \
            num_non_text_entities, relation_embs, topic_entity_one_hot, \
            target_triple_probs, a_entity_id_list = prepare_sample(device, sample)

        if len(h_id_tensor) == 0:
            continue

        pred_triple_logits = model(
            h_id_tensor, r_id_tensor, t_id_tensor, q_emb, entity_embs,
            num_non_text_entities, relation_embs, topic_entity_one_hot)
        target_triple_probs = target_triple_probs.to(device).unsqueeze(-1)
        loss = F.binary_cross_entropy_with_logits(
            pred_triple_logits, target_triple_probs)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss = loss.item()
        epoch_loss += loss

    epoch_loss /= len(train_loader)

    log_dict = {'loss': epoch_loss}
    return log_dict


def main(args):
    # 1. 加载配置
    config_file = f'configs/retriever/{args.dataset}.yaml'
    config = load_yaml(config_file)

    # 2. 定义 device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # 3. 设置环境
    torch.set_num_threads(config['env']['num_threads'])
    set_seed(config['env']['seed'])

    # 4. 读取配置
    use_llm_entities = config['dataset'].get('use_llm_extracted_entities', True)
    use_kg_enhanced = config['dataset'].get('use_kg_enhanced', True)
    entity_mapping_file = config['dataset'].get('entity_mapping_file', None)

    # 5. 打印配置
    print("\n" + "=" * 60)
    print("训练配置:")
    print("=" * 60)
    print(f"数据集: {config['dataset']['name']}")
    print(f"设备: {device}")
    print(f"使用 LLM 实体: {use_llm_entities}")
    print(f"使用 KG 增强嵌入: {use_kg_enhanced}")
    if entity_mapping_file:
        print(f"实体映射文件: {entity_mapping_file}")
    print("=" * 60 + "\n")

    # 6. 🔑 创建实验名称（必须在 wandb.init 之前）
    ts = time.strftime('%b%d-%H:%M:%S', time.gmtime())
    config_df = pd.json_normalize(config, sep='/')
    exp_prefix = config['train']['save_prefix']

    # 根据增强功能添加后缀
    exp_suffix = ""
    if use_llm_entities:
        exp_suffix += "_llm"
    if use_kg_enhanced:
        exp_suffix += "_kg"

    exp_name = f'{exp_prefix}{exp_suffix}_{ts}'
    print(f"实验名称: {exp_name}\n")

    # 7. 初始化 wandb
    wandb.init(
        project=f'{args.dataset}',
        name=exp_name,
        config=config_df.to_dict(orient='records')[0]
    )

    # 8. 创建实验目录
    os.makedirs(exp_name, exist_ok=True)
    print(f"✓ 实验目录已创建: {exp_name}\n")

    # 9. 加载数据集
    train_set = RetrieverDataset(
        config=config,
        split='train',
        use_llm_extracted_entities=use_llm_entities,
        entity_mapping_file=entity_mapping_file if use_llm_entities else None,
        emb_dir_suffix='_kg_enhanced' if use_kg_enhanced else None
    )
    print(f'✓ 训练集加载完成: {len(train_set)} 个样本\n')

    val_set = RetrieverDataset(
        config=config,
        split='val',
        use_llm_extracted_entities=use_llm_entities,
        entity_mapping_file=entity_mapping_file if use_llm_entities else None,
        emb_dir_suffix='_kg_enhanced' if use_kg_enhanced else None
    )
    print(f'✓ 验证集加载完成: {len(val_set)} 个样本\n')

    # 10. 创建数据加载器
    train_loader = DataLoader(
        train_set, batch_size=1, shuffle=True, collate_fn=collate_retriever)
    val_loader = DataLoader(
        val_set, batch_size=1, collate_fn=collate_retriever)

    # 11. 创建模型
    emb_size = train_set[0]['q_emb'].shape[-1]
    model = Retriever(emb_size, **config['retriever']).to(device)  # ✓ device 已定义
    optimizer = Adam(model.parameters(), **config['optimizer'])

    # 12. 训练循环
    num_patient_epochs = 0
    best_val_metric = 0

    for epoch in range(config['train']['num_epochs']):
        num_patient_epochs += 1

        val_eval_dict = eval_epoch(config, device, val_loader, model)
        target_val_metric = val_eval_dict['triple_recall@100']

        if target_val_metric > best_val_metric:
            num_patient_epochs = 0
            best_val_metric = target_val_metric
            best_state_dict = {
                'config': config,
                'model_state_dict': model.state_dict()
            }
            # ✓ exp_name 已定义
            torch.save(best_state_dict, os.path.join(exp_name, f'cpt.pth'))

            val_log = {'val/epoch': epoch}
            for key, val in val_eval_dict.items():
                val_log[f'val/{key}'] = val
            wandb.log(val_log)

        train_log_dict = train_epoch(device, train_loader, model, optimizer)

        train_log_dict.update({
            'num_patient_epochs': num_patient_epochs,
            'epoch': epoch
        })
        wandb.log(train_log_dict)

        if num_patient_epochs == config['train']['patience']:
            break

    print(f"\n✓ 训练完成！最佳模型已保存到: {exp_name}/cpt.pth")


if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser('Retriever Training with Hybrid Enhancements')
    parser.add_argument('-d', '--dataset', type=str, required=True,
                        choices=['webqsp', 'cwq'], help='Dataset name')

    # 🔑 新增命令行参数
    parser.add_argument('--use_llm_entities', action='store_true',
                        help='使用大模型提取的问题实体（覆盖配置文件）')
    parser.add_argument('--use_kg_enhanced', action='store_true',
                        help='使用KG增强的嵌入（覆盖配置文件）')
    parser.add_argument('--entity_mapping_file', type=str,
                        default=None,
                        help='实体映射文件路径（覆盖配置文件）')
    parser.add_argument('--config', type=str,
                        default="/home/shu1004/lyx/subgraph/SubgraphRAG-main/retrieve/configs/retriever/webqsp.yaml",
                        help='配置文件路径（默认：configs/retriever/{dataset}.yaml）')

    args = parser.parse_args()

    main(args)
