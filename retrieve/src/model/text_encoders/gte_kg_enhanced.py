import torch
import torch.nn.functional as F
from typing import List, Tuple, Dict
from .gte_large_en_new import GTELargeEN_new
from src.utils.entity_linker import EntityLinker


class GTEKGEnhanced:
    """
    增强版文本编码器：融合文本嵌入和KG嵌入
    """

    def __init__(
            self,
            device,
            kg_emb_path: str,
            entity_mapping_path: str = None,
            fusion_method: str = 'weighted',
            text_weight: float = 0.7,
            kg_weight: float = 0.3,
            similarity_threshold: float = 0.7,
            projection_dim: int = 1024,
            normalize: bool = True
    ):
        """
        Parameters
        ----------
        device : torch.device
            设备
        kg_emb_path : str
            KG嵌入文件路径
        entity_mapping_path : str
            实体映射文件路径
        fusion_method : str
            融合方法: 'weighted', 'concat', 'gated'
        text_weight : float
            文本嵌入权重 (用于weighted方法)
        kg_weight : float
            KG嵌入权重 (用于weighted方法)
        similarity_threshold : float
            链接相似度阈值
        projection_dim : int
            投影维度（用于concat方法）
        normalize : bool
            是否归一化最终嵌入
        """
        self.device = device
        self.fusion_method = fusion_method
        self.text_weight = text_weight
        self.kg_weight = kg_weight
        self.normalize = normalize

        # 初始化文本编码器
        self.text_encoder = GTELargeEN_new(device, normalize=False)

        # 初始化实体链接器
        self.entity_linker = EntityLinker(
            kg_emb_path=kg_emb_path,
            entity_mapping_path=entity_mapping_path,
            similarity_threshold=similarity_threshold,
            use_mapping_file=(entity_mapping_path is not None)
        )

        # 文本嵌入维度 (GTE-Large: 1024)
        self.text_dim = 1024
        # KG嵌入维度 (RotatE: 2048 for complex embeddings)
        self.kg_dim = self.entity_linker.get_kg_embedding_dim()

        # 根据融合方法初始化投影层
        if fusion_method == 'concat':
            # 拼接后投影到目标维度
            self.projection = torch.nn.Linear(
                self.text_dim + self.kg_dim,
                projection_dim
            ).to(device)
            self.output_dim = projection_dim
        elif fusion_method == 'gated':
            # 门控机制
            self.gate = torch.nn.Sequential(
                torch.nn.Linear(self.text_dim + self.kg_dim, 1),
                torch.nn.Sigmoid()
            ).to(device)
            # KG嵌入投影到文本维度
            self.kg_projection = torch.nn.Linear(
                self.kg_dim,
                self.text_dim
            ).to(device)
            self.output_dim = self.text_dim
        elif fusion_method == 'weighted':
            # 加权融合，需要将KG嵌入投影到文本维度
            self.kg_projection = torch.nn.Linear(
                self.kg_dim,
                self.text_dim
            ).to(device)
            self.output_dim = self.text_dim
        else:
            raise ValueError(f"Unknown fusion method: {fusion_method}")

        # 统计信息
        self.stats = {
            'total_entities': 0,
            'linked_entities': 0,
            'not_linked_entities': 0
        }

    def _fuse_embeddings(
            self,
            text_emb: torch.Tensor,
            kg_emb: torch.Tensor
    ) -> torch.Tensor:
        """
        融合文本嵌入和KG嵌入

        Parameters
        ----------
        text_emb : torch.Tensor
            文本嵌入 (batch_size, text_dim)
        kg_emb : torch.Tensor
            KG嵌入 (batch_size, kg_dim)

        Returns
        -------
        fused_emb : torch.Tensor
            融合后的嵌入
        """
        text_emb = text_emb.to(self.device)
        kg_emb = kg_emb.to(self.device)

        if self.fusion_method == 'weighted':
            # 加权融合
            kg_emb_projected = self.kg_projection(kg_emb)
            fused_emb = self.text_weight * text_emb + self.kg_weight * kg_emb_projected

        elif self.fusion_method == 'concat':
            # 拼接后投影
            concatenated = torch.cat([text_emb, kg_emb], dim=-1)
            fused_emb = self.projection(concatenated)

        elif self.fusion_method == 'gated':
            # 门控融合
            kg_emb_projected = self.kg_projection(kg_emb)
            gate_input = torch.cat([text_emb, kg_emb], dim=-1)
            gate_value = self.gate(gate_input)
            fused_emb = gate_value * text_emb + (1 - gate_value) * kg_emb_projected

        return fused_emb

    def embed_entities(
            self,
            text_entity_list: List[str]
    ) -> Tuple[torch.Tensor, Dict]:
        """
        编码实体列表，支持KG增强

        Returns
        -------
        entity_embs : torch.Tensor
            实体嵌入矩阵 (num_entities, output_dim)
        link_info : dict
            链接信息统计
        """
        if len(text_entity_list) == 0:
            return torch.zeros(0, self.output_dim), {}

        # 1. 获取文本嵌入
        text_embs = self.text_encoder.embed(text_entity_list)

        # 2. 尝试链接每个实体并融合
        enhanced_embs = []
        link_statuses = []

        for i, entity_name in enumerate(text_entity_list):
            text_emb = text_embs[i:i + 1]  # (1, text_dim)

            # 尝试链接到KG
            kg_emb, is_linked, status = self.entity_linker.link_entity(entity_name)

            self.stats['total_entities'] += 1

            if is_linked and kg_emb is not None:
                # 链接成功，融合嵌入
                kg_emb = kg_emb.unsqueeze(0)  # (1, kg_dim)
                fused_emb = self._fuse_embeddings(text_emb, kg_emb)
                enhanced_embs.append(fused_emb.cpu())
                link_statuses.append('linked')
                self.stats['linked_entities'] += 1
            else:
                # 未链接，只使用文本嵌入
                if self.fusion_method in ['weighted', 'gated']:
                    # 保持维度一致
                    enhanced_embs.append(text_emb)
                elif self.fusion_method == 'concat':
                    # 需要通过投影层保持维度一致
                    zero_kg = torch.zeros(1, self.kg_dim).to(self.device)
                    fused_emb = self._fuse_embeddings(text_emb, zero_kg)
                    enhanced_embs.append(fused_emb.cpu())
                link_statuses.append('not_linked')
                self.stats['not_linked_entities'] += 1

        # 3. 拼接所有嵌入
        entity_embs = torch.cat(enhanced_embs, dim=0)

        # 4. 归一化
        if self.normalize:
            entity_embs = F.normalize(entity_embs, p=2, dim=1)

        # 5. 返回链接信息
        link_info = {
            'link_statuses': link_statuses,
            'num_linked': sum(1 for s in link_statuses if s == 'linked'),
            'num_not_linked': sum(1 for s in link_statuses if s == 'not_linked'),
            'link_rate': sum(1 for s in link_statuses if s == 'linked') / len(link_statuses) if link_statuses else 0
        }

        return entity_embs, link_info

    # 在 __call__ 方法中添加对问题主题实体的处理

    def __call__(self, q_text: str, text_entity_list: List[str],
                 relation_list: List[str], q_entity_list: List[str] = None):
        """编码问题、实体、关系和问题主题实体"""

        # 问题用文本编码
        q_emb = self.text_encoder.embed([q_text])

        # 实体用增强编码
        entity_embs, link_info = self.embed_entities(text_entity_list)

        # 关系用文本编码
        relation_embs = self.text_encoder.embed(relation_list)

        # 问题主题实体用增强编码
        if q_entity_list is not None and len(q_entity_list) > 0:
            q_topic_embs, q_topic_link_info = self.embed_entities(q_entity_list)
            # 合并链接信息
            link_info['q_topic_link_info'] = q_topic_link_info
        else:
            q_topic_embs = torch.zeros(0, self.output_dim)

        return q_emb, entity_embs, relation_embs, q_topic_embs, link_info

    def print_stats(self):
        """打印统计信息"""
        total = self.stats['total_entities']
        linked = self.stats['linked_entities']
        not_linked = self.stats['not_linked_entities']

        print("\n" + "=" * 60)
        print("实体链接统计")
        print("=" * 60)
        print(f"总实体数: {total}")
        print(f"已链接: {linked} ({linked / total * 100:.2f}%)")
        print(f"未链接: {not_linked} ({not_linked / total * 100:.2f}%)")
        print("=" * 60 + "\n")