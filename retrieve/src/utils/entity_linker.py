import pickle
import torch
import torch.nn.functional as F
import json
from typing import Dict, Tuple, Optional


class EntityLinker:
    """实体链接器：将数据集实体链接到知识图谱实体"""

    def __init__(
            self,
            kg_emb_path: str,
            entity_mapping_path: Optional[str] = None,
            similarity_threshold: float = 0.7,
            use_mapping_file: bool = True
    ):
        """
        Parameters
        ----------
        kg_emb_path : str
            KG嵌入文件路径 (pkl格式)
        entity_mapping_path : str, optional
            实体映射文件路径 (格式: ["entity"]\tfb_entity)
        similarity_threshold : float
            文本相似度阈值，低于此值标记为 not linked
        use_mapping_file : bool
            是否使用映射文件，False则使用字符串匹配
        """
        self.similarity_threshold = similarity_threshold
        self.use_mapping_file = use_mapping_file

        # 加载KG嵌入
        print(f"加载KG嵌入: {kg_emb_path}")
        with open(kg_emb_path, 'rb') as f:
            kg_data = pickle.load(f)

        self.kg_embeddings = torch.from_numpy(kg_data['entity_embeddings']).float()
        self.entity2id = kg_data['entity2id']
        self.id2entity = kg_data['id2entity']
        self.entity_emb_dict = kg_data['entity_emb_dict']

        print(f"  ✓ 加载了 {len(self.entity2id)} 个KG实体嵌入")
        print(f"  ✓ KG嵌入维度: {self.kg_embeddings.shape[1]}")

        # ========== 添加调试 ==========
        print(f"\nKG实体格式示例（前5个）:")
        for i, entity in enumerate(list(self.entity2id.keys())[:5]):
            print(f"  {i + 1}. '{entity}'")

        # 加载映射文件（如果提供）
        self.entity_mapping = {}
        if use_mapping_file and entity_mapping_path:
            self._load_mapping(entity_mapping_path)

    def _load_mapping(self, mapping_path: str):
        """
        加载实体映射文件（新格式：包含 Freebase MID）

        文件格式：
        entity_name	entity_name_fb	freebase_mid
        ["实体名"]	Wikipedia名称	/m/xxxxx
        """
        print(f"加载实体映射文件: {mapping_path}")

        linked_count = 0
        not_linked_count = 0
        mid_found_count = 0

        with open(mapping_path, 'r', encoding='utf-8') as f:
            # 跳过表头
            header = f.readline()

            for line_num, line in enumerate(f, start=2):
                if not line.strip():
                    continue

                parts = line.strip().split('\t')
                if len(parts) != 3:  # 新格式：3列
                    print(f"  警告: 第{line_num}行格式错误: {line.strip()}")
                    continue

                dataset_entity_raw, wiki_name, freebase_mid = parts

                # 解析JSON数组格式的实体名
                try:
                    dataset_entity_list = json.loads(dataset_entity_raw)
                    if isinstance(dataset_entity_list, list) and len(dataset_entity_list) > 0:
                        dataset_entity = dataset_entity_list[0]
                    else:
                        dataset_entity = dataset_entity_raw.strip('[]"')
                except json.JSONDecodeError:
                    dataset_entity = dataset_entity_raw.strip('[]"')

                # 只保存成功找到 Freebase MID 的映射
                if freebase_mid not in ['not_linked', 'not_in_fb15k']:
                    # 使用 Freebase MID 作为映射值（这是关键！）
                    self.entity_mapping[dataset_entity] = freebase_mid
                    linked_count += 1
                    mid_found_count += 1
                elif wiki_name != 'not_linked':
                    linked_count += 1
                else:
                    not_linked_count += 1

        total = linked_count + not_linked_count
        print(f"  ✓ 总共处理 {total} 个实体")
        print(f"  ✓ 成功链接到Wikipedia: {linked_count} ({linked_count / total * 100:.2f}%)")
        print(f"  ✓ 找到Freebase MID: {mid_found_count} ({mid_found_count / total * 100:.2f}%)")
        print(f"  ✓ 可用于KG增强: {mid_found_count}")
        print(f"  ✗ 未链接: {not_linked_count} ({not_linked_count / total * 100:.2f}%)")

    def link_entity(self, entity_name: str) -> Tuple[Optional[torch.Tensor], bool, str]:
        """
        链接单个实体到KG
        """
        # 策略1：使用映射文件（现在映射到 Freebase MID）
        if self.use_mapping_file and entity_name in self.entity_mapping:
            freebase_mid = self.entity_mapping[entity_name]  # 这是 Freebase MID 如 /m/xxxxx

            # 直接用 MID 查找 KG 嵌入
            if freebase_mid in self.entity_emb_dict:
                kg_emb = torch.from_numpy(self.entity_emb_dict[freebase_mid]).float()
                return kg_emb, True, 'linked'
            else:
                return None, False, 'not_in_kg'

        # 策略2：直接字符串匹配（保留作为后备）
        if entity_name in self.entity_emb_dict:
            kg_emb = torch.from_numpy(self.entity_emb_dict[entity_name]).float()
            return kg_emb, True, 'linked'

        # 未找到链接
        return None, False, 'not_linked'

    def get_kg_embedding_dim(self) -> int:
        """获取KG嵌入维度"""
        return self.kg_embeddings.shape[1]

    def get_stats(self) -> Dict:
        """获取链接统计信息"""
        return {
            'total_kg_entities': len(self.entity2id),
            'total_mappings': len(self.entity_mapping),
            'kg_embedding_dim': self.kg_embeddings.shape[1]
        }