import networkx as nx
import numpy as np
import os
import pickle
import torch
import torch.nn.functional as F
import json

from tqdm import tqdm


class RetrieverDataset:
    def __init__(
            self,
            config,
            split,
            skip_no_path=True,
            use_llm_extracted_entities=False,  # 新增：是否使用大模型提取的实体
            entity_mapping_file=None,  # 新增：实体映射文件路径
            emb_dir_suffix=None,  # 新增：嵌入目录后缀
            data_base_dir=None
    ):
        """
        Parameters
        ----------
        config : dict
            配置字典
        split : str
            数据集分割 (train/val/test)
        skip_no_path : bool
            是否跳过没有路径的样本
        use_llm_extracted_entities : bool
            是否使用大模型提取的问题实体 (q_entity_list)
        entity_mapping_file : str
            实体映射文件路径，格式：entity_name	entity_name_fb	freebase_mid
        """
        self.use_llm_extracted_entities = use_llm_extracted_entities
        self.emb_dir_suffix = emb_dir_suffix

        # Load entity mapping if provided
        self.entity_mapping = {}
        self.reverse_entity_mapping = {}  # 从图实体到文本实体的反向映射
        if entity_mapping_file and os.path.exists(entity_mapping_file):
            self._load_entity_mapping(entity_mapping_file)
            print(f"✓ 加载了 {len(self.entity_mapping)} 条实体映射")
        elif use_llm_extracted_entities:
            print("⚠️  警告: 启用了LLM实体提取但未提供映射文件")

        if data_base_dir:
            self.data_base_dir = data_base_dir
        elif 'data_base_dir' in config:
            self.data_base_dir = config['data_base_dir']
        else:
            self.data_base_dir = 'data_files'  # 默认值

        print(f"✓ 使用数据基础目录: {self.data_base_dir}")

        # Load pre-processed data
        dataset_name = config['dataset']['name']
        processed_dict_list = self._load_processed(dataset_name, split)

        # Extract directed shortest paths from topic entities to answer
        # entities or vice versa as weak supervision signals for triple scoring
        triple_score_dict = self._get_triple_scores(
            dataset_name, split, processed_dict_list)

        # Load pre-computed embeddings
        # emb_dict = self._load_emb(
        #     dataset_name, config['dataset']['text_encoder_name'], split)
        emb_dict = self._load_emb(
            dataset_name,
            config['dataset']['text_encoder_name'],
            split
        )

        # Put everything together
        self._assembly(
            processed_dict_list, triple_score_dict, emb_dict, skip_no_path)

    def _load_processed(self, dataset_name, split):
        """
        加载预处理的数据

        路径格式: {data_base_dir}/{dataset_name}/processed/{split}.pkl
        """
        processed_file = os.path.join(
            self.data_base_dir,  # 🔑 使用实例变量
            dataset_name,
            'processed',
            f'{split}.pkl'
        )

        print(f"加载处理数据: {processed_file}")

        if not os.path.exists(processed_file):
            raise FileNotFoundError(
                f"处理数据文件不存在: {processed_file}\n"
                f"请确认:\n"
                f"1. 数据基础目录: {self.data_base_dir}\n"
                f"2. 数据集名称: {dataset_name}\n"
                f"3. 已运行 emb.py 生成处理数据"
            )

        with open(processed_file, 'rb') as f:
            return pickle.load(f)

    def _load_entity_mapping(self, mapping_file):
        """
        加载实体映射文件

        文件格式：
        entity_name	entity_name_fb	freebase_mid
        ["实体名"]	Wikipedia名称	/m/xxxxx 或 not_linked
        """
        print(f"\n加载实体映射文件: {mapping_file}")

        with open(mapping_file, 'r', encoding='utf-8') as f:
            # 跳过表头
            header = f.readline()

            for line in f:
                if not line.strip():
                    continue

                parts = line.strip().split('\t')
                if len(parts) < 2:
                    continue

                # 解析数据集实体名
                dataset_entity_raw = parts[0]
                wiki_name = parts[1]
                freebase_mid = parts[2] if len(parts) >= 3 else None

                try:
                    dataset_entity_list = json.loads(dataset_entity_raw)
                    if isinstance(dataset_entity_list, list) and len(dataset_entity_list) > 0:
                        dataset_entity = dataset_entity_list[0]
                    else:
                        dataset_entity = dataset_entity_raw.strip('[]"')
                except json.JSONDecodeError:
                    dataset_entity = dataset_entity_raw.strip('[]"')

                # 构建映射字典
                # dataset_entity -> [wiki_name, freebase_mid]
                if wiki_name not in ['not_linked', '']:
                    self.entity_mapping[dataset_entity] = {
                        'wiki_name': wiki_name,
                        'freebase_mid': freebase_mid if freebase_mid not in ['not_linked', 'not_in_fb15k',
                                                                             None] else None
                    }

                    # 建立反向映射
                    self.reverse_entity_mapping[wiki_name] = dataset_entity
                    if freebase_mid and freebase_mid not in ['not_linked', 'not_in_fb15k']:
                        self.reverse_entity_mapping[freebase_mid] = dataset_entity

    def _map_text_entity_to_graph_id(self, text_entity, sample):
        """
        将文本实体映射到图中的实体ID

        Parameters
        ----------
        text_entity : str
            文本形式的实体
        sample : dict
            样本字典，包含 text_entity_list 和 entity2id 等

        Returns
        -------
        entity_id : int or None
            图中的实体ID，如果找不到返回 None
        """
        # 策略1: 直接匹配 text_entity_list
        if text_entity in sample.get('text_entity_list', []):
            # 找到在列表中的索引
            try:
                entity_idx = sample['text_entity_list'].index(text_entity)
                return entity_idx  # text_entity_list 的索引就是实体ID
            except ValueError:
                pass

        # 策略2: 使用映射文件
        if text_entity in self.entity_mapping:
            mapping_info = self.entity_mapping[text_entity]
            wiki_name = mapping_info['wiki_name']
            freebase_mid = mapping_info['freebase_mid']

            # 尝试用 wiki_name 匹配（下划线替换空格）
            wiki_name_with_space = wiki_name.replace('_', ' ')
            if wiki_name_with_space in sample.get('text_entity_list', []):
                entity_idx = sample['text_entity_list'].index(wiki_name_with_space)
                return entity_idx

            # 尝试用 wiki_name 直接匹配
            if wiki_name in sample.get('text_entity_list', []):
                entity_idx = sample['text_entity_list'].index(wiki_name)
                return entity_idx

            # 尝试用 freebase_mid 匹配（可能在 non_text_entity_list 中）
            if freebase_mid:
                # non_text_entity_list 的实体ID从 len(text_entity_list) 开始
                if freebase_mid in sample.get('non_text_entity_list', []):
                    non_text_idx = sample['non_text_entity_list'].index(freebase_mid)
                    return len(sample['text_entity_list']) + non_text_idx

        # 策略3: 模糊匹配（处理大小写、标点等差异）
        text_entity_lower = text_entity.lower().strip()
        for idx, entity in enumerate(sample.get('text_entity_list', [])):
            entity_lower = entity.lower().strip()
            if text_entity_lower == entity_lower:
                return idx
            # 尝试去除标点后匹配
            if text_entity_lower.replace("'", "").replace("-", " ") == entity_lower.replace("'", "").replace("-", " "):
                return idx

        return None

    def _load_emb(self, dataset_name, text_encoder_name, split):
        if self.emb_dir_suffix:
            emb_dir = f"{text_encoder_name}{self.emb_dir_suffix}"
        else:
            emb_dir = text_encoder_name

        # 🔑 使用可配置的基础路径
        file_path = os.path.join(
            self.data_base_dir, dataset_name, 'emb', emb_dir, f'{split}.pth')

        print(f"加载嵌入文件: {file_path}")

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"嵌入文件不存在: {file_path}")

        dict_file = torch.load(file_path)
        return dict_file

    def _get_triple_scores(
            self,
            dataset_name,
            split,
            processed_dict_list
    ):
        save_dir = os.path.join(
            self.data_base_dir, dataset_name, 'triple_scores')
        os.makedirs(save_dir, exist_ok=True)

        # 根据是否使用LLM实体，使用不同的缓存文件
        if self.use_llm_extracted_entities:
            save_file = os.path.join(save_dir, f'{split}_with_llm_entities.pth')
        else:
            save_file = os.path.join(save_dir, f'{split}.pth')

        if os.path.exists(save_file):
            print(f"加载缓存的三元组分数: {save_file}")
            return torch.load(save_file)

        print(f"计算三元组分数...")
        triple_score_dict = dict()

        # 统计信息
        stats = {
            'total_samples': 0,
            'samples_with_llm_entities': 0,
            'llm_entities_found': 0,
            'llm_entities_not_found': 0,
            'additional_paths_found': 0
        }

        for i in tqdm(range(len(processed_dict_list)), desc="提取路径"):
            sample_i = processed_dict_list[i]
            sample_i_id = sample_i['id']

            triple_scores_i, max_path_length_i, sample_stats = self._extract_paths_and_score(
                sample_i)

            triple_score_dict[sample_i_id] = {
                'triple_scores': triple_scores_i,
                'max_path_length': max_path_length_i
            }

            # 更新统计
            stats['total_samples'] += 1
            if sample_stats.get('has_llm_entities'):
                stats['samples_with_llm_entities'] += 1
            stats['llm_entities_found'] += sample_stats.get('llm_entities_found', 0)
            stats['llm_entities_not_found'] += sample_stats.get('llm_entities_not_found', 0)
            if sample_stats.get('additional_paths', 0) > 0:
                stats['additional_paths_found'] += 1

        torch.save(triple_score_dict, save_file)

        # 打印统计信息
        if self.use_llm_extracted_entities:
            print(f"\n{'=' * 60}")
            print("LLM实体提取统计:")
            print(f"{'=' * 60}")
            print(f"总样本数: {stats['total_samples']}")
            print(f"包含LLM实体的样本: {stats['samples_with_llm_entities']}")
            print(f"成功映射的LLM实体: {stats['llm_entities_found']}")
            print(f"未能映射的LLM实体: {stats['llm_entities_not_found']}")
            print(f"找到额外路径的样本: {stats['additional_paths_found']}")
            if stats['llm_entities_found'] > 0:
                print(
                    f"映射成功率: {stats['llm_entities_found'] / (stats['llm_entities_found'] + stats['llm_entities_not_found']) * 100:.2f}%")
            print(f"{'=' * 60}\n")

        return triple_score_dict

    def _extract_paths_and_score(
            self,
            sample
    ):
        """
        提取路径并评分

        增强功能：
        1. 使用原始的 q_entity_id_list
        2. 如果启用，使用大模型提取的 q_entity_list 中的新实体
        3. 通过映射文件将新实体映射到图中
        """
        nx_g = self._get_nx_g(
            sample['h_id_list'],
            sample['r_id_list'],
            sample['t_id_list']
        )

        # 统计信息
        sample_stats = {
            'has_llm_entities': False,
            'llm_entities_found': 0,
            'llm_entities_not_found': 0,
            'additional_paths': 0
        }

        # 收集所有问题实体ID
        q_entity_ids = list(sample['q_entity_id_list'])  # 原始问题实体

        # 如果启用LLM提取的实体
        if self.use_llm_extracted_entities and 'q_entity_list' in sample:
            llm_entities = sample['q_entity_list']

            if isinstance(llm_entities, list) and len(llm_entities) > 0:
                sample_stats['has_llm_entities'] = True

                # 获取原始问题实体的文本表示
                original_q_entities = set(sample.get('q_entity', []))

                # 找出不重合的实体
                new_entities = [e for e in llm_entities if e not in original_q_entities]

                if new_entities:
                    # 尝试映射新实体到图中
                    for text_entity in new_entities:
                        entity_id = self._map_text_entity_to_graph_id(text_entity, sample)

                        if entity_id is not None and entity_id not in q_entity_ids:
                            q_entity_ids.append(entity_id)
                            sample_stats['llm_entities_found'] += 1
                        else:
                            sample_stats['llm_entities_not_found'] += 1

        # 提取从所有问题实体到答案实体的路径
        path_list_ = []
        original_path_count = 0

        for q_entity_id in q_entity_ids:
            for a_entity_id in sample['a_entity_id_list']:
                paths_q_a = self._shortest_path(nx_g, q_entity_id, a_entity_id)
                if len(paths_q_a) > 0:
                    if q_entity_id in sample['q_entity_id_list']:
                        original_path_count += len(paths_q_a)
                    path_list_.extend(paths_q_a)

        # 计算额外找到的路径数
        sample_stats['additional_paths'] = len(path_list_) - original_path_count

        if len(path_list_) == 0:
            max_path_length = None
        else:
            max_path_length = 0

        # 将路径转换为三元组ID列表
        path_list = []

        for path in path_list_:
            num_triples_path = len(path) - 1
            max_path_length = max(max_path_length, num_triples_path)
            triples_path = []

            for i in range(num_triples_path):
                h_id_i = path[i]
                t_id_i = path[i + 1]
                triple_id_i_list = [
                    nx_g[h_id_i][t_id_i]['triple_id']
                ]
                triples_path.append(triple_id_i_list)

            path_list.append(triples_path)

        num_triples = len(sample['h_id_list'])
        triple_scores = self._score_triples(
            path_list,
            num_triples
        )

        return triple_scores, max_path_length, sample_stats

    def _get_nx_g(
            self,
            h_id_list,
            r_id_list,
            t_id_list
    ):
        nx_g = nx.DiGraph()
        num_triples = len(h_id_list)
        for i in range(num_triples):
            h_i = h_id_list[i]
            r_i = r_id_list[i]
            t_i = t_id_list[i]
            nx_g.add_edge(h_i, t_i, triple_id=i, relation_id=r_i)

        return nx_g

    def _shortest_path(
            self,
            nx_g,
            q_entity_id,
            a_entity_id
    ):
        try:
            forward_paths = list(nx.all_shortest_paths(nx_g, q_entity_id, a_entity_id))
        except:
            forward_paths = []

        try:
            backward_paths = list(nx.all_shortest_paths(nx_g, a_entity_id, q_entity_id))
        except:
            backward_paths = []

        full_paths = forward_paths + backward_paths
        if (len(forward_paths) == 0) or (len(backward_paths) == 0):
            return full_paths

        min_path_len = min([len(path) for path in full_paths])
        refined_paths = []
        for path in full_paths:
            if len(path) == min_path_len:
                refined_paths.append(path)

        return refined_paths

    def _score_triples(
            self,
            path_list,
            num_triples
    ):
        triple_scores = torch.zeros(num_triples)

        for path in path_list:
            for triple_id_list in path:
                triple_scores[triple_id_list] = 1.

        return triple_scores

    def _load_emb(self, dataset_name, text_encoder_name, split):
        # 如果有后缀，添加到路径中
        if self.emb_dir_suffix:
            emb_dir = f"{text_encoder_name}{self.emb_dir_suffix}"
        else:
            emb_dir = text_encoder_name

        file_path = f'data_files/{dataset_name}/emb/{emb_dir}/{split}.pth'

        print(f"加载嵌入文件: {file_path}")

        if not os.path.exists(file_path):
            raise FileNotFoundError(
                f"嵌入文件不存在: {file_path}\n"
                f"请确认:\n"
                f"1. 已运行 emb.py 生成嵌入\n"
                f"2. 路径配置正确\n"
                f"3. 如果使用 KG 增强，设置 emb_dir_suffix='_kg_enhanced'"
            )

        dict_file = torch.load(file_path)
        return dict_file

    def _assembly(self, processed_dict_list, triple_score_dict, emb_dict, skip_no_path):
        self.processed_dict_list = []

        # 🔑 添加调试信息
        print("\n" + "=" * 60)
        print("数据一致性检查:")
        print("=" * 60)
        print(f"处理数据样本数: {len(processed_dict_list)}")
        print(f"嵌入数据样本数: {len(emb_dict)}")

        # 检查样本ID
        processed_ids = set(s['id'] for s in processed_dict_list)
        emb_ids = set(emb_dict.keys())

        missing_in_emb = processed_ids - emb_ids
        extra_in_emb = emb_ids - processed_ids

        if missing_in_emb:
            print(f"\n⚠️  处理数据中有 {len(missing_in_emb)} 个样本在嵌入文件中缺失:")
            print(f"   示例: {list(missing_in_emb)[:10]}")

        if extra_in_emb:
            print(f"\n⚠️  嵌入文件中有 {len(extra_in_emb)} 个额外样本:")
            print(f"   示例: {list(extra_in_emb)[:10]}")

        if not missing_in_emb and not extra_in_emb:
            print("✓ 样本ID完全匹配")

        print("=" * 60 + "\n")

        # 原有代码
        num_relevant_triples = []
        num_skipped = 0
        num_missing_emb = 0  # 🔑 统计缺失嵌入的样本

        for i in tqdm(range(len(processed_dict_list)), desc="组装数据"):
            sample_i = processed_dict_list[i]
            sample_i_id = sample_i['id']
            assert sample_i_id in triple_score_dict

            triple_score_i = triple_score_dict[sample_i_id]['triple_scores']
            max_path_length_i = triple_score_dict[sample_i_id]['max_path_length']

            num_relevant_triples_i = len(triple_score_i.nonzero())
            num_relevant_triples.append(num_relevant_triples_i)

            sample_i['target_triple_probs'] = triple_score_i
            sample_i['max_path_length'] = max_path_length_i

            if skip_no_path and (max_path_length_i in [None, 0]):
                num_skipped += 1
                continue

            # 🔑 检查嵌入是否存在
            if sample_i_id not in emb_dict:
                print(f"⚠️  警告: 样本 {sample_i_id} 在嵌入文件中不存在，跳过")
                num_missing_emb += 1
                continue

            sample_i.update(emb_dict[sample_i_id])

            sample_i['a_entity'] = list(set(sample_i['a_entity']))
            sample_i['a_entity_id_list'] = list(set(sample_i['a_entity_id_list']))

            # PE for topic entities
            num_entities_i = len(sample_i['text_entity_list']) + len(sample_i['non_text_entity_list'])
            topic_entity_mask = torch.zeros(num_entities_i)
            topic_entity_mask[sample_i['q_entity_id_list']] = 1.
            topic_entity_one_hot = F.one_hot(topic_entity_mask.long(), num_classes=2)
            sample_i['topic_entity_one_hot'] = topic_entity_one_hot.float()

            self.processed_dict_list.append(sample_i)

        median_num_relevant = int(np.median(num_relevant_triples))
        mean_num_relevant = int(np.mean(num_relevant_triples))
        max_num_relevant = int(np.max(num_relevant_triples))

        print(f'# skipped samples (no path): {num_skipped}')
        print(f'# skipped samples (missing emb): {num_missing_emb}')  # 🔑 新增
        print(
            f'# relevant triples | median: {median_num_relevant} | mean: {mean_num_relevant} | max: {max_num_relevant}')

    def __len__(self):
        return len(self.processed_dict_list)

    def __getitem__(self, i):
        return self.processed_dict_list[i]


def collate_retriever(data):
    sample = data[0]

    h_id_list = sample['h_id_list']
    h_id_tensor = torch.tensor(h_id_list)

    r_id_list = sample['r_id_list']
    r_id_tensor = torch.tensor(r_id_list)

    t_id_list = sample['t_id_list']
    t_id_tensor = torch.tensor(t_id_list)

    num_non_text_entities = len(sample['non_text_entity_list'])

    return h_id_tensor, r_id_tensor, t_id_tensor, sample['q_emb'], \
        sample['entity_embs'], num_non_text_entities, sample['relation_embs'], \
        sample['topic_entity_one_hot'], sample['target_triple_probs'], sample['a_entity_id_list']