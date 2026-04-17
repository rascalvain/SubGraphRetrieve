import os
import pickle

from tqdm import tqdm


class EmbInferDataset_Hybrid:
    def __init__(
            self,
            raw_set,
            entity_identifiers,
            save_path,
            skip_no_topic=False,
            skip_no_ans=False,
            use_q_topic_entities=True  # 新增参数
    ):
        """
        Parameters
        ----------
        entity_identifiers : set
            Set of entity identifiers
        skip_no_topic : bool
            Whether to skip samples without topic entities in the graph
        skip_no_ans : bool
            Whether to skip samples without answer entities in the graph
        use_q_topic_entities : bool
            是否使用问题主题实体字段（q_entity_list）
        """
        self.use_q_topic_entities = use_q_topic_entities

        self.processed_dict_list = self._process(
            raw_set,
            entity_identifiers,
            save_path)

        self.skip_no_topic = skip_no_topic
        self.skip_no_ans = skip_no_ans

        processed_dict_list = []
        for processed_dict_i in self.processed_dict_list:
            if (len(processed_dict_i['q_entity_id_list']) == 0) and skip_no_topic:
                continue

            if (len(processed_dict_i['a_entity_id_list']) == 0) and skip_no_ans:
                continue

            processed_dict_list.append(processed_dict_i)
        self.processed_dict_list = processed_dict_list

        print(f'# raw samples: {len(raw_set)} | # processed samples: {len(self.processed_dict_list)}')

    def _process(
            self,
            raw_set,
            entity_identifiers,
            save_path
    ):
        if os.path.exists(save_path):
            with open(save_path, 'rb') as f:
                return pickle.load(f)

        processed_dict_list = []
        for i in tqdm(range(len(raw_set)), desc="处理数据集"):
            sample_i = raw_set[i]
            processed_dict_i = self._process_sample(
                sample_i,
                entity_identifiers)
            processed_dict_list.append(processed_dict_i)

        with open(save_path, 'wb') as f:
            pickle.dump(processed_dict_list, f)

        return processed_dict_list

    def _process_sample(
            self,
            sample,
            entity_identifiers
    ):
        # Model input (0) question
        question = sample['question']

        triples = sample['graph']

        all_entities = set()
        all_relations = set()
        for (h, r, t) in triples:
            all_entities.add(h)
            all_relations.add(r)
            all_entities.add(t)

        # Sort for deterministic entity IDs
        entity_list = sorted(all_entities)

        # Model input (1) text of entities
        #             (2) number of entities without text
        text_entity_list = []
        non_text_entity_list = []
        for entity in entity_list:
            if entity in entity_identifiers:
                non_text_entity_list.append(entity)
            else:
                text_entity_list.append(entity)

        # Create entity IDs
        entity2id = dict()
        entity_id = 0
        for entity in text_entity_list:
            entity2id[entity] = entity_id
            entity_id += 1
        for entity in non_text_entity_list:
            entity2id[entity] = entity_id
            entity_id += 1

        # Model input (3) text of relations
        relation_list = sorted(all_relations)

        # Create relation IDs
        rel2id = dict()
        rel_id = 0
        for rel in relation_list:
            rel2id[rel] = rel_id
            rel_id += 1

        # Convert triples to entity and relation IDs
        h_id_list = []
        r_id_list = []
        t_id_list = []
        for (h, r, t) in triples:
            h_id_list.append(entity2id[h])
            r_id_list.append(rel2id[r])
            t_id_list.append(entity2id[t])

        # Model input (5) list of question entity IDs
        q_entity_id_list = []
        for entity in sample['q_entity']:
            if entity in entity2id:
                q_entity_id_list.append(entity2id[entity])

        # Prepare output labels
        assert sample['a_entity'] == sample['answer']
        a_entity_id_list = []
        for entity in sample['a_entity']:
            entity_id = entity2id.get(entity, None)
            if entity_id is not None:
                a_entity_id_list.append(entity_id)

        # ========== 新增：处理问题主题实体 ==========
        q_entity_list = []
        if self.use_q_topic_entities and 'q_entity_list' in sample:
            # 从 extract_quest_ent.py 添加的字段中读取
            q_entity_list = sample['q_entity_list']
            if not isinstance(q_entity_list, list):
                q_entity_list = []

        processed_dict = {
            'id': sample['id'],
            'question': question,
            'q_entity': sample['q_entity'],
            'q_entity_id_list': q_entity_id_list,
            'text_entity_list': text_entity_list,
            'non_text_entity_list': non_text_entity_list,
            'relation_list': relation_list,
            'h_id_list': h_id_list,
            'r_id_list': r_id_list,
            't_id_list': t_id_list,
            'a_entity': sample['a_entity'],
            'a_entity_id_list': a_entity_id_list,
            'q_entity_list': q_entity_list  # 新增
        }

        return processed_dict

    def __len__(self):
        return len(self.processed_dict_list)

    def __getitem__(self, i):
        sample = self.processed_dict_list[i]

        # ========== 添加调试（仅前3个样本） ==========
        if i < 3 and self.use_q_topic_entities:
            print(f"\n样本 {i} 调试信息:")
            print(f"  text_entity_list: {sample['text_entity_list'][:3]}...")  # 前3个
            print(f"  实体类型: {type(sample['text_entity_list'][0]) if sample['text_entity_list'] else 'empty'}")

        return {
            'id': sample['id'],
            'question': sample['question'],
            'text_entity_list': sample['text_entity_list'],
            'relation_list': sample['relation_list'],
            'q_entity_list': sample.get('q_entity_list', [])
        }