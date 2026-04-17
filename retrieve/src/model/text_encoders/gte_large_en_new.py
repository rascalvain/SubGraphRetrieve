import torch
import torch.nn.functional as F

from transformers import AutoModel, AutoTokenizer


class GTELargeEN_new:
    def __init__(self,
                 device,
                 normalize=True):
        self.device = device
        # model_path = 'Alibaba-NLP/gte-large-en-v1.5'
        model_path = '/home/shu1004/lyx/subgraph/pretrain_model'
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(
            model_path,
            trust_remote_code=True,
            unpad_inputs=True,
            use_memory_efficient_attention=True).to(device)
        self.normalize = normalize

    @torch.no_grad()
    def embed(self, text_list):
        if len(text_list) == 0:
            return torch.zeros(0, 1024)

        batch_dict = self.tokenizer(
            text_list, max_length=8192, padding=True,
            truncation=True, return_tensors='pt').to(self.device)

        outputs = self.model(**batch_dict).last_hidden_state
        emb = outputs[:, 0]

        if self.normalize:
            emb = F.normalize(emb, p=2, dim=1)

        return emb.cpu()

    def __call__(self, q_text, text_entity_list, relation_list, q_entity_list=None):
        """
        编码问题、实体、关系和问题主题实体

        Parameters
        ----------
        q_text : str
            问题文本
        text_entity_list : list
            实体列表
        relation_list : list
            关系列表
        q_entity_list : list, optional
            问题主题实体列表（新增）
        """
        q_emb = self.embed([q_text])
        entity_embs = self.embed(text_entity_list)
        relation_embs = self.embed(relation_list)

        # 计算问题主题实体嵌入
        if q_entity_list is not None and len(q_entity_list) > 0:
            q_topic_embs = self.embed(q_entity_list)
        else:
            q_topic_embs = torch.zeros(0, 1024)

        return q_emb, entity_embs, relation_embs, q_topic_embs
