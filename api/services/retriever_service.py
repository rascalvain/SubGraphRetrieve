"""
Retriever 推理服务
封装 GNN Retriever 模型的动态图推理逻辑
严格参照 retrieve/inference_hybrid.py::evaluate_split 的推理流程
"""
import math
import sys
import os
from collections import defaultdict, OrderedDict
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

# ── 路径设置，使 retrieve/ 子模块可被导入 ──────────────────────
_BASE_DIR = Path(__file__).resolve().parents[2]  # 第一章/
_RETRIEVE_DIR = _BASE_DIR / "retrieve"
for _p in [str(_BASE_DIR), str(_RETRIEVE_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from src.model.retriever import Retriever  # noqa: E402


HOP_COLORS = {0: "#ff5252", 1: "#40c4ff", 2: "#69f0ae", 3: "#ffeb3b"}
HOP_DIST   = {0: 0.0, 1: 0.2, 2: 0.4, 3: 0.6}


class RetrieverService:
    """
    单例式检索服务，服务启动时调用 load() 初始化模型和编码器，
    之后通过 retrieve() 方法处理每条 API 请求。
    """

    def __init__(self, checkpoint_path: str = "", device: str = "cpu",
                 kg_emb_path: str = "", entity_mapping_path: str = ""):
        self.device = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
        self.checkpoint_path = checkpoint_path
        self.kg_emb_path = kg_emb_path
        self.entity_mapping_path = entity_mapping_path
        self.model: Retriever | None = None
        self.encoder = None
        self.use_kg_enhanced = bool(kg_emb_path)

    # ──────────────────────────────────────────────────────────
    #  模型加载（服务启动时调用一次）
    # ──────────────────────────────────────────────────────────

    def load(self) -> None:
        """加载 Retriever 检查点和文本编码器"""
        if not self.checkpoint_path or not Path(self.checkpoint_path).exists():
            raise FileNotFoundError(
                f"Retriever checkpoint not found: {self.checkpoint_path}\n"
                "Please set RETRIEVER_CHECKPOINT_PATH environment variable."
            )

        print(f"[RetrieverService] Loading checkpoint: {self.checkpoint_path}")
        cpt = torch.load(self.checkpoint_path, map_location="cpu")
        config = cpt["config"]

        # 初始化编码器，先加载以获取 emb_size
        self._load_encoder()
        emb_size = self._get_emb_size()

        # 构造 Retriever 模型
        retriever_cfg = config.get("retriever", {})
        self.model = Retriever(emb_size, **retriever_cfg).to(self.device)
        self.model.load_state_dict(cpt["model_state_dict"])
        self.model.eval()

        print(f"[RetrieverService] Model loaded. emb_size={emb_size}, "
              f"params={sum(p.numel() for p in self.model.parameters()):,}")

    def _load_encoder(self) -> None:
        """选择并加载文本编码器"""
        if self.use_kg_enhanced:
            try:
                from src.model.text_encoders.gte_kg_enhanced import GTEKGEnhanced
                self.encoder = GTEKGEnhanced(
                    device=self.device,
                    kg_emb_path=self.kg_emb_path,
                    entity_mapping_path=self.entity_mapping_path or None,
                )
                print("[RetrieverService] Using GTEKGEnhanced encoder")
                return
            except Exception as e:
                print(f"[RetrieverService] GTEKGEnhanced failed ({e}), falling back to GTELargeEN_new")

        from src.model.text_encoders.gte_large_en_new import GTELargeEN_new
        self.encoder = GTELargeEN_new(device=self.device)
        print("[RetrieverService] Using GTELargeEN_new encoder")

    def _get_emb_size(self) -> int:
        """通过编码一个空字符串来获取嵌入维度"""
        with torch.no_grad():
            result = self.encoder("test", ["entity"], ["relation"], ["entity"])
            # result[0] = q_emb shape (1, emb_size)
            if isinstance(result, tuple):
                return result[0].shape[-1]
            return result.shape[-1]

    # ──────────────────────────────────────────────────────────
    #  核心推理（严格参照 evaluate_split 逻辑）
    # ──────────────────────────────────────────────────────────

    @torch.no_grad()
    def retrieve(
        self,
        question: str,
        triples: list[list],
        topic_entities: list[str] | None = None,
        top_k: int = 100,
    ) -> dict[str, Any]:
        """
        对输入三元组进行相关性打分，返回 Stage2 所需的中间数据。

        Parameters
        ----------
        question       : 自然语言问题
        triples        : [(head, relation, tail), ...]  来自 Neo4j 全局图谱
        topic_entities : 问题主题实体列表（可选，为空时自动匹配）
        top_k          : 返回 Top-K 三元组数量

        Returns
        -------
        dict 包含:
            scored_triples  : [(h, r, t, score), ...]
            entity_list     : 实体列表
            relation_list   : 关系列表
            topic_entity_ids: 主题实体 ID 列表
            dde_hop_map     : {entity_name: hop_level(0-3)}
            triple_ids      : [(h_id, r_id, t_id), ...]  与 scored_triples 对齐
        """
        if not triples:
            return self._empty_result()

        # ── 步骤 1：构建动态词表 ───────────────────────────────
        entity_list, relation_list, entity_to_id, relation_to_id = \
            self._build_vocab(triples)

        # ── 步骤 2：识别主题实体 ──────────────────────────────
        if not topic_entities:
            topic_entities = self._auto_detect_topic_entities(question, entity_list)
        topic_entity_ids = [entity_to_id[e] for e in topic_entities if e in entity_to_id]
        if not topic_entity_ids:
            # 如果仍未找到，取度数最高的实体（出现次数最多）
            degree = defaultdict(int)
            for h, _, t in triples:
                degree[h] += 1
                degree[t] += 1
            top_ents = sorted(entity_list, key=lambda e: degree[e], reverse=True)[:2]
            topic_entity_ids = [entity_to_id[e] for e in top_ents]

        # ── 步骤 3：GTE 编码 ──────────────────────────────────
        enc_result = self.encoder(
            question, entity_list, relation_list,
            [entity_list[i] for i in topic_entity_ids]
        )
        # 兼容 GTELargeEN_new（4值）和 GTEKGEnhanced（5值）
        q_emb, entity_embs, relation_embs = enc_result[0], enc_result[1], enc_result[2]

        # ── 步骤 4：构造三元组 ID 张量（模拟 collate_retriever）
        h_ids = [entity_to_id[h] for h, r, t in triples]
        r_ids = [relation_to_id[r] for h, r, t in triples]
        t_ids = [entity_to_id[t] for h, r, t in triples]

        h_id_tensor = torch.LongTensor(h_ids).to(self.device)
        r_id_tensor = torch.LongTensor(r_ids).to(self.device)
        t_id_tensor = torch.LongTensor(t_ids).to(self.device)

        q_emb        = q_emb.to(self.device)
        entity_embs  = entity_embs.to(self.device)
        relation_embs = relation_embs.to(self.device)

        # ── 步骤 5：构造 topic_entity_one_hot ────────────────
        n_ent = len(entity_list)
        one_hot = torch.zeros(n_ent, 2, device=self.device)
        for eid in topic_entity_ids:
            one_hot[eid, 0] = 1.0
            one_hot[eid, 1] = 1.0

        # ── 步骤 6：前向传播（evaluate_split 核心逻辑）────────
        pred_logits = self.model(
            h_id_tensor, r_id_tensor, t_id_tensor,
            q_emb, entity_embs,
            0,            # num_non_text_entities = 0
            relation_embs,
            one_hot,
        )
        scores = torch.sigmoid(pred_logits).reshape(-1)

        # ── 步骤 7：Top-K 结果构建（evaluate_split 结果格式）──
        actual_top_k = min(top_k, len(scores))
        top_k_result = torch.topk(scores, actual_top_k)
        top_k_indices = top_k_result.indices.cpu().tolist()
        top_k_scores  = top_k_result.values.cpu().tolist()

        scored_triples = []
        triple_ids = []
        for idx, score in zip(top_k_indices, top_k_scores):
            h = entity_list[h_ids[idx]]
            r = relation_list[r_ids[idx]]
            t = entity_list[t_ids[idx]]
            scored_triples.append((h, r, t, float(score)))
            triple_ids.append((h_ids[idx], r_ids[idx], t_ids[idx]))

        # ── 步骤 8：BFS 计算实体 hop 距离（供 Stage2 可视化）──
        dde_hop_map = self._compute_hop_map(
            entity_list, topic_entity_ids, triples, entity_to_id
        )

        return {
            "scored_triples":   scored_triples,
            "entity_list":      entity_list,
            "relation_list":    relation_list,
            "topic_entity_ids": topic_entity_ids,
            "dde_hop_map":      dde_hop_map,
            "triple_ids":       triple_ids,
        }

    # ──────────────────────────────────────────────────────────
    #  辅助方法
    # ──────────────────────────────────────────────────────────

    @staticmethod
    def _build_vocab(triples: list) -> tuple:
        """从三元组构建去重有序的实体/关系列表和映射字典"""
        entity_set: OrderedDict = OrderedDict()
        relation_set: OrderedDict = OrderedDict()
        for h, r, t in triples:
            entity_set[str(h)] = None
            entity_set[str(t)] = None
            relation_set[str(r)] = None
        entity_list   = list(entity_set.keys())
        relation_list = list(relation_set.keys())
        entity_to_id   = {e: i for i, e in enumerate(entity_list)}
        relation_to_id = {r: i for i, r in enumerate(relation_list)}
        return entity_list, relation_list, entity_to_id, relation_to_id

    @staticmethod
    def _auto_detect_topic_entities(question: str, entity_list: list[str]) -> list[str]:
        """字符串包含匹配，找出出现在 question 中的实体名（不区分大小写）"""
        q_lower = question.lower()
        return [e for e in entity_list if e.lower() in q_lower]

    @staticmethod
    def _compute_hop_map(
        entity_list: list[str],
        topic_entity_ids: list[int],
        triples: list,
        entity_to_id: dict,
        max_hops: int = 3,
    ) -> dict[str, int]:
        """
        BFS 计算每个实体距离主题实体的最小跳数（0-3）
        超出 max_hops 的实体归入 hop=3
        """
        from collections import deque

        # 构建无向邻接表（id → set of neighbor ids）
        adj: dict[int, set] = defaultdict(set)
        for h, r, t in triples:
            h_id = entity_to_id.get(str(h))
            t_id = entity_to_id.get(str(t))
            if h_id is not None and t_id is not None:
                adj[h_id].add(t_id)
                adj[t_id].add(h_id)

        hop_map: dict[int, int] = {}
        queue: deque = deque()
        for eid in topic_entity_ids:
            if eid not in hop_map:
                hop_map[eid] = 0
                queue.append((eid, 0))

        while queue:
            cur, depth = queue.popleft()
            if depth >= max_hops:
                continue
            for nb in adj.get(cur, set()):
                if nb not in hop_map:
                    hop_map[nb] = depth + 1
                    queue.append((nb, depth + 1))

        # 未被 BFS 覆盖的实体归入 hop=3
        return {
            entity_list[eid]: hop_map.get(eid, max_hops)
            for eid in range(len(entity_list))
        }

    @staticmethod
    def _empty_result() -> dict[str, Any]:
        return {
            "scored_triples":   [],
            "entity_list":      [],
            "relation_list":    [],
            "topic_entity_ids": [],
            "dde_hop_map":      {},
            "triple_ids":       [],
        }

    # ──────────────────────────────────────────────────────────
    #  Stage2 格式输出辅助
    # ──────────────────────────────────────────────────────────

    def build_stage2(
        self,
        retrieval_result: dict,
        max_draft_nodes: int = 10,
    ) -> dict[str, Any]:
        """
        将 retrieve() 结果转换为 Stage2 前端格式：
        DDE_NODES, DDE_EDGES, DRAFT_NODES, DRAFT_LINKS
        """
        from api.graph_utils import (
            HOP_COLORS, HOP_DIST, _make_id, infer_type
        )

        scored   = retrieval_result["scored_triples"]
        hop_map  = retrieval_result["dde_hop_map"]
        entity_list = retrieval_result["entity_list"]
        topic_ids   = retrieval_result["topic_entity_ids"]

        # ── DDE_NODES：按 hop 层级分组 ─────────────────────────
        hop_buckets: dict[int, list[str]] = {0: [], 1: [], 2: [], 3: []}
        for ename, hop in hop_map.items():
            h = min(hop, 3)
            hop_buckets[h].append(ename)

        dde_nodes: list[dict] = []
        node_label_to_dde_idx: dict[str, int] = {}
        for hop in (0, 1, 2, 3):
            entities_at_hop = hop_buckets[hop]
            cnt = max(1, len(entities_at_hop))
            for i, ename in enumerate(entities_at_hop):
                idx = len(dde_nodes)
                node_label_to_dde_idx[ename] = idx
                dde_nodes.append({
                    "label": ename,
                    "hop":   hop,
                    "ang":   None if hop == 0 else round(2 * math.pi * i / cnt + hop * 0.31, 4),
                    "dist":  HOP_DIST[hop],
                    "col":   HOP_COLORS[hop],
                })

        # ── DDE_EDGES：有边相连的 DDE 节点对 ─────────────────
        seen_pairs: set = set()
        dde_edges: list[list[int]] = []
        for h, r, t, _s in scored:
            hi = node_label_to_dde_idx.get(h)
            ti = node_label_to_dde_idx.get(t)
            if hi is not None and ti is not None:
                pair = (min(hi, ti), max(hi, ti))
                if pair not in seen_pairs:
                    seen_pairs.add(pair)
                    dde_edges.append([hi, ti])
                if len(dde_edges) >= 24:
                    break

        # ── DRAFT_NODES：Top scored 实体 + 2 噪声节点 ─────────
        unique_entities: list[str] = []
        seen_ents: set = set()
        for h, r, t, _s in scored:
            for e in (h, t):
                if e not in seen_ents and len(unique_entities) < max_draft_nodes:
                    seen_ents.add(e)
                    unique_entities.append(e)

        used_ids: set = set()
        id_map: dict[str, str] = {}
        draft_nodes: list[dict] = []
        for e in unique_entities:
            nid = _make_id(e, used_ids)
            id_map[e] = nid
            draft_nodes.append({"id": nid, "label": e, "type": infer_type(e)})
        # 追加两个固定噪声节点
        draft_nodes.append({"id": "Noise1", "label": "Civilian Casualties", "type": "noise"})
        draft_nodes.append({"id": "Noise2", "label": "Diplomatic Talks",    "type": "noise"})
        draft_set = {n["id"] for n in draft_nodes}

        # ── DRAFT_LINKS：正常边 + 2 条噪声边 ─────────────────
        draft_links: list[dict] = []
        for h, r, t, _s in scored:
            h_id = id_map.get(h)
            t_id = id_map.get(t)
            if h_id and t_id and h_id in draft_set and t_id in draft_set:
                draft_links.append({"s": h_id, "t": t_id, "lb": str(r).upper(), "noise": False})
            if len(draft_links) >= 14:
                break
        # 追加噪声边
        normal_ids = [n["id"] for n in draft_nodes if n["type"] != "noise"]
        if len(normal_ids) >= 2:
            draft_links.append({"s": normal_ids[0], "t": "Noise1", "lb": "CAUSED", "noise": True})
            draft_links.append({"s": normal_ids[1], "t": "Noise2", "lb": "SOUGHT", "noise": True})

        return {
            "DDE_NODES":   dde_nodes,
            "DDE_EDGES":   dde_edges,
            "DRAFT_NODES": draft_nodes,
            "DRAFT_LINKS": draft_links,
        }
