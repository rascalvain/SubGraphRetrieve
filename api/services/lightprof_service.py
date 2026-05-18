"""
LightPROF 推理子图采样服务
封装 LightPROF_Sampler，结合 PoG 指示器精炼检索到的草稿子图
"""
import sys
from pathlib import Path
from typing import Any

# ── 路径设置 ────────────────────────────────────────────────
_BASE_DIR = Path(__file__).resolve().parents[2]
_LIGHTPROF_DIR = _BASE_DIR / "lightprof"
for _p in [str(_BASE_DIR), str(_LIGHTPROF_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)


class LightprofService:
    """
    LightPROF 推理子图采样服务
    输入：scored_triples + PoG 指示器 + 问题文本
    输出：精炼后的 lightprof_gr_triples，及 Stage3 前端格式数据
    """

    def __init__(
        self,
        sbert_model:  str = "all-MiniLM-L6-v2",
        api_key:      str = "",
        base_url:     str = "",
        model_name:   str = "",
        use_mock_llm: bool = False,
        top_k_chains: int = 5,
        w1:           int = 80,
        max_hops:     int = 3,
        verbose:      bool = False,
    ):
        self.sbert_model  = sbert_model
        self.api_key      = api_key
        self.base_url     = base_url
        self.model_name   = model_name
        self.use_mock_llm = use_mock_llm
        self.top_k_chains = top_k_chains
        self.w1           = w1
        self.max_hops     = max_hops
        self.verbose      = verbose
        self._fuzzy_selector = None
        self._llm_scorer     = None

    def load(self) -> None:
        """预初始化 FuzzySelector 和 LLMScorer（可选）"""
        from lightprof_sampling import FuzzySelector, LLMScorer

        # FuzzySelector 使用本地 SBERT，加载失败自动跳过
        try:
            self._fuzzy_selector = FuzzySelector(sbert_model=self.sbert_model)
            print(f"[LightprofService] FuzzySelector loaded ({self.sbert_model})")
        except Exception as e:
            print(f"[LightprofService] FuzzySelector skipped: {e}")
            self._fuzzy_selector = None

        # LLMScorer：有 api_key 且不强制 mock 时才启用
        if not self.use_mock_llm and self.api_key:
            try:
                self._llm_scorer = LLMScorer(
                    model_name=self.model_name or None,
                    api_key=self.api_key,
                    base_url=self.base_url or None,
                )
                print("[LightprofService] LLMScorer initialized")
            except Exception as e:
                print(f"[LightprofService] LLMScorer failed: {e}, using mock")
                self._llm_scorer = None
        else:
            print("[LightprofService] Using mock LLM scorer")

    def sample(
        self,
        question:        str,
        scored_triples:  list,
        topic_entities:  list[str],
        i_llm:           dict | None = None,
    ) -> tuple[list, dict]:
        """
        运行 LightPROF 采样流程

        Parameters
        ----------
        question        : 自然语言问题
        scored_triples  : [(h, r, t, score), ...]  来自 RetrieverService
        topic_entities  : 主题实体列表
        i_llm           : PoG 推理指示器 {R_IA, D_predict, List_T}

        Returns
        -------
        (lightprof_gr_triples, stats)
            lightprof_gr_triples : [(h, r, t), ...]
            stats                : {num_chains_total, num_chains_after_fuzzy, ...}
        """
        from lightprof_sampling import SimpleKG, LightPROF_Sampler

        if not scored_triples:
            return [], {}

        # 构建 SimpleKG
        kg = SimpleKG(scored_triples, bidirectional=True)

        # 构建 LightPROF_Sampler
        sampler = LightPROF_Sampler(
            kg=kg,
            llm_scorer=self._llm_scorer,
            fuzzy_selector=self._fuzzy_selector,
            verbose=self.verbose,
        )

        # 提取 I_LLM 参数
        i_llm_text  = (i_llm or {}).get("R_IA", "")
        d_predict   = (i_llm or {}).get("D_predict", 2)
        h_q         = min(max(int(d_predict), 1), self.max_hops)

        try:
            gr_triples, stats = sampler.run_pipeline(
                question=question,
                anchor_entities=topic_entities,
                h_q=h_q,
                top_k_chains=self.top_k_chains,
                max_paths_per_chain=10,
                i_llm=i_llm_text or None,
                w1=self.w1,
            )
            return gr_triples, stats
        except Exception as e:
            print(f"[LightprofService] run_pipeline failed: {e}")
            # Fallback：直接返回 scored_triples 中分数最高的三元组
            fallback = [(h, r, t) for h, r, t, s in scored_triples[:50]]
            return fallback, {"error": str(e)}

    # ──────────────────────────────────────────────────────────
    #  Stage3 格式输出辅助
    # ──────────────────────────────────────────────────────────

    @staticmethod
    def build_stage3(
        question:           str,
        gr_triples:         list,
        scored_triples:     list,
        pog_result:         dict,
        draft_links_count:  int = 0,
        topic_entities:     list[str] | None = None,
    ) -> dict[str, Any]:
        """
        将 LightPROF + PoG 结果转换为 Stage3 前端格式：
        CHAINS, INDICATOR_DATA, PRUNE_STEPS, FINAL_NODES, FINAL_LINKS
        """
        from api.graph_utils import (
            _make_id, infer_type,
            build_adj, find_paths, score_path,
            build_entity_chain, extract_relation_chain,
            build_prune_steps,
        )

        # 若 gr_triples 为空，回退到 scored_triples 前 50 条
        use_triples = gr_triples if gr_triples else [(h, r, t) for h, r, t, s in scored_triples[:50]]

        # 构建 id_map
        unique_ents: list[str] = []
        seen: set = set()
        for item in use_triples:
            for e in (str(item[0]), str(item[2])):
                if e not in seen:
                    seen.add(e)
                    unique_ents.append(e)

        used_ids: set = set()
        id_map: dict[str, str] = {}
        for e in unique_ents:
            id_map[e] = _make_id(e, used_ids)

        # ── CHAINS：BFS 提取 2-3 跳路径 ───────────────────────
        adj = build_adj([(str(i[0]), str(i[1]), str(i[2])) for i in use_triples])
        anchors = topic_entities[:3] if topic_entities else unique_ents[:2]
        raw_paths = find_paths(anchors, adj, max_hops=3, max_paths=30)
        raw_paths.sort(key=lambda p: score_path(p), reverse=True)

        chains: list[dict] = []
        for i, p in enumerate(raw_paths[:6]):
            s = score_path(p)
            chains.append({
                "hops":  len(p["rels"]),
                "hi":    i < 3 or s >= 0.9,
                "score": round(s, 2),
                "nodes": p["nodes"],
                "rels":  p["rels"],
            })

        # ── INDICATOR_DATA：来自 PoG ──────────────────────────
        split_q    = pog_result.get("Split_q", [question])
        i_llm_data = pog_result.get("I_LLM", {})
        list_t     = i_llm_data.get("List_T", [])
        r_ia       = i_llm_data.get("R_IA",   "")
        d_predict  = int(i_llm_data.get("D_predict", 3))

        indicator_data = {
            "splitQuestions": [
                {"id": f"Q{i+1}", "text": q}
                for i, q in enumerate(split_q[:2])
            ],
            "entityChain":   build_entity_chain(list_t, r_ia),
            "relationChain": extract_relation_chain(
                [(str(i[0]), str(i[1]), str(i[2])) for i in use_triples], top_n=5
            ),
            "dPredict": d_predict,
        }

        # ── FINAL_NODES / FINAL_LINKS ─────────────────────────
        score_lookup: dict = {}
        for h, r, t, s in scored_triples:
            score_lookup[(str(h), str(r), str(t))] = float(s)

        final_links_raw = []
        for item in use_triples:
            h, r, t = str(item[0]), str(item[1]), str(item[2])
            w = score_lookup.get((h, r, t), 0.8)
            if w >= 0.76 and h in id_map and t in id_map:
                final_links_raw.append({"s": id_map[h], "t": id_map[t],
                                        "lb": r.upper(), "w": round(w, 2)})

        # 按 w 降序，最多 6 条
        final_links = sorted(final_links_raw, key=lambda x: x["w"], reverse=True)[:6]

        # 收集 FINAL_NODES 中用到的实体
        final_ids: list[str] = []
        for lnk in final_links:
            for fid in (lnk["s"], lnk["t"]):
                if fid not in final_ids:
                    final_ids.append(fid)

        # 反查 id → label
        id_to_label = {v: k for k, v in id_map.items()}
        final_nodes = [
            {"id": fid, "label": id_to_label.get(fid, fid), "type": infer_type(id_to_label.get(fid, fid))}
            for fid in final_ids
        ]

        # ── PRUNE_STEPS ───────────────────────────────────────
        non_noise = len([lnk for lnk in final_links_raw if not lnk.get("noise", False)])
        prune_steps = build_prune_steps(
            num_draft_links=draft_links_count or len(use_triples),
            num_non_noise_links=non_noise,
            num_final_links=len(final_links),
        )

        return {
            "CHAINS":         chains,
            "INDICATOR_DATA": indicator_data,
            "PRUNE_STEPS":    prune_steps,
            "FINAL_NODES":    final_nodes,
            "FINAL_LINKS":    final_links,
        }
