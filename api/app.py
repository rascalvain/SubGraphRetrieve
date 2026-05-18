"""
Flask 后端接口入口
提供子图检索 Stage2（草稿子图）和 Stage3（优化子图）端点
无缝替换原型系统中基于 LLM 模拟的 Page2PipelineService

使用方式:
    export RETRIEVER_CHECKPOINT_PATH=/path/to/cpt.pth
    export OPENAI_API_KEY=sk-xxx
    export OPENAI_BASE_URL=https://api.openai-proxy.org/v1
    python api/app.py
"""
from __future__ import annotations

import sys
import os
from pathlib import Path

# ── 路径注册（确保 api/ 本身可被 import）────────────────────
_API_DIR  = Path(__file__).resolve().parent
_BASE_DIR = _API_DIR.parent
for _p in [str(_BASE_DIR), str(_API_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from flask import Flask, jsonify, request
from flask import Response

import api.config as cfg
from api.services.retriever_service  import RetrieverService
from api.services.pog_service        import PogService
from api.services.lightprof_service  import LightprofService


app = Flask(__name__)

# ── 全局服务实例（进程启动时加载一次）──────────────────────
_retriever_svc: RetrieverService | None = None
_pog_svc:       PogService       | None = None
_lightprof_svc: LightprofService | None = None
_init_error:    str | None = None


def _init_services() -> None:
    """在 Flask 首次请求前（或启动时）初始化所有服务"""
    global _retriever_svc, _pog_svc, _lightprof_svc, _init_error

    try:
        _retriever_svc = RetrieverService(
            checkpoint_path   = cfg.RETRIEVER_CHECKPOINT_PATH,
            device            = cfg.DEVICE,
            kg_emb_path       = cfg.KG_EMB_PATH,
            entity_mapping_path = cfg.ENTITY_MAPPING_PATH,
        )
        _retriever_svc.load()
    except Exception as e:
        _init_error = str(e)
        print(f"[WARN] RetrieverService init failed: {e}")
        print("       Stage2/Stage3 will return empty results until a valid checkpoint is configured.")

    try:
        _pog_svc = PogService(
            model    = cfg.OPENAI_MODEL,
            api_key  = cfg.OPENAI_API_KEY,
            base_url = cfg.OPENAI_BASE_URL,
        )
        _pog_svc.load()
    except Exception as e:
        print(f"[WARN] PogService init failed: {e}")

    try:
        _lightprof_svc = LightprofService(
            sbert_model  = cfg.SBERT_MODEL_NAME,
            api_key      = cfg.OPENAI_API_KEY,
            base_url     = cfg.OPENAI_BASE_URL,
            model_name   = cfg.OPENAI_MODEL,
            top_k_chains = cfg.LIGHTPROF_K,
            w1           = cfg.LIGHTPROF_W1,
            max_hops     = cfg.LIGHTPROF_MAX_HOPS,
        )
        _lightprof_svc.load()
    except Exception as e:
        print(f"[WARN] LightprofService init failed: {e}")


# ── 工具函数 ──────────────────────────────────────────────

def _parse_request() -> tuple[str, list, list[str], int] | Response:
    """解析并校验请求体，返回 (question, triples, topic_entities, top_k)"""
    data = request.get_json(force=True, silent=True) or {}
    question = str(data.get("question", "")).strip()
    if not question:
        return jsonify({"error": "Missing required field: question"}), 400

    raw_triples = data.get("triples", [])
    if not isinstance(raw_triples, list) or not raw_triples:
        return jsonify({"error": "Missing or empty field: triples"}), 400

    # 兼容 [[h,r,t], ...] 和 [{"source":h,"target":t,"label":r}, ...] 两种格式
    triples: list = []
    for item in raw_triples:
        if isinstance(item, (list, tuple)) and len(item) >= 3:
            triples.append((str(item[0]), str(item[1]), str(item[2])))
        elif isinstance(item, dict):
            h = item.get("source") or item.get("h") or item.get("head", "")
            r = item.get("label")  or item.get("r") or item.get("relation", "")
            t = item.get("target") or item.get("t") or item.get("tail", "")
            if h and r and t:
                triples.append((str(h), str(r), str(t)))

    if not triples:
        return jsonify({"error": "No valid triples found in request"}), 400

    topic_entities: list[str] = data.get("topic_entities") or []
    top_k: int = int(data.get("top_k", cfg.DEFAULT_TOP_K))
    return question, triples, topic_entities, top_k


def _make_meta(source: str, llm_succeeded: bool = True,
               fallback_used: bool = False, llm_error: str | None = None) -> dict:
    """构建与原型系统格式一致的 meta 字段"""
    return {
        "source":       source,
        "model":        f"Retriever+{'GTEKGEnhanced' if cfg.KG_EMB_PATH else 'GTELargeEN_new'}",
        "llm_succeeded": llm_succeeded,
        "fallback_used": fallback_used,
        "llm_error":     llm_error,
    }


# ────────────────────────────────────────────────────────────
#  端点：健康检查
# ────────────────────────────────────────────────────────────

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "retriever_ready": _retriever_svc is not None and _retriever_svc.model is not None,
        "pog_ready":       _pog_svc is not None and _pog_svc._generator is not None,
        "lightprof_ready": _lightprof_svc is not None,
        "init_error":      _init_error,
    })


# ────────────────────────────────────────────────────────────
#  端点：Stage2 草稿子图检索
# ────────────────────────────────────────────────────────────

@app.route("/api/stage2", methods=["POST"])
def stage2():
    """
    Retriever GNN 检索 → 草稿子图

    Request JSON:
        question       (str, required)  自然语言问题
        triples        (list, required) Neo4j 查回的三元组
        topic_entities (list, optional) 主题实体（为空时自动检测）
        top_k          (int, optional)  默认 100

    Response JSON (与 page2_pipeline.stage2_draft_subgraph 格式一致):
        DDE_NODES, DDE_EDGES, DRAFT_NODES, DRAFT_LINKS, meta
    """
    parsed = _parse_request()
    if isinstance(parsed, tuple) and len(parsed) == 2 and not isinstance(parsed[0], str):
        return parsed  # error response

    question, triples, topic_entities, top_k = parsed

    if _retriever_svc is None or _retriever_svc.model is None:
        return jsonify({"error": "Retriever service not initialized. Check RETRIEVER_CHECKPOINT_PATH."}), 503

    try:
        retrieval = _retriever_svc.retrieve(
            question=question,
            triples=triples,
            topic_entities=topic_entities or None,
            top_k=top_k,
        )
        stage2_data = _retriever_svc.build_stage2(retrieval)
        stage2_data["meta"] = _make_meta("retriever")
        # 在响应中附加 retrieval_result 供 stage3 端点复用（前端可选传入）
        stage2_data["_retrieval"] = {
            "scored_triples":   retrieval["scored_triples"],
            "topic_entity_ids": retrieval["topic_entity_ids"],
            "entity_list":      retrieval["entity_list"],
        }
        return jsonify(stage2_data)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# ────────────────────────────────────────────────────────────
#  端点：Stage3 子图优化（PoG + LightPROF）
# ────────────────────────────────────────────────────────────

@app.route("/api/stage3", methods=["POST"])
def stage3():
    """
    PoG + LightPROF 优化 → 精炼行为子图

    Request JSON:
        question       (str, required)  自然语言问题
        triples        (list, required) Neo4j 三元组（与 stage2 相同）
        topic_entities (list, optional)
        top_k          (int, optional)
        stage2_data    (dict, optional) stage2 返回的 _retrieval 字段（有则复用，无则重新推理）

    Response JSON (与 page2_pipeline.stage3_optimize_subgraph 格式一致):
        CHAINS, INDICATOR_DATA, PRUNE_STEPS, FINAL_NODES, FINAL_LINKS, meta
    """
    parsed = _parse_request()
    if isinstance(parsed, tuple) and len(parsed) == 2 and not isinstance(parsed[0], str):
        return parsed

    question, triples, topic_entities, top_k = parsed
    body = request.get_json(force=True, silent=True) or {}

    if _retriever_svc is None or _retriever_svc.model is None:
        return jsonify({"error": "Retriever service not initialized."}), 503

    try:
        # ① 尝试复用 stage2 的检索结果
        stage2_cache = body.get("stage2_data", {}).get("_retrieval", {})
        if stage2_cache.get("scored_triples"):
            scored_triples  = stage2_cache["scored_triples"]
            topic_ent_ids   = stage2_cache.get("topic_entity_ids", [])
            entity_list     = stage2_cache.get("entity_list", [])
            topic_ents_text = [entity_list[i] for i in topic_ent_ids if i < len(entity_list)]
        else:
            retrieval       = _retriever_svc.retrieve(question, triples, topic_entities or None, top_k)
            scored_triples  = retrieval["scored_triples"]
            topic_ent_ids   = retrieval["topic_entity_ids"]
            entity_list     = retrieval["entity_list"]
            topic_ents_text = [entity_list[i] for i in topic_ent_ids if i < len(entity_list)]

        if topic_entities:
            topic_ents_text = topic_entities  # 用户显式提供优先

        # ② PoG 生成推理指示器
        pog_result = {"Split_q": [question], "I_LLM": {}, "pog_is_fallback": True}
        llm_succeeded = False
        llm_error = None
        if _pog_svc is not None:
            try:
                pog_result = _pog_svc.generate(question, topic_ents_text)
                llm_succeeded = not pog_result.get("pog_is_fallback", True)
            except Exception as e:
                llm_error = str(e)

        # ③ LightPROF 采样
        gr_triples: list = []
        if _lightprof_svc is not None:
            gr_triples, _ = _lightprof_svc.sample(
                question=question,
                scored_triples=scored_triples,
                topic_entities=topic_ents_text,
                i_llm=pog_result.get("I_LLM"),
            )

        # ④ 构建 Stage3 格式
        draft_links_count = len(scored_triples)
        stage3_data = LightprofService.build_stage3(
            question=question,
            gr_triples=gr_triples,
            scored_triples=scored_triples,
            pog_result=pog_result,
            draft_links_count=draft_links_count,
            topic_entities=topic_ents_text,
        )
        stage3_data["meta"] = _make_meta(
            source="retriever+lightprof" if gr_triples else "retriever-only",
            llm_succeeded=llm_succeeded,
            fallback_used=not bool(gr_triples),
            llm_error=llm_error,
        )
        return jsonify(stage3_data)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# ────────────────────────────────────────────────────────────
#  端点：Pipeline（一次性返回 stage2 + stage3）
# ────────────────────────────────────────────────────────────

@app.route("/api/pipeline", methods=["POST"])
def pipeline():
    """
    完整流水线：Retriever → PoG → LightPROF

    一次请求返回 {stage2: {...}, stage3: {...}}
    """
    parsed = _parse_request()
    if isinstance(parsed, tuple) and len(parsed) == 2 and not isinstance(parsed[0], str):
        return parsed

    question, triples, topic_entities, top_k = parsed

    if _retriever_svc is None or _retriever_svc.model is None:
        return jsonify({"error": "Retriever service not initialized."}), 503

    try:
        # Stage2
        retrieval = _retriever_svc.retrieve(question, triples, topic_entities or None, top_k)
        s2 = _retriever_svc.build_stage2(retrieval)
        s2["meta"] = _make_meta("retriever")

        scored_triples  = retrieval["scored_triples"]
        entity_list     = retrieval["entity_list"]
        topic_ent_ids   = retrieval["topic_entity_ids"]
        topic_ents_text = topic_entities or [entity_list[i] for i in topic_ent_ids if i < len(entity_list)]

        # PoG
        pog_result = {"Split_q": [question], "I_LLM": {}, "pog_is_fallback": True}
        llm_succeeded = False
        llm_error = None
        if _pog_svc is not None:
            try:
                pog_result    = _pog_svc.generate(question, topic_ents_text)
                llm_succeeded = not pog_result.get("pog_is_fallback", True)
            except Exception as e:
                llm_error = str(e)

        # LightPROF
        gr_triples: list = []
        if _lightprof_svc is not None:
            gr_triples, _ = _lightprof_svc.sample(
                question=question,
                scored_triples=scored_triples,
                topic_entities=topic_ents_text,
                i_llm=pog_result.get("I_LLM"),
            )

        # Stage3
        s3 = LightprofService.build_stage3(
            question=question,
            gr_triples=gr_triples,
            scored_triples=scored_triples,
            pog_result=pog_result,
            draft_links_count=len(s2.get("DRAFT_LINKS", [])),
            topic_entities=topic_ents_text,
        )
        s3["meta"] = _make_meta(
            source="retriever+lightprof" if gr_triples else "retriever-only",
            llm_succeeded=llm_succeeded,
            fallback_used=not bool(gr_triples),
            llm_error=llm_error,
        )

        return jsonify({"stage2": s2, "stage3": s3})

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# ────────────────────────────────────────────────────────────
#  启动入口
# ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  子图检索后端服务")
    print("=" * 60)
    print(f"  Checkpoint : {cfg.RETRIEVER_CHECKPOINT_PATH or '(未配置)'}")
    print(f"  KG Emb     : {cfg.KG_EMB_PATH or '(未配置，使用 GTELargeEN_new)'}")
    print(f"  Device     : {cfg.DEVICE}")
    print(f"  OpenAI     : {'已配置' if cfg.OPENAI_API_KEY else '(未配置，PoG/LightPROF 将使用 fallback)'}")
    print(f"  Host:Port  : {cfg.FLASK_HOST}:{cfg.FLASK_PORT}")
    print("=" * 60)

    _init_services()

    app.run(
        host=cfg.FLASK_HOST,
        port=cfg.FLASK_PORT,
        debug=cfg.FLASK_DEBUG,
    )
