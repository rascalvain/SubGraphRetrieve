"""
图谱工具函数
提供节点 ID 生成、类型推断、可视化辅助等功能
工具函数参照原型系统 page2_pipeline.py，与前端数据格式保持一致
"""
import math
import re
from collections import Counter, defaultdict
from typing import Any

# ── 前端可视化常量（与原型系统保持一致）────────────────────────
HOP_COLORS = {0: "#ff5252", 1: "#40c4ff", 2: "#69f0ae", 3: "#ffeb3b"}
HOP_DIST   = {0: 0.0, 1: 0.2, 2: 0.4, 3: 0.6}

VALID_TYPES = {"actor", "weapon", "country", "org", "person", "location", "noise"}

TYPE_HINTS: dict[str, list[str]] = {
    "weapon":   ["atacms", "missile", "weapon", "air defense", "tomahawk", "防空", "导弹", "武器"],
    "country":  ["russia", "ukraine", "usa", "united states", "俄罗斯", "乌克兰", "美国", "government", "政府"],
    "org":      ["fleet", "nato", "eu", "command", "administration", "force", "军队", "舰队", "欧盟", "北约"],
    "location": ["kyiv", "moscow", "crimea", "sevastopol", "novorossiysk", "voronezh", "海", "港", "city"],
    "person":   ["trump", "putin", "zelensky", "rubio", "biden", "harris", "普京", "泽连斯基"],
}


def infer_type(label: str) -> str:
    """按关键词推断实体类型，与原型系统保持一致"""
    low = str(label).lower()
    for t in ("weapon", "person", "country", "org", "location"):
        if any(k in low for k in TYPE_HINTS[t]):
            return t
    if re.match(r"^[A-Z][a-z]+(?: [A-Z][a-z]+){1,2}$", str(label)):
        return "person"
    return "actor"


def _safe_type(v: Any) -> str:
    t = str(v or "").strip().lower()
    return t if t in VALID_TYPES else infer_type(t)


def _make_id(label: str, used: set) -> str:
    """从实体名生成唯一合法的节点 ID"""
    base = re.sub(r"[^A-Za-z0-9]+", "", label)[:22]
    if not base:
        base = f"N{abs(hash(label)) % 10000000}"
    if base[0].isdigit():
        base = f"N{base}"
    cand = base
    i = 2
    while cand in used:
        cand = f"{base[:18]}{i}"
        i += 1
    used.add(cand)
    return cand


def _find_id(node_map: dict, label: str) -> str | None:
    """在 node_map 中按 label 找到对应 id"""
    for nid, n in node_map.items():
        if n.get("label") == label:
            return nid
    return None


def build_entity_chain(list_t: list, r_ia: str) -> list[dict]:
    """
    根据 PoG 输出构建 entityChain
    交替出现 entity（蓝色）和 rel（灰色），最后追加 answer 节点
    """
    chain: list[dict] = []
    if not list_t and not r_ia:
        return [{"text": "[Predicted Behavior]", "type": "answer"}]

    # 从 r_ia 中提取关系词（大写下划线格式），与实体交替拼接
    rel_tokens = re.findall(r'\b[A-Z][A-Z_]{2,}\b', r_ia or "")
    entities = list_t or []

    max_len = max(len(entities), len(rel_tokens) + 1)
    for i in range(min(max_len, 8)):
        if i < len(entities):
            chain.append({"text": entities[i], "type": "entity"})
        if i < len(rel_tokens):
            chain.append({"text": rel_tokens[i], "type": "rel"})

    chain.append({"text": "[Predicted Behavior]", "type": "answer"})
    return chain


def extract_relation_chain(triples: list, top_n: int = 5) -> list[str]:
    """从 lightprof_gr_triples 中按频次提取关系链"""
    counter: Counter = Counter()
    for item in triples:
        if len(item) >= 3:
            counter[str(item[1]).upper()] += 1
    return [r for r, _ in counter.most_common(top_n)]


def build_adj(triples: list) -> dict[str, list[dict]]:
    """构建有向邻接表，供 BFS 路径提取使用"""
    adj: dict[str, list[dict]] = defaultdict(list)
    for item in triples:
        h, r, t = str(item[0]), str(item[1]), str(item[2])
        adj[h].append({"node": t, "rel": r.upper()})
    return adj


def find_paths(
    start_nodes: list[str],
    adj: dict[str, list[dict]],
    max_hops: int = 3,
    max_paths: int = 20,
) -> list[dict]:
    """
    从 start_nodes 出发 BFS 提取 2-3 跳路径
    返回 list of {nodes: [...], rels: [...]}
    """
    from collections import deque

    paths: list[dict] = []
    for start in start_nodes:
        queue: deque = deque()
        queue.append({"path": [start], "rels": []})
        while queue and len(paths) < max_paths:
            state = queue.popleft()
            cur_path = state["path"]
            cur_rels = state["rels"]
            depth = len(cur_rels)
            if depth >= 2:  # 至少 2 跳才记录
                paths.append({"nodes": cur_path[:], "rels": cur_rels[:]})
            if depth < max_hops:
                cur_node = cur_path[-1]
                for edge in adj.get(cur_node, []):
                    nxt = edge["node"]
                    if nxt not in cur_path:  # 避免环
                        queue.append({
                            "path": cur_path + [nxt],
                            "rels": cur_rels + [edge["rel"]],
                        })
    # 去重
    seen: set = set()
    unique: list[dict] = []
    for p in paths:
        key = (tuple(p["nodes"]), tuple(p["rels"]))
        if key not in seen:
            seen.add(key)
            unique.append(p)
    return unique


def score_path(path: dict, rel_keys: tuple = ("ATTACK", "THREAT", "SUPPLIED", "DEPLOY", "ORDER")) -> float:
    """为路径打分，含军事关键词的路径得分更高"""
    base = 0.72 + 0.06 * len(path["rels"])
    bonus = 0.04 * sum(any(k in r for k in rel_keys) for r in path["rels"])
    return min(0.97, base + bonus)


def build_prune_steps(
    num_draft_links: int,
    num_non_noise_links: int,
    num_final_links: int,
) -> list[dict]:
    """构建 PRUNE_STEPS（4 个固定步骤，填入真实统计数字）"""
    return [
        {
            "icon": "📊", "iClass": "blue",
            "state": "初始状态",
            "desc": "草稿子图包含全部候选关系链（含噪声边）",
            "cnt": f"{num_draft_links} 条", "cClass": "",
        },
        {
            "icon": "✂️", "iClass": "orange",
            "state": "步骤一：结构过滤",
            "desc": "剪除偏离主题的噪声边，保留军事行为关联路径",
            "cnt": f"{num_non_noise_links} 条", "cClass": "",
        },
        {
            "icon": "🤖", "iClass": "orange",
            "state": "步骤二：LLM 语义打分",
            "desc": "对候选关系链执行语义一致性评分并排序",
            "cnt": "得分分配", "cClass": "",
        },
        {
            "icon": "🎯", "iClass": "green",
            "state": "步骤三：阈值剪枝 (w > 0.75)",
            "desc": "保留高置信关系链并实例化最终行为子图",
            "cnt": f"{num_final_links} 条 ✓", "cClass": "green",
        },
    ]
