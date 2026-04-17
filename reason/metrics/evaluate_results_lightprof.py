#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LightPROF 专用评估脚本
直接读取 main_lightprof.py 的输出文件（predictions.jsonl），
利用其中的 lightprof_gr_triples 字段计算 a_entity_in_graph，
无需依赖外部 scored_triples/*.pth 文件。

评估指标与原始脚本保持一致：
  - Hit@1
  - Macro F1 / Precision / Recall
  - Exact Match / Totally Wrong
  - Micro F1 / Precision / Recall
  - Hal Score（使用 lightprof_gr_triples 作为子图实体来源）

用法（在 reason/ 目录下执行）：
  python metrics/evaluate_results_lightprof.py \
      -p ./results/KGQA/webqsp/LightPROF/.../predictions.jsonl

  # 只评估全集（默认）
  python metrics/evaluate_results_lightprof.py -p <pred_file>

  # 同时评估全集和子集
  python metrics/evaluate_results_lightprof.py -p <pred_file> --eval_subset
"""

import sys
import os
import re
import json
import string
import argparse
import numpy as np
from tqdm import tqdm
from copy import deepcopy

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ==========================================
# 工具函数（与原始脚本保持一致）
# ==========================================

def normalize(s: str) -> str:
    s = s.lower()
    exclude = set(string.punctuation)
    s = "".join(char for char in s if char not in exclude)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = re.sub(r"\b(<pad>)\b", " ", s)
    s = " ".join(s.split())
    return s


def match(s1: str, s2: str) -> bool:
    return normalize(s2) in normalize(s1)


# ==========================================
# 语义匹配器（sentence-transformers）
# ==========================================

class SemanticMatcher:
    """
    基于 sentence-transformers 的语义相似度匹配器
    对每个样本批量编码，避免重复计算

    用法：
        matcher = SemanticMatcher(model_name='all-MiniLM-L6-v2', threshold=0.85)
        sim = matcher.similarity("ans: Jamaican English", "Jamaican Creole")
        hit = matcher.is_match("ans: Jamaican English", "Jamaican Creole")
    """

    def __init__(self, model_name: str = 'all-MiniLM-L6-v2', threshold: float = 0.85):
        self.threshold = threshold
        self.model = None
        self._cache = {}  # 文本 → 向量的缓存
        self._load_model(model_name)

    def _load_model(self, model_name: str):
        try:
            from sentence_transformers import SentenceTransformer
            print(f"  加载语义匹配模型: {model_name} (阈值={self.threshold})")
            self.model = SentenceTransformer(model_name)
            print(f"  ✓ 语义匹配器初始化成功")
        except ImportError:
            print("  ⚠️  未安装 sentence-transformers，语义匹配将被跳过")
            print("       安装命令: pip install sentence-transformers")
        except RuntimeError as e:
            if "size mismatch" in str(e):
                print(f"  ⚠️  语义匹配模型缓存与当前库版本不兼容: {e}")
                print("       解决方法（任选其一）：")
                print("       1. 清除缓存: rm -rf ~/.cache/torch/sentence_transformers/"
                      f"sentence-transformers_{model_name.replace('/', '_')}/")
                print("       2. 更新库: pip install -U sentence-transformers")
                print("       3. 指定其他本地模型: --semantic_model /your/local/model/path")
            else:
                print(f"  ⚠️  语义匹配模型加载失败: {e}")
        except Exception as e:
            print(f"  ⚠️  语义匹配模型加载失败: {e}")

    def _clean_pred(self, text: str) -> str:
        """提取预测中 ans: 后面的实际答案文本"""
        if 'ans:' in text.lower():
            return text.split('ans:')[-1].strip()
        return text.strip()

    def encode_batch(self, texts: list) -> 'np.ndarray':
        """批量编码文本（带缓存）"""
        if self.model is None:
            return None
        new_texts = [t for t in texts if t not in self._cache]
        if new_texts:
            vecs = self.model.encode(new_texts, normalize_embeddings=True,
                                     show_progress_bar=False)
            for t, v in zip(new_texts, vecs):
                self._cache[t] = v
        return np.array([self._cache[t] for t in texts])

    def similarity(self, pred: str, answer: str) -> float:
        """计算单对文本的余弦相似度"""
        if self.model is None:
            return 0.0
        pred_clean = self._clean_pred(pred)
        vecs = self.encode_batch([pred_clean, normalize(answer)])
        if vecs is None:
            return 0.0
        return float(vecs[0] @ vecs[1])

    def is_match(self, pred: str, answer: str) -> bool:
        return self.similarity(pred, answer) >= self.threshold

    def build_sim_matrix(self, predictions: list, answers: list) -> 'np.ndarray':
        """
        构建预测 × 答案的相似度矩阵（批量计算，高效）

        Returns:
            sim_matrix: shape (len(predictions), len(answers))
        """
        if self.model is None or not predictions or not answers:
            return np.zeros((len(predictions), len(answers)))

        pred_texts = [self._clean_pred(p) for p in predictions]
        ans_texts = [normalize(a) for a in answers]

        pred_vecs = self.encode_batch(pred_texts)   # (P, D)
        ans_vecs = self.encode_batch(ans_texts)     # (A, D)

        return pred_vecs @ ans_vecs.T               # (P, A)


def remove_duplicates(input_list):
    seen = set()
    result = []
    for item in input_list:
        if item not in seen:
            result.append(item)
            seen.add(item)
    return result


def get_pred(prediction, split=None):
    if split is not None:
        return prediction.split(split)
    res = [p for p in prediction.split("\n") if 'ans:' in p and 'none' not in p.lower()]
    if len(res) >= 1:
        res = [p for p in res if
               "ans: not available" not in p.lower() and
               "ans: no information available" not in p.lower()]
    return remove_duplicates(res)


def _is_hit(pred: str, answer: str, double_check: bool,
            sim_matrix=None, pred_idx: int = 0, ans_idx: int = 0,
            semantic_threshold: float = 0.0) -> bool:
    """
    统一命中判断：字符串匹配 OR 语义匹配（任一满足即命中）

    Args:
        pred:               预测文本
        answer:             标准答案文本
        double_check:       是否启用反向子串检查
        sim_matrix:         预计算的语义相似度矩阵（可选）
        pred_idx/ans_idx:   在 sim_matrix 中的索引
        semantic_threshold: 语义匹配阈值（0 表示不启用语义匹配）
    """
    # 条件1：字符串子串匹配（预测包含答案）
    if match(pred, answer):
        return True
    # 条件2：反向子串匹配（double_check 时启用）
    if double_check:
        pred_clean = pred.split('ans:')[-1].strip()
        if match(answer, pred_clean) or match(answer, pred):
            return True
    # 条件3：语义相似度匹配（sim_matrix 不为 None 且阈值 > 0 时启用）
    if sim_matrix is not None and semantic_threshold > 0:
        if sim_matrix[pred_idx, ans_idx] >= semantic_threshold:
            return True
    return False


def eval_recall(prediction, answer, double_check,
                sim_matrix=None, semantic_threshold: float = 0.0):
    prediction = deepcopy(prediction)
    prediction = sorted(prediction, key=len, reverse=True)
    # orig_preds 保留删除前的完整有序列表，用于在 sim_matrix 中定位行索引
    # 调用方需确保传入时已按 len 降序排好，sim_matrix 行顺序才能与此对应
    orig_preds = list(prediction)
    matched = 0.
    for ans_idx, a in enumerate(answer):
        for pred in prediction:
            pred_idx = orig_preds.index(pred) if sim_matrix is not None else 0
            if _is_hit(pred, a, double_check, sim_matrix, pred_idx, ans_idx,
                       semantic_threshold):
                matched += 1
                prediction.remove(pred)
                break
    return matched / len(answer), matched, len(answer)


def eval_precision(prediction, answer, double_check,
                   sim_matrix=None, semantic_threshold: float = 0.0):
    prediction = deepcopy(prediction)
    prediction = sorted(prediction, key=len, reverse=True)
    orig_preds = list(prediction)
    num_pred = len(prediction)
    if num_pred == 0:
        return 0, 0, 0
    matched = 0.
    for ans_idx, a in enumerate(answer):
        for pred in prediction:
            pred_idx = orig_preds.index(pred) if sim_matrix is not None else 0
            if _is_hit(pred, a, double_check, sim_matrix, pred_idx, ans_idx,
                       semantic_threshold):
                matched += 1
                prediction.remove(pred)
                break
    return matched / num_pred, matched, num_pred


def eval_f1(precision, recall):
    if precision + recall == 0:
        return 0
    return 2 * precision * recall / (precision + recall)


def eval_hit(prediction, answer, double_check,
             sim_matrix=None, semantic_threshold: float = 0.0):
    if len(prediction) == 0:
        return 0
    for ans_idx, a in enumerate(answer):
        # 只看第一条预测（pred_idx=0）
        if _is_hit(prediction[0], a, double_check, sim_matrix, 0, ans_idx,
                   semantic_threshold):
            return 1
    return 0


def get_all_retrieved_entities(triplet_list):
    """从三元组列表中提取所有实体"""
    all_ent = set()
    for triplet in triplet_list:
        if len(triplet) >= 3:
            all_ent.add(str(triplet[0]))
            all_ent.add(str(triplet[2]))
    return list(all_ent)


def compute_a_entity_in_graph(ground_truth, gr_triples):
    """
    利用 lightprof_gr_triples 计算答案实体是否在推理子图中

    Args:
        ground_truth: 标准答案列表
        gr_triples:   lightprof_gr_triples [(h, r, t), ...]

    Returns:
        bool: 至少一个答案实体在子图中
    """
    if not gr_triples:
        return False

    all_ents = set()
    for triplet in gr_triples:
        if len(triplet) >= 3:
            all_ents.add(str(triplet[0]).lower())
            all_ents.add(str(triplet[2]).lower())

    for a in ground_truth:
        a_lower = normalize(a)
        for e in all_ents:
            if a_lower in e or e in a_lower:
                return True
    return False


def eval_hal_score(prediction, answer, double_check, a_entity_in_graph,
                   no_ans, subgraph_ent, stats):
    """Hal Score 计算（与原始脚本逻辑完全一致）"""
    answer = deepcopy(answer)
    score = 0
    stats['total_samples'] += 1

    if a_entity_in_graph:
        stats['total_g_samples'] += 1
        if no_ans:
            stats['g_no_ans'] += 1
            return 0, stats

        for pred in prediction:
            stats['total_ans'] += 1
            stats['total_g_ans'] += 1
            no_match = True
            for a in answer:
                if (match(pred, a) or
                        (double_check and match(a, pred.split('ans:')[-1].strip())) or
                        (double_check and match(a, pred))):
                    score += 1
                    stats['g_c'] += 1
                    no_match = False
                    answer.remove(a)

                    not_in_graph = True
                    for ent in subgraph_ent:
                        if (pred.lower().split('ans:')[-1].strip() in ent.lower() or
                                ent.lower() in pred.lower()):
                            not_in_graph = False
                            stats['g_c_in_graph'] += 1
                            break
                    if not_in_graph:
                        stats['g_c_out_graph'] += 1
                    break

            if no_match:
                score += -1
                stats['g_w'] += 1
                not_in_graph = True
                for ent in subgraph_ent:
                    if (pred.lower().split('ans:')[-1].strip() in ent.lower() or
                            ent.lower() in pred.lower()):
                        not_in_graph = False
                        stats['g_w_in_graph'] += 1
                        break
                if not_in_graph:
                    stats['g_w_out_graph'] += 1

        return score / len(prediction), stats

    else:
        stats['total_b_samples'] += 1
        if no_ans:
            stats['b_no_ans'] += 1
            return 1, stats
        else:
            for pred in prediction:
                stats['total_ans'] += 1
                stats['total_b_ans'] += 1
                no_match = True
                for ent in subgraph_ent:
                    if (pred.lower().split('ans:')[-1].strip() in ent.lower() or
                            ent.lower() in pred.lower()):
                        score += -1
                        stats['b_in_graph'] += 1
                        no_match = False
                        break
                if no_match:
                    score += -1.5
                    no_match_ans = True
                    for a in answer:
                        if (match(pred, a) or
                                (double_check and match(a, pred.split('ans:')[-1].strip())) or
                                (double_check and match(a, pred))):
                            stats['b_out_graph_c'] += 1
                            no_match_ans = False
                            answer.remove(a)
                            break
                    if no_match_ans:
                        stats['b_out_graph_w'] += 1
            return score / len(prediction), stats


# ==========================================
# 核心评估函数
# ==========================================

def eval_results_lightprof(predict_file: str,
                            subset: bool = False,
                            bad_samples: bool = False,
                            eval_hops: int = -1,
                            save_detail: bool = True,
                            semantic_matcher: 'SemanticMatcher | None' = None):
    """
    LightPROF 专用评估函数

    Args:
        predict_file:     main_lightprof.py 输出的 predictions.jsonl 路径
        subset:           只评估答案实体在子图中的样本（好样本）
        bad_samples:      只评估答案实体不在子图中的样本（坏样本）
        eval_hops:        按跳数筛选（-1 表示不筛选）
        save_detail:      是否保存逐条评估结果
        semantic_matcher: SemanticMatcher 实例；不为 None 时启用语义匹配兜底

    Returns:
        评估指标字典
    """
    assert not (subset and bad_samples), "subset 和 bad_samples 不能同时为 True"

    # 确定详细评估文件名
    tag = "subset" if subset else ("badSamples" if bad_samples else "full")
    detail_name = f"lightprof_{tag}_hop{eval_hops}_detailed_eval.jsonl"
    detail_file = predict_file.replace("predictions.jsonl", detail_name)
    result_txt = predict_file.replace("predictions.jsonl", f"lightprof_{tag}_eval_result.txt")

    hit_list, f1_list, prec_list, recall_list, hal_list = [], [], [], [], []
    total_pred = total_answer = total_match = 0
    total_cnt = no_ans_cnt = 0

    stats = {
        'g_no_ans': 0, 'g_c': 0, 'g_w': 0,
        'b_no_ans': 0, 'b_in_graph': 0, 'b_out_graph_c': 0, 'b_out_graph_w': 0,
        'total_ans': 0, 'total_g_samples': 0, 'total_b_samples': 0, 'total_samples': 0,
        'total_g_ans': 0, 'total_b_ans': 0,
        'g_c_out_graph': 0, 'g_w_out_graph': 0, 'g_c_in_graph': 0, 'g_w_in_graph': 0
    }

    detail_lines = []

    with open(predict_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    for line in tqdm(lines, desc=f"评估 [{tag}]"):
        try:
            data = json.loads(line.strip())
        except Exception:
            continue

        sid = data.get('id') or data.get('question', '')[:40]  # id 为 None 时用问题前40字符代替
        question = data.get('question', '')
        ground_truth = remove_duplicates(
            data.get('ground_truth') or data.get('answers') or data.get('a_entity') or []
        )
        answer = sorted(ground_truth, key=len, reverse=True)
        if not answer:  # ← 新增这两行
            continue  # ← 跳过没有答案的样本
        prediction_raw = data.get('prediction', '')
        gr_triples = data.get('lightprof_gr_triples', [])

        # 年份类问题截断答案
        if 'when' in question.lower() or 'what year' in question.lower():
            for idx in range(len(answer)):
                if '-' in answer[idx] and answer[idx].split('-')[0].isdigit():
                    answer[idx] = answer[idx].split('-')[0]

        # ── 计算 a_entity_in_graph（优先使用文件中已有的字段）──────────
        if 'a_entity_in_graph' in data and data['a_entity_in_graph'] is not None:
            # inference_hybrid.py 已计算好，是实体列表，判断是否非空即可
            a_entity_in_graph = bool(data['a_entity_in_graph'])
        else:
            a_entity_in_graph = compute_a_entity_in_graph(answer, gr_triples)

        # subset / bad_samples 筛选
        if subset and not a_entity_in_graph:
            continue
        if bad_samples and a_entity_in_graph:
            continue

        # hop 筛选（需要样本中有 lightprof_stats 或 max_path_length 字段）
        if eval_hops > 0:
            lp_stats = data.get('lightprof_stats', {})
            num_paths = lp_stats.get('num_paths', -1)
            if eval_hops == 3:
                if num_paths < 3:
                    continue
            elif num_paths != eval_hops:
                continue

        # ── 解析预测答案 ──────────────────────────────────────────────
        double_check = any(k in question.lower() for k in [
            'when', 'what year', 'which year', 'where', 'sport',
            'what countr', 'language', 'nba finals', 'world series'
        ])

        prediction = get_pred(prediction_raw)
        # eval_precision/recall 内部会按长度降序排列，提前排好
        # 使 sim_matrix 的行顺序与函数内部 orig_preds 的顺序完全一致
        prediction = sorted(prediction, key=len, reverse=True)
        total_cnt += 1

        no_ans_flag = (
            len(prediction) == 0 or
            'ans:' not in prediction_raw or
            'ans: not available' in prediction_raw.lower() or
            'ans: no information available' in prediction_raw.lower()
        )
        if no_ans_flag:
            no_ans_cnt += 1

        # ── 语义相似度矩阵（可选）────────────────────────────────────
        # 注意：必须在 prediction 排序后再构建，行索引才与 eval 函数内部一致
        sim_matrix = None
        sem_threshold = 0.0
        if semantic_matcher is not None and semantic_matcher.model is not None and prediction:
            sim_matrix = semantic_matcher.build_sim_matrix(prediction, answer)
            sem_threshold = semantic_matcher.threshold

        # ── 计算各指标 ────────────────────────────────────────────────
        prec_score, matched_1, num_pred = eval_precision(
            prediction, answer, double_check, sim_matrix, sem_threshold)
        recall_score, matched_2, num_answer = eval_recall(
            prediction, answer, double_check, sim_matrix, sem_threshold)
        f1_score = eval_f1(prec_score, recall_score)
        hit = eval_hit(prediction, answer, double_check, sim_matrix, sem_threshold)

        # Hal Score：全集时计算，子集/坏样本时跳过
        if not subset and not bad_samples:
            subgraph_ent = get_all_retrieved_entities(gr_triples)
            hal_score, stats = eval_hal_score(
                prediction, answer, double_check,
                a_entity_in_graph, no_ans_flag, subgraph_ent, stats
            )
        else:
            hal_score = 0

        assert matched_1 == matched_2
        total_pred += num_pred
        total_answer += num_answer
        total_match += matched_1

        hit_list.append(hit)
        f1_list.append(f1_score)
        prec_list.append(prec_score)
        recall_list.append(recall_score)
        hal_list.append(hal_score)

        detail_lines.append(json.dumps({
            'id': sid,
            'prediction': prediction,
            'ground_truth': answer,
            'hit': hit,
            'f1': f1_score,
            'precision': prec_score,
            'recall': recall_score,
            'hal_score': hal_score,
            'a_entity_in_graph': a_entity_in_graph
        }, ensure_ascii=False))

    if len(hit_list) == 0:
        print(f"  ⚠️  [{tag}] 无有效样本，跳过")
        return {}

    # ── 汇总指标 ──────────────────────────────────────────────────────
    avg_hit = sum(hit_list) * 100 / len(hit_list)
    avg_f1 = sum(f1_list) * 100 / len(f1_list)
    avg_prec = sum(prec_list) * 100 / len(prec_list)
    avg_recall = sum(recall_list) * 100 / len(recall_list)
    avg_hal = (sum(hal_list) / len(hal_list) + 1.5) / (1 + 1.5) * 100

    num_exact_match = (np.array(f1_list) == 1).sum() / len(f1_list) * 100
    num_totally_wrong = (np.array(recall_list) == 0).sum() / len(recall_list) * 100

    micro_prec = total_match / total_pred if total_pred > 0 else 0
    micro_recall = total_match / total_answer if total_answer > 0 else 0
    micro_f1 = (2 * micro_prec * micro_recall / (micro_prec + micro_recall)
                if (micro_prec + micro_recall) > 0 else 0)

    # ── 打印结果 ──────────────────────────────────────────────────────
    sep = "=" * 60
    print(f"\n{sep}")
    print(f"  评估范围: {tag.upper()}  (共 {total_cnt} 条样本)")
    print(f"{sep}")
    print(f"  Hit@1:          {avg_hit:.2f}%")
    print(f"  Macro F1:       {avg_f1:.2f}%")
    print(f"  Macro Precision:{avg_prec:.2f}%")
    print(f"  Macro Recall:   {avg_recall:.2f}%")
    print(f"  Exact Match:    {num_exact_match:.2f}%")
    print(f"  Totally Wrong:  {num_totally_wrong:.2f}%")
    print(f"  Micro F1:       {micro_f1*100:.2f}%")
    print(f"  Micro Precision:{micro_prec*100:.2f}%")
    print(f"  Micro Recall:   {micro_recall*100:.2f}%")
    if not subset and not bad_samples:
        print(f"  Hal Score:      {avg_hal:.2f}%")
    print(f"  无答案样本:     {no_ans_cnt}/{total_cnt} ({no_ans_cnt/total_cnt*100:.1f}%)")
    print(f"{sep}\n")

    # ── 保存结果 ──────────────────────────────────────────────────────
    result_str = (
        f"Hit@1: {avg_hit:.4f}, Macro F1: {avg_f1:.4f}, "
        f"Macro Precision: {avg_prec:.4f}, Macro Recall: {avg_recall:.4f}, "
        f"Exact Match: {num_exact_match:.4f}, Totally Wrong: {num_totally_wrong:.4f}, "
        f"Hal Score: {avg_hal:.4f}"
    )
    with open(result_txt, 'w', encoding='utf-8') as f:
        f.write(result_str + "\n")
    print(f"  评估结果已保存: {result_txt}")

    if save_detail:
        with open(detail_file, 'w', encoding='utf-8') as f:
            f.write("\n".join(detail_lines))
        print(f"  详细评估已保存: {detail_file}")

    return {
        'hit@1': avg_hit, 'macro_f1': avg_f1,
        'macro_precision': avg_prec, 'macro_recall': avg_recall,
        'exact_match': num_exact_match, 'totally_wrong': num_totally_wrong,
        'micro_f1': micro_f1 * 100, 'micro_precision': micro_prec * 100,
        'micro_recall': micro_recall * 100, 'hal_score': avg_hal,
        'total_cnt': total_cnt, 'no_ans_cnt': no_ans_cnt,
    }


# ==========================================
# 命令行入口
# ==========================================

def main():
    parser = argparse.ArgumentParser(
        description="LightPROF 评估脚本（基于 lightprof_gr_triples 字段）"
    )
    parser.add_argument("-p", "--pred_file", type=str, required=True,
                        help="main_lightprof.py 输出的 predictions.jsonl 路径")
    parser.add_argument("--eval_subset", action="store_true",
                        help="额外评估子集（答案实体在子图中的样本）")
    parser.add_argument("--eval_bad", action="store_true",
                        help="额外评估坏样本（答案实体不在子图中）")
    parser.add_argument("--no_detail", action="store_true",
                        help="不保存逐条评估详情文件")
    # ── 语义匹配参数 ─────────────────────────────────────────────────
    parser.add_argument("--use_semantic", action="store_true",
                        help="启用 sentence-transformers 语义相似度兜底匹配")
    parser.add_argument("--semantic_model", type=str,
                        default="/home/shu1004/lyx/GCA/GCA-main/all-mpnet-base-v2",
                        help="sentence-transformers 模型名称或本地路径 "
                             "（默认: all-MiniLM-L6-v2）")
    parser.add_argument("--semantic_threshold", type=float, default=0.85,
                        help="语义相似度命中阈值，取值 0~1（默认: 0.85）")
    args = parser.parse_args()

    if not os.path.exists(args.pred_file):
        print(f"❌ 文件不存在: {args.pred_file}")
        sys.exit(1)

    print(f"\n预测文件: {args.pred_file}")

    # ── 初始化语义匹配器（只初始化一次，所有评估共享）────────────────
    semantic_matcher = None
    if args.use_semantic:
        semantic_matcher = SemanticMatcher(
            model_name=args.semantic_model,
            threshold=args.semantic_threshold
        )

    # 全集评估（必做）
    eval_results_lightprof(
        args.pred_file,
        subset=False,
        bad_samples=False,
        save_detail=not args.no_detail,
        semantic_matcher=semantic_matcher
    )

    # 子集评估（可选）
    if args.eval_subset:
        eval_results_lightprof(
            args.pred_file,
            subset=True,
            bad_samples=False,
            save_detail=not args.no_detail,
            semantic_matcher=semantic_matcher
        )

    # 坏样本评估（可选）
    if args.eval_bad:
        eval_results_lightprof(
            args.pred_file,
            subset=False,
            bad_samples=True,
            save_detail=not args.no_detail,
            semantic_matcher=semantic_matcher
        )


if __name__ == "__main__":
    main()
