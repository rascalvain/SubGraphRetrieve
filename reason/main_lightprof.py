#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LightPROF + PoG 推理主脚本（新增文件，不修改原始 main.py）

与原始 main.py 的主要区别：
1. 直接加载 lightprof_sampling.py 输出的 .pth 文件（每个样本已含 lightprof_gr_triples）
2. 新增 prompt_mode 前缀 'lightprof'，调用 prepare_prompts_lightprof.py
3. 支持 --use_pog_prompt 标志，可将 I_LLM/Split_q 注入 LLM prompt
4. 不依赖 HuggingFace 数据集，不需要 RoG 子图
"""

import os
import sys
import json
import torch
import wandb
import argparse
from tqdm import tqdm
from pathlib import Path

# 确保 reason 目录在 sys.path 中（在 reason/ 目录下运行脚本时自动生效）
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from llm_utils import llm_init, llm_inf_all
from preprocess.prepare_prompts import get_prompts_for_data
from preprocess.prepare_prompts_lightprof import (
    POG_ENHANCED_SYS_PROMPT,
    get_prompts_for_data_lightprof
)
from metrics.evaluate_results_corrected import eval_results as eval_results_corrected
from metrics.evaluate_results import eval_results as eval_results_original


# ==========================================
# 辅助函数
# ==========================================

def is_lightprof_mode(prompt_mode: str) -> bool:
    return prompt_mode.startswith('lightprof')


def get_defined_prompts(prompt_mode: str, model_name: str, llm_mode: str,
                        use_pog_prompt: bool = False, rank_first: bool = False):
    """
    根据 prompt_mode 选择系统提示词，支持 lightprof 模式

    重要：llm_mode 中含 'icl' 时（如 sys_icl_dc），llm_inf 会自动注入少样本示例
    （icl_user_prompt + icl_ass_prompt），cot_prompt 也需换用 icl_cot_prompt，
    其中包含 "Return the most possible answers"，有助于模型输出多个答案。

    rank_first=True 时，cot_query 改用 cot_prompt_rm_rank：
      "先删错误答案，再把最有信心的放第一"，直接靶向提升 Hit@1。
    """
    if is_lightprof_mode(prompt_mode):
        # ── 选择 cot_prompt ──────────────────────────────────────────
        if rank_first:
            from prompts import cot_prompt_rm_rank
            cot = cot_prompt_rm_rank        # "最有信心的答案放第一" → Hit@1↑
        elif 'icl' in llm_mode:
            from prompts import icl_cot_prompt
            cot = icl_cot_prompt            # "返回所有可能答案" → Recall↑
        else:
            from prompts import cot_prompt
            cot = cot_prompt

        # ── 选择 sys_prompt ──────────────────────────────────────────
        if use_pog_prompt or 'pog' in prompt_mode:
            return POG_ENHANCED_SYS_PROMPT, cot
        elif 'icl' in llm_mode:
            from prompts import icl_sys_prompt
            return icl_sys_prompt, cot
        else:
            from prompts import sys_prompt
            return sys_prompt, cot

    # 兼容原始 main.py 的模式
    if 'gpt' in model_name or 'gpt' in prompt_mode:
        if 'gptLabel' in prompt_mode:
            from prompts import sys_prompt_gpt, cot_prompt_gpt
            return sys_prompt_gpt, cot_prompt_gpt
        else:
            from prompts import icl_sys_prompt, icl_cot_prompt
            return icl_sys_prompt, icl_cot_prompt
    elif 'noevi' in prompt_mode:
        from prompts import noevi_sys_prompt, noevi_cot_prompt
        return noevi_sys_prompt, noevi_cot_prompt
    elif 'icl' in llm_mode:
        from prompts import icl_sys_prompt, icl_cot_prompt
        return icl_sys_prompt, icl_cot_prompt
    else:
        from prompts import sys_prompt, cot_prompt
        return sys_prompt, cot_prompt


def load_lightprof_data(score_dict_path: str) -> list:
    """
    从 lightprof_sampling.py 输出的 .pth 文件加载数据

    每个样本结构：
      - id, question, answers (原始字段)
      - q_entity, q_entity_list (主题实体)
      - scored_triplets (检索器打分的三元组)
      - lightprof_gr_triples (LightPROF 精炼的推理子图三元组)
      - lightprof_stats (采样统计)
      - Split_q, I_LLM (PoG 指标，若有)
    """
    print(f"加载 LightPROF 数据: {score_dict_path}")
    data = torch.load(score_dict_path, weights_only=False)

    if isinstance(data, dict):
        # 兼容字典格式（key=id, value=sample）
        data = list(data.values())

    print(f"共加载 {len(data)} 条样本")

    # 统计 lightprof_gr_triples 覆盖率
    has_gr = sum(1 for d in data if d.get('lightprof_gr_triples'))
    has_pog = sum(1 for d in data if d.get('I_LLM') or d.get('Split_q'))
    print(f"  含 lightprof_gr_triples: {has_gr}/{len(data)} ({has_gr/len(data)*100:.1f}%)")
    print(f"  含 PoG 指标 (I_LLM/Split_q): {has_pog}/{len(data)} ({has_pog/len(data)*100:.1f}%)")

    return data


def save_checkpoint(file_handle, data: dict):
    file_handle.write(json.dumps(data, ensure_ascii=False) + "\n")


def load_checkpoint(file_path) -> list:
    if os.path.exists(file_path):
        print("*" * 50)
        print(f"从断点续跑: {file_path}")
        with open(file_path, "r", encoding='utf-8') as f:
            ckpt = [json.loads(line) for line in f]
        try:
            print(f"已处理到: {ckpt[-1]['id']}")
        except (IndexError, KeyError):
            pass
        print("*" * 50)
        return ckpt
    return []


def eval_all(pred_file_path, run, subset: bool, split: str = None, eval_hops: int = -1):
    print("=" * 50)
    print(f"评估子集: {'subset' if subset else 'all'}")

    hit1, f1, prec, recall, em, tw, mi_f1, mi_prec, mi_recall, total_cnt, no_ans_cnt, no_ans_ratio, hal_score, stats = \
        eval_results_corrected(str(pred_file_path), cal_f1=True, subset=subset, split=split, eval_hops=eval_hops)

    postfix = "_sub" if subset else ""
    run.log({
        f"results{postfix}/hit@1": hit1,
        f"results{postfix}/macro_f1": f1,
        f"results{postfix}/macro_precision": prec,
        f"results{postfix}/macro_recall": recall,
        f"results{postfix}/exact_match": em,
        f"results{postfix}/totally_wrong": tw,
        f"results{postfix}/micro_f1": mi_f1,
        f"results{postfix}/micro_precision": mi_prec,
        f"results{postfix}/micro_recall": mi_recall,
        f"results{postfix}/total_cnt": total_cnt,
        f"results{postfix}/no_ans_cnt": no_ans_cnt,
        f"results{postfix}/no_ans_ratio": no_ans_ratio,
        f"results{postfix}/hal_score": hal_score,
    })
    if stats is not None:
        for k, v in stats.items():
            run.log({f"stats{postfix}/{k}": v})

    hit, _, _, _ = eval_results_original(str(pred_file_path), cal_f1=True, subset=subset, eval_hops=eval_hops)
    run.log({f"results{postfix}/hit": hit})
    print("=" * 50)


# ==========================================
# 主函数
# ==========================================

def main():
    parser = argparse.ArgumentParser(description="LightPROF + PoG RAG for KGQA")
    parser.add_argument("-d", "--dataset_name", type=str, default="webqsp",
                        help="数据集名称 (webqsp / cwq)")
    parser.add_argument("--prompt_mode", type=str, default="lightprof_100",
                        help=(
                            "Prompt 模式，支持:\n"
                            "  lightprof_100          : 使用 lightprof_gr_triples，最多 100 条三元组\n"
                            "  lightprof_pog_100      : 同上 + PoG I_LLM/Split_q 注入\n"
                            "  lightprof_fallback_100 : lightprof_gr_triples 为空时回退到 scored_triplets\n"
                            "  scored_100             : 原始模式（兼容）\n"
                        ))
    parser.add_argument("-p", "--score_dict_path", type=str, required=True,
                        help="lightprof_sampling.py 输出的 .pth 文件路径")
    parser.add_argument("--llm_mode", type=str, default="sys_icl_dc",
                        help="LLM 推理模式")
    parser.add_argument("-m", "--model_name", type=str,
                        default="meta-llama/Meta-Llama-3.1-8B-Instruct",
                        help="LLM 模型名称")
    parser.add_argument("--split", type=str, default="test",
                        help="数据集分片 (test / val)")
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--max_seq_len_to_capture", type=int, default=8192 * 2)
    parser.add_argument("--max_tokens", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--frequency_penalty", type=float, default=0.16)
    parser.add_argument("--thres", type=float, default=0.0,
                        help="三元组分数阈值（lightprof 模式下无效）")

    # LightPROF 专属参数
    parser.add_argument("--use_pog_prompt", action="store_true",
                        help="在 prompt 中注入 PoG 推理指导（Split_q/I_LLM）")
    parser.add_argument("--no_fallback", action="store_true",
                        help="lightprof_gr_triples 为空时不回退到 scored_triplets")
    parser.add_argument("--rank_first", action="store_true",
                        help=(
                            "使用 cot_prompt_rm_rank：要求模型删除错误答案并将最有信心的"
                            "答案放第一位，直接提升 Hit@1（与 icl_cot 互斥，优先级更高）"
                        ))
    parser.add_argument("--no_sort_triples", action="store_true",
                        help="禁用三元组按锚点相关性排序（默认开启，有助于 Hit@1）")
    parser.add_argument("--wandb_project", type=str, default=None,
                        help="WandB 项目名，默认为 LightPROF-{dataset_name}")

    args = parser.parse_args()

    # ── 参数解析 ──────────────────────────────────────────────────────
    dataset_name = args.dataset_name
    prompt_mode = args.prompt_mode
    llm_mode = args.llm_mode
    model_name = args.model_name
    split = args.split
    thres = args.thres
    use_pog_prompt = args.use_pog_prompt
    fallback_to_scored = not args.no_fallback
    rank_first = args.rank_first
    sort_triples = not args.no_sort_triples

    # ── WandB 初始化 ──────────────────────────────────────────────────
    project_name = args.wandb_project or f"LightPROF-{dataset_name}"
    run_name = (
        f"{model_name.split('/')[-1]}-{prompt_mode}"
        f"{'_pog' if use_pog_prompt else ''}"
        f"{'_rank1st' if rank_first else ''}"
        f"-{llm_mode}-fp{args.frequency_penalty}-thres{thres}-{split}"
    )
    run = wandb.init(project=project_name, name=run_name, config=vars(args))

    # ── 输出路径 ──────────────────────────────────────────────────────
    pog_tag = "_pog" if (use_pog_prompt or 'pog' in prompt_mode) else ""
    raw_pred_folder = Path(f"./results/KGQA/{dataset_name}/LightPROF/{model_name.split('/')[-1]}")
    raw_pred_folder.mkdir(parents=True, exist_ok=True)
    raw_pred_file = raw_pred_folder / (
        f"{prompt_mode}{pog_tag}-{llm_mode}-fp{args.frequency_penalty}"
        f"-thres{thres}-{split}-predictions-resume.jsonl"
    )

    # ── 加载数据 ──────────────────────────────────────────────────────
    data = load_lightprof_data(args.score_dict_path)

    # 按 split 筛选（若样本含 split 字段）
    if any('split' in d for d in data):
        data_split = [d for d in data if d.get('split', split) == split]
        if data_split:
            print(f"按 split='{split}' 筛选后: {len(data_split)} 条样本")
            data = data_split

    # ── 构建 Prompt ───────────────────────────────────────────────────
    sys_prompt, cot_prompt = get_defined_prompts(
        prompt_mode, model_name, llm_mode, use_pog_prompt, rank_first
    )

    print("生成 Prompts...")
    if rank_first:
        print("  ✓ cot 模式: rank_first（最有信心的答案放第一 → Hit@1 优化）")
    if sort_triples:
        print("  ✓ 三元组排序: 锚点 1-hop 优先（有助于模型聚焦关键证据）")

    if is_lightprof_mode(prompt_mode):
        data = get_prompts_for_data_lightprof(
            data, prompt_mode, sys_prompt, cot_prompt, thres,
            use_pog_prompt=use_pog_prompt,
            fallback_to_scored=fallback_to_scored,
            sort_by_relevance=sort_triples
        )
    else:
        # 兼容原始 scored_xxx / rog_xxx 等模式
        data = get_prompts_for_data(data, prompt_mode, sys_prompt, cot_prompt, thres)

    # ── LLM 初始化 ────────────────────────────────────────────────────
    llm = llm_init(
        model_name,
        args.tensor_parallel_size,
        args.max_seq_len_to_capture,
        args.max_tokens,
        args.seed,
        args.temperature,
        args.frequency_penalty
    )

    # ── 推理循环 ──────────────────────────────────────────────────────
    print("开始推理...")
    start_idx = len(load_checkpoint(raw_pred_file))

    # 需要在保存前删除的大字段（节省磁盘空间）
    fields_to_drop = ['lightprof_gr_triples', 'scored_triplets', 'graph',
                      'good_paths_rog', 'good_triplets_rog']

    with open(raw_pred_file, "a", encoding='utf-8') as pred_file:
        for each_qa in tqdm(data[start_idx:], initial=start_idx, total=len(data)):
            res = llm_inf_all(llm, each_qa, llm_mode, model_name)

            # 删除大字段
            for field in fields_to_drop:
                each_qa.pop(field, None)

            each_qa["prediction"] = res[0]
            save_checkpoint(pred_file, each_qa)

    # ── 重命名（去掉 -resume 标记）────────────────────────────────────
    final_pred_file = raw_pred_file.with_name(
        raw_pred_file.stem.replace("-resume", "") + raw_pred_file.suffix
    )
    os.rename(raw_pred_file, final_pred_file)
    print(f"\n预测结果已保存到: {final_pred_file}")

    # ── 评估 ──────────────────────────────────────────────────────────
    eval_all(final_pred_file, run, subset=True)
    eval_all(final_pred_file, run, subset=False)

    run.finish()


if __name__ == "__main__":
    main()
