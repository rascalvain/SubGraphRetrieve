#!/bin/bash
# run_lightprof.sh - 带时间戳日志的推理脚本

# ── 配置区域（按需修改）────────────────────────────────────────────────
DATA_PATH="/media/shu1004/pytorch/projects/lyx/subgraph/SubgraphRAG-main/lightprof/runs/20260312_105037/lightprof_output.pth"
MODEL_PATH="/media/shu1004/pytorch/projects/lyx/Meta-Llama-3.1-8B-Instruct"
DATASET="webqsp"
PROMPT_MODE="lightprof_pog_200"
SPLIT="test"

USE_POG_PROMPT="true"   # "true" 则注入 PoG 推理指导（Split_q/I_LLM）
RANK_FIRST="true"       # "true" 则要求模型把最有信心的答案放第一（↑ Hit@1）
SORT_TRIPLES="true"     # "true" 则对三元组按锚点相关性排序（1-hop 优先，↑ Hit@1）
                        # 设为 "false" 会传入 --no_sort_triples

SCRIPT_DIR="/media/shu1004/pytorch/projects/lyx/subgraph/SubgraphRAG-main/reason"
LOG_DIR="${SCRIPT_DIR}/logs"
# ────────────────────────────────────────────────────────────────────────

# 生成时间戳（格式：20260309_193824）
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOG_DIR}/reasoning_${TIMESTAMP}.log"

# 确保日志目录存在
mkdir -p "${LOG_DIR}"

# ── 构建可选参数 ─────────────────────────────────────────────────────
POG_FLAG=""
POG_TAG="关闭"
if [ "${USE_POG_PROMPT}" = "true" ]; then
    POG_FLAG="--use_pog_prompt"
    POG_TAG="开启"
fi

RANK_FLAG=""
RANK_TAG="关闭"
if [ "${RANK_FIRST}" = "true" ]; then
    RANK_FLAG="--rank_first"
    RANK_TAG="开启（最有信心的答案放第一）"
fi

SORT_FLAG=""
SORT_TAG="开启（1-hop 优先）"
if [ "${SORT_TRIPLES}" = "false" ]; then
    SORT_FLAG="--no_sort_triples"
    SORT_TAG="关闭"
fi

# ── 打印启动信息 ─────────────────────────────────────────────────────
echo "========================================"
echo "  LightPROF 推理任务启动"
echo "  时间戳:      ${TIMESTAMP}"
echo "  日志文件:    ${LOG_FILE}"
echo "  PoG 增强:    ${POG_TAG}"
echo "  Rank First:  ${RANK_TAG}"
echo "  三元组排序:  ${SORT_TAG}"
echo "========================================"

# 切换到脚本目录
cd "${SCRIPT_DIR}" || exit 1

# 写入日志头部（含配置信息）
{
    echo "========================================"
    echo "  任务开始时间: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "  时间戳标识:   ${TIMESTAMP}"
    echo "  数据文件:     ${DATA_PATH}"
    echo "  模型路径:     ${MODEL_PATH}"
    echo "  数据集:       ${DATASET}"
    echo "  Prompt模式:   ${PROMPT_MODE}"
    echo "  数据分片:     ${SPLIT}"
    echo "  PoG 增强:     ${POG_TAG}"
    echo "  Rank First:   ${RANK_TAG}"
    echo "  三元组排序:   ${SORT_TAG}"
    echo "========================================"
    echo ""
} >> "${LOG_FILE}"

export WANDB_MODE=disabled

# 执行推理（nohup 后台运行，输出追加到日志文件）
nohup python main_lightprof.py \
    -d "${DATASET}" \
    -p "${DATA_PATH}" \
    --prompt_mode "${PROMPT_MODE}" \
    --split "${SPLIT}" \
    -m "${MODEL_PATH}" \
    ${POG_FLAG} \
    ${RANK_FLAG} \
    ${SORT_FLAG} \
    >> "${LOG_FILE}" 2>&1 &

PID=$!
echo "  进程 PID: ${PID}"
echo "  进程 PID: ${PID}" >> "${LOG_FILE}"
echo ""
echo "  后台运行中，实时查看日志："
echo "  tail -f ${LOG_FILE}"
echo "========================================"

# 将 PID 写入 pid 文件，方便后续管理
echo "${PID}" > "${LOG_DIR}/reasoning_${TIMESTAMP}.pid"

## ── 快速参考：常用命令组合 ──────────────────────────────────────────
## 当前最优（PoG + rank_first + 三元组排序）
#USE_POG_PROMPT="true" RANK_FIRST="true" SORT_TRIPLES="true"
#
## 对照组：关闭 rank_first，观察 Hit@1 变化
#USE_POG_PROMPT="true" RANK_FIRST="false" SORT_TRIPLES="true"
#
## 对照组：关闭三元组排序，单独验证排序效果
#USE_POG_PROMPT="true" RANK_FIRST="true" SORT_TRIPLES="false"
#
## 基础模式（lightprof_gr_triples，无 PoG，无优化）
#python main_lightprof.py \
#  -d webqsp \
#  -p ../lightprof/output/lightprof_output.pth \
#  --prompt_mode lightprof_100 \
#  --split test \
#  -m "${MODEL_PATH}"
#
## 含回退的安全模式
#python main_lightprof.py \
#  -d webqsp \
#  -p ../lightprof/output/lightprof_output.pth \
#  --prompt_mode lightprof_fallback_100 \
#  --use_pog_prompt \
#  --rank_first \
#  --split test \
#  -m "${MODEL_PATH}"
