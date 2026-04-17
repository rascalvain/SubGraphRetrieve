#!/bin/bash
# run_sampling.sh - LightPROF 采样脚本（带时间戳目录，日志与结果统一存放）
#
# 用法：
#   bash run_sampling.sh              # 前台运行（输出到终端）
#   bash run_sampling.sh --bg         # 后台运行（nohup）
#   bash run_sampling.sh --no-fuzzy   # 禁用 Fuzzy Selection
#   bash run_sampling.sh --mock       # 使用模拟 LLM（测试用）

# ── 配置区域（按需修改）────────────────────────────────────────────────
INPUT_FILE="/home/shu1004/lyx/subgraph/SubgraphRAG-main/reason/data/test_retrieval_result.pth"
SCRIPT_DIR="/home/shu1004/lyx/subgraph/SubgraphRAG-main/lightprof"
RUNS_DIR="${SCRIPT_DIR}/runs"       # 所有运行结果的根目录

# 采样参数
H_Q=2                               # 推理跳数
TOP_K_CHAINS=5                      # LLM 精筛保留的 Top-K 关系链
MAX_PATHS=10                        # 每条关系链最多采样路径数
BIDIRECTIONAL="true"                # 是否构建双向图（true/false）

# Fuzzy Selection 参数
ENABLE_FUZZY="true"                 # 是否启用模糊筛选（true/false）
W1=80                               # Fuzzy Selection 保留链数量
SBERT_MODEL="/home/shu1004/lyx/subgraph/SubgraphRAG-main/all-mpnet-base-v2"

# LLM 配置
USE_MOCK_LLM="false"                # 是否使用模拟 LLM（true=测试用）
LLM_MODEL="gpt-4o-mini"
LLM_API_KEY="sk-IQ8vi7XzSOgnTAW805DchQy2YVSOA8q6WYb7vUZRYOHKN6vN"
LLM_BASE_URL="https://api.openai-proxy.org/v1"

NUM_DISPLAY=2                       # 结果展示样本数
# ────────────────────────────────────────────────────────────────────────

# ── 解析命令行参数 ────────────────────────────────────────────────────
BG_MODE="false"
for arg in "$@"; do
    case "$arg" in
        --bg)        BG_MODE="true" ;;
        --no-fuzzy)  ENABLE_FUZZY="false" ;;
        --mock)      USE_MOCK_LLM="true" ;;
    esac
done

# ── 生成时间戳，创建本次运行专属目录 ─────────────────────────────────
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RUN_DIR="${RUNS_DIR}/${TIMESTAMP}"
LOG_FILE="${RUN_DIR}/sampling.log"
OUTPUT_FILE="${RUN_DIR}/lightprof_output.pth"   # 采样结果保存在同一目录

mkdir -p "${RUN_DIR}"

# ── 构建可选参数 ─────────────────────────────────────────────────────
EXTRA_FLAGS=""

if [ "${BIDIRECTIONAL}" = "true" ]; then
    EXTRA_FLAGS="${EXTRA_FLAGS} --bidirectional"
fi

FUZZY_TAG="开启 (W1=${W1})"
if [ "${ENABLE_FUZZY}" = "false" ]; then
    EXTRA_FLAGS="${EXTRA_FLAGS} --no_fuzzy"
    FUZZY_TAG="关闭"
fi

LLM_TAG="${LLM_MODEL}"
if [ "${USE_MOCK_LLM}" = "true" ]; then
    EXTRA_FLAGS="${EXTRA_FLAGS} --use_mock_llm"
    LLM_TAG="Mock（模拟）"
fi

# ── 打印启动信息 ─────────────────────────────────────────────────────
echo "========================================"
echo "  LightPROF 采样任务"
echo "  时间戳:       ${TIMESTAMP}"
echo "  运行目录:     ${RUN_DIR}"
echo "  日志文件:     ${LOG_FILE}"
echo "  输出文件:     ${OUTPUT_FILE}"
echo "  模糊筛选:     ${FUZZY_TAG}"
echo "  LLM 模型:     ${LLM_TAG}"
echo "  后台模式:     ${BG_MODE}"
echo "========================================"

# ── 写入日志头部 ─────────────────────────────────────────────────────
{
    echo "========================================"
    echo "  任务开始时间:  $(date '+%Y-%m-%d %H:%M:%S')"
    echo "  时间戳标识:    ${TIMESTAMP}"
    echo "  运行目录:      ${RUN_DIR}"
    echo ""
    echo "  输入文件:      ${INPUT_FILE}"
    echo "  输出文件:      ${OUTPUT_FILE}"
    echo ""
    echo "  推理跳数 h_q:  ${H_Q}"
    echo "  Top-K 链数:    ${TOP_K_CHAINS}"
    echo "  每链最大路径:  ${MAX_PATHS}"
    echo "  双向图:        ${BIDIRECTIONAL}"
    echo ""
    echo "  模糊筛选:      ${FUZZY_TAG}"
    echo "  SBERT 模型:    ${SBERT_MODEL}"
    echo ""
    echo "  LLM 模型:      ${LLM_TAG}"
    echo "  LLM API URL:   ${LLM_BASE_URL}"
    echo "========================================"
    echo ""
} >> "${LOG_FILE}"

# ── 切换到脚本目录 ───────────────────────────────────────────────────
cd "${SCRIPT_DIR}" || exit 1

# ── 构建 Python 命令 ─────────────────────────────────────────────────
CMD="python lightprof_sampling.py \
    -i \"${INPUT_FILE}\" \
    -o \"${OUTPUT_FILE}\" \
    --h_q ${H_Q} \
    --top_k_chains ${TOP_K_CHAINS} \
    --max_paths_per_chain ${MAX_PATHS} \
    --w1 ${W1} \
    --sbert_model \"${SBERT_MODEL}\" \
    --llm_model \"${LLM_MODEL}\" \
    --llm_api_key \"${LLM_API_KEY}\" \
    --llm_base_url \"${LLM_BASE_URL}\" \
    --num_display_samples ${NUM_DISPLAY} \
    ${EXTRA_FLAGS}"

# ── 执行 ─────────────────────────────────────────────────────────────
if [ "${BG_MODE}" = "true" ]; then
    # 后台模式：nohup，2>&1 重定向到日志
    echo "  后台运行，日志实时查看："
    echo "  tail -f ${LOG_FILE}"
    echo "========================================"

    eval "nohup ${CMD} >> \"${LOG_FILE}\" 2>&1 &"
    PID=$!

    echo "  进程 PID: ${PID}"
    echo "  进程 PID: ${PID}" >> "${LOG_FILE}"
    echo "${PID}" > "${RUN_DIR}/sampling.pid"

    echo ""
    echo "  结束后结果在: ${RUN_DIR}/"
    echo "    ├── sampling.log       # 运行日志"
    echo "    ├── lightprof_output.pth  # 采样结果"
    echo "    └── lightprof_output_stats.json  # 统计信息"
else
    # 前台模式：同时输出到终端和日志文件（tee）
    echo "  前台运行（输出同时写入日志）..."
    echo "========================================"
    eval "${CMD} 2>&1 | tee -a \"${LOG_FILE}\""

    EXIT_CODE=${PIPESTATUS[0]}
    {
        echo ""
        echo "========================================"
        echo "  任务结束时间: $(date '+%Y-%m-%d %H:%M:%S')"
        echo "  退出码: ${EXIT_CODE}"
        echo "========================================"
    } >> "${LOG_FILE}"

    echo ""
    echo "========================================"
    echo "  运行完成！结果目录: ${RUN_DIR}/"
    echo "    ├── sampling.log"
    echo "    ├── lightprof_output.pth"
    echo "    └── lightprof_output_stats.json"
    echo "========================================"
fi
