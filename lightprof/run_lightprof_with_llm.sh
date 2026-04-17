#!/bin/bash
# LightPROF Pipeline with Real LLM（使用真实LLM）
# 使用方法: bash run_lightprof_with_llm.sh <checkpoint_path> <api_key>

set -e

# 检查参数
if [ -z "$1" ]; then
    echo "错误: 请提供检查点路径"
    echo "使用方法: bash run_lightprof_with_llm.sh <checkpoint_path> [api_key]"
    echo "示例: bash run_lightprof_with_llm.sh webqsp_hybrid_llm_kg_Feb05-09_40_35/cpt.pth sk-xxx"
    exit 1
fi

CHECKPOINT="$1"
CHECKPOINT_DIR=$(dirname "$CHECKPOINT")

# API密钥（可以从参数或环境变量获取）
if [ ! -z "$2" ]; then
    export OPENAI_API_KEY="$2"
elif [ -z "$OPENAI_API_KEY" ]; then
    echo "错误: 未设置OPENAI_API_KEY"
    echo "方法1: export OPENAI_API_KEY='your_key'"
    echo "方法2: bash run_lightprof_with_llm.sh checkpoint.pth 'your_key'"
    exit 1
fi

echo "================================================================================"
echo "LightPROF Pipeline with Real LLM"
echo "================================================================================"
echo "检查点: $CHECKPOINT"
echo "LLM模型: gpt-4o-mini"
echo "API密钥: ${OPENAI_API_KEY:0:10}..."
echo "================================================================================"
echo

# 激活conda环境
if command -v conda &> /dev/null; then
    source $(conda info --base)/etc/profile.d/conda.sh
    conda activate retriever 2>/dev/null || echo "警告: conda环境激活失败"
fi

# 步骤1: Retriever推理
if [ ! -f "${CHECKPOINT_DIR}/val_retrieval_result.pth" ]; then
    echo "[步骤1/3] 运行Retriever推理..."
    python inference_hybrid.py \
        -p "$CHECKPOINT" \
        --splits val test \
        --enable_entity_mapping \
        --num_display_samples 0
    echo
else
    echo "[步骤1/3] 检索结果已存在，跳过..."
    echo
fi

# 步骤2a: LightPROF采样（验证集）- 使用真实LLM
echo "[步骤2/3] 对验证集应用LightPROF采样（使用真实LLM）..."
python lightprof_sampling.py \
    -i "${CHECKPOINT_DIR}/val_retrieval_result.pth" \
    --h_q 2 \
    --top_k_chains 5 \
    --max_paths_per_chain 10 \
    --llm_model gpt-4o-mini \
    --num_display_samples 2

echo

# 步骤2b: LightPROF采样（测试集）- 使用真实LLM
echo "[步骤3/3] 对测试集应用LightPROF采样（使用真实LLM）..."
python lightprof_sampling.py \
    -i "${CHECKPOINT_DIR}/test_retrieval_result.pth" \
    --h_q 2 \
    --top_k_chains 5 \
    --max_paths_per_chain 10 \
    --llm_model gpt-4o-mini \
    --num_display_samples 2

echo

echo "================================================================================"
echo "✅ 完成！"
echo "================================================================================"
echo "生成的文件："
echo "  验证集: ${CHECKPOINT_DIR}/val_retrieval_result_lightprof.pth"
echo "  测试集: ${CHECKPOINT_DIR}/test_retrieval_result_lightprof.pth"
echo "================================================================================"
