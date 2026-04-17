#!/bin/bash
# LightPROF 完整Pipeline脚本（Linux版本）
# 使用方法: bash run_lightprof_pipeline.sh <checkpoint_path>

set -e  # 遇到错误立即退出

# 检查参数
if [ -z "$1" ]; then
    echo "错误: 请提供检查点路径"
    echo "使用方法: bash run_lightprof_pipeline.sh webqsp_hybrid_llm_kg_Feb05-09_40_35/cpt.pth"
    exit 1
fi

CHECKPOINT="$1"
CHECKPOINT_DIR=$(dirname "$CHECKPOINT")

echo "================================================================================"
echo "LightPROF 完整Pipeline"
echo "================================================================================"
echo "检查点: $CHECKPOINT"
echo "输出目录: $CHECKPOINT_DIR"
echo "================================================================================"
echo

# 激活conda环境（根据实际情况修改）
if command -v conda &> /dev/null; then
    echo "激活conda环境: retriever"
    source $(conda info --base)/etc/profile.d/conda.sh
    conda activate retriever || echo "警告: conda环境激活失败，尝试继续..."
else
    echo "警告: 未找到conda，使用系统Python"
fi

# 步骤1: Retriever推理（如果结果不存在）
if [ ! -f "${CHECKPOINT_DIR}/val_retrieval_result.pth" ]; then
    echo "[步骤1/3] 运行Retriever推理..."
    python inference_hybrid.py \
        -p "$CHECKPOINT" \
        --splits val test \
        --enable_entity_mapping \
        --num_display_samples 0
    
    if [ $? -ne 0 ]; then
        echo "错误: Retriever推理失败"
        exit 1
    fi
    echo
else
    echo "[步骤1/3] 检索结果已存在，跳过..."
    echo
fi

# 步骤2a: LightPROF采样（验证集）
echo "[步骤2/3] 对验证集应用LightPROF采样..."
python lightprof_sampling.py \
    -i "${CHECKPOINT_DIR}/val_retrieval_result.pth" \
    --h_q 2 \
    --top_k_chains 5 \
    --max_paths_per_chain 10 \
    --use_mock_llm \
    --num_display_samples 2

if [ $? -ne 0 ]; then
    echo "错误: 验证集采样失败"
    exit 1
fi
echo

# 步骤2b: LightPROF采样（测试集）
echo "[步骤3/3] 对测试集应用LightPROF采样..."
python lightprof_sampling.py \
    -i "${CHECKPOINT_DIR}/test_retrieval_result.pth" \
    --h_q 2 \
    --top_k_chains 5 \
    --max_paths_per_chain 10 \
    --use_mock_llm \
    --num_display_samples 2

if [ $? -ne 0 ]; then
    echo "错误: 测试集采样失败"
    exit 1
fi
echo

echo "================================================================================"
echo "完成！生成的文件："
echo "================================================================================"
echo "验证集:"
echo "  - ${CHECKPOINT_DIR}/val_retrieval_result_lightprof.pth"
echo "  - ${CHECKPOINT_DIR}/val_retrieval_result_lightprof_stats.json"
echo
echo "测试集:"
echo "  - ${CHECKPOINT_DIR}/test_retrieval_result_lightprof.pth"
echo "  - ${CHECKPOINT_DIR}/test_retrieval_result_lightprof_stats.json"
echo "================================================================================"
echo

echo "✅ 全部完成！"
