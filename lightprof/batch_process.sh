#!/bin/bash
# 批量处理多个检查点的LightPROF采样
# 使用方法: bash batch_process.sh

set -e

echo "================================================================================"
echo "批量处理 LightPROF 采样"
echo "================================================================================"

# 配置参数
H_Q=2
TOP_K_CHAINS=5
MAX_PATHS=10
USE_MOCK=true  # 改为false以使用真实LLM

# 激活环境
if command -v conda &> /dev/null; then
    source $(conda info --base)/etc/profile.d/conda.sh
    conda activate retriever 2>/dev/null || true
fi

# 查找所有retrieval_result.pth文件
echo "正在查找检索结果文件..."
retrieval_files=$(find . -name "*_retrieval_result.pth" ! -name "*_lightprof.pth" 2>/dev/null)

if [ -z "$retrieval_files" ]; then
    echo "错误: 未找到任何检索结果文件"
    echo "请先运行 inference_hybrid.py 生成检索结果"
    exit 1
fi

echo "找到以下文件:"
echo "$retrieval_files"
echo

# 统计
total_files=$(echo "$retrieval_files" | wc -l)
processed=0

# 处理每个文件
for input_file in $retrieval_files; do
    processed=$((processed + 1))
    echo "================================================================================"
    echo "[$processed/$total_files] 处理: $input_file"
    echo "================================================================================"
    
    # 构建输出文件名
    output_file="${input_file%.pth}_lightprof.pth"
    
    # 检查是否已处理
    if [ -f "$output_file" ]; then
        echo "⚠️  输出文件已存在，跳过: $output_file"
        echo
        continue
    fi
    
    # 构建命令
    if [ "$USE_MOCK" = true ]; then
        python lightprof_sampling.py \
            -i "$input_file" \
            -o "$output_file" \
            --h_q $H_Q \
            --top_k_chains $TOP_K_CHAINS \
            --max_paths_per_chain $MAX_PATHS \
            --use_mock_llm \
            --num_display_samples 0
    else
        python lightprof_sampling.py \
            -i "$input_file" \
            -o "$output_file" \
            --h_q $H_Q \
            --top_k_chains $TOP_K_CHAINS \
            --max_paths_per_chain $MAX_PATHS \
            --llm_model gpt-4o-mini \
            --num_display_samples 0
    fi
    
    if [ $? -eq 0 ]; then
        echo "✅ 成功: $output_file"
    else
        echo "❌ 失败: $input_file"
        exit 1
    fi
    echo
done

echo "================================================================================"
echo "✅ 批量处理完成！"
echo "================================================================================"
echo "总共处理: $processed 个文件"
echo "================================================================================"
