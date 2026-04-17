#!/bin/bash

#==============================================================================
# WebQSP KG增强嵌入计算脚本 (完善版)
# 支持：本地数据集、命令行参数、自动化管理、自定义输出目录
#==============================================================================

# ====== 默认配置 ======
DATASET="webqsp"
USE_LOCAL_DATASET=true
LOCAL_DATASET_PATH="/home/shu1004/lyx/subgraph/SubgraphRAG-main/retrieve/processed_dataset/webqsp_processed"
KG_EMB_PATH="./src/data/entity_embeddings.pkl"
ENTITY_MAPPING_PATH="./src/data/entity_mapping_with_mid.txt"
FUSION_METHOD="weighted"
TEXT_WEIGHT=0.7
KG_WEIGHT=0.3
OUTPUT_BASE_DIR="./outputs"
LOG_DIR="./logs"
PYTHON_SCRIPT="emb_hybrid.py"

# ====== 帮助信息 ======
show_help() {
    cat << EOF
用法: $0 [选项]

WebQSP KG增强嵌入计算脚本

选项:
    -d, --dataset DATASET           数据集名称 (默认: webqsp)
    --local-data PATH               本地数据集路径 (默认: ./datasets/webqsp_processed)
    --no-local                      不使用本地数据集，从HuggingFace加载
    -k, --kg-emb PATH              KG嵌入文件路径 (默认: ./src/data/entity_embeddings.pkl)
    -m, --mapping PATH              实体映射文件路径 (默认: ./src/data/entity_mapping.txt)
    -f, --fusion METHOD             融合方法 (weighted/concat/gated, 默认: weighted)
    -t, --text-weight WEIGHT        文本权重 (默认: 0.7)
    -g, --kg-weight WEIGHT          KG权重 (默认: 0.3)
    -o, --output-dir DIR            输出根目录 (默认: ./outputs)
    -l, --log-dir DIR               日志目录 (默认: ./logs)
    -h, --help                      显示此帮助信息

示例:
    $0                              # 使用默认配置
    $0 -t 0.8 -g 0.2               # 自定义权重
    $0 --fusion concat              # 使用拼接融合
    $0 --no-local                   # 从HuggingFace加载数据

EOF
}

# ====== 解析命令行参数 ======
while [[ $# -gt 0 ]]; do
    case $1 in
        -d|--dataset)
            DATASET="$2"
            shift 2
            ;;
        --local-data)
            LOCAL_DATASET_PATH="$2"
            shift 2
            ;;
        --no-local)
            USE_LOCAL_DATASET=false
            shift
            ;;
        -k|--kg-emb)
            KG_EMB_PATH="$2"
            shift 2
            ;;
        -m|--mapping)
            ENTITY_MAPPING_PATH="$2"
            shift 2
            ;;
        -f|--fusion)
            FUSION_METHOD="$2"
            shift 2
            ;;
        -t|--text-weight)
            TEXT_WEIGHT="$2"
            shift 2
            ;;
        -g|--kg-weight)
            KG_WEIGHT="$2"
            shift 2
            ;;
        -o|--output-dir)
            OUTPUT_BASE_DIR="$2"
            shift 2
            ;;
        -l|--log-dir)
            LOG_DIR="$2"
            shift 2
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            echo "❌ 错误: 未知参数 '$1'"
            show_help
            exit 1
            ;;
    esac
done

# ====== 验证配置 ======
echo "=========================================="
echo "WebQSP KG增强嵌入计算任务"
echo "=========================================="
echo "正在验证配置..."

# 验证Python脚本
if [ ! -f "${PYTHON_SCRIPT}" ]; then
    echo "❌ 错误: Python脚本不存在: ${PYTHON_SCRIPT}"
    exit 1
fi
echo "✓ Python脚本: ${PYTHON_SCRIPT}"

# 验证KG嵌入文件
if [ ! -f "${KG_EMB_PATH}" ]; then
    echo "❌ 错误: KG嵌入文件不存在: ${KG_EMB_PATH}"
    exit 1
fi
echo "✓ KG嵌入文件: ${KG_EMB_PATH}"

# 验证实体映射文件
if [ ! -f "${ENTITY_MAPPING_PATH}" ]; then
    echo "❌ 错误: 实体映射文件不存在: ${ENTITY_MAPPING_PATH}"
    exit 1
fi
echo "✓ 实体映射文件: ${ENTITY_MAPPING_PATH}"

# 验证本地数据集
if [ "$USE_LOCAL_DATASET" = true ]; then
    if [ ! -d "${LOCAL_DATASET_PATH}" ]; then
        echo "❌ 错误: 本地数据集目录不存在: ${LOCAL_DATASET_PATH}"
        echo "提示: 使用 --no-local 参数可以从HuggingFace加载数据"
        exit 1
    fi

    # 检查parquet文件
    PARQUET_COUNT=$(find "${LOCAL_DATASET_PATH}" -name "*.parquet" 2>/dev/null | wc -l)
    if [ ${PARQUET_COUNT} -eq 0 ]; then
        echo "❌ 错误: 本地数据集目录中没有parquet文件: ${LOCAL_DATASET_PATH}"
        echo "提示: 先运行 extract_quest_ent.py 处理数据集"
        exit 1
    fi
    echo "✓ 本地数据集: ${LOCAL_DATASET_PATH} (${PARQUET_COUNT} 个文件)"
else
    echo "ℹ️  将从HuggingFace加载数据集"
fi

echo "=========================================="

# ====== 生成时间戳 ======
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
echo ""
echo "启动时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "时间戳: ${TIMESTAMP}"
echo ""

# ====== 创建输出目录 ======
OUTPUT_DIR="${OUTPUT_BASE_DIR}/emb_${FUSION_METHOD}_t${TEXT_WEIGHT}_k${KG_WEIGHT}_${TIMESTAMP}"
mkdir -p "${OUTPUT_DIR}"
mkdir -p "${LOG_DIR}"

LOG_FILE="${LOG_DIR}/emb_${TIMESTAMP}.log"
PID_FILE="${LOG_DIR}/emb_${TIMESTAMP}.pid"

echo "✓ 输出目录: ${OUTPUT_DIR}"
echo "✓ 日志文件: ${LOG_FILE}"
echo "✓ PID文件: ${PID_FILE}"

# ====== 记录配置到日志 ======
{
    echo "========================================"
    echo "WebQSP KG增强嵌入计算任务"
    echo "========================================"
    echo "启动时间: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "时间戳: ${TIMESTAMP}"
    echo "主机: $(hostname)"
    echo "用户: $(whoami)"
    echo "工作目录: $(pwd)"
    echo ""
    echo "配置参数:"
    echo "  数据集: ${DATASET}"
    echo "  使用本地数据集: ${USE_LOCAL_DATASET}"
    if [ "$USE_LOCAL_DATASET" = true ]; then
        echo "  本地数据集路径: ${LOCAL_DATASET_PATH}"
    fi
    echo "  KG嵌入路径: ${KG_EMB_PATH}"
    echo "  实体映射路径: ${ENTITY_MAPPING_PATH}"
    echo "  融合方法: ${FUSION_METHOD}"
    echo "  文本权重: ${TEXT_WEIGHT}"
    echo "  KG权重: ${KG_WEIGHT}"
    echo "  输出目录: ${OUTPUT_DIR}"
    echo ""
    echo "========================================"
    echo ""
} > "${LOG_FILE}"

# ====== 构建Python命令 ======
PYTHON_CMD="python ${PYTHON_SCRIPT} -d ${DATASET}"

# 添加本地数据集参数
if [ "$USE_LOCAL_DATASET" = true ]; then
    PYTHON_CMD="${PYTHON_CMD} --use_local_dataset --local_dataset_path ${LOCAL_DATASET_PATH}"
fi

# 🔑 添加输出目录参数（放在这里，便于调试）
PYTHON_CMD="${PYTHON_CMD} --output_dir ${OUTPUT_DIR}"

# 添加KG增强参数
PYTHON_CMD="${PYTHON_CMD} \
    --use_kg_enhancement \
    --kg_emb_path ${KG_EMB_PATH} \
    --entity_mapping_path ${ENTITY_MAPPING_PATH} \
    --fusion_method ${FUSION_METHOD} \
    --text_weight ${TEXT_WEIGHT} \
    --kg_weight ${KG_WEIGHT}"

# 设置HuggingFace镜像
export HF_ENDPOINT=https://hf-mirror.com
echo "✓ 使用 HuggingFace 镜像: ${HF_ENDPOINT}" | tee -a "${LOG_FILE}"

# ====== 显示并记录执行命令 ======
echo ""
echo "=========================================="
echo "执行命令:"
echo "=========================================="
echo "${PYTHON_CMD}"
echo "=========================================="
echo ""

{
    echo "执行命令:"
    echo "${PYTHON_CMD}"
    echo ""
    echo "开始执行..."
    echo "========================================"
    echo ""
} >> "${LOG_FILE}"

# ====== 使用nohup后台执行 ======
nohup ${PYTHON_CMD} >> "${LOG_FILE}" 2>&1 &
PID=$!

# 保存PID
echo ${PID} > "${PID_FILE}"

# ====== 创建快捷管理脚本 ======
# 1. 查看日志脚本
VIEW_LOG_SCRIPT="${LOG_DIR}/view_log_${TIMESTAMP}.sh"
cat > "${VIEW_LOG_SCRIPT}" << EOF
#!/bin/bash
# 查看日志文件
echo "正在查看日志: ${LOG_FILE}"
echo "按 Ctrl+C 退出"
echo ""
tail -f ${LOG_FILE}
EOF
chmod +x "${VIEW_LOG_SCRIPT}"

# 2. 停止任务脚本
STOP_SCRIPT="${LOG_DIR}/stop_${TIMESTAMP}.sh"
cat > "${STOP_SCRIPT}" << EOF
#!/bin/bash
# 停止嵌入计算任务

PID=${PID}
echo "=========================================="
echo "停止任务"
echo "=========================================="
echo "PID: \${PID}"
echo ""

if ps -p \${PID} > /dev/null 2>&1; then
    echo "正在停止进程..."
    kill \${PID}

    # 等待进程停止
    for i in {1..5}; do
        sleep 1
        if ! ps -p \${PID} > /dev/null 2>&1; then
            echo "✓ 任务已成功停止"
            exit 0
        fi
        echo "等待进程停止... (\${i}/5)"
    done

    # 强制停止
    echo "进程未响应，强制停止..."
    kill -9 \${PID} 2>/dev/null
    sleep 1

    if ! ps -p \${PID} > /dev/null 2>&1; then
        echo "✓ 任务已强制停止"
    else
        echo "❌ 无法停止进程"
        exit 1
    fi
else
    echo "进程已经停止 (PID: \${PID})"
fi
EOF
chmod +x "${STOP_SCRIPT}"

# 3. 检查状态脚本
STATUS_SCRIPT="${LOG_DIR}/status_${TIMESTAMP}.sh"
cat > "${STATUS_SCRIPT}" << EOF
#!/bin/bash
# 检查任务状态

PID=${PID}
LOG_FILE="${LOG_FILE}"
OUTPUT_DIR="${OUTPUT_DIR}"

echo "=========================================="
echo "任务状态"
echo "=========================================="
echo "PID: \${PID}"
echo "日志: \${LOG_FILE}"
echo "输出目录: \${OUTPUT_DIR}"
echo ""

if ps -p \${PID} > /dev/null 2>&1; then
    echo "状态: ✓ 运行中"
    echo ""

    # 显示进程信息
    echo "进程信息:"
    ps -p \${PID} -o pid,ppid,cmd,etime,%cpu,%mem
    echo ""

    # 检查输出目录
    echo "输出目录内容:"
    echo "----------------------------------------"
    if [ -d "\${OUTPUT_DIR}" ]; then
        find "\${OUTPUT_DIR}" -type f -exec ls -lh {} \; 2>/dev/null | head -20
    else
        echo "输出目录尚未创建"
    fi
    echo "----------------------------------------"
    echo ""

    # 显示最新日志
    echo "最新日志 (最后20行):"
    echo "=========================================="
    tail -n 20 "\${LOG_FILE}"
    echo "=========================================="
else
    echo "状态: ✗ 已停止"
    echo ""

    # 检查输出文件
    echo "输出文件统计:"
    echo "----------------------------------------"
    if [ -d "\${OUTPUT_DIR}" ]; then
        echo "目录: \${OUTPUT_DIR}"
        du -sh "\${OUTPUT_DIR}"
        echo ""
        find "\${OUTPUT_DIR}" -type f -exec ls -lh {} \;
    else
        echo "输出目录不存在"
    fi
    echo "----------------------------------------"
    echo ""

    echo "完整日志:"
    echo "=========================================="
    cat "\${LOG_FILE}"
    echo "=========================================="
fi
EOF
chmod +x "${STATUS_SCRIPT}"

# 4. 查看输出目录脚本
VIEW_OUTPUT_SCRIPT="${LOG_DIR}/view_output_${TIMESTAMP}.sh"
cat > "${VIEW_OUTPUT_SCRIPT}" << EOF
#!/bin/bash
# 查看输出目录内容

OUTPUT_DIR="${OUTPUT_DIR}"

echo "=========================================="
echo "输出目录: \${OUTPUT_DIR}"
echo "=========================================="
echo ""

if [ ! -d "\${OUTPUT_DIR}" ]; then
    echo "❌ 输出目录不存在"
    exit 1
fi

# 显示目录结构
echo "目录结构:"
echo "----------------------------------------"
tree -L 3 "\${OUTPUT_DIR}" 2>/dev/null || find "\${OUTPUT_DIR}" -type f -o -type d | head -50
echo "----------------------------------------"
echo ""

# 显示磁盘使用
echo "磁盘使用:"
echo "----------------------------------------"
du -sh "\${OUTPUT_DIR}"
du -sh "\${OUTPUT_DIR}"/*
echo "----------------------------------------"
echo ""

# 显示文件列表
echo "文件详情:"
echo "----------------------------------------"
find "\${OUTPUT_DIR}" -type f -exec ls -lh {} \;
echo "----------------------------------------"
EOF
chmod +x "${VIEW_OUTPUT_SCRIPT}"

# ====== 输出任务信息 ======
echo ""
echo "=========================================="
echo "✓ 任务已成功启动！"
echo "=========================================="
echo "  进程ID (PID): ${PID}"
echo "  输出目录: ${OUTPUT_DIR}"
echo "  日志文件: ${LOG_FILE}"
echo ""
echo "管理命令:"
echo "  查看日志:     ${VIEW_LOG_SCRIPT}"
echo "                或: tail -f ${LOG_FILE}"
echo ""
echo "  检查状态:     ${STATUS_SCRIPT}"
echo "                或: ps -p ${PID}"
echo ""
echo "  查看输出:     ${VIEW_OUTPUT_SCRIPT}"
echo "                或: ls -lh ${OUTPUT_DIR}"
echo ""
echo "  停止任务:     ${STOP_SCRIPT}"
echo "                或: kill ${PID}"
echo ""
echo "  实时监控:     watch -n 2 '${STATUS_SCRIPT}'"
echo "=========================================="
echo ""

# ====== 等待并检查进程状态 ======
echo "等待进程启动..."
sleep 3

if ps -p ${PID} > /dev/null 2>&1; then
    echo "✓ 进程运行正常 (PID: ${PID})"
    echo ""
    echo "正在读取初始日志..."
    echo "=========================================="
    head -n 50 "${LOG_FILE}"
    echo "=========================================="
    echo ""
    echo "提示: 使用 ${VIEW_LOG_SCRIPT} 查看完整日志"
    echo "提示: 使用 ${STATUS_SCRIPT} 检查任务状态和输出"
else
    echo "❌ 警告: 进程可能已退出或启动失败"
    echo ""
    echo "完整日志:"
    echo "=========================================="
    cat "${LOG_FILE}"
    echo "=========================================="
    exit 1
fi

echo ""
echo "任务正在后台运行中..."
echo "使用上述命令管理任务"
echo ""