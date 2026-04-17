#!/bin/bash
# Linux服务器快速配置脚本
# 使用方法: bash setup_linux.sh

echo "================================================================================"
echo "LightPROF Linux服务器环境配置"
echo "================================================================================"
echo

# 1. 检查Python版本
echo "[1/5] 检查Python环境..."
if command -v python &> /dev/null; then
    PYTHON_VERSION=$(python --version 2>&1)
    echo "  ✓ Python版本: $PYTHON_VERSION"
else
    echo "  ❌ 未找到Python"
    exit 1
fi

# 2. 检查conda
echo
echo "[2/5] 检查conda..."
if command -v conda &> /dev/null; then
    CONDA_VERSION=$(conda --version 2>&1)
    echo "  ✓ Conda版本: $CONDA_VERSION"
    
    # 检查retriever环境是否存在
    if conda env list | grep -q "retriever"; then
        echo "  ✓ retriever环境已存在"
    else
        echo "  创建retriever环境..."
        conda create -n retriever python=3.10 -y
        echo "  ✓ retriever环境创建完成"
    fi
else
    echo "  ⚠️  未找到conda，将使用系统Python"
fi

# 3. 激活环境并安装依赖
echo
echo "[3/5] 安装Python依赖..."
if command -v conda &> /dev/null; then
    source $(conda info --base)/etc/profile.d/conda.sh
    conda activate retriever
fi

# 检查必需的包
REQUIRED_PACKAGES="torch numpy tqdm"
OPTIONAL_PACKAGES="sentence-transformers openai"

echo "  检查必需包..."
for pkg in $REQUIRED_PACKAGES; do
    if python -c "import $pkg" 2>/dev/null; then
        echo "    ✓ $pkg"
    else
        echo "    ✗ $pkg (未安装)"
        INSTALL_NEEDED=true
    fi
done

echo "  检查可选包..."
for pkg in $OPTIONAL_PACKAGES; do
    pkg_import=$(echo $pkg | sed 's/-/_/g')
    if python -c "import $pkg_import" 2>/dev/null; then
        echo "    ✓ $pkg"
    else
        echo "    ⚠️  $pkg (未安装，某些功能可能不可用)"
    fi
done

if [ "$INSTALL_NEEDED" = true ]; then
    echo
    read -p "是否安装缺失的依赖? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        pip install torch numpy tqdm sentence-transformers openai
    fi
fi

# 4. 设置shell脚本权限
echo
echo "[4/5] 设置执行权限..."
for script in *.sh; do
    if [ -f "$script" ]; then
        chmod +x "$script"
        echo "  ✓ $script"
    fi
done

# 5. 测试功能
echo
echo "[5/5] 测试LightPROF功能..."
if python test_lightprof.py 2>&1 | grep -q "所有测试通过"; then
    echo "  ✓ 功能测试通过"
else
    echo "  ⚠️  功能测试失败，请检查依赖"
fi

echo
echo "================================================================================"
echo "✅ 配置完成！"
echo "================================================================================"
echo
echo "下一步:"
echo "1. 检查模型检查点文件是否存在"
echo "2. 运行测试: python test_lightprof.py"
echo "3. 处理数据: bash run_lightprof_pipeline.sh checkpoint.pth"
echo
echo "常用命令:"
echo "  conda activate retriever        # 激活环境"
echo "  python test_lightprof.py        # 测试功能"
echo "  bash run_lightprof_pipeline.sh  # 运行完整流程"
echo "================================================================================"
