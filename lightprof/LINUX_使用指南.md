# Linux 服务器使用指南

## 📦 文件传输到服务器

### **方式1: 使用scp**

```bash
# 从Windows上传到Linux服务器
scp -r "g:\小论文\第一章\新建文件夹\SubgraphRAG-main" username@server:/path/to/destination/

# 或只上传retrieve目录
scp -r "g:\小论文\第一章\新建文件夹\SubgraphRAG-main\retrieve" username@server:/path/to/SubgraphRAG-main/
```

### **方式2: 使用git**

```bash
# 在Windows上
cd "g:\小论文\第一章\新建文件夹\SubgraphRAG-main"
git init
git add .
git commit -m "Initial commit"
git remote add origin <your_repo_url>
git push -u origin main

# 在Linux服务器上
git clone <your_repo_url>
cd SubgraphRAG-main/retrieve
```

### **方式3: 使用rsync（推荐）**

```bash
# 增量同步，更快
rsync -avz --progress "g:\小论文\第一章\新建文件夹\SubgraphRAG-main" username@server:/path/to/destination/
```

---

## 🔧 服务器环境配置

### **1. 创建conda环境**

```bash
# SSH登录到服务器
ssh username@your-server

# 创建环境
conda create -n retriever python=3.10 -y
conda activate retriever

# 安装依赖
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121
pip install numpy tqdm sentence-transformers openai

# 验证安装
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import sentence_transformers; print('sentence-transformers: OK')"
```

### **2. 设置权限**

```bash
cd SubgraphRAG-main/retrieve

# 给shell脚本添加执行权限
chmod +x run_lightprof_pipeline.sh
chmod +x run_lightprof_with_llm.sh
chmod +x batch_process.sh

# 验证权限
ls -l *.sh
```

---

## 🚀 运行命令（Linux版本）

### **基础用法：单个命令**

```bash
cd ~/SubgraphRAG-main/retrieve

# Step 1: Retriever推理
python inference_hybrid.py \
    -p webqsp_hybrid_llm_kg_Feb05-09_40_35/cpt.pth \
    --splits val test \
    --enable_entity_mapping \
    --num_display_samples 3

# Step 2: LightPROF采样（验证集）
python lightprof_sampling.py \
    -i webqsp_hybrid_llm_kg_Feb05-09_40_35/val_retrieval_result.pth \
    --use_mock_llm \
    --num_display_samples 2

# Step 3: LightPROF采样（测试集）
python lightprof_sampling.py \
    -i webqsp_hybrid_llm_kg_Feb05-09_40_35/test_retrieval_result.pth \
    --use_mock_llm \
    --num_display_samples 2
```

### **使用Shell脚本：一键运行**

```bash
# 方式1: 使用模拟LLM（快速测试）
bash run_lightprof_pipeline.sh webqsp_hybrid_llm_kg_Feb05-09_40_35/cpt.pth

# 方式2: 使用真实LLM
export OPENAI_API_KEY='your_key_here'
bash run_lightprof_with_llm.sh webqsp_hybrid_llm_kg_Feb05-09_40_35/cpt.pth

# 或直接传入API密钥
bash run_lightprof_with_llm.sh webqsp_hybrid_llm_kg_Feb05-09_40_35/cpt.pth 'your_key_here'
```

### **批量处理**

```bash
# 批量处理所有检索结果文件
bash batch_process.sh
```

---

## 🎯 后台运行（适合长时间任务）

### **方式1: 使用nohup**

```bash
# 后台运行，输出重定向到日志文件
nohup python lightprof_sampling.py \
    -i test_retrieval_result.pth \
    --use_mock_llm \
    > lightprof_sampling.log 2>&1 &

# 查看进程
ps aux | grep lightprof_sampling

# 实时查看日志
tail -f lightprof_sampling.log

# 终止进程
kill <pid>
```

### **方式2: 使用screen**

```bash
# 创建新session
screen -S lightprof

# 在screen中运行
python lightprof_sampling.py -i test_retrieval_result.pth --use_mock_llm

# 分离session: Ctrl+A, 然后按D
# 重新连接: screen -r lightprof
# 查看所有session: screen -ls
# 终止session: exit（在screen内）
```

### **方式3: 使用tmux**

```bash
# 创建新session
tmux new -s lightprof

# 运行任务
python lightprof_sampling.py -i test_retrieval_result.pth --use_mock_llm

# 分离: Ctrl+B, 然后按D
# 重新连接: tmux attach -t lightprof
# 查看sessions: tmux ls
```

---

## 📊 监控和调试

### **查看GPU使用情况**

```bash
# 实时监控GPU
watch -n 1 nvidia-smi

# 或使用gpustat（需要安装）
pip install gpustat
watch -n 1 gpustat -cpu
```

### **查看进程状态**

```bash
# 查看Python进程
ps aux | grep python

# 查看资源使用
top -u $USER

# 查看特定进程
top -p <pid>
```

### **查看输出文件**

```bash
# 实时查看最新生成的文件
watch -n 5 'ls -lht *.pth | head -10'

# 检查文件大小
du -h *_retrieval_result*.pth

# 快速查看文件内容
python -c "
import torch
data = torch.load('test_retrieval_result_lightprof.pth')
print(f'样本数: {len(data)}')
sample = list(data.values())[0]
print(f'推理图三元组: {len(sample.get(\"lightprof_gr_triples\", []))}')
"
```

---

## 🔐 API密钥管理

### **方式1: 环境变量（推荐）**

```bash
# 在 ~/.bashrc 或 ~/.zshrc 中添加
echo 'export OPENAI_API_KEY="your_key_here"' >> ~/.bashrc
source ~/.bashrc

# 验证
echo $OPENAI_API_KEY
```

### **方式2: .env文件**

```bash
# 创建.env文件
cat > .env << EOF
OPENAI_API_KEY=your_key_here
DEEPSEEK_API_KEY=your_deepseek_key
EOF

# 在脚本中使用
python -c "
from dotenv import load_dotenv
load_dotenv()
import os
print(os.getenv('OPENAI_API_KEY'))
"
```

### **方式3: 配置文件**

```bash
# 创建config.json
cat > config.json << EOF
{
    "openai_api_key": "your_key_here",
    "llm_model": "gpt-4o-mini"
}
EOF

# 在lightprof_sampling.py中读取配置
```

---

## 🎮 常用命令组合

### **调试流程**

```bash
# 1. 测试功能
python test_lightprof.py

# 2. 处理小样本
python -c "
import torch
data = torch.load('test_retrieval_result.pth')
small = dict(list(data.items())[:10])
torch.save(small, 'test_small.pth')
"

python lightprof_sampling.py -i test_small.pth --use_mock_llm

# 3. 检查结果
python -c "
import torch
data = torch.load('test_small_lightprof.pth')
for sid, s in list(data.items())[:2]:
    print(f'样本: {sid}')
    print(f'  原始三元组: {len(s[\"scored_triples\"])}')
    print(f'  推理图三元组: {len(s[\"lightprof_gr_triples\"])}')
    print(f'  压缩率: {len(s[\"lightprof_gr_triples\"])/len(s[\"scored_triples\"]):.2%}')
    print()
"
```

### **完整流程（串联执行）**

```bash
#!/bin/bash
# 完整的端到端流程

CHECKPOINT="webqsp_hybrid_llm_kg_Feb05-09_40_35/cpt.pth"

# Step 1: Retriever
python inference_hybrid.py \
    -p $CHECKPOINT \
    --splits test \
    --enable_entity_mapping \
    --num_display_samples 0

# Step 2: LightPROF
python lightprof_sampling.py \
    -i webqsp_hybrid_llm_kg_Feb05-09_40_35/test_retrieval_result.pth \
    --use_mock_llm \
    --num_display_samples 2

# Step 3: 验证输出
python -c "
import torch
data = torch.load('webqsp_hybrid_llm_kg_Feb05-09_40_35/test_retrieval_result_lightprof.pth')
print(f'✅ 成功生成 {len(data)} 个样本的推理图')
"

echo "✅ 完整流程执行成功！"
```

---

## 📈 性能优化（服务器环境）

### **1. 多GPU并行**

如果有多个GPU，可以分割数据并行处理：

```bash
# 分割数据
python -c "
import torch
data = torch.load('test_retrieval_result.pth')
items = list(data.items())
n = len(items) // 4  # 假设4个GPU

for i in range(4):
    start = i * n
    end = (i + 1) * n if i < 3 else len(items)
    chunk = dict(items[start:end])
    torch.save(chunk, f'test_chunk_{i}.pth')
    print(f'Chunk {i}: {len(chunk)} 样本')
"

# 在不同GPU上运行
CUDA_VISIBLE_DEVICES=0 python lightprof_sampling.py -i test_chunk_0.pth --use_mock_llm &
CUDA_VISIBLE_DEVICES=1 python lightprof_sampling.py -i test_chunk_1.pth --use_mock_llm &
CUDA_VISIBLE_DEVICES=2 python lightprof_sampling.py -i test_chunk_2.pth --use_mock_llm &
CUDA_VISIBLE_DEVICES=3 python lightprof_sampling.py -i test_chunk_3.pth --use_mock_llm &

wait  # 等待所有进程完成

# 合并结果
python -c "
import torch
merged = {}
for i in range(4):
    data = torch.load(f'test_chunk_{i}_lightprof.pth')
    merged.update(data)
torch.save(merged, 'test_retrieval_result_lightprof.pth')
print(f'✅ 合并完成: {len(merged)} 个样本')
"
```

### **2. 限制CPU使用**

```bash
# 限制PyTorch线程数
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

python lightprof_sampling.py -i input.pth --use_mock_llm
```

### **3. 使用SSD加速**

```bash
# 如果有SSD，将数据复制到SSD
cp -r data /tmp/data
cd /tmp/data

# 在SSD上处理
python lightprof_sampling.py -i test_retrieval_result.pth --use_mock_llm

# 处理完成后复制回去
cp *_lightprof.pth ~/SubgraphRAG-main/retrieve/
```

---

## 🔍 日志和调试

### **详细日志**

```bash
# 将stdout和stderr都保存到日志
python lightprof_sampling.py \
    -i test_retrieval_result.pth \
    --use_mock_llm \
    2>&1 | tee lightprof_$(date +%Y%m%d_%H%M%S).log

# 只保存错误日志
python lightprof_sampling.py \
    -i test_retrieval_result.pth \
    --use_mock_llm \
    2> lightprof_error.log
```

### **进度监控**

```bash
# 实时显示处理进度
tail -f lightprof_sampling.log | grep "处理进度"

# 统计已处理数量
grep "样本" lightprof_sampling.log | tail -1
```

---

## 🐛 常见问题（Linux特定）

### **问题1: 权限被拒绝**

```bash
# 错误
bash: ./run_lightprof_pipeline.sh: Permission denied

# 解决
chmod +x run_lightprof_pipeline.sh
chmod +x run_lightprof_with_llm.sh
chmod +x batch_process.sh
```

### **问题2: conda命令未找到**

```bash
# 错误
conda: command not found

# 解决方案1: 手动激活conda
source ~/anaconda3/etc/profile.d/conda.sh
conda activate retriever

# 解决方案2: 使用完整路径
~/anaconda3/envs/retriever/bin/python lightprof_sampling.py -i input.pth --use_mock_llm

# 解决方案3: 添加到PATH
echo 'export PATH="$HOME/anaconda3/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
```

### **问题3: 内存不足**

```bash
# 错误
RuntimeError: CUDA out of memory

# 解决：批量处理
python -c "
import torch
data = torch.load('test_retrieval_result.pth')
batch_size = 100
items = list(data.items())

for i in range(0, len(items), batch_size):
    batch = dict(items[i:i+batch_size])
    torch.save(batch, f'batch_{i//batch_size}.pth')
"

# 逐批处理
for file in batch_*.pth; do
    python lightprof_sampling.py -i $file --use_mock_llm
done

# 合并结果
python merge_results.py
```

### **问题4: 文件路径编码问题**

```bash
# 如果有中文路径问题
export LANG=zh_CN.UTF-8
export LC_ALL=zh_CN.UTF-8

# 或使用英文路径
mv "g:\小论文\第一章\新建文件夹\SubgraphRAG-main" ~/SubgraphRAG
```

---

## 📋 服务器运行最佳实践

### **1. 使用screen/tmux运行长时间任务**

```bash
# 创建session
screen -S lightprof_job

# 激活环境并运行
conda activate retriever
cd ~/SubgraphRAG-main/retrieve
bash run_lightprof_pipeline.sh webqsp_hybrid_llm_kg_Feb05-09_40_35/cpt.pth

# 分离session: Ctrl+A, D
# 断开SSH连接（任务继续运行）

# 稍后重新连接
ssh username@server
screen -r lightprof_job
```

### **2. 定时任务（crontab）**

```bash
# 编辑crontab
crontab -e

# 添加定时任务（每天凌晨2点运行）
0 2 * * * cd ~/SubgraphRAG-main/retrieve && bash run_lightprof_pipeline.sh checkpoint.pth >> cron.log 2>&1
```

### **3. 资源监控**

```bash
# 创建监控脚本
cat > monitor.sh << 'EOF'
#!/bin/bash
while true; do
    clear
    echo "=== 系统资源监控 ==="
    echo "时间: $(date)"
    echo
    echo "=== GPU ==="
    nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader
    echo
    echo "=== Python进程 ==="
    ps aux | grep python | grep -v grep
    echo
    echo "=== 磁盘空间 ==="
    df -h | grep -E '/$|/home'
    sleep 5
done
EOF

chmod +x monitor.sh
./monitor.sh
```

---

## 🌐 使用DeepSeek API（国内服务器）

如果在国内服务器，推荐使用DeepSeek API：

```bash
# 设置DeepSeek API
export OPENAI_API_KEY="your_deepseek_key"
export OPENAI_BASE_URL="https://api.deepseek.com/v1"

# 运行
python lightprof_sampling.py \
    -i test_retrieval_result.pth \
    --llm_model deepseek-chat \
    --llm_base_url https://api.deepseek.com/v1
```

---

## 📦 完整示例脚本

创建一个完整的运行脚本：

```bash
cat > run_complete_pipeline.sh << 'EOF'
#!/bin/bash
set -e

# 配置
CHECKPOINT="webqsp_hybrid_llm_kg_Feb05-09_40_35/cpt.pth"
USE_MOCK_LLM=true

# 激活环境
source $(conda info --base)/etc/profile.d/conda.sh
conda activate retriever

cd ~/SubgraphRAG-main/retrieve

echo "开始完整流程..."

# Step 1: Retriever推理
if [ ! -f "webqsp_hybrid_llm_kg_Feb05-09_40_35/test_retrieval_result.pth" ]; then
    echo "[1/2] Retriever推理..."
    python inference_hybrid.py \
        -p $CHECKPOINT \
        --splits test \
        --enable_entity_mapping \
        --num_display_samples 0
else
    echo "[1/2] 跳过Retriever推理（结果已存在）"
fi

# Step 2: LightPROF采样
echo "[2/2] LightPROF采样..."
if [ "$USE_MOCK_LLM" = true ]; then
    python lightprof_sampling.py \
        -i webqsp_hybrid_llm_kg_Feb05-09_40_35/test_retrieval_result.pth \
        --use_mock_llm \
        --num_display_samples 2
else
    python lightprof_sampling.py \
        -i webqsp_hybrid_llm_kg_Feb05-09_40_35/test_retrieval_result.pth \
        --llm_model gpt-4o-mini \
        --num_display_samples 2
fi

echo "✅ 完成！"
EOF

chmod +x run_complete_pipeline.sh
bash run_complete_pipeline.sh
```

---

## 📊 性能基准（服务器环境）

### **处理速度参考**

| 配置 | 处理速度 | 总时间（1639样本） |
|------|---------|-------------------|
| CPU only | ~2 samples/s | ~14分钟 |
| 1x GPU (T4) | ~8 samples/s | ~3.5分钟 |
| 4x GPU (T4) | ~30 samples/s | ~1分钟 |

### **内存需求**

| 阶段 | GPU内存 | 系统内存 |
|------|---------|---------|
| Retriever推理 | ~2GB | ~4GB |
| LightPROF采样 | ~500MB | ~2GB |

---

## 🔄 完整的Linux命令（针对你的需求）

### **你的原始命令（已修正）**

```bash
# 进入工作目录
cd ~/SubgraphRAG-main/retrieve

# 激活环境
conda activate retriever

# Step 1: Retriever推理
python inference_hybrid.py \
    -p webqsp_hybrid_llm_kg_Feb05-09_40_35/cpt.pth \
    --eval_k_list 100 \
    --splits val test \
    --enable_entity_mapping \
    --num_display_samples 3

# Step 2: LightPROF采样（验证集）
python lightprof_sampling.py \
    -i webqsp_hybrid_llm_kg_Feb05-09_40_35/val_retrieval_result.pth \
    --h_q 2 \
    --top_k_chains 5 \
    --use_mock_llm \
    --num_display_samples 2

# Step 3: LightPROF采样（测试集）
python lightprof_sampling.py \
    -i webqsp_hybrid_llm_kg_Feb05-09_40_35/test_retrieval_result.pth \
    --h_q 2 \
    --top_k_chains 5 \
    --use_mock_llm \
    --num_display_samples 2

echo "✅ 全部完成！"
```

### **保存为脚本文件**

```bash
# 创建脚本
cat > my_experiment.sh << 'EOF'
#!/bin/bash
set -e

cd ~/SubgraphRAG-main/retrieve
source $(conda info --base)/etc/profile.d/conda.sh
conda activate retriever

CHECKPOINT="webqsp_hybrid_llm_kg_Feb05-09_40_35/cpt.pth"

# Retriever推理
python inference_hybrid.py \
    -p $CHECKPOINT \
    --eval_k_list 100 \
    --splits val test \
    --enable_entity_mapping \
    --num_display_samples 3

# LightPROF采样
for split in val test; do
    python lightprof_sampling.py \
        -i webqsp_hybrid_llm_kg_Feb05-09_40_35/${split}_retrieval_result.pth \
        --h_q 2 \
        --top_k_chains 5 \
        --use_mock_llm \
        --num_display_samples 2
done

echo "✅ 完成！"
EOF

# 添加执行权限
chmod +x my_experiment.sh

# 后台运行
nohup bash my_experiment.sh > experiment.log 2>&1 &

# 查看日志
tail -f experiment.log
```

---

## 📚 快速参考

### **SSH相关**

```bash
# 上传文件
scp local_file.pth username@server:/path/to/retrieve/

# 下载结果
scp username@server:/path/to/retrieve/*_lightprof.pth ./results/

# SSH保持连接
# 在 ~/.ssh/config 添加:
Host myserver
    HostName your-server.com
    User username
    ServerAliveInterval 60
    ServerAliveCountMax 3
```

### **常用别名**

添加到 `~/.bashrc`：

```bash
alias cdret='cd ~/SubgraphRAG-main/retrieve'
alias actret='conda activate retriever'
alias gpustat='nvidia-smi'
alias checklog='tail -f lightprof_sampling.log'
```

---

## ✅ 迁移检查清单

- [ ] 已将代码上传到服务器
- [ ] 已创建conda环境并安装依赖
- [ ] 已给shell脚本添加执行权限（`chmod +x *.sh`）
- [ ] 已测试Python脚本可以运行（`python test_lightprof.py`）
- [ ] 已设置API密钥（如使用真实LLM）
- [ ] 已验证检查点文件路径正确
- [ ] 已了解如何使用screen/tmux进行后台运行

完成这些检查后，你就可以在Linux服务器上顺畅运行了！🚀
