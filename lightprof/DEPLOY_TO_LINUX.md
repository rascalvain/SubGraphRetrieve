# 🚀 Linux服务器部署完全指南

## 📋 部署步骤

### **Step 1: 上传代码到服务器**

#### **方式A: 使用scp（推荐）**

在Windows PowerShell中执行：

```powershell
# 上传整个项目
scp -r "g:\小论文\第一章\新建文件夹\SubgraphRAG-main" username@server-ip:~/

# 或只上传retrieve目录
scp -r "g:\小论文\第一章\新建文件夹\SubgraphRAG-main\retrieve" username@server-ip:~/SubgraphRAG-main/
```

#### **方式B: 使用WinSCP/FileZilla（图形界面）**

1. 下载并安装 WinSCP
2. 连接到服务器
3. 拖拽文件夹上传

---

### **Step 2: SSH登录并配置环境**

```bash
# 登录服务器
ssh username@server-ip

# 进入项目目录
cd ~/SubgraphRAG-main/retrieve

# 运行配置脚本
bash setup_linux.sh
```

配置脚本会自动：
- ✅ 检查Python和conda环境
- ✅ 创建/激活retriever环境
- ✅ 安装必需的依赖
- ✅ 设置脚本执行权限
- ✅ 运行功能测试

---

### **Step 3: 运行测试**

```bash
# 激活环境
conda activate retriever

# 测试功能
python test_lightprof.py
```

预期输出：
```
✅ 所有测试通过！
```

---

### **Step 4: 处理实际数据**

#### **4.1 检查输入文件**

```bash
# 确认检查点文件存在
ls -lh webqsp_hybrid_llm_kg_Feb05-09_40_35/cpt.pth

# 如果检查点在Windows上，需要先上传
# scp checkpoint.pth username@server:~/SubgraphRAG-main/retrieve/
```

#### **4.2 运行完整流程**

```bash
# 使用模拟LLM（快速测试）
bash run_lightprof_pipeline.sh webqsp_hybrid_llm_kg_Feb05-09_40_35/cpt.pth
```

或手动执行：

```bash
# Retriever推理
python inference_hybrid.py \
    -p webqsp_hybrid_llm_kg_Feb05-09_40_35/cpt.pth \
    --splits val test \
    --enable_entity_mapping \
    --num_display_samples 3

# LightPROF采样
python lightprof_sampling.py \
    -i webqsp_hybrid_llm_kg_Feb05-09_40_35/val_retrieval_result.pth \
    --use_mock_llm

python lightprof_sampling.py \
    -i webqsp_hybrid_llm_kg_Feb05-09_40_35/test_retrieval_result.pth \
    --use_mock_llm
```

---

## 🔑 使用真实LLM（可选）

### **配置API密钥**

```bash
# 方式1: 临时设置
export OPENAI_API_KEY='sk-your-key-here'

# 方式2: 永久设置（推荐）
echo 'export OPENAI_API_KEY="sk-your-key-here"' >> ~/.bashrc
source ~/.bashrc

# 验证
echo $OPENAI_API_KEY
```

### **运行**

```bash
# 使用真实LLM
bash run_lightprof_with_llm.sh webqsp_hybrid_llm_kg_Feb05-09_40_35/cpt.pth

# 或指定API密钥
bash run_lightprof_with_llm.sh webqsp_hybrid_llm_kg_Feb05-09_40_35/cpt.pth 'your_key'
```

---

## 🎯 后台运行（重要！）

对于长时间任务，**必须**使用后台运行方式：

### **推荐: 使用screen（最简单）**

```bash
# 1. 创建screen session
screen -S lightprof_job

# 2. 在screen中运行任务
conda activate retriever
cd ~/SubgraphRAG-main/retrieve
bash run_lightprof_pipeline.sh webqsp_hybrid_llm_kg_Feb05-09_40_35/cpt.pth

# 3. 分离screen（按键）
# Ctrl+A, 然后按 D

# 4. 断开SSH（任务继续运行！）
exit

# 5. 稍后重新连接
ssh username@server
screen -r lightprof_job

# 查看所有screen session
screen -ls

# 终止session（在screen内）
exit
```

### **备选: 使用nohup**

```bash
# 后台运行
nohup bash run_lightprof_pipeline.sh checkpoint.pth > pipeline.log 2>&1 &

# 查看进程ID
echo $!

# 查看日志
tail -f pipeline.log

# 断开SSH后，任务继续运行

# 稍后查看进程
ps aux | grep lightprof

# 终止进程
kill <pid>
```

---

## 📥 从服务器下载结果

### **下载单个文件**

```bash
# 在Windows PowerShell中
scp username@server:~/SubgraphRAG-main/retrieve/test_retrieval_result_lightprof.pth ./results/
```

### **下载所有结果**

```bash
# 下载所有LightPROF结果
scp username@server:~/SubgraphRAG-main/retrieve/*_lightprof.pth ./results/

# 下载整个检查点目录
scp -r username@server:~/SubgraphRAG-main/retrieve/webqsp_hybrid_llm_kg_Feb05-09_40_35 ./results/
```

---

## 🎮 实战示例

### **场景1: 快速测试（10分钟内完成）**

```bash
# SSH登录
ssh username@server

# 配置环境
cd ~/SubgraphRAG-main/retrieve
bash setup_linux.sh

# 测试功能
python test_lightprof.py

# 处理小样本
python -c "
import torch
d = torch.load('test_retrieval_result.pth')
small = dict(list(d.items())[:10])
torch.save(small, 'test_small.pth')
"

python lightprof_sampling.py -i test_small.pth --use_mock_llm

echo "✅ 测试完成！"
```

### **场景2: 正式实验（后台运行）**

```bash
# SSH登录
ssh username@server

# 创建screen session
screen -S my_experiment

# 激活环境
conda activate retriever
cd ~/SubgraphRAG-main/retrieve

# 运行完整流程
bash run_lightprof_pipeline.sh webqsp_hybrid_llm_kg_Feb05-09_40_35/cpt.pth

# 分离screen: Ctrl+A, D
# 现在可以安全地关闭SSH连接

# 稍后查看进度
ssh username@server
screen -r my_experiment
```

### **场景3: 多GPU并行处理**

```bash
# 分割数据
python -c "
import torch
data = torch.load('test_retrieval_result.pth')
items = list(data.items())
n = len(items) // 4

for i in range(4):
    start, end = i * n, (i+1) * n if i < 3 else len(items)
    torch.save(dict(items[start:end]), f'test_part{i}.pth')
"

# 并行处理
for i in {0..3}; do
    CUDA_VISIBLE_DEVICES=$i python lightprof_sampling.py \
        -i test_part${i}.pth \
        --use_mock_llm \
        --num_display_samples 0 &
done

wait  # 等待所有任务完成

# 合并结果
python merge_results.py \
    -i test_part*_lightprof.pth \
    -o test_retrieval_result_lightprof.pth
```

---

## 📊 监控和调试

### **实时监控脚本**

创建 `monitor.sh`：

```bash
#!/bin/bash
while true; do
    clear
    echo "=== 时间: $(date) ==="
    echo
    echo "=== GPU使用情况 ==="
    nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader
    echo
    echo "=== Python进程 ==="
    ps aux | grep python | grep -v grep | head -5
    echo
    echo "=== 最新生成的文件 ==="
    ls -lht *_lightprof.pth 2>/dev/null | head -3
    echo
    echo "=== 磁盘使用 ==="
    df -h | grep -E '/$|/home'
    sleep 5
done
```

运行：

```bash
chmod +x monitor.sh
./monitor.sh
```

### **查看实时日志**

```bash
# 如果使用nohup
tail -f pipeline.log

# 动态监控关键信息
tail -f pipeline.log | grep -E "处理进度|完成|错误"

# 统计进度
grep "处理进度" pipeline.log | tail -1
```

---

## 🔧 故障排除

### **问题1: ImportError**

```bash
# 错误
ModuleNotFoundError: No module named 'sentence_transformers'

# 解决
conda activate retriever
pip install sentence-transformers
```

### **问题2: CUDA错误**

```bash
# 错误
RuntimeError: CUDA out of memory

# 解决1: 使用CPU
export CUDA_VISIBLE_DEVICES=""
python lightprof_sampling.py -i input.pth --use_mock_llm

# 解决2: 减少batch size（在代码中调整）
# 解决3: 使用更小的GPU或分批处理
```

### **问题3: 权限问题**

```bash
# 错误
Permission denied: *.sh

# 解决
chmod +x *.sh

# 或使用bash明确执行
bash run_lightprof_pipeline.sh checkpoint.pth
```

### **问题4: 路径不存在**

```bash
# 错误
FileNotFoundError: test_retrieval_result.pth

# 检查当前目录
pwd
ls -lh *.pth

# 使用绝对路径
python lightprof_sampling.py \
    -i ~/SubgraphRAG-main/retrieve/test_retrieval_result.pth \
    --use_mock_llm
```

---

## 📝 完整的部署检查清单

### **上传前（Windows）**

- [ ] 确认所有Python脚本无语法错误
- [ ] 确认检查点文件准备好
- [ ] 记录服务器IP、用户名、密码

### **服务器配置**

- [ ] SSH登录成功
- [ ] 代码已上传到服务器
- [ ] conda环境已创建（`conda activate retriever`）
- [ ] 依赖已安装（`pip list | grep torch`）
- [ ] shell脚本有执行权限（`ls -l *.sh`）
- [ ] 功能测试通过（`python test_lightprof.py`）

### **运行任务**

- [ ] 检查点文件路径正确
- [ ] API密钥已设置（如使用真实LLM）
- [ ] 使用screen/tmux后台运行
- [ ] 监控任务运行状态

### **结果验证**

- [ ] 输出文件已生成（`ls *_lightprof.pth`）
- [ ] 文件大小合理（`du -h *.pth`）
- [ ] 数据格式正确（快速加载测试）
- [ ] 下载结果到本地

---

## 🎯 你的具体命令（Linux版本）

基于你的需求，这是在Linux服务器上运行的完整命令：

```bash
# 1. 登录服务器
ssh your-username@your-server

# 2. 进入工作目录
cd ~/SubgraphRAG-main/retrieve

# 3. 激活环境
conda activate retriever

# 4. 运行Retriever推理
python inference_hybrid.py \
    -p webqsp_hybrid_llm_kg_Feb05-09_40_35/cpt.pth \
    --eval_k_list 100 \
    --splits val test \
    --enable_entity_mapping \
    --num_display_samples 3

# 5. 运行LightPROF采样（验证集）
python lightprof_sampling.py \
    -i webqsp_hybrid_llm_kg_Feb05-09_40_35/val_retrieval_result.pth \
    --h_q 2 \
    --top_k_chains 5 \
    --use_mock_llm \
    --num_display_samples 2

# 6. 运行LightPROF采样（测试集）
python lightprof_sampling.py \
    -i webqsp_hybrid_llm_kg_Feb05-09_40_35/test_retrieval_result.pth \
    --h_q 2 \
    --top_k_chains 5 \
    --use_mock_llm \
    --num_display_samples 2

# 7. 验证结果
ls -lh webqsp_hybrid_llm_kg_Feb05-09_40_35/*_lightprof.pth
```

### **一键运行版本**

```bash
# 创建并运行脚本
cat > run_my_experiment.sh << 'EOF'
#!/bin/bash
set -e

cd ~/SubgraphRAG-main/retrieve
source $(conda info --base)/etc/profile.d/conda.sh
conda activate retriever

CHECKPOINT="webqsp_hybrid_llm_kg_Feb05-09_40_35/cpt.pth"

# Retriever
python inference_hybrid.py \
    -p $CHECKPOINT \
    --eval_k_list 100 \
    --splits val test \
    --enable_entity_mapping \
    --num_display_samples 3

# LightPROF
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

chmod +x run_my_experiment.sh

# 后台运行
nohup bash run_my_experiment.sh > experiment.log 2>&1 &

# 查看日志
tail -f experiment.log
```

---

## 🌐 网络配置（国内服务器）

### **使用国内镜像加速**

```bash
# PyTorch清华镜像
pip install torch -i https://pypi.tuna.tsinghua.edu.cn/simple

# 通用pip镜像
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple sentence-transformers openai

# 永久配置
mkdir -p ~/.pip
cat > ~/.pip/pip.conf << EOF
[global]
index-url = https://pypi.tuna.tsinghua.edu.cn/simple
[install]
trusted-host = pypi.tuna.tsinghua.edu.cn
EOF
```

### **使用DeepSeek API（国内）**

```bash
# 设置DeepSeek
export OPENAI_API_KEY="your_deepseek_key"

# 运行
python lightprof_sampling.py \
    -i test_retrieval_result.pth \
    --llm_model deepseek-chat \
    --llm_base_url https://api.deepseek.com/v1
```

---

## 📊 预计运行时间

基于1639个样本（WebQSP测试集）：

| 阶段 | 配置 | 预计时间 |
|------|------|---------|
| Retriever推理 | 1x GPU (V100) | ~5分钟 |
| LightPROF采样（模拟LLM） | CPU | ~3分钟 |
| LightPROF采样（真实LLM） | API调用 | ~30分钟* |
| **总计（模拟LLM）** | - | **~8分钟** |

*使用GPT-4o-mini，取决于API速率限制

---

## 🎁 提供的Linux脚本

| 文件 | 用途 |
|------|------|
| `setup_linux.sh` | 🔧 一键配置环境 |
| `run_lightprof_pipeline.sh` | 🚀 完整流程（模拟LLM） |
| `run_lightprof_with_llm.sh` | 🤖 完整流程（真实LLM） |
| `batch_process.sh` | 📦 批量处理多个文件 |
| `merge_results.py` | 🔗 合并并行处理结果 |
| `LINUX_使用指南.md` | 📚 详细使用说明 |
| `LINUX_COMMANDS.txt` | 📋 快速命令参考 |

---

## 🚨 重要提示

### **1. 务必使用后台运行**

❌ **错误做法**：
```bash
ssh username@server
python lightprof_sampling.py -i input.pth --use_mock_llm
# 关闭SSH → 任务终止！
```

✅ **正确做法**：
```bash
ssh username@server
screen -S job
python lightprof_sampling.py -i input.pth --use_mock_llm
# Ctrl+A, D 分离
# 关闭SSH → 任务继续运行
```

### **2. 定期检查进度**

```bash
# 重新连接查看
ssh username@server
screen -r job

# 或查看日志
tail -100 experiment.log
```

### **3. 保存结果**

任务完成后，及时下载结果到本地：

```bash
scp username@server:~/SubgraphRAG-main/retrieve/*_lightprof.pth ./local_results/
```

---

## ✅ 快速开始（30秒）

```bash
# 在服务器上执行
cd ~/SubgraphRAG-main/retrieve && \
bash setup_linux.sh && \
python test_lightprof.py && \
echo "✅ 环境配置完成，可以开始处理数据！"
```

---

## 📞 获取帮助

如果遇到问题：

1. 查看日志文件
2. 运行 `python test_lightprof.py` 诊断
3. 检查 `LINUX_COMMANDS.txt` 快速参考
4. 阅读 `LINUX_使用指南.md` 详细说明

祝你在服务器上运行顺利！🎉
