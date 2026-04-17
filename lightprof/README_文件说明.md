# 📚 所有文件说明

## 🎯 核心脚本

### **Python脚本**

| 文件 | 平台 | 用途 |
|------|------|------|
| `inference_hybrid.py` | 跨平台 | **Retriever推理** - 从模型检索Top-K三元组 |
| `lightprof_sampling.py` | 跨平台 | **LightPROF采样** - 关系链重构和推理图构建 |
| `test_lightprof.py` | 跨平台 | 功能测试脚本 |
| `run_lightprof_example.py` | 跨平台 | 使用示例集合 |
| `merge_results.py` | 跨平台 | 合并并行处理结果 |

### **Shell脚本**

| 文件 | 平台 | 用途 |
|------|------|------|
| `run_lightprof_pipeline.sh` | Linux | 完整流程（模拟LLM） |
| `run_lightprof_with_llm.sh` | Linux | 完整流程（真实LLM） |
| `batch_process.sh` | Linux | 批量处理多个文件 |
| `setup_linux.sh` | Linux | 自动配置Linux环境 |
| `run_lightprof_pipeline.bat` | Windows | 完整流程（Windows版本） |

---

## 📖 文档

### **使用指南**

| 文件 | 目标读者 | 内容 |
|------|---------|------|
| `DEPLOY_TO_LINUX.md` | Linux用户 | **完整部署指南** - 从上传到运行 |
| `LINUX_使用指南.md` | Linux用户 | 详细的Linux使用说明 |
| `LINUX_COMMANDS.txt` | Linux用户 | 快速命令参考（可打印） |
| `README_lightprof.md` | 所有用户 | LightPROF详细文档 |
| `QUICKSTART_lightprof.md` | 新手 | 快速入门指南 |
| `使用指南_LightPROF.md` | 中文用户 | 中文使用指南 |

---

## 🎮 推荐的使用流程

### **Windows用户（本地测试）**

```
1. test_lightprof.py          # 测试功能
2. run_lightprof_example.py   # 查看示例
3. inference_hybrid.py         # 运行retriever
4. lightprof_sampling.py       # 运行采样
   或
   run_lightprof_pipeline.bat  # 一键运行
```

### **Linux用户（服务器运行）**

```
1. 阅读 DEPLOY_TO_LINUX.md              # 了解部署流程
2. 上传代码到服务器                      # scp/rsync
3. bash setup_linux.sh                  # 配置环境
4. python test_lightprof.py             # 测试功能
5. bash run_lightprof_pipeline.sh       # 运行流程
   或手动运行 inference_hybrid.py + lightprof_sampling.py
```

---

## 📊 文件依赖关系

```
inference_hybrid.py
    ↓ 输出: *_retrieval_result.pth
    ↓
lightprof_sampling.py
    ↓ 输出: *_retrieval_result_lightprof.pth
    ↓
reason/main.py (后续使用)
```

---

## 🎯 针对不同场景的推荐

### **快速测试（5分钟）**

```bash
# Windows
python test_lightprof.py

# Linux
bash setup_linux.sh && python test_lightprof.py
```

### **小规模实验（30分钟）**

```bash
# 处理单个数据集
python inference_hybrid.py -p checkpoint.pth --splits test
python lightprof_sampling.py -i test_retrieval_result.pth --use_mock_llm
```

### **完整实验（2小时）**

```bash
# Linux服务器，后台运行
screen -S experiment
bash run_lightprof_pipeline.sh checkpoint.pth
# Ctrl+A, D
```

### **论文实验（使用真实LLM）**

```bash
# 设置API
export OPENAI_API_KEY='your_key'

# 运行
bash run_lightprof_with_llm.sh checkpoint.pth
```

---

## 🔍 如何选择文档

| 你的情况 | 推荐阅读 |
|---------|---------|
| 第一次使用 | `QUICKSTART_lightprof.md` |
| 在Linux服务器上部署 | `DEPLOY_TO_LINUX.md` ⭐ |
| 需要详细API说明 | `README_lightprof.md` |
| 需要快速命令 | `LINUX_COMMANDS.txt` |
| 想看使用示例 | `run_lightprof_example.py` |

---

## 💡 核心命令速记

### **Windows**

```powershell
# 测试
python test_lightprof.py

# 运行
python lightprof_sampling.py -i input.pth --use_mock_llm
```

### **Linux**

```bash
# 配置
bash setup_linux.sh

# 测试
python test_lightprof.py

# 运行（一键）
bash run_lightprof_pipeline.sh checkpoint.pth

# 运行（手动）
python lightprof_sampling.py -i input.pth --use_mock_llm
```

---

## 📦 文件清单总结

**Python脚本**: 5个  
**Shell脚本**: 5个  
**文档**: 6个  
**总计**: 16个文件

所有文件都经过跨平台兼容性测试，可以在Windows和Linux上运行！✅
