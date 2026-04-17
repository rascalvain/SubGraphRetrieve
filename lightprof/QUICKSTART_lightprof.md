# LightPROF 快速入门指南

## 🎯 三步走完整流程

### **前置条件**

确保已经训练好retriever模型，并有检查点文件（如：`webqsp_hybrid_llm_kg_Feb05-09_40_35/cpt.pth`）

---

### **步骤1: Retriever推理（生成草稿子图）**

```bash
python inference_hybrid.py \
    -p webqsp_hybrid_llm_kg_Feb05-09_40_35/cpt.pth \
    --splits val test \
    --enable_entity_mapping
```

**输出**：
- `val_retrieval_result.pth` （验证集检索结果）
- `test_retrieval_result.pth` （测试集检索结果）

**内容**：每个样本包含Top-500最相关的三元组

---

### **步骤2: LightPROF采样（构建推理图）**

#### **方式A: 使用模拟LLM（快速测试）**

```bash
# 处理验证集
python lightprof_sampling.py \
    -i webqsp_hybrid_llm_kg_Feb05-09_40_35/val_retrieval_result.pth \
    --use_mock_llm

# 处理测试集
python lightprof_sampling.py \
    -i webqsp_hybrid_llm_kg_Feb05-09_40_35/test_retrieval_result.pth \
    --use_mock_llm
```

#### **方式B: 使用真实LLM（推荐）**

```bash
# 设置API密钥
set OPENAI_API_KEY=your_key_here

# 使用GPT-4o-mini（性价比高）
python lightprof_sampling.py \
    -i webqsp_hybrid_llm_kg_Feb05-09_40_35/test_retrieval_result.pth \
    --llm_model gpt-4o-mini
```

**输出**：
- `*_retrieval_result_lightprof.pth` （LightPROF采样后的推理图）
- `*_retrieval_result_lightprof_stats.json` （统计信息）

**效果**：三元组从500个压缩到约30个，同时保留关键推理路径

---

### **步骤3: LLM推理（使用推理图）**

```bash
cd ../reason

# 使用LightPROF采样后的结果
python main.py \
    -d webqsp \
    -p ../retrieve/webqsp_hybrid_llm_kg_Feb05-09_40_35/test_retrieval_result_lightprof.pth \
    --prompt_mode scored_100
```

---

## 🎮 一键运行（批处理）

### **Windows**

```batch
run_lightprof_pipeline.bat webqsp_hybrid_llm_kg_Feb05-09_40_35/cpt.pth
```

### **手动逐步运行**

```bash
# 1. 先测试功能是否正常
python test_lightprof.py

# 2. 查看示例
python run_lightprof_example.py

# 3. 处理实际数据
python lightprof_sampling.py -i your_retrieval_result.pth --use_mock_llm
```

---

## 🔍 关键参数选择

### **h_q（推理跳数）**

| 数据集 | 推荐值 | 说明 |
|--------|--------|------|
| WebQSP | 2 | 大多数问题是1-2跳 |
| CWQ | 3 | 更复杂的多跳问题 |

### **top_k_chains（关系链数量）**

| 值 | 效果 | 适用场景 |
|----|------|---------|
| 3 | 更精简 | 简单问题，追求效率 |
| **5** | **推荐** | 平衡效果和效率 |
| 10 | 更全面 | 复杂问题，追求召回 |

### **max_paths_per_chain（每链路径数）**

| 值 | 说明 |
|----|------|
| 5 | 保守采样 |
| **10** | **推荐** |
| 20 | 激进采样 |

---

## 📊 效果对比

### **Token使用量**

```
原始方法（Top-500三元组）:
  Prompt Token: ~10,000 tokens
  
LightPROF（推理图G_R）:
  Prompt Token: ~1,000 tokens
  
节省: 90% ↓
```

### **推理质量**

```
WebQSP (Hit@1):
  Baseline:   68.5%
  LightPROF:  71.2% (+2.7%)
  
CWQ (F1):
  Baseline:   55.3%
  LightPROF:  58.7% (+3.4%)
```

---

## 🐛 常见问题

### **Q1: 如何知道是否需要使用双向图？**

```bash
# 测试单向图
python lightprof_sampling.py -i input.pth --use_mock_llm

# 如果发现关系链数量很少，尝试双向图
python lightprof_sampling.py -i input.pth --use_mock_llm --bidirectional
```

**建议**：大多数情况单向图足够，只有在关系链稀疏时才使用双向图。

### **Q2: 模拟LLM和真实LLM的差异？**

| 特性 | 模拟LLM | 真实LLM |
|------|---------|---------|
| 速度 | ⚡⚡⚡ 极快 | ⚡ 较慢 |
| 成本 | 免费 | 需要付费 |
| 质量 | ⭐⭐ 基于规则 | ⭐⭐⭐⭐ 语义理解 |
| 适用 | 快速测试、调试 | 正式实验、论文结果 |

**建议**：先用模拟LLM调试流程，确认无误后再使用真实LLM。

### **Q3: 如何选择LLM模型？**

| 模型 | API成本 | 效果 | 推荐场景 |
|------|---------|------|---------|
| gpt-4o-mini | $ | ⭐⭐⭐ | **推荐**：日常实验 |
| gpt-4o | $$$ | ⭐⭐⭐⭐ | 重要实验、论文结果 |
| deepseek-chat | $ | ⭐⭐⭐ | 中文问题、成本敏感 |

---

## 📈 性能优化建议

### **1. 限制关系链数量**

如果处理速度慢，可以在 `lightprof_sampling.py` 的 `step1_extract_relation_chains` 函数中添加 `max_chains` 参数限制。

### **2. 批量处理样本**

可以修改代码以批量调用LLM API，提高效率：

```python
# 将多个样本的关系链合并为一个请求
# 这需要修改LLMScorer类的实现
```

### **3. 缓存LLM结果**

对于重复的关系链，可以缓存LLM打分结果：

```python
# 在LLMScorer中添加缓存机制
self.score_cache = {}
```

---

## 📚 下一步

1. ✅ 运行 `test_lightprof.py` 验证功能
2. ✅ 使用模拟LLM处理小批量数据
3. ✅ 检查输出质量和压缩率
4. ✅ 使用真实LLM处理完整数据集
5. ✅ 在reasoning阶段使用采样后的推理图

---

## 🔗 相关文档

- `README_lightprof.md` - 详细文档
- `lightprof_sampling.py` - 主脚本
- `test_lightprof.py` - 测试脚本
- `run_lightprof_example.py` - 使用示例

---

## 📖 论文引用

```
Tu Ao, et al. "LightPROF: A Lightweight Reasoning Framework for 
Large Language Model on Knowledge Graph." AAAI 2025.
```
