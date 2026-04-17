# 🚀 LightPROF 完整使用指南

## 📦 已创建的文件

| 文件 | 类型 | 说明 |
|------|------|------|
| `lightprof_sampling.py` | 主脚本 | LightPROF推理图采样实现 |
| `test_lightprof.py` | 测试脚本 | 功能测试和验证 |
| `run_lightprof_example.py` | 示例脚本 | 使用示例集合 |
| `run_lightprof_pipeline.bat` | 批处理 | Windows一键运行 |
| `README_lightprof.md` | 文档 | 详细说明文档 |
| `QUICKSTART_lightprof.md` | 快速入门 | 快速开始指南 |

---

## 🎯 你的具体命令调整

### **原始命令**

你提供的命令有格式错误（重复了参数）。正确的命令应该是：

```bash
python inference_hybrid.py \
    -p webqsp_hybrid_llm_kg_Feb05-09_40_35/cpt.pth \
    --max_K 500 \
    --eval_k_list 50 100 200 500 \
    --splits val test \
    --enable_entity_mapping \
    --num_display_samples 3
```

### **完整的两步流程**

#### **Step 1: Retriever推理**

```bash
python inference_hybrid.py \
    -p webqsp_hybrid_llm_kg_Feb05-09_40_35/cpt.pth \
    --eval_k_list 100 \
    --splits val test \
    --enable_entity_mapping \
    --num_display_samples 3
```

**输出**：
- `webqsp_hybrid_llm_kg_Feb05-09_40_35/val_retrieval_result.pth`
- `webqsp_hybrid_llm_kg_Feb05-09_40_35/test_retrieval_result.pth`

#### **Step 2: LightPROF采样**

```bash
# 验证集
python lightprof_sampling.py \
    -i webqsp_hybrid_llm_kg_Feb05-09_40_35/val_retrieval_result.pth \
    --h_q 2 \
    --top_k_chains 5 \
    --use_mock_llm \
    --num_display_samples 2

# 测试集
python lightprof_sampling.py \
    -i webqsp_hybrid_llm_kg_Feb05-09_40_35/test_retrieval_result.pth \
    --h_q 2 \
    --top_k_chains 5 \
    --use_mock_llm \
    --num_display_samples 2
```

**输出**：
- `webqsp_hybrid_llm_kg_Feb05-09_40_35/val_retrieval_result_lightprof.pth`
- `webqsp_hybrid_llm_kg_Feb05-09_40_35/test_retrieval_result_lightprof.pth`

---

## 📊 数据格式对比

### **inference_hybrid.py 输出**

```python
{
    'sample_id': {
        'question': "问题文本",
        'scored_triples': [
            (head, relation, tail, score),  # 500个三元组
            ...
        ],
        'q_entity': ['原始实体'],
        'q_entity_list': ['扩展实体列表'],
        ...
    }
}
```

### **lightprof_sampling.py 输出**

```python
{
    'sample_id': {
        # === 保留原有所有字段 ===
        'question': "问题文本",
        'scored_triples': [...],  # 保留原始500个
        'q_entity': ['原始实体'],
        'q_entity_list': ['扩展实体列表'],
        ...
        
        # === 新增LightPROF字段 ===
        'lightprof_gr_triples': [
            (head, relation, tail),  # 约30个精选三元组
            ...
        ],
        'lightprof_stats': {
            'num_chains_total': 52,
            'num_chains_selected': 5,
            'num_paths': 15,
            'num_triples': 32
        }
    }
}
```

---

## 🎨 可视化理解

### **算法流程图**

```
┌─────────────────────────────────────────────────────────┐
│  输入: inference_hybrid.py 的输出                        │
│  - 每个问题有500个scored_triples                        │
│  - q_entity_list（扩展的锚点实体）                      │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│  步骤1: 关系链检索（深度受限BFS）                       │
│  ─────────────────────────────────────────────          │
│  Lou Seal ──[sports.mascot.team]──> SF Giants          │
│            ──[winner_of]──> 2010 WS                     │
│                                                          │
│  提取关系链:                                            │
│    • (sports.mascot.team,)                              │
│    • (sports.mascot.team, winner_of)                    │
│    • (sports.mascot.team, located_in)                   │
│    • ...                                                 │
│  共52条链                                               │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│  步骤2a: LLM语义相关性排序                              │
│  ─────────────────────────────────────────────          │
│  问题: "What year did ... win the World Series?"       │
│                                                          │
│  LLM打分:                                               │
│    1. (mascot_team, winner_of)        [0.95] ✅         │
│    2. (mascot_team, located_in)       [0.30]            │
│    3. (mascot_team, sport)            [0.25]            │
│    ...                                                   │
│                                                          │
│  选择Top-5关系链                                        │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│  步骤2b: 推理路径采样                                   │
│  ─────────────────────────────────────────────          │
│  将抽象关系链实例化为具体路径:                          │
│                                                          │
│  链: (mascot_team, winner_of)                           │
│  路径1: Lou Seal → SF Giants → 2010 WS                  │
│  路径2: Lou Seal → SF Giants → 2012 WS                  │
│  路径3: Lou Seal → SF Giants → 2014 WS                  │
│                                                          │
│  构建推理图 G_R: 收集所有路径的三元组                   │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│  输出: lightprof_gr_triples                             │
│  - 从500个压缩到32个三元组                              │
│  - 保留了最关键的推理路径                               │
│  - 可直接用于LLM推理                                    │
└─────────────────────────────────────────────────────────┘
```

---

## 🧪 测试和验证

### **1. 功能测试**

```bash
# 运行完整的测试套件
python test_lightprof.py
```

预期输出：
```
✅ 所有测试通过！
```

### **2. 示例演示**

```bash
# 查看各种使用示例
python run_lightprof_example.py
```

### **3. 小批量验证**

在处理完整数据集之前，先用小样本测试：

```bash
# 创建一个只包含10个样本的测试文件
python -c "
import torch
data = torch.load('test_retrieval_result.pth')
small_data = dict(list(data.items())[:10])
torch.save(small_data, 'test_small.pth')
"

# 处理小样本
python lightprof_sampling.py -i test_small.pth --use_mock_llm
```

---

## 📋 检查清单

在运行完整流程前，请确认：

- [ ] ✅ 已训练好retriever模型（有cpt.pth文件）
- [ ] ✅ 已运行inference_hybrid.py生成检索结果
- [ ] ✅ 检索结果文件存在且格式正确
- [ ] ✅ 已安装必要依赖（torch, tqdm）
- [ ] ✅ 如使用真实LLM，已设置API密钥
- [ ] ✅ 已运行test_lightprof.py验证功能

---

## 💻 实战示例

### **场景1: 快速测试（使用模拟LLM）**

```bash
# 1. 检查输入文件是否存在
dir webqsp_hybrid_llm_kg_Feb05-09_40_35\test_retrieval_result.pth

# 2. 运行LightPROF采样
python lightprof_sampling.py ^
    -i webqsp_hybrid_llm_kg_Feb05-09_40_35/test_retrieval_result.pth ^
    --use_mock_llm ^
    --num_display_samples 3

# 3. 检查输出
dir webqsp_hybrid_llm_kg_Feb05-09_40_35\test_retrieval_result_lightprof.pth
```

### **场景2: 正式实验（使用GPT-4o-mini）**

```bash
# 1. 设置API密钥
set OPENAI_API_KEY=sk-your-key-here

# 2. 处理验证集
python lightprof_sampling.py ^
    -i webqsp_hybrid_llm_kg_Feb05-09_40_35/val_retrieval_result.pth ^
    --llm_model gpt-4o-mini ^
    --h_q 2 ^
    --top_k_chains 5

# 3. 处理测试集
python lightprof_sampling.py ^
    -i webqsp_hybrid_llm_kg_Feb05-09_40_35/test_retrieval_result.pth ^
    --llm_model gpt-4o-mini ^
    --h_q 2 ^
    --top_k_chains 5
```

### **场景3: 参数调优实验**

```bash
# 对比不同的top_k设置
for k in 3 5 10; do
    python lightprof_sampling.py \
        -i test_retrieval_result.pth \
        -o test_result_k${k}.pth \
        --top_k_chains $k \
        --use_mock_llm
done

# 分析不同参数的效果
python analyze_lightprof_results.py test_result_k*.pth
```

---

## 🎓 理解关键概念

### **关系链 vs 推理路径**

**关系链（抽象）**：
```
(r1, r2, r3)
例如: (mascot_of, winner_of)
```

**推理路径（具体）**：
```
(e0, r1, e1, r2, e2, r3, e3)
例如: Lou Seal -[mascot_of]-> SF Giants -[winner_of]-> 2010 WS
```

### **为什么需要两步？**

1. **步骤1（关系链）**：探索所有可能的关系组合
2. **步骤2a（LLM排序）**：找出最相关的关系组合
3. **步骤2b（实例化）**：将相关的关系链转换为具体的实体路径

**优势**：分离抽象和具体，让LLM专注于语义相关性判断。

---

## 🔬 实验建议

### **消融实验**

对比不同方法的效果：

| 方法 | 三元组数 | 命令 |
|------|---------|------|
| Baseline | 500 | 直接使用scored_triples |
| Top-100 | 100 | scored_triples[:100] |
| **LightPROF** | ~30 | lightprof_gr_triples |

### **参数敏感性分析**

```bash
# 测试不同的top_k
for k in 3 5 7 10; do
    python lightprof_sampling.py -i input.pth --top_k_chains $k --use_mock_llm
done

# 测试不同的h_q
for h in 1 2 3; do
    python lightprof_sampling.py -i input.pth --h_q $h --use_mock_llm
done
```

---

## 🎉 总结

你现在拥有完整的LightPROF实现：

1. ✅ **主脚本**：`lightprof_sampling.py` - 完整的采样实现
2. ✅ **测试脚本**：`test_lightprof.py` - 验证功能
3. ✅ **示例脚本**：`run_lightprof_example.py` - 学习如何使用
4. ✅ **批处理**：`run_lightprof_pipeline.bat` - 一键运行
5. ✅ **文档**：完整的使用说明和快速入门

### **核心改进**

相比你提供的初始代码，新实现包含：

- ✅ 更完整的BFS实现（支持环路检测）
- ✅ 真实LLM API集成（OpenAI、DeepSeek等）
- ✅ 双向图支持（可选反向遍历）
- ✅ 详细的统计和可视化
- ✅ 批量处理能力
- ✅ 完善的错误处理

### **下一步行动**

```bash
# 1. 快速测试
python test_lightprof.py

# 2. 处理你的数据
python lightprof_sampling.py \
    -i webqsp_hybrid_llm_kg_Feb05-09_40_35/test_retrieval_result.pth \
    --use_mock_llm

# 3. 查看结果
python -c "import torch; d=torch.load('webqsp_hybrid_llm_kg_Feb05-09_40_35/test_retrieval_result_lightprof.pth'); print(f'样本数: {len(d)}'); sample=list(d.values())[0]; print(f'推理图三元组: {len(sample[\"lightprof_gr_triples\"])}')"
```

开始使用吧！🚀
