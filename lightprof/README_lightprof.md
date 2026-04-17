# LightPROF 推理图采样

## 📖 概述

`lightprof_sampling.py` 实现了 LightPROF 论文中的**关系链重构与推理子图采样**算法。

该脚本以 `inference_hybrid.py` 的输出为输入，通过以下三个步骤构建精炼的推理图：

1. **关系链检索**：从锚点实体出发，使用深度受限BFS收集所有关系链
2. **语义相关性排序**：使用LLM对关系链进行语义相关性打分，选择Top-K最相关的链
3. **推理图采样**：将选中的关系链实例化为具体的推理路径，构建推理图 G_R

---

## 🎯 算法流程

### **步骤1: 关系链检索**

从锚点实体集合 `q_entity_list` 出发，使用深度受限的BFS算法：

```
输入: 
  - 锚点实体: ['Lou Seal']
  - 最大跳数: h_q = 2
  - 草稿子图: scored_triples (来自retriever)

输出:
  - 关系链集合: {
      ('mascot_of',),
      ('mascot_of', 'winner_of'),
      ('mascot_of', 'located_in'),
      ...
  }
```

### **步骤2a: 语义相关性排序与筛选**

使用LLM对所有关系链进行打分：

```
输入:
  - 问题: "What year did the team with mascot named Lou Seal win the World Series?"
  - 关系链列表

LLM打分:
  1. mascot_of -> winner_of  [Score: 0.95]
  2. mascot_of -> located_in  [Score: 0.30]
  ...

输出:
  - Top-K关系链: [('mascot_of', 'winner_of'), ...]
```

### **步骤2b: 推理路径采样与推理图构建**

将抽象的关系链实例化为具体的实体路径：

```
关系链: ('mascot_of', 'winner_of')

实例化路径:
  Path 1: Lou Seal -[mascot_of]-> San Francisco Giants -[winner_of]-> 2010 World Series
  Path 2: Lou Seal -[mascot_of]-> San Francisco Giants -[winner_of]-> 2012 World Series
  Path 3: Lou Seal -[mascot_of]-> San Francisco Giants -[winner_of]-> 2014 World Series

推理图 G_R:
  [(Lou Seal, mascot_of, San Francisco Giants),
   (San Francisco Giants, winner_of, 2010 World Series),
   (San Francisco Giants, winner_of, 2012 World Series),
   (San Francisco Giants, winner_of, 2014 World Series)]
```

---

## 🚀 使用方法

### **基础用法（模拟LLM）**

```bash
# 处理test集的检索结果
python lightprof_sampling.py \
    -i webqsp_hybrid_llm_kg_Feb05-09_40_35/test_retrieval_result.pth \
    --use_mock_llm

# 处理val集的检索结果
python lightprof_sampling.py \
    -i webqsp_hybrid_llm_kg_Feb05-09_40_35/val_retrieval_result.pth \
    --use_mock_llm
```

### **使用真实LLM（OpenAI API）**

```bash
# 设置API密钥
export OPENAI_API_KEY="your_api_key_here"

# 使用GPT-4o-mini
python lightprof_sampling.py \
    -i test_retrieval_result.pth \
    --llm_model gpt-4o-mini

# 使用GPT-4o
python lightprof_sampling.py \
    -i test_retrieval_result.pth \
    --llm_model gpt-4o
```

### **使用DeepSeek API**

```bash
python lightprof_sampling.py \
    -i test_retrieval_result.pth \
    --llm_model deepseek-chat \
    --llm_api_key "your_deepseek_key" \
    --llm_base_url "https://api.deepseek.com/v1"
```

### **完整参数示例**

```bash
python lightprof_sampling.py \
    -i webqsp_hybrid_llm_kg_Feb05-09_40_35/test_retrieval_result.pth \
    -o webqsp_hybrid_llm_kg_Feb05-09_40_35/test_lightprof_result.pth \
    --h_q 2 \
    --top_k_chains 5 \
    --max_paths_per_chain 10 \
    --bidirectional \
    --use_mock_llm \
    --num_display_samples 3
```

---

## 📋 参数说明

### **必需参数**

| 参数 | 说明 | 示例 |
|------|------|------|
| `-i`, `--input` | 输入文件路径（inference_hybrid的输出） | `test_retrieval_result.pth` |

### **可选参数**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `-o`, `--output` | `{input}_lightprof.pth` | 输出文件路径 |
| `--h_q` | 2 | 推理跳数 |
| `--top_k_chains` | 5 | 选择的Top-K关系链数量 |
| `--max_paths_per_chain` | 10 | 每条链最多采样的路径数 |
| `--bidirectional` | False | 使用双向图（允许反向遍历） |
| `--use_mock_llm` | False | 使用模拟LLM（测试用） |
| `--llm_model` | `gpt-4o-mini` | LLM模型名称 |
| `--llm_api_key` | 环境变量 | LLM API密钥 |
| `--llm_base_url` | None | LLM API基础URL |
| `--num_display_samples` | 2 | 展示的样本数量 |

---

## 📦 输出格式

### **输出文件结构**

输出的 `.pth` 文件保留原有所有字段，并新增两个字段：

```python
{
    'sample_id': {
        # === 原有字段（来自inference_hybrid.py） ===
        'question': str,
        'scored_triples': [(h, r, t, score), ...],
        'q_entity': [str, ...],
        'q_entity_list': [str, ...],
        'a_entity': [str, ...],
        ...
        
        # === 新增字段（LightPROF采样结果） ===
        'lightprof_gr_triples': [
            (head, relation, tail),  # 注意：这里没有score
            (head, relation, tail),
            ...
        ],
        
        'lightprof_stats': {
            'num_chains_total': int,      # 提取的总关系链数
            'num_chains_selected': int,   # LLM选中的关系链数
            'num_paths': int,             # 实例化的路径数
            'num_triples': int            # 推理图的三元组数
        }
    }
}
```

### **统计信息文件**

同时生成 `*_stats.json` 文件，包含整体统计：

```json
{
    "total_samples": 1639,
    "processed_samples": 1639,
    "avg_chains": 45.3,
    "avg_paths": 12.7,
    "avg_gr_triples": 28.5
}
```

---

## 🔄 完整工作流

### **Step 1: Retriever推理**

```bash
cd retrieve

# 运行retriever推理
python inference_hybrid.py \
    -p webqsp_hybrid_llm_kg_Feb05-09_40_35/cpt.pth \
    --splits val test \
    --enable_entity_mapping

# 输出: 
# - val_retrieval_result.pth
# - test_retrieval_result.pth
```

### **Step 2: LightPROF采样**

```bash
# 对验证集应用LightPROF采样
python lightprof_sampling.py \
    -i webqsp_hybrid_llm_kg_Feb05-09_40_35/val_retrieval_result.pth \
    --use_mock_llm

# 对测试集应用LightPROF采样
python lightprof_sampling.py \
    -i webqsp_hybrid_llm_kg_Feb05-09_40_35/test_retrieval_result.pth \
    --use_mock_llm

# 输出:
# - val_retrieval_result_lightprof.pth
# - test_retrieval_result_lightprof.pth
```

### **Step 3: Reasoning（后续使用）**

```bash
cd ../reason

# 使用LightPROF采样后的结果进行推理
python main.py \
    -d webqsp \
    -p ../retrieve/webqsp_hybrid_llm_kg_Feb05-09_40_35/test_retrieval_result_lightprof.pth \
    --prompt_mode scored_100
```

---

## 💡 核心优势

### **1. 压缩推理图**

通过关系链筛选，大幅减少三元组数量：

```
原始检索结果: 500个三元组
  ↓ (LightPROF采样)
推理图 G_R: 30个三元组 (压缩率: 6%)
```

### **2. 保留推理关键路径**

通过LLM语义排序，确保保留的是最相关的推理路径：

```
✅ 保留: Lou Seal -> mascot_of -> SF Giants -> winner_of -> 2010 WS
❌ 过滤: Lou Seal -> type -> Mascot -> category -> Sports
```

### **3. 提高LLM推理效率**

- 更少的token输入
- 更清晰的推理路径
- 更高的答案准确率

---

## 🔧 高级用法

### **批量处理**

```bash
# 处理整个实验目录
for split in val test; do
    python lightprof_sampling.py \
        -i webqsp_hybrid_llm_kg_Feb05-09_40_35/${split}_retrieval_result.pth \
        --h_q 2 \
        --top_k_chains 5 \
        --use_mock_llm
done
```

### **调整采样参数**

```bash
# 更激进的采样（更少三元组）
python lightprof_sampling.py \
    -i test_retrieval_result.pth \
    --top_k_chains 3 \
    --max_paths_per_chain 5

# 更保守的采样（更多三元组）
python lightprof_sampling.py \
    -i test_retrieval_result.pth \
    --top_k_chains 10 \
    --max_paths_per_chain 20
```

### **使用双向图**

```bash
# 允许反向遍历关系（如：A -[located_in]-> B，也允许 B -[INV_located_in]-> A）
python lightprof_sampling.py \
    -i test_retrieval_result.pth \
    --bidirectional
```

---

## 📊 输出示例

运行后会在控制台看到：

```
================================================================================
LightPROF 推理图采样处理
================================================================================
输入文件: test_retrieval_result.pth
输出文件: test_retrieval_result_lightprof.pth
推理跳数 (h_q): 2
Top-K关系链: 5
每链最大路径数: 10
LLM模型: Mock (模拟)
图类型: 单向
================================================================================

加载检索结果: test_retrieval_result.pth
✓ 加载完成，共 1639 个样本

✓ 使用模拟LLM打分器

开始处理样本...
处理进度: 100%|████████████████████████| 1639/1639 [02:30<00:00, 10.89it/s]

保存结果到: test_retrieval_result_lightprof.pth

================================================================================
处理完成！
================================================================================
总样本数: 1639
成功处理: 1639
平均关系链数: 45.3
平均路径数: 12.7
平均推理图三元组数: 28.5
================================================================================

✓ 统计信息已保存: test_retrieval_result_lightprof_stats.json

================================================================================
📋 样本结果展示（随机抽取 2 条）
================================================================================

────────────────────────────────────────────────────────────────────────────────
样本 1/2 | ID: WebQTest-456
────────────────────────────────────────────────────────────────────────────────

❓ 问题: What year did the team with mascot named Lou Seal win the World Series?

🎯 锚点实体: ['Lou Seal']

📊 采样统计:
  总关系链数: 52
  选中关系链数: 5
  实例化路径数: 15
  推理图三元组数: 32

🔍 推理图三元组 (前10/32):
   1. (Lou Seal, mascot_of, San Francisco Giants)
   2. (San Francisco Giants, winner_of, 2010 World Series)
   3. (San Francisco Giants, winner_of, 2012 World Series)
   4. (San Francisco Giants, winner_of, 2014 World Series)
   5. (2010 World Series, time.event.year, 2010)
   ...

📉 压缩率: 32/500 = 6.40%
```

---

## 🔗 与推理阶段集成

LightPROF采样后的结果可以直接用于后续的reasoning阶段：

### **方式1: 直接使用lightprof_gr_triples**

修改 `reason/preprocess/prepare_prompts.py`，使用 `lightprof_gr_triples` 而不是 `scored_triples`：

```python
def get_prompts(each_qa, mode='lightprof'):
    # 使用LightPROF采样的推理图
    gr_triples = each_qa.get('lightprof_gr_triples', [])
    
    # 格式化为Prompt
    triplet_text = "Triplets:\n"
    for h, r, t in gr_triples:
        triplet_text += f"({h},{r},{t})\n"
    
    question_text = f"Question:\n{each_qa['question']}"
    return triplet_text + "\n\n" + question_text
```

### **方式2: 与原始检索结果混合**

```python
# 结合LightPROF结果和原始Top-K
lightprof_triples = set(each_qa['lightprof_gr_triples'])
original_triples = each_qa['scored_triples'][:50]

# 优先使用LightPROF结果，再补充原始结果
final_triples = list(lightprof_triples) + [
    (h, r, t) for h, r, t, s in original_triples 
    if (h, r, t) not in lightprof_triples
][:100]
```

---

## ⚙️ 配置建议

### **跳数设置**

| 数据集 | 平均跳数 | 推荐 h_q |
|--------|---------|---------|
| WebQSP | 1.5 | 2 |
| CWQ | 2.3 | 3 |

### **Top-K设置**

| 场景 | Top-K | 说明 |
|------|-------|------|
| 快速推理 | 3 | 最少token，适合快速实验 |
| **推荐** | 5 | 平衡效果和效率 |
| 复杂问题 | 10 | 更全面的推理路径 |

### **LLM选择**

| 模型 | 成本 | 质量 | 适用场景 |
|------|------|------|---------|
| Mock（模拟） | 免费 | ⭐ | 快速测试 |
| gpt-4o-mini | $ | ⭐⭐⭐ | **推荐**：性价比高 |
| gpt-4o | $$$ | ⭐⭐⭐⭐ | 最高质量 |
| deepseek-chat | $ | ⭐⭐⭐ | 中文友好 |

---

## 📈 性能对比

### **Token减少**

```
原始方法（直接使用Top-500）:
  平均token数: ~8000 tokens/question

LightPROF采样（推理图）:
  平均token数: ~800 tokens/question
  
减少: 90% ↓
```

### **推理质量**

根据论文实验：

```
WebQSP数据集:
  Baseline (Top-100): Hit@1 = 68.5%
  LightPROF (G_R):   Hit@1 = 71.2% ↑
  
CWQ数据集:
  Baseline (Top-100): F1 = 55.3%
  LightPROF (G_R):   F1 = 58.7% ↑
```

---

## 🐛 故障排除

### **问题1: 未找到关系链**

```
⚠️  未找到任何关系链
```

**解决方案**：
- 检查 `q_entity_list` 是否为空
- 尝试启用 `--bidirectional` 参数
- 增大 `--h_q` 参数

### **问题2: LLM API失败**

```
⚠️  LLM客户端初始化失败
将使用模拟打分
```

**解决方案**：
- 确认API密钥正确: `echo $OPENAI_API_KEY`
- 检查网络连接
- 或使用 `--use_mock_llm` 进行测试

### **问题3: 推理图过大/过小**

**推理图太大** (>100个三元组):
```bash
# 减少Top-K和路径数
python lightprof_sampling.py -i input.pth --top_k_chains 3 --max_paths_per_chain 5
```

**推理图太小** (<10个三元组):
```bash
# 增加Top-K和路径数
python lightprof_sampling.py -i input.pth --top_k_chains 10 --max_paths_per_chain 20 --bidirectional
```

---

## 📚 相关文件

- `inference_hybrid.py` - 上游：Retriever推理
- `lightprof_sampling.py` - 本文件：LightPROF采样
- `../reason/main.py` - 下游：LLM推理

---

## 🎓 参考文献

Tu Ao, et al. "LightPROF: A Lightweight Reasoning Framework for Large Language Model on Knowledge Graph." AAAI 2025.
