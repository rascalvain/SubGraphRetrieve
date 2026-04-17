# Subgraph Retrieval for KGQA (with LightPROF Reasoning)

一个面向知识图谱问答（KGQA）的子图检索与推理工程。
项目核心流程包括：

1. **子图检索（Retriever）**：从候选图结构中为三元组打分并召回 Top-K 子图证据
2. **Prompt 构建（含 LightPROF / PoG 增强）**：将检索结果组织为 LLM 输入
3. **LLM 推理**：生成最终答案（`ans:` 格式）
4. **指标评估**：计算 Hit@1、F1、Hal Score 等指标

---

## 1. 项目结构

```text
.
├─ retrieve/                          # 子图检索训练/推理/评估主模块
│  ├─ train.py
│  ├─ train_hybrid.py
│  ├─ inference.py
│  ├─ inference_hybrid.py
│  ├─ eval.py
│  ├─ emb.py
│  ├─ emb_hybrid.py
│  ├─ generate_weak_supervision.py
│  ├─ configs/
│  │  ├─ retriever/*.yaml
│  │  └─ emb/**/**/*.yaml
│  ├─ src/
│  │  ├─ model/retriever.py
│  │  ├─ dataset/retriever.py
│  │  ├─ dataset/retriever_hybrid.py
│  │  ├─ dataset/emb*.py
│  │  ├─ model/text_encoders/*.py
│  │  └─ utils/entity_linker.py
│  ├─ requirements/*.txt
│  └─ data_files/{webqsp,cwq}/...
│
├─ reason/                            # LLM 推理与答案评估
│  ├─ main.py
│  ├─ main_lightprof.py
│  ├─ llm_utils.py
│  ├─ prompts.py
│  ├─ preprocess/
│  │  ├─ prepare_data.py
│  │  ├─ prepare_prompts.py
│  │  └─ prepare_prompts_lightprof.py
│  └─ metrics/
│     ├─ evaluate_results.py
│     ├─ evaluate_results_corrected.py
│     └─ evaluate_results_lightprof.py
│
├─ lightprof/                         # LightPROF 说明文档
│  ├─ README.md
│  ├─ README_lightprof.md
│  └─ QUICKSTART_lightprof.md
│
├─ emb/                               # 数据预处理（实体/三元组抽取、对齐）
│  ├─ extract_quest_ent.py
│  ├─ extract_triples.py
│  ├─ entity_alignment.py
│  └─ create_name2id.py
│
├─ pog_pruning/
│  └─ pog_indicator.py
│
├─ datasets/                          # 数据目录（原始/处理后）
└─ requirements/
   └─ subgraph/requirements_*.txt
```

---

## 2. 功能概览

- 支持 **WebQSP / CWQ** 等 KGQA 数据流程
- 支持 **检索器训练与推理**（包括 hybrid 版本）
- 支持 **LightPROF 样本格式**加载、prompt 构建与 LLM 推理
- 支持 **PoG 提示增强**、三元组排序、回退补全（fallback）
- 支持较完整的评估指标：
  - 检索阶段：`triple_recall@K`、`ans_recall@K`
  - QA 阶段：`Hit@1`、`Macro/Micro F1`、`Precision/Recall`、`Hal Score`

---

## 3. 环境准备

> 建议使用独立虚拟环境（conda 或 venv）。

### 3.1 Python 版本
代码中未严格写死版本，建议使用：

- **Python 3.10+**（推荐 3.10 / 3.11）

### 3.2 安装依赖

根据你的实际流程选择依赖文件（检索 / 子图 / LightPROF）：

```bash
pip install -r retrieve/requirements/<your_requirements>.txt
# 或
pip install -r requirements/subgraph/<your_requirements>.txt
```

如果你使用 GPU，需确保本地 `PyTorch + CUDA` 版本匹配。

---

## 4. 数据与输入格式

项目中常见中间文件格式：

- `*.pth`：检索输出、LightPROF 采样输出等
- `predictions.jsonl`：LLM 推理结果
- 评估输出：`*_eval_result.txt` 与 `*_detailed_eval.jsonl`

LightPROF 常用样本字段示例（按代码读取逻辑）：

- `id`
- `question`
- `answers`（或 `ground_truth`）
- `q_entity` / `q_entity_list`
- `scored_triplets`
- `lightprof_gr_triples`
- `lightprof_stats`
- `Split_q`
- `I_LLM`

---

## 5. 快速开始（最小可复现流程）

> 默认在项目根目录执行命令。

### Step 1: 训练子图检索器（Hybrid）

```bash
python retrieve/train_hybrid.py -d webqsp
```

### Step 2: 检索推理并保存结果

```bash
python retrieve/inference_hybrid.py \
  -p <checkpoint_path> \
  --splits test \
  --max_K 500 \
  --eval_k_list 50 100 200 500
```

输出示例：
- `*_retrieval_result.pth`
- `*_metrics.json / *_metrics.txt`

### Step 3: LightPROF + LLM 推理

```bash
python reason/main_lightprof.py \
  -d webqsp \
  -p <lightprof_sampling_output.pth> \
  --prompt_mode lightprof_100 \
  --llm_mode sys_icl_dc \
  -m meta-llama/Meta-Llama-3.1-8B-Instruct
```

输出示例：
- `results/KGQA/<dataset>/LightPROF/<model>/.../predictions.jsonl`

### Step 4: 结果评估

```bash
python reason/metrics/evaluate_results_lightprof.py -p <predictions.jsonl>
```

可选：对 bad/subset 进行附加评估

```bash
python reason/metrics/evaluate_results_lightprof.py \
  -p <predictions.jsonl> \
  --eval_subset \
  --eval_bad
```

---

## 6. 关键脚本说明

### 检索模块

- `retrieve/train_hybrid.py`
  训练入口：加载 YAML 配置，执行 train/eval，保存 checkpoint。
- `retrieve/inference_hybrid.py`
  推理入口：加载 checkpoint，计算 recall@K，导出检索结果。
- `retrieve/src/model/retriever.py`
  检索核心模型（消息传递 + 三元组相关性预测）。

### 推理模块

- `reason/main_lightprof.py`
  LightPROF 版本推理主入口（加载 `.pth`、构造 prompt、调用 LLM、保存预测）。
- `reason/preprocess/prepare_prompts_lightprof.py`
  Prompt 构建：支持三元组重排、PoG 引导块、fallback 机制。

### 评估模块

- `reason/metrics/evaluate_results_lightprof.py`
  计算 Hit@1、Macro/Micro F1、Hal Score，并输出汇总与逐条结果。

---

## 7. 常用参数

### `reason/main_lightprof.py`（常见）

- `--prompt_mode`（默认：`lightprof_100`）
- `--llm_mode`（默认：`sys_icl_dc`）
- `--model_name`（默认：`meta-llama/Meta-Llama-3.1-8B-Instruct`）
- `--frequency_penalty`（默认：`0.16`）
- `--thres`（默认：`0.0`）
- `--use_pog_prompt`（启用 PoG 引导）
- `--no_fallback`（禁用证据不足时补充）
- `--rank_first`（优先排序）
- `--no_sort_triples`（禁用排序）

### `retrieve/inference_hybrid.py`（常见）

- `-p/--path`：checkpoint 路径
- `--splits`：`val`/`test`
- `--max_K`：最大检索深度
- `--eval_k_list`：评估 K 列表（例如 `50 100 200 500`）

---

## 8. 评估指标说明

### 检索阶段

- `triple_recall@K`：Top-K 是否覆盖黄金三元组
- `ans_recall@K`：Top-K 证据是否可触达答案实体

### QA 阶段

- `Hit@1`
- `Macro Precision / Recall / F1`
- `Micro Precision / Recall / F1`
- `Hal Score`（用于衡量幻觉倾向）

---

## 9. 结果文件说明

- `predictions.jsonl`：每条问题的预测答案与辅助信息
- `*_eval_result.txt`：整体指标汇总
- `*_detailed_eval.jsonl`：逐样本评估细节（命中、错误类型等）

---

## 10. 常见问题与注意事项

1. **硬编码绝对路径**
   部分脚本默认参数可能写了本地绝对路径，迁移机器后需要修改。

2. **密钥与敏感信息**
   请不要在代码中硬编码 API Key。建议使用环境变量（如 `OPENAI_API_KEY`）。

3. **显存占用**
   长上下文 + 大模型配置（如较大 `max_tokens`）会显著提高显存需求。

4. **版本兼容**
   `transformers` / `sentence-transformers` / `torch` 版本不匹配时可能报错（尤其是权重 shape mismatch）。

5. **复现随机性**
   即使设定 seed，不同硬件和库版本仍可能带来轻微波动。

---

## 11. 安全与合规建议

- 提交到 GitHub 前检查：
  - API Key / Token / 私钥
  - 本地绝对路径
  - 大体积临时文件（建议 `.gitignore`）
- 建议在 release 前增加：
  - `requirements.txt` 固定版本
  - 统一的数据下载与预处理脚本说明
  - 一键复现实验命令文档

---

## 12. 引用与致谢

若你在论文或项目中使用本仓库，请在 README/论文中注明该工程与具体实验设置（数据集、模型、prompt 模式、评估脚本版本）。
