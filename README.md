# Multi-Granularity Semantic-Aware Subgraph Retrieval for KGQA

> **基于多粒度语义感知与动态路径规划的推理证据图谱检索方法**

A Knowledge Graph Question Answering (KGQA) system that retrieves and refines answer subgraphs from large-scale knowledge graphs using a GNN-based Retriever, LightPROF path sampling, and PoG reasoning indicators.

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Project Structure](#project-structure)
4. [Environment Setup](#environment-setup)
5. [Quick Start](#quick-start)
   - [Training](#step-1-train-the-retriever)
   - [Inference & Evaluation](#step-2-retrieval-inference--evaluation)
   - [LightPROF Sampling](#step-3-lightprof-subgraph-sampling)
   - [LLM Reasoning](#step-4-llm-reasoning)
6. [Flask API Service](#flask-api-service)
   - [Startup](#startup)
   - [Endpoints](#endpoints)
   - [Request & Response Format](#request--response-format)
   - [Configuration](#configuration)
7. [Module Reference](#module-reference)
8. [Evaluation Metrics](#evaluation-metrics)
9. [Common Issues](#common-issues)

---

## Overview

This project implements a complete subgraph retrieval pipeline for KGQA:

| Stage | Module | Description |
|-------|--------|-------------|
| **Stage 1** | Neo4j / External | Global graph recall — fetch candidate triples from the KG (handled externally) |
| **Stage 2** | `retrieve/` + `api/` | Draft subgraph retrieval — score candidate triples with a GNN Retriever |
| **Stage 3** | `lightprof/` + `pog_pruning/` + `api/` | Subgraph refinement — LightPROF path sampling guided by PoG indicators |
| **Stage 4** | `reason/` | LLM reasoning — generate final answers from the refined subgraph |

The `api/` module exposes Stage 2 and Stage 3 as a Flask REST service that **seamlessly replaces** LLM-simulated subgraph generation in prototype systems.

---

## Architecture

```
                     ┌─────────────────────────────────────────────┐
                     │              Flask API Service               │
                     │                 api/app.py                   │
                     └───────────────────┬─────────────────────────┘
                                         │
              ┌──────────────────────────┼──────────────────────────┐
              ▼                          ▼                           ▼
   ┌──────────────────┐      ┌───────────────────┐      ┌──────────────────────┐
   │  RetrieverService│      │    PogService      │      │  LightprofService    │
   │  (Stage 2)       │      │    (Stage 3a)      │      │  (Stage 3b)          │
   └────────┬─────────┘      └─────────┬──────────┘      └──────────┬───────────┘
            │                          │                             │
            ▼                          ▼                             ▼
   ┌──────────────────┐      ┌──────────────────┐      ┌────────────────────────┐
   │ Retriever GNN    │      │PoGIndicatorGen   │      │ LightPROF_Sampler      │
   │ + GTE Encoder    │      │(question split + │      │ BFS chains             │
   │ DDE hop scoring  │      │ I_LLM indicator) │      │ + SBERT fuzzy filter   │
   └──────────────────┘      └──────────────────┘      │ + LLM chain scoring    │
                                                        └────────────────────────┘

Input:  question (str) + triples [(h, r, t), ...]  ← from Neo4j global graph
Output: Stage2 draft subgraph  +  Stage3 refined subgraph  (frontend-ready JSON)
```

### Retriever Model

The Retriever uses a message-passing GNN with **DDE (Distance Direction Encoding)**:

```
q_emb, entity_embs, relation_embs  ←  GTE-large-en-v1.5 encoder
                                       (GTELargeEN_new  or  GTEKGEnhanced)

topic_entity_one_hot  →  DDE propagation  →  per-triple relevance score
                                              (sigmoid output, [0, 1])
```

Text encoders:
- **`GTELargeEN_new`** — standard GTE-large-en-v1.5 (1024-dim), no external KG embeddings required
- **`GTEKGEnhanced`** — GTE-large-en-v1.5 fused with pre-computed KG entity embeddings for improved entity grounding

### LightPROF Sampling (3-stage)

1. **BFS chain extraction** — enumerate reasoning chains from topic entities
2. **SBERT fuzzy filtering** — retain semantically relevant chains (`w1` parameter)
3. **LLM chain scoring** — top-k chains ranked by LLM relevance score

---

## Project Structure

```text
第一章/
│
├── api/                              # Flask REST API (Stage 2 + Stage 3)
│   ├── __init__.py
│   ├── app.py                        # Flask entry point, 4 endpoints
│   ├── config.py                     # All config via env vars
│   ├── graph_utils.py                # Shared graph utility functions
│   └── services/
│       ├── __init__.py
│       ├── retriever_service.py      # GNN Retriever inference service
│       ├── pog_service.py            # PoG indicator generator service
│       └── lightprof_service.py      # LightPROF sampler + Stage3 builder
│
├── retrieve/                         # Retriever training & offline inference
│   ├── train.py                      # Standard training
│   ├── train_hybrid.py               # Hybrid training (text + KG embeddings)
│   ├── inference.py                  # Standard inference
│   ├── inference_hybrid.py           # Hybrid inference (used as API reference)
│   ├── eval.py                       # Evaluation utilities
│   ├── emb.py                        # Entity embedding generation
│   ├── emb_hybrid.py                 # Hybrid embedding generation
│   ├── generate_weak_supervision.py  # Weak supervision label generation
│   ├── configs/
│   │   ├── retriever/                # YAML configs for retriever training
│   │   └── emb/                      # YAML configs for embedding models
│   ├── src/
│   │   ├── model/
│   │   │   ├── retriever.py          # Core GNN Retriever model
│   │   │   └── text_encoders/
│   │   │       ├── gte_large_en_new.py   # GTELargeEN_new encoder
│   │   │       ├── gte_kg_enhanced.py    # GTEKGEnhanced encoder
│   │   │       └── ...
│   │   ├── dataset/
│   │   │   ├── retriever.py          # Standard dataset
│   │   │   ├── retriever_hybrid.py   # Hybrid dataset
│   │   │   └── emb*.py               # Embedding datasets
│   │   └── utils/
│   │       └── entity_linker.py
│   ├── requirements/                 # Per-component requirements files
│   └── data_files/
│       ├── webqsp/                   # WebQSP dataset files
│       └── cwq/                      # ComplexWebQuestions dataset files
│
├── lightprof/                        # LightPROF sampling module
│   ├── lightprof_sampling.py         # Core: SimpleKG, LightPROF_Sampler, LLMScorer
│   ├── run_lightprof_example.py      # Standalone sampling example
│   ├── run_lightprof_pipeline.sh
│   ├── README.md
│   └── README_lightprof.md
│
├── pog_pruning/                      # PoG reasoning indicator generation
│   └── pog_indicator.py             # PoGIndicatorGenerator (question split + I_LLM)
│
├── reason/                           # LLM reasoning & answer evaluation
│   ├── main.py                       # Standard reasoning entry point
│   ├── main_lightprof.py             # LightPROF reasoning entry point
│   ├── llm_utils.py                  # LLM API wrappers
│   ├── prompts.py                    # Prompt templates
│   ├── preprocess/
│   │   ├── prepare_data.py
│   │   ├── prepare_prompts.py
│   │   └── prepare_prompts_lightprof.py
│   └── metrics/
│       ├── evaluate_results.py
│       ├── evaluate_results_corrected.py
│       └── evaluate_results_lightprof.py
│
├── emb/                              # Data preprocessing utilities
│   ├── extract_quest_ent.py          # Question entity extraction
│   ├── extract_triples.py            # Triple extraction
│   ├── entity_alignment.py           # Entity alignment across sources
│   └── create_name2id.py             # Entity name-to-ID mapping
│
├── datasets/                         # Raw and processed datasets
├── requirements/
│   └── subgraph/                     # Subgraph-specific requirements
└── README.md
```

---

## Environment Setup

**Requirements:** Python 3.10+ (3.10 or 3.11 recommended)

### Install dependencies

```bash
# Core retriever + API service
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118  # adjust CUDA version
pip install flask transformers sentence-transformers openai

# Or install from the bundled requirements
pip install -r retrieve/requirements/<your_requirements>.txt
pip install -r requirements/subgraph/<your_requirements>.txt
```

### Environment variables (API service)

Create a `.env` file or export the following before starting the API server:

```bash
# Required
export RETRIEVER_CHECKPOINT_PATH=/path/to/checkpoint.pth

# Optional: use GTEKGEnhanced encoder (otherwise GTELargeEN_new is used)
export KG_EMB_PATH=/path/to/kg_embeddings.pth
export ENTITY_MAPPING_PATH=/path/to/entity_mapping.json

# Optional: enable LLM-based PoG + LightPROF scoring
export OPENAI_API_KEY=sk-xxx
export OPENAI_BASE_URL=https://api.openai.com/v1
export OPENAI_MODEL=gpt-4o-mini

# Optional: device and inference params
export DEVICE=cuda          # or cpu
export DEFAULT_TOP_K=100
export SBERT_MODEL_NAME=all-MiniLM-L6-v2
export LIGHTPROF_W1=80
export LIGHTPROF_K=5
export LIGHTPROF_MAX_HOPS=3

# Optional: Flask server
export FLASK_HOST=0.0.0.0
export FLASK_PORT=5001
```

---

## Quick Start

### Step 1: Train the Retriever

```bash
# Standard training
python retrieve/train.py -d webqsp

# Hybrid training (with KG embeddings)
python retrieve/train_hybrid.py -d webqsp
```

Training outputs a checkpoint to `retrieve/outputs/<run_name>/best_model.pth`.

### Step 2: Retrieval Inference & Evaluation

```bash
python retrieve/inference_hybrid.py \
  -p retrieve/outputs/<run_name>/best_model.pth \
  --splits test \
  --max_K 500 \
  --eval_k_list 50 100 200 500
```

**Output files:**
- `*_retrieval_result.pth` — retrieval results per sample
- `*_metrics.json` / `*_metrics.txt` — recall@K metrics

### Step 3: LightPROF Subgraph Sampling

```bash
python lightprof/run_lightprof_example.py \
  --input_path <retrieval_result.pth> \
  --output_path <lightprof_output.pth> \
  --top_k_chains 5 \
  --max_hops 3
```

Or using the full pipeline script:

```bash
bash lightprof/run_lightprof_pipeline.sh
```

### Step 4: LLM Reasoning

```bash
python reason/main_lightprof.py \
  -d webqsp \
  -p <lightprof_output.pth> \
  --prompt_mode lightprof_100 \
  --llm_mode sys_icl_dc \
  -m meta-llama/Meta-Llama-3.1-8B-Instruct
```

**Output:** `results/KGQA/<dataset>/LightPROF/<model>/.../predictions.jsonl`

### Step 5: Evaluate Results

```bash
python reason/metrics/evaluate_results_lightprof.py -p <predictions.jsonl>

# With subset and bad-case analysis
python reason/metrics/evaluate_results_lightprof.py \
  -p <predictions.jsonl> \
  --eval_subset \
  --eval_bad
```

---

## Flask API Service

The `api/` module wraps Stage 2 and Stage 3 as a REST service for integration with frontend prototype systems.

### Startup

```bash
export RETRIEVER_CHECKPOINT_PATH=/path/to/checkpoint.pth
export OPENAI_API_KEY=sk-xxx          # optional, enables PoG + LightPROF LLM scoring
python api/app.py
```

The server starts on `http://0.0.0.0:5001` by default.

### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Health check — returns readiness of each service |
| `POST` | `/api/stage2` | Draft subgraph retrieval (Retriever GNN) |
| `POST` | `/api/stage3` | Refined subgraph (PoG + LightPROF) |
| `POST` | `/api/pipeline` | Full pipeline: returns both Stage 2 and Stage 3 in one request |

### Request & Response Format

#### Common Request Body

All `POST` endpoints accept the same request schema:

```json
{
  "question": "Which country does the president of France lead?",
  "triples": [
    ["Emmanuel Macron", "president of", "France"],
    ["France", "located in", "Europe"],
    ["Emmanuel Macron", "born in", "Amiens"]
  ],
  "topic_entities": ["Emmanuel Macron"],
  "top_k": 100
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `question` | string | Yes | Natural language question |
| `triples` | array | Yes | Candidate triples from Neo4j global graph. Accepts `[[h, r, t], ...]` or `[{"source": h, "label": r, "target": t}, ...]` |
| `topic_entities` | array | No | Topic entities (auto-detected from question if omitted) |
| `top_k` | int | No | Max triples to return (default: 100) |

---

#### `GET /health`

```json
{
  "status": "ok",
  "retriever_ready": true,
  "pog_ready": true,
  "lightprof_ready": true,
  "init_error": null
}
```

---

#### `POST /api/stage2` — Draft Subgraph

**Response:**

```json
{
  "DDE_NODES": [
    {"label": "Emmanuel Macron", "hop": 0, "ang": null, "dist": 0.0, "col": "#ff5252"},
    {"label": "France",          "hop": 1, "ang": 1.57, "dist": 0.2, "col": "#40c4ff"}
  ],
  "DDE_EDGES": [[0, 1], [1, 2]],
  "DRAFT_NODES": [
    {"id": "EmmanuelMacron", "label": "Emmanuel Macron", "type": "person"},
    {"id": "France",         "label": "France",          "type": "location"},
    {"id": "Noise1",         "label": "Civilian Casualties", "type": "noise"},
    {"id": "Noise2",         "label": "Diplomatic Talks",    "type": "noise"}
  ],
  "DRAFT_LINKS": [
    {"s": "EmmanuelMacron", "t": "France", "lb": "PRESIDENT OF", "noise": false},
    {"s": "EmmanuelMacron", "t": "Noise1", "lb": "CAUSED",       "noise": true}
  ],
  "meta": {
    "source": "retriever",
    "model": "Retriever+GTELargeEN_new",
    "llm_succeeded": true,
    "fallback_used": false,
    "llm_error": null
  },
  "_retrieval": {
    "scored_triples": [["Emmanuel Macron", "president of", "France", 0.93]],
    "topic_entity_ids": [0],
    "entity_list": ["Emmanuel Macron", "France"]
  }
}
```

**DDE_NODES fields:**

| Field | Description |
|-------|-------------|
| `label` | Entity name |
| `hop` | Distance from topic entities (0–3) |
| `ang` | Angular position for radial layout (null for hop=0) |
| `dist` | Radial distance (0.0 / 0.2 / 0.4 / 0.6) |
| `col` | Color by hop level |

---

#### `POST /api/stage3` — Refined Subgraph

Accepts an optional `stage2_data` field to reuse Stage 2 retrieval results (avoiding duplicate inference):

```json
{
  "question": "...",
  "triples": [...],
  "stage2_data": { "_retrieval": { ... } }
}
```

**Response:**

```json
{
  "CHAINS": [
    {
      "hops": 2,
      "hi": true,
      "score": 0.91,
      "nodes": ["Emmanuel Macron", "France", "Europe"],
      "rels":  ["president of", "located in"]
    }
  ],
  "INDICATOR_DATA": {
    "splitQuestions": [
      {"id": "Q1", "text": "Who is the president of France?"},
      {"id": "Q2", "text": "Which country does he lead?"}
    ],
    "entityChain": [
      {"text": "Emmanuel Macron", "type": "entity"},
      {"text": "president of",   "type": "rel"},
      {"text": "France",         "type": "answer"}
    ],
    "relationChain": ["president of", "located in", "born in"],
    "dPredict": 2
  },
  "PRUNE_STEPS": [
    {"icon": "search", "iClass": "step-icon blue",   "state": "done", "desc": "Draft subgraph retrieved",    "cnt": "87 triples", "cClass": "cnt-blue"},
    {"icon": "filter", "iClass": "step-icon orange", "state": "done", "desc": "Noise triples filtered",      "cnt": "23 removed", "cClass": "cnt-orange"},
    {"icon": "route",  "iClass": "step-icon green",  "state": "done", "desc": "Reasoning paths sampled",     "cnt": "5 chains",   "cClass": "cnt-green"},
    {"icon": "check",  "iClass": "step-icon purple", "state": "done", "desc": "Final subgraph constructed",  "cnt": "6 triples",  "cClass": "cnt-purple"}
  ],
  "FINAL_NODES": [
    {"id": "EmmanuelMacron", "label": "Emmanuel Macron", "type": "person"},
    {"id": "France",         "label": "France",          "type": "location"}
  ],
  "FINAL_LINKS": [
    {"s": "EmmanuelMacron", "t": "France", "lb": "PRESIDENT OF", "w": 0.93}
  ],
  "meta": {
    "source": "retriever+lightprof",
    "model": "Retriever+GTELargeEN_new",
    "llm_succeeded": true,
    "fallback_used": false,
    "llm_error": null
  }
}
```

---

#### `POST /api/pipeline` — Full Pipeline

Returns both Stage 2 and Stage 3 in a single request:

```json
{
  "stage2": { ... },
  "stage3": { ... }
}
```

---

### Configuration

All configuration is controlled by environment variables. Full reference:

| Variable | Default | Description |
|----------|---------|-------------|
| `RETRIEVER_CHECKPOINT_PATH` | `""` | **Required.** Path to trained Retriever `.pth` checkpoint |
| `KG_EMB_PATH` | `""` | Path to KG entity embeddings. If set, uses `GTEKGEnhanced`; otherwise `GTELargeEN_new` |
| `ENTITY_MAPPING_PATH` | `""` | Optional entity name→ID mapping for `GTEKGEnhanced` |
| `OPENAI_API_KEY` | `""` | OpenAI API key. Required for PoG + LightPROF LLM scoring |
| `OPENAI_BASE_URL` | `""` | Custom API base URL (e.g., for proxies or local deployments) |
| `OPENAI_MODEL` | `gpt-4o-mini` | LLM model name |
| `SBERT_MODEL_NAME` | `all-MiniLM-L6-v2` | Sentence-BERT model for LightPROF fuzzy filtering |
| `DEFAULT_TOP_K` | `100` | Default number of top-K triples returned by the Retriever |
| `DEVICE` | `cpu` | PyTorch device (`cpu` or `cuda`) |
| `LIGHTPROF_W1` | `80` | Number of chains retained after SBERT fuzzy filtering |
| `LIGHTPROF_K` | `5` | Number of top-K chains after LLM scoring |
| `LIGHTPROF_MAX_HOPS` | `3` | Maximum hop depth for chain extraction |
| `FLASK_HOST` | `0.0.0.0` | Flask server host |
| `FLASK_PORT` | `5001` | Flask server port |
| `FLASK_DEBUG` | `false` | Enable Flask debug mode |

**Degradation behavior** (when optional components are unavailable):
- No `OPENAI_API_KEY` → PoG uses fallback (simple entity chain), LightPROF skips LLM scoring
- No `KG_EMB_PATH` → falls back to `GTELargeEN_new` encoder automatically
- `RETRIEVER_CHECKPOINT_PATH` not found → Stage 2 / Stage 3 endpoints return `503`

---

## Module Reference

### `retrieve/src/model/retriever.py`

Core GNN Retriever. Key arguments from checkpoint `config['retriever']`:

| Param | Description |
|-------|-------------|
| `emb_size` | Input embedding dimension (auto-detected from encoder) |
| `topic_pe` | Whether to use topic-entity positional encoding |
| `DDE_*` | DDE propagation hyperparameters |

### `lightprof/lightprof_sampling.py`

Key classes:

| Class | Description |
|-------|-------------|
| `SimpleKG` | Lightweight KG built from scored triples; supports bidirectional traversal |
| `FuzzySelector` | SBERT-based chain semantic filtering |
| `LLMScorer` | LLM-based chain relevance scoring via OpenAI API |
| `LightPROF_Sampler` | Orchestrates BFS → fuzzy filter → LLM score → path sampling |

Main method: `LightPROF_Sampler.run_pipeline(question, anchor_entities, h_q, top_k_chains, ...)`

### `pog_pruning/pog_indicator.py`

`PoGIndicatorGenerator.generate(question, entities)` returns:

```python
{
    "Split_q":  ["sub-question 1", "sub-question 2"],
    "I_LLM": {
        "R_IA":      "entity1 - relation - entity2",  # reasoning path description
        "D_predict": 2,                                # predicted hop depth
        "List_T":    ["entity1", "entity2"]            # intermediate entities
    },
    "is_fallback": False
}
```

---

## Evaluation Metrics

### Retrieval Stage (`retrieve/inference_hybrid.py`)

| Metric | Description |
|--------|-------------|
| `triple_recall@K` | Fraction of gold triples covered by top-K retrieved triples |
| `ans_recall@K` | Fraction of samples where the answer entity appears in top-K evidence |

### QA Stage (`reason/metrics/evaluate_results_lightprof.py`)

| Metric | Description |
|--------|-------------|
| `Hit@1` | Exact match: at least one predicted answer equals a gold answer |
| `Macro F1` | F1 averaged over all questions |
| `Micro F1` | F1 computed on aggregated TP/FP/FN counts |
| `Precision / Recall` | Standard P/R for multi-answer questions |
| `Hal Score` | Hallucination score — penalizes answers with no KG grounding |

---

## Common Issues

**1. `FileNotFoundError: Retriever checkpoint not found`**
Set `RETRIEVER_CHECKPOINT_PATH` to the correct `.pth` file path.

**2. `503 Retriever service not initialized`**
The checkpoint failed to load on startup. Check the server logs for the actual error (wrong config keys, shape mismatch, etc.).

**3. Encoder shape mismatch**
Ensure `KG_EMB_PATH` matches the encoder used during training. If the checkpoint was trained with `GTELargeEN_new`, leave `KG_EMB_PATH` empty.

**4. PoG / LightPROF LLM scoring disabled**
Normal behavior when `OPENAI_API_KEY` is not set. Stage 3 still returns results using a rule-based fallback.

**5. CUDA out of memory**
Reduce `DEFAULT_TOP_K` or switch to `DEVICE=cpu` for large input graphs.

**6. `transformers` / `sentence-transformers` version conflicts**
Pin versions as specified in the requirements files. The GTE-large encoder requires `transformers>=4.36`.

---

## Citation

If you use this repository in your research, please cite the corresponding paper and acknowledge the dataset sources (WebQSP, ComplexWebQuestions).

---

## Acknowledgments

- GTE text encoder: [Alibaba-NLP/gte-large-en-v1.5](https://huggingface.co/Alibaba-NLP/gte-large-en-v1.5)
- LightPROF sampling methodology
- PoG (Paths over Graphs) indicator generation
- WebQSP and CWQ benchmark datasets
