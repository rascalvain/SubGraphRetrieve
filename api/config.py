"""
API 服务配置
所有配置项均通过环境变量读取，支持 .env 文件方式注入
"""
import os

# ── 检索器模型 ─────────────────────────────────────────────────
# 已训练好的 Retriever 检查点路径（必须配置）
RETRIEVER_CHECKPOINT_PATH: str = os.getenv("RETRIEVER_CHECKPOINT_PATH", "")

# 文本编码器名称（与训练时保持一致）
TEXT_ENCODER_NAME: str = os.getenv("TEXT_ENCODER_NAME", "gte-large-en-v1.5")

# KG 嵌入文件路径，若非空则使用 GTEKGEnhanced；否则降级到 GTELargeEN_new
KG_EMB_PATH: str = os.getenv("KG_EMB_PATH", "")

# 实体映射文件（GTEKGEnhanced 可选，用于实体链接）
ENTITY_MAPPING_PATH: str = os.getenv("ENTITY_MAPPING_PATH", "")

# ── OpenAI / LLM（PoG + LightPROF stage2a 使用）────────────────
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL: str = os.getenv("OPENAI_BASE_URL", "")
OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# ── SentenceBERT（LightPROF Fuzzy 筛选使用）────────────────────
SBERT_MODEL_NAME: str = os.getenv("SBERT_MODEL_NAME", "all-MiniLM-L6-v2")

# ── 推理参数 ──────────────────────────────────────────────────
DEFAULT_TOP_K: int = int(os.getenv("DEFAULT_TOP_K", "100"))
DEVICE: str = os.getenv("DEVICE", "cpu")

# LightPROF 参数
LIGHTPROF_W1: int = int(os.getenv("LIGHTPROF_W1", "80"))   # Fuzzy 保留链数
LIGHTPROF_K: int = int(os.getenv("LIGHTPROF_K", "5"))      # LLM 精筛 Top-K
LIGHTPROF_MAX_HOPS: int = int(os.getenv("LIGHTPROF_MAX_HOPS", "3"))

# ── Flask 服务 ────────────────────────────────────────────────
FLASK_HOST: str = os.getenv("FLASK_HOST", "0.0.0.0")
FLASK_PORT: int = int(os.getenv("FLASK_PORT", "5001"))
FLASK_DEBUG: bool = os.getenv("FLASK_DEBUG", "false").lower() == "true"
