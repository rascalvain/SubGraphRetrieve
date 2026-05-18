"""
PoG 推理指示器服务
封装 PoGIndicatorGenerator，提供 Stage3 所需的子问题拆分和推理链生成
"""
import sys
from pathlib import Path
from typing import Any

# ── 路径设置 ────────────────────────────────────────────────
_BASE_DIR = Path(__file__).resolve().parents[2]
for _p in [str(_BASE_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)


class PogService:
    """
    PoG 推理指示器生成服务
    调用 OpenAI API 对问题进行子问题拆分和推理链构建
    """

    def __init__(
        self,
        model: str = "",
        api_key: str = "",
        base_url: str = "",
        verbose: bool = False,
    ):
        self.model    = model
        self.api_key  = api_key
        self.base_url = base_url
        self.verbose  = verbose
        self._generator = None

    def load(self) -> None:
        """初始化 PoGIndicatorGenerator"""
        from pog_pruning.pog_indicator import PoGIndicatorGenerator
        self._generator = PoGIndicatorGenerator(
            model    = self.model or None,
            api_key  = self.api_key or None,
            base_url = self.base_url or None,
            verbose  = self.verbose,
        )
        print("[PogService] PoGIndicatorGenerator initialized")

    def generate(self, question: str, entities: list[str]) -> dict[str, Any]:
        """
        生成 PoG 推理指示器

        Returns
        -------
        {
            "Split_q": [...],      # 子问题列表
            "I_LLM": {             # 推理指示器
                "R_IA": str,
                "D_predict": int,
                "List_T": [...]
            },
            "pog_is_fallback": bool
        }
        """
        if self._generator is None:
            return self._make_fallback(question, entities)

        try:
            result = self._generator.generate(question, entities)
            return {
                "Split_q":        result.get("Split_q", [question]),
                "I_LLM":          result.get("I_LLM", self._default_i_llm(entities)),
                "pog_is_fallback": result.get("is_fallback", False),
            }
        except Exception as e:
            print(f"[PogService] generate failed: {e}")
            return self._make_fallback(question, entities)

    @staticmethod
    def _default_i_llm(entities: list[str]) -> dict:
        return {
            "R_IA":      " - ".join(entities[:3]) if entities else "",
            "D_predict": 2,
            "List_T":    entities[:3],
        }

    @staticmethod
    def _make_fallback(question: str, entities: list[str]) -> dict[str, Any]:
        return {
            "Split_q":        [question],
            "I_LLM":          {
                "R_IA":      " - ".join(entities[:3]) if entities else "",
                "D_predict": 2,
                "List_T":    entities[:3],
            },
            "pog_is_fallback": True,
        }
