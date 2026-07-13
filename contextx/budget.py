"""Stage 7 — Token Budget Manager.

Count tokens, reserve output, trim the lowest-ranked tail to fit the window.

Tokenizer note: `tiktoken` (cl100k) is OpenAI's tokenizer, NOT Claude's, so
counts drift ~10-20%. We therefore apply a `budget_safety_margin` headroom so
the assembled prompt stays comfortably under the real model limit.
"""

from __future__ import annotations

from dataclasses import dataclass

from .config import Config
from .types import ContextItem, Request

try:
    import tiktoken

    _ENC = tiktoken.get_encoding("cl100k_base")
except Exception:  # pragma: no cover
    _ENC = None


def count_tokens(text: str) -> int:
    """Fast local estimate (tiktoken cl100k, or ~4 chars/token fallback)."""
    if _ENC is not None:
        return len(_ENC.encode(text))
    return max(1, (len(text) + 3) // 4)


@dataclass
class BudgetPlan:
    total: int
    reserved_output: int
    system: int
    fixed_user: int
    available_for_context: int


class BudgetManager:
    def __init__(self, config: Config | None = None) -> None:
        self.cfg = config or Config()

    def plan(self, request: Request, system_prompt: str) -> BudgetPlan:
        total = request.max_context_tokens
        reserved = request.reserve_output_tokens
        system_toks = count_tokens(system_prompt)
        user_toks = count_tokens(request.user_message)
        raw = total - reserved - system_toks - user_toks
        # keep headroom for tokenizer drift + block delimiters we add in build
        available = int(raw * (1.0 - self.cfg.budget_safety_margin))
        return BudgetPlan(
            total=total,
            reserved_output=reserved,
            system=system_toks,
            fixed_user=user_toks,
            available_for_context=max(0, available),
        )

    def fit(
        self, items: list[ContextItem], plan: BudgetPlan
    ) -> tuple[list[ContextItem], int, int]:
        """Greedily include highest-ranked items until the budget is spent."""
        used = 0
        kept: list[ContextItem] = []
        trimmed = 0
        for it in items:
            if it.tokens == 0:
                it.tokens = count_tokens(it.text)
            if used + it.tokens <= plan.available_for_context:
                it.included = True
                used += it.tokens
                kept.append(it)
            else:
                it.included = False
                trimmed += 1
        return kept, used, trimmed
