"""Stage 10 — LLM (resilient Claude client).

Adds the production concerns the toy skipped:
  * retries with exponential backoff + jitter on transient errors (429/529/
    connection), bounded by `llm_max_retries`
  * request timeout
  * prompt caching: passes structured `system` blocks with cache_control through
    untouched, and surfaces cache read/write token usage
  * graceful mock fallback (no SDK / no key) so the pipeline always runs

`complete(system, user)` accepts `system` as a plain string or a list of content
blocks (with cache_control). `__call__` is the simple text-in/text-out hook the
compressor uses for abstractive summaries.
"""

from __future__ import annotations

import os
import random
import re
import time
from dataclasses import dataclass

from .config import Config


@dataclass
class LLMResponse:
    text: str
    backend: str
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    retries: int = 0


class LLM:
    def __init__(self, config: Config | None = None) -> None:
        self.cfg = config or Config()
        self.model = self.cfg.llm_model
        self._client = None
        self._errors: tuple = ()
        self.backend = "mock"
        if os.environ.get("ANTHROPIC_API_KEY"):
            try:
                import anthropic

                self._client = anthropic.Anthropic(timeout=self.cfg.llm_timeout_s)
                self._errors = (
                    anthropic.RateLimitError,
                    anthropic.APIConnectionError,
                    anthropic.InternalServerError,
                    anthropic.APIStatusError,
                )
                self.backend = "anthropic"
            except Exception:
                self._client = None

    @property
    def client(self):
        return self._client

    def complete(self, system, user: str) -> LLMResponse:
        if self._client is not None:
            return self._anthropic(system, user)
        return self._mock(system, user)

    def __call__(self, prompt: str) -> str:
        return self.complete("You are a concise summarizer.", prompt).text

    # --- real client with retry/backoff -----------------------------------
    def _anthropic(self, system, user: str) -> LLMResponse:
        last_exc = None
        for attempt in range(self.cfg.llm_max_retries + 1):
            try:
                msg = self._client.messages.create(
                    model=self.model,
                    max_tokens=self.cfg.llm_max_tokens,
                    system=system,
                    messages=[{"role": "user", "content": user}],
                )
                text = "".join(b.text for b in msg.content if b.type == "text")
                u = msg.usage
                return LLMResponse(
                    text=text,
                    backend="anthropic",
                    input_tokens=u.input_tokens,
                    output_tokens=u.output_tokens,
                    cache_read_tokens=getattr(u, "cache_read_input_tokens", 0) or 0,
                    cache_write_tokens=getattr(u, "cache_creation_input_tokens", 0) or 0,
                    retries=attempt,
                )
            except self._errors as exc:  # transient — back off and retry
                last_exc = exc
                if attempt >= self.cfg.llm_max_retries:
                    break
                sleep = self.cfg.llm_backoff_base_s * (2**attempt) + random.uniform(0, 0.3)
                time.sleep(sleep)
        raise RuntimeError(f"LLM call failed after retries: {last_exc}")

    # --- mock -------------------------------------------------------------
    def _mock(self, system, user: str) -> LLMResponse:
        bullets = re.findall(r"^- (.+)$", user, flags=re.MULTILINE)
        req = re.search(r"<user_request>\s*(.+?)\s*</user_request>", user, re.DOTALL)
        question = req.group(1).strip() if req else "(no request found)"
        top = bullets[:4]
        body = "\n".join(f"  * {b}" for b in top) if top else "  (no context included)"
        text = (
            "[MOCK LLM - set ANTHROPIC_API_KEY for a real answer]\n\n"
            f"Request: {question}\n\n"
            f"Answering from the {len(bullets)} context item(s) assembled, "
            f"the most relevant being:\n{body}"
        )
        return LLMResponse(text=text, backend="mock")
