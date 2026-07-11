"""Stage 9 — Validation.

Last gate before the model call. Deterministic checks:
  * token ceiling (with the safety margin already applied upstream)
  * non-empty essentials; user message actually present
  * balanced XML-ish delimiters
  * leftover duplicate lines (dedup should have caught them)
  * PROMPT-INJECTION scan over untrusted content — flags known override
    patterns ("ignore previous instructions", "you are now", etc.). Detection
    is heuristic; the real mitigation is the delimiting done in build.py. A hit
    is a warning by default (content is already fenced), escalatable to a hard
    fail via `Config`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .budget import count_tokens
from .build import BuiltPrompt
from .config import Config
from .types import UNTRUSTED_SOURCES, ContextItem, Request

_INJECTION_PATTERNS = [
    r"ignore (all |the |your )?(previous|prior|above) (instructions|prompts?)",
    r"disregard (the |all |your )?(previous|above|system)",
    r"you are now (a|an|the)\b",
    r"forget (everything|all|your) (instructions|context)",
    r"system prompt\b",
    r"</?(system|assistant)>",
    r"reveal (your|the) (system|instructions|prompt)",
    r"do not follow",
]
_INJECTION_RE = re.compile("|".join(f"(?:{p})" for p in _INJECTION_PATTERNS), re.I)


@dataclass
class ValidationReport:
    ok: bool = True
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    prompt_tokens: int = 0
    injection_flags: list[str] = field(default_factory=list)

    def fail(self, msg: str) -> None:
        self.ok = False
        self.errors.append(msg)


class Validator:
    def __init__(self, config: Config | None = None) -> None:
        self.cfg = config or Config()

    def validate(
        self,
        request: Request,
        prompt: BuiltPrompt,
        items: list[ContextItem] | None = None,
    ) -> ValidationReport:
        rep = ValidationReport()
        rep.prompt_tokens = count_tokens(prompt.system) + count_tokens(prompt.user)

        ceiling = request.max_context_tokens - request.reserve_output_tokens
        if rep.prompt_tokens > ceiling:
            rep.fail(f"prompt {rep.prompt_tokens} tok exceeds ceiling {ceiling}")

        if not prompt.user.strip():
            rep.fail("empty user content")
        if request.user_message not in prompt.user:
            rep.fail("user message missing from final prompt")

        for tag in ("memory", "conversation", "untrusted_context", "user_request"):
            if prompt.user.count(f"<{tag}>") != prompt.user.count(f"</{tag}>"):
                rep.fail(f"unbalanced <{tag}> block")

        lines = [ln.strip() for ln in prompt.user.splitlines() if ln.startswith("- ")]
        if len(lines) != len(set(lines)):
            rep.warnings.append("duplicate context lines survived to the prompt")

        # prompt-injection scan on untrusted items
        if self.cfg.injection_scan and items:
            for it in items:
                if not it.included or it.source not in UNTRUSTED_SOURCES:
                    continue
                if _INJECTION_RE.search(it.text):
                    flag = f"{it.source.value}: '{it.preview(50)}'"
                    rep.injection_flags.append(flag)
            if rep.injection_flags:
                msg = f"{len(rep.injection_flags)} injection pattern(s) in untrusted context"
                rep.warnings.append(msg)  # fenced + instructed; warn, don't block

        return rep
