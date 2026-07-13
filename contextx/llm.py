"""Stage 10 — LLM (Anthropic / Ollama / mock).

Backends, selected by `Config.llm_provider` ("auto" by default):
  * anthropic — Claude Messages API (used when ANTHROPIC_API_KEY is set).
  * ollama    — a local Ollama server (http://localhost:11434), no API key, no
                cost. Used automatically when reachable and no Claude key is set.
  * mock      — extractive stand-in so the pipeline always runs.

All backends share retries with exponential backoff + timeout. `complete` takes
`system` as a plain string OR a list of Anthropic content blocks (cache markers);
non-Anthropic backends flatten it to text. `__call__` is the text-in/text-out
hook the compressor uses.
"""

from __future__ import annotations

import json
import os
import random
import time
import urllib.error
import urllib.request
from dataclasses import dataclass

from .config import Config


@dataclass
class LLMResponse:
    text: str
    backend: str
    model: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    retries: int = 0


def _system_text(system) -> str:
    if isinstance(system, str):
        return system
    if isinstance(system, list):
        return "\n".join(b.get("text", "") for b in system if isinstance(b, dict))
    return str(system)


class LLM:
    def __init__(self, config: Config | None = None) -> None:
        self.cfg = config or Config()
        self.model = self.cfg.llm_model
        self._client = None
        self._errors: tuple = ()
        self.ollama_model = self.cfg.ollama_model
        self.backend = self._select_backend()

    # --- backend selection ------------------------------------------------
    def _select_backend(self) -> str:
        provider = self.cfg.llm_provider
        if provider == "auto":
            if os.environ.get("ANTHROPIC_API_KEY"):
                provider = "anthropic"
            elif os.environ.get("OPENAI_API_KEY"):
                provider = "openai"
            elif self._ollama_reachable():
                provider = "ollama"
            else:
                provider = "mock"

        if provider == "anthropic" and os.environ.get("ANTHROPIC_API_KEY"):
            try:
                import anthropic

                self._client = anthropic.Anthropic(timeout=self.cfg.llm_timeout_s)
                self._errors = (
                    anthropic.RateLimitError, anthropic.APIConnectionError,
                    anthropic.InternalServerError, anthropic.APIStatusError,
                )
                return "anthropic"
            except Exception:
                pass
        if provider == "openai" and os.environ.get("OPENAI_API_KEY"):
            return "openai"
        if provider == "ollama":
            self._resolve_ollama_model()
            return "ollama"
        return "mock"

    def _ollama_reachable(self) -> bool:
        try:
            with urllib.request.urlopen(self.cfg.ollama_host + "/api/tags", timeout=1.5):
                return True
        except Exception:
            return False

    def _resolve_ollama_model(self) -> None:
        """Use the configured model if pulled, else the first available one."""
        try:
            with urllib.request.urlopen(self.cfg.ollama_host + "/api/tags", timeout=2.0) as r:
                names = [m["name"] for m in json.loads(r.read()).get("models", [])]
            if names and self.cfg.ollama_model not in names and not any(
                n.split(":")[0] == self.cfg.ollama_model for n in names
            ):
                self.ollama_model = names[0]
        except Exception:
            pass

    # --- public API -------------------------------------------------------
    def complete(self, system, user: str) -> LLMResponse:
        if self.backend == "anthropic":
            return self._anthropic(system, user)
        if self.backend == "openai":
            return self._openai(_system_text(system), user)
        if self.backend == "ollama":
            return self._ollama(_system_text(system), user)
        return self._mock(system, user)

    def __call__(self, prompt: str) -> str:
        return self.complete("You are a concise summarizer.", prompt).text

    # --- streaming --------------------------------------------------------
    def stream(self, system, user: str):
        """Yield the answer as text chunks. Mirrors `complete`'s backend choice."""
        if self.backend == "anthropic":
            yield from self._anthropic_stream(system, user)
        elif self.backend == "openai":
            yield from self._openai_stream(_system_text(system), user)
        elif self.backend == "ollama":
            yield from self._ollama_stream(_system_text(system), user)
        else:
            yield from self._mock_stream(system, user)

    def _mock_stream(self, system, user: str):
        text = self._mock(system, user).text
        for i, tok in enumerate(text.split(" ")):
            yield tok if i == 0 else " " + tok

    def _anthropic_stream(self, system, user: str):  # pragma: no cover - needs key
        with self._client.messages.stream(
            model=self.model, max_tokens=self.cfg.llm_max_tokens,
            system=system, messages=[{"role": "user", "content": user}],
        ) as s:
            yield from s.text_stream

    def _openai_stream(self, system: str, user: str):  # pragma: no cover - needs key
        body = json.dumps({
            "model": self.cfg.openai_model,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
            "max_tokens": self.cfg.llm_max_tokens, "stream": True,
        }).encode("utf-8")
        req = urllib.request.Request(
            self.cfg.openai_base_url.rstrip("/") + "/chat/completions", data=body,
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY', '')}"})
        with urllib.request.urlopen(req, timeout=self.cfg.llm_timeout_s) as r:
            for raw in r:
                line = raw.decode("utf-8").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    delta = json.loads(data)["choices"][0]["delta"].get("content")
                except Exception:
                    continue
                if delta:
                    yield delta

    def _ollama_stream(self, system: str, user: str):  # pragma: no cover - needs server
        body = json.dumps({
            "model": self.ollama_model,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
            "stream": True, "options": {"num_predict": self.cfg.llm_max_tokens},
        }).encode("utf-8")
        req = urllib.request.Request(
            self.cfg.ollama_host + "/api/chat", data=body,
            headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=self.cfg.llm_timeout_s) as r:
                for raw in r:
                    line = raw.strip()
                    if not line:
                        continue
                    piece = json.loads(line).get("message", {}).get("content", "")
                    if piece:
                        yield piece
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            yield from self._mock_stream(system, user)  # degrade if server is down

    # --- anthropic --------------------------------------------------------
    def _anthropic(self, system, user: str) -> LLMResponse:
        last = None
        for attempt in range(self.cfg.llm_max_retries + 1):
            try:
                msg = self._client.messages.create(
                    model=self.model, max_tokens=self.cfg.llm_max_tokens,
                    system=system, messages=[{"role": "user", "content": user}],
                )
                text = "".join(b.text for b in msg.content if b.type == "text")
                u = msg.usage
                return LLMResponse(
                    text=text, backend="anthropic", model=self.model,
                    input_tokens=u.input_tokens, output_tokens=u.output_tokens,
                    cache_read_tokens=getattr(u, "cache_read_input_tokens", 0) or 0,
                    cache_write_tokens=getattr(u, "cache_creation_input_tokens", 0) or 0,
                    retries=attempt,
                )
            except self._errors as exc:
                last = exc
                if attempt >= self.cfg.llm_max_retries:
                    break
                time.sleep(self.cfg.llm_backoff_base_s * (2**attempt) + random.uniform(0, 0.3))
        raise RuntimeError(f"Anthropic call failed after retries: {last}")

    # --- openai-compatible (stdlib HTTP) ----------------------------------
    def _openai(self, system: str, user: str) -> LLMResponse:
        key = os.environ.get("OPENAI_API_KEY", "")
        body = json.dumps({
            "model": self.cfg.openai_model,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
            "max_tokens": self.cfg.llm_max_tokens,
        }).encode("utf-8")
        last = None
        for attempt in range(self.cfg.llm_max_retries + 1):
            try:
                req = urllib.request.Request(
                    self.cfg.openai_base_url.rstrip("/") + "/chat/completions",
                    data=body,
                    headers={"Content-Type": "application/json",
                             "Authorization": f"Bearer {key}"})
                with urllib.request.urlopen(req, timeout=self.cfg.llm_timeout_s) as r:
                    data = json.loads(r.read())
                usage = data.get("usage", {})
                return LLMResponse(
                    text=data["choices"][0]["message"]["content"],
                    backend="openai", model=self.cfg.openai_model,
                    input_tokens=usage.get("prompt_tokens", 0),
                    output_tokens=usage.get("completion_tokens", 0),
                    retries=attempt,
                )
            except (urllib.error.URLError, TimeoutError, ConnectionError, KeyError) as exc:
                last = exc
                if attempt >= self.cfg.llm_max_retries:
                    break
                time.sleep(self.cfg.llm_backoff_base_s * (2**attempt) + random.uniform(0, 0.3))
        raise RuntimeError(f"OpenAI call failed after retries: {last}")

    # --- ollama (local, stdlib HTTP) --------------------------------------
    def _ollama(self, system: str, user: str) -> LLMResponse:
        body = json.dumps({
            "model": self.ollama_model,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
            "stream": False,
            "options": {"num_predict": self.cfg.llm_max_tokens},
        }).encode("utf-8")
        last = None
        for attempt in range(self.cfg.llm_max_retries + 1):
            try:
                req = urllib.request.Request(
                    self.cfg.ollama_host + "/api/chat", data=body,
                    headers={"Content-Type": "application/json"})
                with urllib.request.urlopen(req, timeout=self.cfg.llm_timeout_s) as r:
                    data = json.loads(r.read())
                return LLMResponse(
                    text=data.get("message", {}).get("content", ""),
                    backend="ollama", model=self.ollama_model,
                    input_tokens=data.get("prompt_eval_count", 0),
                    output_tokens=data.get("eval_count", 0),
                    retries=attempt,
                )
            except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
                last = exc
                if attempt >= self.cfg.llm_max_retries:
                    break
                time.sleep(self.cfg.llm_backoff_base_s * (2**attempt) + random.uniform(0, 0.3))
        # local server unavailable — degrade to mock rather than crash
        resp = self._mock(system, user)
        resp.text = f"[ollama unavailable: {last}]\n\n" + resp.text
        return resp

    # --- mock -------------------------------------------------------------
    def _mock(self, system, user: str) -> LLMResponse:
        import re

        bullets = re.findall(r"^(?:- |\[\d+\] )(.+)$", user, flags=re.MULTILINE)
        req = re.search(r"<user_request>\s*(.+?)\s*</user_request>", user, re.DOTALL)
        question = req.group(1).strip() if req else "(no request found)"
        top = bullets[:4]
        body = "\n".join(f"  * {b}" for b in top) if top else "  (no context included)"
        text = (
            "[MOCK LLM - set ANTHROPIC_API_KEY or run Ollama for a real answer]\n\n"
            f"Request: {question}\n\n"
            f"Answering from the {len(bullets)} context item(s) assembled, "
            f"the most relevant being:\n{body}"
        )
        return LLMResponse(text=text, backend="mock")
