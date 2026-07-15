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
from typing import Any, Callable

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
    tool_calls: int = 0        # number of tool invocations in a tool run


@dataclass
class Tool:
    """A callable the model may invoke. `parameters` is a JSON Schema object."""
    name: str
    description: str
    parameters: dict
    func: Callable[..., Any]


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

    # --- tool use (function calling) --------------------------------------
    def run_tools(self, system, user: str, tools: list[Tool],
                  max_iters: int = 6) -> LLMResponse:
        """Agentic loop: the model may call the provided tools until it produces
        a final answer. Supported on the openai and anthropic backends; the mock
        backend answers without tools."""
        if self.backend == "openai":
            return self._openai_tools(_system_text(system), user, tools, max_iters)
        if self.backend == "anthropic":
            return self._anthropic_tools(system, user, tools, max_iters)
        return self._mock(system, user)

    @staticmethod
    def _dispatch(tools: list[Tool], name: str, args: dict) -> str:
        tool = next((t for t in tools if t.name == name), None)
        if tool is None:
            return f"error: no such tool '{name}'"
        try:
            return str(tool.func(**args))
        except Exception as exc:  # tool errors are returned to the model, not raised
            return f"error: {exc}"

    def _openai_chat_once(self, messages: list[dict], tool_specs: list[dict]) -> dict:
        """One OpenAI chat call with tools; returns the assistant message dict.
        Isolated so the tool loop can be unit-tested by patching this method."""
        body = json.dumps({
            "model": self.cfg.openai_model, "messages": messages,
            "tools": tool_specs, "max_tokens": self.cfg.llm_max_tokens,
        }).encode("utf-8")
        req = urllib.request.Request(
            self.cfg.openai_base_url.rstrip("/") + "/chat/completions", data=body,
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY', '')}"})
        with urllib.request.urlopen(req, timeout=self.cfg.llm_timeout_s) as r:
            return json.loads(r.read())["choices"][0]["message"]

    def _openai_tools(self, system: str, user: str, tools: list[Tool],
                      max_iters: int) -> LLMResponse:
        specs = [{"type": "function", "function": {
            "name": t.name, "description": t.description, "parameters": t.parameters}}
            for t in tools]
        messages = [{"role": "system", "content": system},
                    {"role": "user", "content": user}]
        calls = 0
        for _ in range(max_iters):
            msg = self._openai_chat_once(messages, specs)
            tool_calls = msg.get("tool_calls")
            if not tool_calls:
                return LLMResponse(text=msg.get("content") or "", backend="openai",
                                   model=self.cfg.openai_model, tool_calls=calls)
            messages.append(msg)
            for call in tool_calls:
                calls += 1
                fn = call["function"]
                try:
                    args = json.loads(fn.get("arguments") or "{}")
                except Exception:
                    args = {}
                messages.append({"role": "tool", "tool_call_id": call["id"],
                                 "content": self._dispatch(tools, fn["name"], args)})
        return LLMResponse(text="(max tool iterations reached)", backend="openai",
                           model=self.cfg.openai_model, tool_calls=calls)

    def _anthropic_tools(self, system, user: str, tools: list[Tool],
                         max_iters: int) -> LLMResponse:  # pragma: no cover - needs key
        specs = [{"name": t.name, "description": t.description,
                  "input_schema": t.parameters} for t in tools]
        messages = [{"role": "user", "content": user}]
        calls = 0
        for _ in range(max_iters):
            msg = self._client.messages.create(
                model=self.model, max_tokens=self.cfg.llm_max_tokens,
                system=system, tools=specs, messages=messages)
            if msg.stop_reason != "tool_use":
                text = "".join(b.text for b in msg.content if b.type == "text")
                return LLMResponse(text=text, backend="anthropic", model=self.model,
                                   tool_calls=calls)
            messages.append({"role": "assistant", "content": msg.content})
            results = []
            for block in msg.content:
                if block.type == "tool_use":
                    calls += 1
                    results.append({"type": "tool_result", "tool_use_id": block.id,
                                    "content": self._dispatch(tools, block.name, block.input)})
            messages.append({"role": "user", "content": results})
        return LLMResponse(text="(max tool iterations reached)", backend="anthropic",
                           model=self.model, tool_calls=calls)

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
