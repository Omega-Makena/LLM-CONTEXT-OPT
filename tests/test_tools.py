"""Tests for tool use (function calling) in the LLM stage."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from contextx import Config, Tool  # noqa: E402
from contextx.llm import LLM  # noqa: E402

ADD = Tool("add", "add two numbers",
           {"type": "object", "properties": {"a": {"type": "number"}, "b": {"type": "number"}}},
           lambda a, b: a + b)


def _openai_llm():
    llm = LLM(Config(llm_provider="mock"))
    llm.backend = "openai"  # force the openai tool path (no network — chat is patched)
    return llm


def test_openai_tool_loop_executes_and_finalizes():
    llm = _openai_llm()
    seen = {"n": 0}

    def fake_chat(messages, specs):
        seen["n"] += 1
        if seen["n"] == 1:
            return {"tool_calls": [{"id": "c1", "function": {
                "name": "add", "arguments": '{"a": 2, "b": 3}'}}]}
        return {"content": "The sum is 5.", "tool_calls": None}

    llm._openai_chat_once = fake_chat
    resp = llm.run_tools("sys", "what is 2+3?", [ADD])
    assert resp.text == "The sum is 5."
    assert resp.tool_calls == 1
    assert seen["n"] == 2                     # one tool round + one final round


def test_tool_error_is_returned_not_raised():
    llm = _openai_llm()
    captured = {}

    def fake_chat(messages, specs):
        if "done" not in captured:
            captured["done"] = True
            return {"tool_calls": [{"id": "c", "function": {
                "name": "boom", "arguments": "{}"}}]}
        captured["result"] = messages[-1]["content"]
        return {"content": "handled", "tool_calls": None}

    llm._openai_chat_once = fake_chat
    boom = Tool("boom", "raises", {"type": "object", "properties": {}}, lambda: 1 / 0)
    resp = llm.run_tools("s", "go", [boom])
    assert resp.text == "handled"
    assert "error" in captured["result"].lower()   # tool exception fed back to model


def test_unknown_tool_is_handled():
    llm = _openai_llm()
    state = {}

    def fake_chat(messages, specs):
        if "x" not in state:
            state["x"] = 1
            return {"tool_calls": [{"id": "c", "function": {
                "name": "missing", "arguments": "{}"}}]}
        state["result"] = messages[-1]["content"]
        return {"content": "ok", "tool_calls": None}

    llm._openai_chat_once = fake_chat
    resp = llm.run_tools("s", "go", [ADD])
    assert resp.text == "ok"
    assert "no such tool" in state["result"].lower()


def test_mock_backend_runs_without_tools():
    llm = LLM(Config(llm_provider="mock"))
    resp = llm.run_tools("s", "<user_request>\nhi\n</user_request>", [ADD])
    assert resp.backend == "mock"
