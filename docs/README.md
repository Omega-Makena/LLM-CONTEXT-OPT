# contextx documentation

Detailed reference for the contextx context-optimization engine — read this to
understand *how* it optimizes an LLM's context and *how* to work on the code.

## Start here

| If you want to… | Read |
|---|---|
| Understand the method — how we optimize LLM context, stage by stage | [context-optimization.md](context-optimization.md) |
| Understand the codebase — architecture, data flow, module map | [architecture.md](architecture.md) |
| Get set up and start contributing | [developer-guide.md](developer-guide.md) |
| Tune behaviour — every configuration knob | [configuration.md](configuration.md) |
| Call the HTTP service (ingest/query/stream, auth) | [api.md](api.md) |
| Measure retrieval quality and tune weights | [evaluation.md](evaluation.md) |

The top-level [../README.md](../README.md) is the quick-start; [../FINANCE.md](../FINANCE.md)
documents the finance domain pack.

## Summary

contextx decides what goes into an LLM's context window. It ingests documents
once (chunk → embed → persistent index), then per request: collect → retrieve
(semantic + lexical) → rerank → rank → filter → compress → budget → build →
validate → LLM, with cache, memory, and observability across the flow.

## Conventions

- Stage numbers (1–12) match the pipeline and the trace output.
- File references (`contextx/retrieve.py`) are relative to the repo root.
- "Ingest time" = once per document; "query time" = per request.
