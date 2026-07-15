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

## One-paragraph summary

An LLM only sees what you put in its context window. contextx is the layer that
decides *what goes in*: it **ingests** documents once (chunk → embed → persistent
index), and per request it **collects** candidate context from every source,
**retrieves** the relevant pieces (semantic + lexical), **reranks** them with a
cross-encoder, **ranks/filters/compresses** what survives, fits it to a **token
budget**, **builds** a clean prompt with citations and trust boundaries,
**validates** it, and calls the model — while a **cache**, **memory**, and
**observability** layer wrap the whole flow. The result is a prompt that is
relevant, in-budget, attributable, safe, and cheap.

## Conventions in these docs

- Stage numbers (1–12) match the pipeline and the trace output.
- File references look like `contextx/retrieve.py` and are relative to the repo
  root.
- "Ingest time" = amortized, run once per document; "query time" = per request.
