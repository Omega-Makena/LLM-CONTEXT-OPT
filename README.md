# contextx

A context-optimization engine. It ingests your documents, then for each request
assembles a tight, in-budget, cited prompt and calls an LLM.

```
INGEST:  documents → chunk → embed → persistent index
QUERY:   collect → retrieve → rerank → rank → filter → compress → budget → build → validate → LLM
         (memory read/write and a semantic cache wrap the flow)
```

## Install

```bash
pip install -e ".[full]"   # engine + real backends (faiss, sentence-transformers, anthropic, fastapi, pypdf)
pip install -e ".[dev]"    # minimal: runs on fallbacks, for tests
```

Copy `.env.example` to `.env` and add an LLM key (or run Ollama locally).

## Quick start

```python
from contextx import ContextEngine, Request, Document

engine = ContextEngine()
engine.ingest([Document(text="A refresh token is a long-lived credential ...")])
result = engine.run(Request(user_message="Explain refresh tokens"))
print(result.answer)
print(result.trace.report())     # per-stage timing, counts, cost
```

```bash
python examples/demo.py         # end-to-end demo with a trace
python examples/run_eval.py     # retrieval evaluation
uvicorn --factory contextx.service:create_app --port 8000   # HTTP service
pytest tests/
```

## LLM backends

Claude, OpenAI (and OpenAI-compatible gateways: Azure, OpenRouter, vLLM, LM
Studio), local Ollama, or a mock. `Config.llm_provider="auto"` selects
Claude → OpenAI → Ollama → mock based on available keys/servers.

Each optional backend (embeddings, vector index, reranker, tokenizer, LLM) has a
fallback, so the engine runs with zero heavy installs and upgrades as each
becomes available. The trace footer shows which backend each stage used.

## Features

- Ingest/query split with a persistent ANN index (FAISS HNSW; backend pluggable)
- Hybrid retrieval (semantic + BM25/FTS5) with cross-encoder reranking
- Multi-tenancy with per-document ACLs and API-key auth
- Citations, retrieval abstention, PII redaction, audit log
- Streaming answers, USD cost tracking, per-stage trace
- SQLite memory, semantic response cache
- Offline evaluation harness (IR metrics, ablations, weight tuning)
- Finance domain pack (`contextx/domains/`)

## Documentation

Full reference in [docs/](docs/):

- [context-optimization.md](docs/context-optimization.md) — how the pipeline works
- [developer-guide.md](docs/developer-guide.md) — setup, testing, extending
- [architecture.md](docs/architecture.md) · [configuration.md](docs/configuration.md) ·
  [api.md](docs/api.md) · [evaluation.md](docs/evaluation.md)
- [FINANCE.md](FINANCE.md) — finance domain pack

## Layout

```
contextx/   library (one module per stage)
examples/   demos and eval scripts
tests/      pytest suite
docs/       documentation
scripts/    CI eval gate
```

## License

MIT — see [LICENSE](LICENSE).
