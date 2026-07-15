# Developer guide

Onboarding and reference for engineers working on contextx.

## 1. Setup

Requires Python ≥ 3.10.

```bash
git clone https://github.com/Omega-Makena/LLM-CONTEXT-OPT
cd LLM-CONTEXT-OPT
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"        # core + pytest + ruff
pip install -e ".[full]"       # add the real backends (faiss, sentence-transformers, anthropic, fastapi, pypdf)
```

The engine runs with **zero heavy installs** thanks to fallbacks, so `pip install -e .`
is enough to import and test; `[full]` gives you the real embedding/rerank/LLM
stack.

Optional runtime config:
```bash
export ANTHROPIC_API_KEY=sk-ant-...   # use Claude
export OPENAI_API_KEY=sk-...          # use OpenAI
# or run Ollama locally: `ollama serve && ollama pull llama3.1`
```

## 2. Run it

```bash
python examples/demo.py         # ingest a corpus, answer a query, print the trace
python examples/run_eval.py     # retrieval evaluation on the golden set
python examples/eval_custom.py  # eval + tune on your own JSONL data
python examples/finance_demo.py # the finance domain pack end-to-end
uvicorn --factory contextx.service:create_app --port 8000   # the HTTP service
```

Minimal library use:
```python
from contextx import ContextEngine, Request, Document

engine = ContextEngine()
engine.ingest([Document(text="A refresh token is a long-lived credential ...")])
result = engine.run(Request(user_message="Explain refresh tokens"))
print(result.answer)
print(result.trace.report())     # per-stage timing + counts + cost
```

## 3. Project layout

```
contextx/            the library (one module per stage — see architecture.md)
  eval/              evaluation harness (metrics, datasets, faithfulness)
  domains/           domain packs (finance)
examples/            runnable demos + eval scripts + sample data
tests/               pytest suite (unit + integration)
docs/                these documents
scripts/             eval_gate.py (CI regression gate)
pyproject.toml       packaging + extras + ruff/pytest config
.github/workflows/   CI (tests on fallback backends + eval gate)
```

## 4. Testing

```bash
pytest -q                          # full suite
pytest tests/test_pipeline.py -q   # one file
pytest -k redact -q                # by keyword
python -m ruff check contextx tests   # lint (must be clean)
```

Notes:
- The full suite is **slow (~8 min)** because each `ContextEngine` construction
  loads the embedder + cross-encoder. **CI is fast** — it installs *without*
  `sentence-transformers`, so tests run on the hash/identity fallbacks.
- Write tests to pass on **both** the real and fallback backends (CI uses
  fallbacks). Use `Embedder(force_fallback=True)` for fast, deterministic
  store/embedding tests.
- Isolate state per test with `tmp_path` and a `Config(index_dir=..., memory_db_path=...)`.

## 5. How to extend

### Add / change an LLM backend (`llm.py`)
Add a branch in `_select_backend`, implement `_yourbackend(system, user) -> LLMResponse`
(with retries) and `_yourbackend_stream(...)`, and wire them in `complete()` /
`stream()`. Set `response.backend` and `response.model` so cost accounting works.

### Add a vector-store backend (`backends.py`)
Implement the `VectorBackend` protocol (`add / search / count / reset / save /
load`), register it in `make_backend`, and add any config fields. `search` must
return `(row_ids, cosine_sims)`. The store guarantees dense row ids, so
position-based backends (faiss/numpy) can ignore the ids.

### Add a domain pack (`domains/`)
Provide a `Config` preset (`finance_config`), a system prompt, any domain PII
patterns (as `(label, pattern, validator)` tuples passed to `redact_pii`), entity
extractors, and a labeled eval set (`Document` corpus + `EvalExample` queries).
See `domains/finance.py` and `domains/finance_data.py`.

### Add / reorder a pipeline stage (`pipeline.py`)
Stages live in `_prepare()` (1–9) and `run()`/`run_stream()` (10, 5, 11). Wrap new
work in `with trace.stage("name") as rec:` so it shows in the trace. Keep expensive
work late (operate on the smallest surviving set).

### Tune ranking weights
Don't hand-edit weights blindly — run `contextx.tune.tune_weights(engine, queries)`
against labeled data to fit `w_rerank / w_similarity / w_bm25` by nDCG. See
[evaluation.md](evaluation.md).

## 6. Coding conventions

- **All knobs go in `Config`** (`config.py`) — no magic-number literals in code.
- **One `ContextItem` flows through stages**; add fields there, not ad-hoc dicts.
- **Graceful fallback** for every optional dependency — import lazily, degrade,
  and report the backend in the trace.
- **No unverified code.** If you can't test a path here (e.g. a DB backend), mark
  it experimental, gate it behind an import, and test its *failure* mode.
- **Line length 100**, ruff must pass. Match the surrounding comment density and
  naming.
- Prefer the dedicated tests over manual verification; add a test with every
  behaviour change.

## 7. Common pitfalls

- Editing a module while the test suite is running causes a mid-import race and
  spurious failures — let a run finish before editing.
- Changing `Config.embed_model` invalidates an existing index (dimension/space
  mismatch). The store raises `ModelMismatchError`; delete `index_dir` to rebuild.
- The semantic response cache is invalidated on ingest/update/delete — if you add
  a new write path, call `cache.invalidate_responses()`.
- Tenant/ACL filtering is post-ANN; if you touch `store.search`, preserve the
  adaptive over-fetch loop or small tenants will under-return.

## 8. Release / CI

- CI runs on push/PR to `main`: install (core + faiss + tiktoken + anthropic, no
  torch) → ruff (non-blocking) → pytest → `scripts/eval_gate.py` (fails the build
  if retrieval regresses below conservative thresholds).
- Version lives in `pyproject.toml` and `contextx/__init__.__version__`.
