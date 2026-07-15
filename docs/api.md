# HTTP API

The FastAPI service (`contextx/service.py`) exposes the engine over HTTP. It needs
the `serve` extra:

```bash
pip install -e ".[serve,full]"
uvicorn --factory contextx.service:create_app --host 0.0.0.0 --port 8000
# or: docker build -t contextx . && docker run -p 8000:8000 -e ANTHROPIC_API_KEY=... contextx
```

## Authentication

Callers **cannot** assert their own identity. Each API key is bound server-side to
exactly one tenant + set of principals (`contextx/auth.py`); the authenticated
identity is what flows into retrieval, so tenant isolation is real end-to-end.
`tenant_id`/`principals` are not accepted in request bodies.

Configure keys via the `CONTEXTX_API_KEYS` env var (JSON), a file, or by passing
an `APIKeyAuth` to `create_app`:

```bash
export CONTEXTX_API_KEYS='{"sk-acme-123": {"tenant_id": "acme", "principals": ["execs"]}}'
```

Send the key as a bearer token (or `X-API-Key`):

```
Authorization: Bearer sk-acme-123
```

Keys are stored **hashed** (SHA-256) with constant-time comparison. If no keys are
configured the service runs in **open/dev mode** (single `default` tenant, no
auth) — do not do this in production.

## Endpoints

### `GET /health`
Liveness. Open (no auth). → `{"status": "ok", "auth": "enabled" | "open"}`

### `GET /stats`
Index status. → `{"backend", "chunks", "vectors", "fts"}`

### `POST /ingest`  *(auth)*
Add documents into the caller's tenant.
```json
{"documents": [
  {"text": "...", "doc_id": "acme-10k-2024", "source": "knowledge_base",
   "metadata": {"year": 2024}, "acl": ["execs"]}
]}
```
→ `{"chunks_added": 12, "tenant_id": "acme"}`.
`acl` is an optional within-tenant read list (empty = all principals in the tenant).

### `POST /documents/update`  *(auth)*
Replace documents by `doc_id` (delete old chunks, re-ingest). Same body as ingest.
→ `{"chunks": 12, "tenant_id": "acme"}`.

### `DELETE /documents/{doc_id}`  *(auth)*
Remove a document — **scoped to the caller's tenant** (a key cannot delete another
tenant's document by id). → `{"chunks_removed": 12}` or `404`.

### `POST /query`  *(auth)*
Run the pipeline and return the full answer.
```json
{"user_message": "what was revenue?",
 "conversation": ["earlier context ..."],
 "preferences": {"style": "concise"},
 "max_context_tokens": 32000, "reserve_output_tokens": 4000}
```
→
```json
{"answer": "...",
 "sources": [{"n": 1, "doc_id": "acme-10k-2024", "chunk_id": "...", "preview": "..."}],
 "low_confidence": false,
 "backend": "anthropic",
 "prompt_tokens": 812,
 "tenant_id": "acme"}
```

### `POST /query/stream`  *(auth)*
Same input as `/query`, streamed as **Server-Sent Events**:
```
event: meta
data: {"tenant_id": "acme", "sources": [...], "low_confidence": false}

data: {"text": "Revenue "}
data: {"text": "was "}
...
event: done
data: {}
```
The `meta` event (sources + confidence) is sent first, before any answer text.

## Example (curl)

```bash
KEY="sk-acme-123"

curl -s -X POST localhost:8000/ingest \
  -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" \
  -d '{"documents":[{"text":"Acme FY2024 revenue was $5.0M.","doc_id":"acme-fy24"}]}'

curl -s -X POST localhost:8000/query \
  -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" \
  -d '{"user_message":"what was Acme revenue?"}'

curl -N -X POST localhost:8000/query/stream \
  -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" \
  -d '{"user_message":"what was Acme revenue?"}'
```

## Notes

- The engine is synchronous and CPU/GPU-bound; endpoints offload it to a worker
  thread (via `engine.arun` / `arun_stream`) so the event loop is never blocked.
- Index writes are serialized with an async lock.
- Interactive docs are available at `/docs` (Swagger) when the service is running.
