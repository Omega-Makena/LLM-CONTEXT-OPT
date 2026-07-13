"""FastAPI service layer — turns the engine (a library) into an HTTP service.

Endpoints (all mutating/query routes are authenticated):
  GET  /health              liveness (open)
  GET  /stats               index/backends status (open)
  POST /ingest              add documents into the caller's tenant
  POST /documents/update    replace documents by doc_id
  DELETE /documents/{id}    remove a document (scoped to the caller's tenant)
  POST /query               run the pipeline; returns answer + citations

AUTH: the caller's `tenant_id`/`principals` are derived server-side from their
API key (see `contextx.auth.APIKeyAuth`), NOT taken from the request body — so a
client cannot impersonate another tenant. If no keys are configured, the service
runs in OPEN (single-tenant, dev) mode. Configure via `CONTEXTX_API_KEYS` env or
pass an `APIKeyAuth` to `create_app`.

The `ContextEngine` is synchronous and single-instance; endpoints run it in a
threadpool (`asyncio.to_thread`) so one slow embed/LLM call doesn't block the
event loop. A global asyncio lock serializes index writes.

Requires the `serve` extra:  pip install "contextx[serve,full]"
Run:  uvicorn --factory contextx.service:create_app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import asyncio
from typing import Any

try:
    from fastapi import Depends, FastAPI, Header, HTTPException
    from fastapi import Request as HTTPRequest
    from pydantic import BaseModel, Field
except Exception as exc:  # pragma: no cover
    raise ImportError(
        "The service layer needs FastAPI. Install with: pip install 'contextx[serve]'"
    ) from exc

from .auth import APIKeyAuth, AuthContext
from .config import Config
from .pipeline import ContextEngine
from .types import Document, Request, Source


# --- request/response schemas ---------------------------------------------
# Note: tenant_id / principals are intentionally ABSENT from the inbound schemas
# — they come from the authenticated API key, never from the caller.
class DocIn(BaseModel):
    text: str
    doc_id: str = ""
    source: str = "knowledge_base"
    metadata: dict[str, Any] = Field(default_factory=dict)
    acl: list[str] = Field(default_factory=list)  # within-tenant read ACL


class IngestIn(BaseModel):
    documents: list[DocIn]


class QueryIn(BaseModel):
    user_message: str
    conversation: list[str] = Field(default_factory=list)
    preferences: dict[str, Any] = Field(default_factory=dict)
    max_context_tokens: int = 32_000
    reserve_output_tokens: int = 4_000


class QueryOut(BaseModel):
    answer: str
    sources: list[dict]
    low_confidence: bool
    backend: str
    prompt_tokens: int
    tenant_id: str


def _to_document(d: DocIn, tenant_id: str) -> Document:
    return Document(
        text=d.text, doc_id=d.doc_id, source=Source(d.source),
        metadata=d.metadata, tenant_id=tenant_id, acl=d.acl,
    )


def create_app(
    engine: ContextEngine | None = None, auth: APIKeyAuth | None = None
) -> "FastAPI":
    app = FastAPI(title="contextx", version="0.3.0")
    app.state.engine = engine or ContextEngine(Config())
    app.state.auth = auth or APIKeyAuth.from_env()
    app.state.write_lock = asyncio.Lock()

    async def identity(
        request: HTTPRequest,
        authorization: str | None = Header(None),
        x_api_key: str | None = Header(None),
    ) -> AuthContext:
        """Resolve the caller's tenant + principals from their API key."""
        a: APIKeyAuth = request.app.state.auth
        if not a.configured:
            return AuthContext(tenant_id="default", principals=[])  # open/dev mode
        token = None
        if authorization and authorization.lower().startswith("bearer "):
            token = authorization[7:].strip()
        ctx = a.authenticate(token or x_api_key)
        if ctx is None:
            raise HTTPException(status_code=401, detail="invalid or missing API key")
        return ctx

    @app.get("/health")
    async def health():
        return {"status": "ok", "auth": "enabled" if app.state.auth.configured else "open"}

    @app.get("/stats")
    async def stats():
        return app.state.engine.store.stats()

    @app.post("/ingest")
    async def ingest(body: IngestIn, ident: AuthContext = Depends(identity)):
        docs = [_to_document(d, ident.tenant_id) for d in body.documents]
        async with app.state.write_lock:
            n = await asyncio.to_thread(app.state.engine.ingest, docs)
        return {"chunks_added": n, "tenant_id": ident.tenant_id}

    @app.post("/documents/update")
    async def update(body: IngestIn, ident: AuthContext = Depends(identity)):
        docs = [_to_document(d, ident.tenant_id) for d in body.documents]
        async with app.state.write_lock:
            n = await asyncio.to_thread(app.state.engine.update, docs)
        return {"chunks": n, "tenant_id": ident.tenant_id}

    @app.delete("/documents/{doc_id}")
    async def delete(doc_id: str, ident: AuthContext = Depends(identity)):
        async with app.state.write_lock:
            n = await asyncio.to_thread(app.state.engine.delete, doc_id, ident.tenant_id)
        if n == 0:
            raise HTTPException(status_code=404, detail="document not found in your tenant")
        return {"chunks_removed": n}

    @app.post("/query", response_model=QueryOut)
    async def query(body: QueryIn, ident: AuthContext = Depends(identity)):
        req = Request(
            user_message=body.user_message,
            tenant_id=ident.tenant_id,
            principals=ident.principals,
            conversation=body.conversation,
            preferences=body.preferences,
            max_context_tokens=body.max_context_tokens,
            reserve_output_tokens=body.reserve_output_tokens,
        )
        result = await asyncio.to_thread(app.state.engine.run, req)
        return QueryOut(
            answer=result.answer,
            sources=result.sources,
            low_confidence=result.low_confidence,
            backend=result.llm.backend,
            prompt_tokens=result.trace.metrics.get("prompt_tokens", 0),
            tenant_id=ident.tenant_id,
        )

    return app
