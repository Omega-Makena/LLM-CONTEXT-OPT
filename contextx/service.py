"""FastAPI service layer — turns the engine (a library) into an HTTP service.

Endpoints:
  GET  /health              liveness
  GET  /stats               index/backends status
  POST /ingest              add documents (tenant/acl aware)
  POST /documents/update    replace documents by doc_id
  DELETE /documents/{id}    remove a document
  POST /query               run the pipeline; returns answer + citations

The `ContextEngine` is synchronous and single-instance; endpoints run it in a
threadpool (`asyncio.to_thread`) so one slow embed/LLM call doesn't block the
event loop and the server can serve concurrent requests. A global asyncio lock
serializes index writes (the store is not safe for concurrent writers).

Requires the `serve` extra:  pip install "contextx[serve,full]"
Run:  uvicorn contextx.service:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import asyncio
from typing import Any

try:
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel, Field
except Exception as exc:  # pragma: no cover
    raise ImportError(
        "The service layer needs FastAPI. Install with: pip install 'contextx[serve]'"
    ) from exc

from .config import Config
from .pipeline import ContextEngine
from .types import Document, Request, Source

# --- request/response schemas ---------------------------------------------
class DocIn(BaseModel):
    text: str
    doc_id: str = ""
    source: str = "knowledge_base"
    metadata: dict[str, Any] = Field(default_factory=dict)
    tenant_id: str = "default"
    acl: list[str] = Field(default_factory=list)


class IngestIn(BaseModel):
    documents: list[DocIn]


class QueryIn(BaseModel):
    user_message: str
    tenant_id: str = "default"
    principals: list[str] = Field(default_factory=list)
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


def _to_document(d: DocIn) -> Document:
    return Document(
        text=d.text, doc_id=d.doc_id, source=Source(d.source),
        metadata=d.metadata, tenant_id=d.tenant_id, acl=d.acl,
    )


def create_app(engine: ContextEngine | None = None) -> "FastAPI":
    app = FastAPI(title="contextx", version="0.3.0")
    app.state.engine = engine or ContextEngine(Config())
    app.state.write_lock = asyncio.Lock()  # serialize index writers

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.get("/stats")
    async def stats():
        return app.state.engine.store.stats()

    @app.post("/ingest")
    async def ingest(body: IngestIn):
        docs = [_to_document(d) for d in body.documents]
        async with app.state.write_lock:
            n = await asyncio.to_thread(app.state.engine.ingest, docs)
        return {"chunks_added": n}

    @app.post("/documents/update")
    async def update(body: IngestIn):
        docs = [_to_document(d) for d in body.documents]
        async with app.state.write_lock:
            n = await asyncio.to_thread(app.state.engine.update, docs)
        return {"chunks": n}

    @app.delete("/documents/{doc_id}")
    async def delete(doc_id: str):
        async with app.state.write_lock:
            n = await asyncio.to_thread(app.state.engine.delete, doc_id)
        if n == 0:
            raise HTTPException(status_code=404, detail="document not found")
        return {"chunks_removed": n}

    @app.post("/query", response_model=QueryOut)
    async def query(body: QueryIn):
        req = Request(
            user_message=body.user_message,
            tenant_id=body.tenant_id,
            principals=body.principals,
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
        )

    return app


# Served via uvicorn factory mode so importing this module does NOT construct an
# engine (which would load models):  uvicorn --factory contextx.service:create_app
