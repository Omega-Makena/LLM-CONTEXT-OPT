"""The orchestrator.

Two entry points, reflecting the ingest/query split:

  * ingest(documents)  — one-time / incremental: chunk -> embed -> persist into
    the vector store. Amortized cost; survives restarts.
  * run(request, ...)  — per request: collect ephemeral context, retrieve
    durable context from the index, rerank, rank, filter, compress, budget,
    build, validate, call the LLM, write memory. Cache wraps the LLM call
    (semantically) so paraphrased requests hit.

Every stage records into a Trace so you can see counts and latency move.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

from .budget import BudgetManager
from .build import BuiltPrompt, PromptBuilder
from .cache import Cache
from .collect import Collector
from .compress import Compressor
from .config import Config
from .embeddings import Embedder
from .filter import Filter
from .llm import LLM, LLMResponse
from .memory import MemoryManager
from .observability import Trace, estimate_cost
from .rank import Ranker
from .rerank import Reranker
from .retrieve import Retriever
from .security import AuditLog, redact_pii
from .store import VectorStore
from .types import UNTRUSTED_SOURCES, Document, Request, Source
from .validate import Validator

DEFAULT_SYSTEM = (
    "You are a helpful assistant. Use the provided context blocks to answer the "
    "user's request accurately. If the context is insufficient, say so."
)

_STREAM_DONE = object()


def _next_or_done(it):
    try:
        return next(it)
    except StopIteration:
        return _STREAM_DONE


def _memory_scope(request: Request) -> str:
    """Per-requester memory namespace: tenant + first principal (user), so one
    user's remembered facts never leak to another. Default single-user mode
    (no tenant, no principal) keeps the shared 'global' scope."""
    if request.tenant_id == "default" and not request.principals:
        return "global"
    user = request.principals[0] if request.principals else "anon"
    return f"{request.tenant_id}:{user}"


@dataclass
class PipelineResult:
    answer: str
    prompt: BuiltPrompt
    llm: LLMResponse
    trace: Trace
    sources: list[dict] = field(default_factory=list)   # citations for the answer
    low_confidence: bool = False                         # retrieval abstention flag


@dataclass
class _Prepared:
    """Everything stages 1-9 produce; shared by run() and run_stream()."""
    prompt: BuiltPrompt
    report: Any
    low_confidence: bool
    confidence: float
    scope: str
    pii_redacted: int


@dataclass
class StreamResult:
    stream: Any                 # generator yielding answer text chunks
    sources: list[dict]         # known up front, before streaming
    low_confidence: bool
    prompt: BuiltPrompt
    trace: Trace


class ContextEngine:
    def __init__(
        self,
        config: Config | None = None,
        embedder: Embedder | None = None,
        memory: MemoryManager | None = None,
        llm: LLM | None = None,
        system_prompt: str = DEFAULT_SYSTEM,
        abstractive_compression: bool = False,
    ) -> None:
        self.cfg = config or Config()
        self.cache = Cache(self.cfg)
        self.embedder = embedder or Embedder(
            self.cfg.embed_model, batch_size=self.cfg.embed_batch_size)
        self.store = VectorStore(self.embedder, self.cfg)
        self.collector = Collector()
        self.retriever = Retriever(self.embedder, self.store, self.cfg, self.cache)
        self.reranker = Reranker(self.cfg)
        self.ranker = Ranker(self.cfg)
        # similarity cutting happens at retrieval; the filter only dedups /
        # drops stale / resolves contradictions (so hybrid lexical hits with low
        # cosine but high rerank aren't wrongly dropped here).
        self.filter = Filter(self.cfg.dup_threshold, 0.0)
        self.compressor = Compressor(self.embedder, self.cfg)
        self.budget = BudgetManager(self.cfg)
        self.builder = PromptBuilder(self.cfg)
        self.validator = Validator(self.cfg)
        self.memory = memory or MemoryManager(self.embedder, self.cfg)
        self.llm = llm or LLM(self.cfg)
        if self.cfg.llm_memory_extraction:
            self.memory.llm = self.llm  # enable LLM fact extraction
        self.system_prompt = system_prompt
        self.abstractive_compression = abstractive_compression
        self.audit = AuditLog(self.cfg.audit_log_path) if self.cfg.audit_log_path else None

    # --- ingest (amortized) ----------------------------------------------
    def ingest(self, documents: list[Document]) -> int:
        """Chunk + embed + persist durable documents. Returns #chunks added."""
        n = self.store.add_documents(documents)
        if n:
            self.cache.invalidate_responses()  # corpus changed -> drop stale answers
        return n

    def update(self, documents: list[Document]) -> int:
        """Replace documents by doc_id (delete old chunks, re-ingest). Keeps the
        index from going stale when source docs change."""
        n = self.store.update_documents(documents)
        self.cache.invalidate_responses()
        return n

    def delete(self, doc_id: str, tenant_id: str | None = None) -> int:
        """Remove a document and its chunks from the index. Pass `tenant_id` to
        restrict deletion to that tenant's copy. Returns #chunks."""
        n = self.store.delete_document(doc_id, tenant_id=tenant_id)
        if n:
            self.cache.invalidate_responses()
        return n

    # --- retrieval-only (used by the eval harness) -----------------------
    def recall_candidates(self, query: str, rerank: bool = True) -> list:
        """Return corpus candidates for `query` with `.similarity` (bi-encoder)
        and, if `rerank`, `.rerank_score` (cross-encoder) populated. Vector-only
        (hybrid=False) and no ephemeral/memory, so the eval measures the semantic
        retrieval + rerank stages in isolation.
        """
        cands = self.retriever.retrieve(query, [], None, hybrid=False)
        if rerank:
            self.reranker.rerank(query, cands)  # sets .rerank_score in place
        return cands

    # --- query (per request) ---------------------------------------------
    def run(
        self, request: Request, write_memory: bool = True, **ephemeral_sources: Any
    ) -> PipelineResult:
        trace = Trace()
        prep = self._prepare(request, trace, **ephemeral_sources)

        # 10 — LLM (semantic response cache wraps the call)
        with trace.stage("10 llm") as rec:
            if not prep.report.ok:
                response = LLMResponse(
                    text="[BLOCKED] validation failed: " + "; ".join(prep.report.errors),
                    backend="none")
            else:
                qvec = self.embedder.encode_one(prep.prompt.user)
                response = self.cache.semantic_get_or_compute(
                    "llm", qvec, key=prep.prompt.user[:256],
                    compute=lambda: self.llm.complete(
                        prep.prompt.system_blocks, prep.prompt.user))
            rec.notes["backend"] = response.backend
            if response.retries:
                rec.notes["retries"] = response.retries

        # 5 — Memory write (scoped to the requester)
        if write_memory and prep.report.ok:
            with trace.stage("5 memory-write") as rec:
                new = self.memory.extract_and_store(
                    request.user_message, response.text, scope=prep.scope)
                rec.notes["stored"] = len(new)

        self._finalize(request, trace, prep, response)
        return PipelineResult(
            answer=response.text, prompt=prep.prompt, llm=response, trace=trace,
            sources=prep.prompt.sources, low_confidence=prep.low_confidence)

    def run_stream(
        self, request: Request, write_memory: bool = True, **ephemeral_sources: Any
    ) -> StreamResult:
        """Like run(), but streams the answer as text chunks. `sources` and
        `low_confidence` are known before streaming; the response cache is
        bypassed (you can't cache a half-streamed answer). Memory is written once
        the stream is fully consumed."""
        trace = Trace()
        prep = self._prepare(request, trace, **ephemeral_sources)

        def gen():
            chunks: list[str] = []
            if not prep.report.ok:
                msg = "[BLOCKED] validation failed: " + "; ".join(prep.report.errors)
                chunks.append(msg)
                yield msg
            else:
                for piece in self.llm.stream(prep.prompt.system_blocks, prep.prompt.user):
                    chunks.append(piece)
                    yield piece
            if write_memory and prep.report.ok:
                self.memory.extract_and_store(
                    request.user_message, "".join(chunks), scope=prep.scope)

        return StreamResult(
            stream=gen(), sources=prep.prompt.sources,
            low_confidence=prep.low_confidence, prompt=prep.prompt, trace=trace)

    def run_with_tools(
        self, request: Request, tools: list, write_memory: bool = True,
        **ephemeral_sources: Any,
    ) -> PipelineResult:
        """Like run(), but the model may call `tools` (a list of llm.Tool) in an
        agentic loop after the context prompt is built. Bypasses the response
        cache (tool runs are not cacheable)."""
        trace = Trace()
        prep = self._prepare(request, trace, **ephemeral_sources)
        with trace.stage("10 llm+tools") as rec:
            if not prep.report.ok:
                response = LLMResponse(
                    text="[BLOCKED] validation failed: " + "; ".join(prep.report.errors),
                    backend="none")
            else:
                response = self.llm.run_tools(
                    prep.prompt.system_blocks, prep.prompt.user, tools)
            rec.notes["backend"] = response.backend
            rec.notes["tool_calls"] = response.tool_calls
        if write_memory and prep.report.ok:
            self.memory.extract_and_store(
                request.user_message, response.text, scope=prep.scope)
        self._finalize(request, trace, prep, response)
        return PipelineResult(
            answer=response.text, prompt=prep.prompt, llm=response, trace=trace,
            sources=prep.prompt.sources, low_confidence=prep.low_confidence)

    # --- async entry points ----------------------------------------------
    # The pipeline is CPU/GPU-bound (embedding, reranking), so the correct async
    # integration is to offload the sync work to a worker thread — the event
    # loop is never blocked. These are the await-able API for async hosts.
    async def arun(
        self, request: Request, write_memory: bool = True, **ephemeral_sources: Any
    ) -> PipelineResult:
        return await asyncio.to_thread(self.run, request, write_memory, **ephemeral_sources)

    async def arun_stream(
        self, request: Request, write_memory: bool = True, **ephemeral_sources: Any
    ) -> StreamResult:
        sr = await asyncio.to_thread(self.run_stream, request, write_memory, **ephemeral_sources)
        sync_gen = sr.stream

        async def agen():
            while True:  # pull each chunk off the sync generator in a thread
                chunk = await asyncio.to_thread(_next_or_done, sync_gen)
                if chunk is _STREAM_DONE:
                    break
                yield chunk

        sr.stream = agen()
        return sr

    async def arun_with_sources(
        self, request: Request, fetchers: list, write_memory: bool = True,
        **ephemeral_sources: Any,
    ) -> PipelineResult:
        """Fetch ephemeral sources concurrently, then run(). Each fetcher is an
        async callable `(request) -> list[str]`; they run together (asyncio.gather)
        so N remote fetches cost ~one round-trip, not N. Their outputs join the
        request's tool_outputs."""
        fetched: list[str] = list(ephemeral_sources.pop("tool_outputs", []))
        if fetchers:
            for result in await asyncio.gather(*(f(request) for f in fetchers)):
                fetched.extend(result)
        return await self.arun(
            request, write_memory=write_memory, tool_outputs=fetched, **ephemeral_sources)

    # --- shared prepare (stages 1-9) -------------------------------------
    def _prepare(
        self, request: Request, trace: Trace, **ephemeral_sources: Any
    ) -> _Prepared:
        scope = _memory_scope(request)
        principals = set(request.principals)

        # 1 — Collect ephemeral context (conversation, tool outputs) + memory read
        with trace.stage("1 collect") as rec:
            items = self.collector.collect(request, **ephemeral_sources)
            # keep only ephemeral sources here; durable knowledge comes from the index
            items = [it for it in items if it.source != Source.USER_MESSAGE]
            mem_items = self.memory.retrieve(request.user_message, scope=scope)
            items.extend(mem_items)
            rec.items_out = len(items)
            rec.notes["memory_hits"] = len(mem_items)

        # 2 — Retrieve: durable (index) + ephemeral (inline), tenant/ACL enforced
        with trace.stage("2 retrieve", len(items)) as rec:
            candidates = self.retriever.retrieve(
                request.user_message, items, request.metadata_filter or None,
                tenant_id=request.tenant_id, principals=principals,
            )
            bm25 = self.retriever.hybrid_scores(request.user_message, candidates)
            rec.items_out = len(candidates)
            rec.notes["backend"] = self.retriever.backend
            rec.notes["from_index"] = sum(1 for c in candidates if "chunk_id" in c.metadata)

        # 2b — Rerank (precision stage over the recall set) + abstention check
        with trace.stage("2b rerank", len(candidates)) as rec:
            reranked = self.reranker.rerank(request.user_message, candidates)
            reranked = reranked[: self.cfg.rerank_k]
            # confidence = the best RAW cross-encoder score; if nothing clears the
            # bar, flag low-confidence so the answer path can decline to guess.
            confidence = max((it.raw_rerank_score for it in reranked), default=0.0)
            low_confidence = bool(reranked) and confidence < self.cfg.abstain_below
            rec.items_out = len(reranked)
            rec.notes["backend"] = self.reranker.backend
            rec.notes["confidence"] = round(confidence, 2)
            if low_confidence:
                rec.notes["low_confidence"] = True

        # 3 — Rank (blend signals)
        with trace.stage("3 rank", len(reranked)) as rec:
            ranked = self.ranker.rank(request, reranked, bm25)
            rec.items_out = len(ranked)
            if ranked:
                rec.notes["top_score"] = round(ranked[0].score, 3)

        # 4 — Filter / dedup / contradiction
        with trace.stage("4 filter", len(ranked)) as rec:
            filtered, fstats = self.filter.apply(ranked)
            rec.items_out = len(filtered)
            rec.notes.update(dup=fstats.duplicates, conflict=fstats.contradictions)

        # 6 — Compress oversized items
        with trace.stage("6 compress", len(filtered)) as rec:
            llm_fn = self.llm if self.abstractive_compression else None
            compressed, cstats = self.compressor.compress(
                request.user_message, filtered, llm=llm_fn
            )
            rec.items_out = len(compressed)
            rec.notes["ratio"] = round(cstats.ratio, 2)

        # 7 — Budget
        plan = self.budget.plan(request, self.system_prompt)
        with trace.stage("7 budget", len(compressed)) as rec:
            kept, used, trimmed = self.budget.fit(compressed, plan)
            rec.items_out = len(kept)
            rec.notes.update(used_tok=used, avail=plan.available_for_context, trimmed=trimmed)

        # PII redaction (optional): scrub retrieved context before it reaches
        # the model / the provider.
        pii_redacted = 0
        if self.cfg.redact_pii:
            for it in kept:
                if it.source in UNTRUSTED_SOURCES:
                    it.text, counts = redact_pii(it.text)
                    pii_redacted += sum(counts.values())

        # 8 — Build
        with trace.stage("8 build", len(kept)) as rec:
            prompt = self.builder.build(request, kept, self.system_prompt)
            rec.items_out = len(prompt.included_items)

        # 9 — Validate (incl. injection scan)
        with trace.stage("9 validate") as rec:
            report = self.validator.validate(request, prompt, prompt.included_items)
            rec.notes.update(ok=report.ok, tok=report.prompt_tokens)
            if report.injection_flags:
                rec.notes["injection"] = len(report.injection_flags)
            if not report.ok:
                rec.notes["errors"] = "; ".join(report.errors)

        return _Prepared(
            prompt=prompt, report=report, low_confidence=low_confidence,
            confidence=confidence, scope=scope, pii_redacted=pii_redacted)

    # --- shared finalize (stage 11: metrics, cost, audit, log) -----------
    def _finalize(
        self, request: Request, trace: Trace, prep: _Prepared, response: LLMResponse
    ) -> None:
        cost = estimate_cost(
            response.model or request.model,
            response.input_tokens, response.output_tokens,
            response.cache_read_tokens, response.cache_write_tokens,
        ) if response.backend in ("anthropic", "openai") else 0.0
        trace.metrics.update(
            request_id=trace.request_id,
            embed_backend=self.embedder.backend,
            index_backend=self.store.backend,
            rerank_backend=self.reranker.backend,
            llm_backend=response.backend,
            indexed_chunks=self.store.stats()["chunks"],
            cache_hit_rate=round(self.cache.hit_rate, 2),
            cache_read_tokens=response.cache_read_tokens,
            prompt_tokens=prep.report.prompt_tokens,
            output_tokens=response.output_tokens,
            cost_usd=cost,
            context_items_final=len(prep.prompt.included_items),
            injection_flags=len(prep.report.injection_flags),
            pii_redacted=prep.pii_redacted,
            retrieval_confidence=round(prep.confidence, 2),
            low_confidence=prep.low_confidence,
        )
        if self.audit is not None:
            self.audit.record({
                "request_id": trace.request_id,
                "tenant_id": request.tenant_id,
                "principals": request.principals,
                "query": request.user_message,
                "retrieved_chunks": [s.get("chunk_id") for s in prep.prompt.sources],
                "low_confidence": prep.low_confidence,
                "backend": response.backend,
                "cost_usd": cost,
            })
        if self.cfg.log_requests:
            print(trace.log_line())
