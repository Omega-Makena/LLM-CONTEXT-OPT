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

    def delete(self, doc_id: str) -> int:
        """Remove a document and its chunks from the index. Returns #chunks."""
        n = self.store.delete_document(doc_id)
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

        # 10 — LLM (semantic response cache wraps the call)
        with trace.stage("10 llm") as rec:
            if not report.ok:
                response = LLMResponse(
                    text="[BLOCKED] validation failed: " + "; ".join(report.errors),
                    backend="none",
                )
            else:
                qvec = self.embedder.encode_one(prompt.user)
                response = self.cache.semantic_get_or_compute(
                    "llm",
                    qvec,
                    key=prompt.user[:256],
                    compute=lambda: self.llm.complete(prompt.system_blocks, prompt.user),
                )
            rec.notes["backend"] = response.backend
            if response.retries:
                rec.notes["retries"] = response.retries

        # 5 — Memory write (scoped to the requester)
        if write_memory and report.ok:
            with trace.stage("5 memory-write") as rec:
                new = self.memory.extract_and_store(
                    request.user_message, response.text, scope=scope)
                rec.notes["stored"] = len(new)

        # 11 — Metrics (incl. estimated dollar cost; local/mock backends are free)
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
            prompt_tokens=report.prompt_tokens,
            output_tokens=response.output_tokens,
            cost_usd=cost,
            context_items_final=len(prompt.included_items),
            injection_flags=len(report.injection_flags),
            pii_redacted=pii_redacted,
            retrieval_confidence=round(confidence, 2),
            low_confidence=low_confidence,
        )

        # audit trail + structured log (both optional)
        if self.audit is not None:
            self.audit.record({
                "request_id": trace.request_id,
                "tenant_id": request.tenant_id,
                "principals": request.principals,
                "query": request.user_message,
                "retrieved_chunks": [s.get("chunk_id") for s in prompt.sources],
                "low_confidence": low_confidence,
                "backend": response.backend,
                "cost_usd": cost,
            })
        if self.cfg.log_requests:
            print(trace.log_line())

        return PipelineResult(
            answer=response.text,
            prompt=prompt,
            llm=response,
            trace=trace,
            sources=prompt.sources,
            low_confidence=low_confidence,
        )
