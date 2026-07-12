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
from .observability import Trace
from .rank import Ranker
from .rerank import Reranker
from .retrieve import Retriever
from .store import VectorStore
from .types import Document, Request, Source
from .validate import Validator

DEFAULT_SYSTEM = (
    "You are a helpful assistant. Use the provided context blocks to answer the "
    "user's request accurately. If the context is insufficient, say so."
)


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
        self.embedder = embedder or Embedder(self.cfg.embed_model)
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

    # --- ingest (amortized) ----------------------------------------------
    def ingest(self, documents: list[Document]) -> int:
        """Chunk + embed + persist durable documents. Returns #chunks added."""
        return self.store.add_documents(documents)

    def update(self, documents: list[Document]) -> int:
        """Replace documents by doc_id (delete old chunks, re-ingest). Keeps the
        index from going stale when source docs change."""
        return self.store.update_documents(documents)

    def delete(self, doc_id: str) -> int:
        """Remove a document and its chunks from the index. Returns #chunks."""
        return self.store.delete_document(doc_id)

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

        # 1 — Collect ephemeral context (conversation, tool outputs) + memory read
        with trace.stage("1 collect") as rec:
            items = self.collector.collect(request, **ephemeral_sources)
            # keep only ephemeral sources here; durable knowledge comes from the index
            items = [it for it in items if it.source != Source.USER_MESSAGE]
            mem_items = self.memory.retrieve(request.user_message)
            items.extend(mem_items)
            rec.items_out = len(items)
            rec.notes["memory_hits"] = len(mem_items)

        # 2 — Retrieve: durable (index) + ephemeral (inline)
        with trace.stage("2 retrieve", len(items)) as rec:
            candidates = self.retriever.retrieve(
                request.user_message, items, request.metadata_filter or None
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

        # 5 — Memory write
        if write_memory and report.ok:
            with trace.stage("5 memory-write") as rec:
                new = self.memory.extract_and_store(request.user_message, response.text)
                rec.notes["stored"] = len(new)

        # 11 — Metrics
        trace.metrics.update(
            embed_backend=self.embedder.backend,
            index_backend=self.store.backend,
            rerank_backend=self.reranker.backend,
            llm_backend=response.backend,
            indexed_chunks=self.store.stats()["chunks"],
            cache_hit_rate=round(self.cache.hit_rate, 2),
            cache_read_tokens=response.cache_read_tokens,
            prompt_tokens=report.prompt_tokens,
            context_items_final=len(prompt.included_items),
            injection_flags=len(report.injection_flags),
            retrieval_confidence=round(confidence, 2),
            low_confidence=low_confidence,
        )

        return PipelineResult(
            answer=response.text,
            prompt=prompt,
            llm=response,
            trace=trace,
            sources=prompt.sources,
            low_confidence=low_confidence,
        )
