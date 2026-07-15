"""Central configuration — every knob and magic number lives here.

One of the biggest tells of a toy is thresholds scattered as literals across the
code. This gathers them so they can be tuned, versioned, and (eventually)
learned. Construct once and thread through the engine.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Config:
    # --- models -----------------------------------------------------------
    embed_model: str = "all-MiniLM-L6-v2"
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    llm_model: str = "claude-sonnet-5"

    # --- ingest / chunking ------------------------------------------------
    chunk_target_tokens: int = 320
    chunk_overlap_tokens: int = 48
    embed_batch_size: int = 64

    # --- persistent index -------------------------------------------------
    index_dir: str = ".contextx_index"
    # vector backend: "auto" (faiss if installed, else numpy), "faiss", "numpy",
    # or "pgvector" (experimental — needs Postgres + psycopg).
    vector_backend: str = "auto"
    pg_dsn: str = "postgresql://localhost/contextx"
    pg_table: str = "contextx_vectors"
    hnsw_M: int = 32                # graph degree; higher = better recall, more RAM
    hnsw_ef_construction: int = 200
    hnsw_ef_search: int = 64        # query-time recall/latency tradeoff

    # --- retrieval --------------------------------------------------------
    recall_k: int = 40              # candidates pulled from the index (recall stage)
    rerank_k: int = 12              # kept after cross-encoder rerank (precision stage)
    ephemeral_k: int = 8            # top inline (conversation/tool/memory) items kept
    min_similarity: float = 0.15
    enable_hybrid: bool = True      # fuse vector + BM25 lexical recall (RRF)
    rrf_k: int = 60                 # reciprocal-rank-fusion constant
    # abstention: if the top RAW cross-encoder score is below this, the retrieval
    # is flagged low-confidence (the answer path can decline instead of guessing).
    abstain_below: float = -3.0

    # --- ranking weights (linear blend; sum need not be 1) ---------------
    w_rerank: float = 0.55
    w_similarity: float = 0.15
    w_bm25: float = 0.10
    w_recency: float = 0.08
    w_importance: float = 0.07
    w_conversation: float = 0.03
    w_preference: float = 0.02
    recency_half_life_s: float = 7 * 24 * 3600.0

    # --- filtering --------------------------------------------------------
    dup_threshold: float = 0.90

    # --- compression ------------------------------------------------------
    max_item_tokens: int = 220
    compress_target_ratio: float = 0.5

    # --- budget -----------------------------------------------------------
    max_context_tokens: int = 32_000
    reserve_output_tokens: int = 4_000
    budget_safety_margin: float = 0.10   # keep 10% headroom for tokenizer drift

    # --- memory -----------------------------------------------------------
    memory_db_path: str = ".contextx_memory.db"
    memory_max_records: int = 5_000      # bounded; lowest value*recency evicted
    memory_read_k: int = 5
    llm_memory_extraction: bool = False  # extract facts with the LLM (costs a call/turn)

    # --- cache ------------------------------------------------------------
    cache_max_entries: int = 10_000
    cache_ttl_s: float = 3600.0
    semantic_cache_threshold: float = 0.97  # response cache near-dup match

    # --- llm --------------------------------------------------------------
    # provider: "auto" picks anthropic (ANTHROPIC_API_KEY) -> openai
    # (OPENAI_API_KEY) -> ollama (local server reachable) -> mock. Force one with
    # "anthropic" / "openai" / "ollama" / "mock".
    llm_provider: str = "auto"
    ollama_host: str = "http://localhost:11434"
    ollama_model: str = "llama3.1"     # falls back to the first pulled model
    # OpenAI-compatible: works for OpenAI, Azure OpenAI, OpenRouter, vLLM, LM
    # Studio, etc. by changing the base URL. Key read from OPENAI_API_KEY.
    openai_base_url: str = "https://api.openai.com/v1"
    openai_model: str = "gpt-4o-mini"
    llm_max_tokens: int = 1024
    llm_timeout_s: float = 60.0
    llm_max_retries: int = 4
    llm_backoff_base_s: float = 0.5
    enable_prompt_caching: bool = True

    # --- security & privacy ----------------------------------------------
    injection_scan: bool = True
    redact_pii: bool = False              # scrub PII from retrieved context
    audit_log_path: str | None = None     # JSONL provenance trail (None = off)

    # --- observability ----------------------------------------------------
    log_requests: bool = False            # emit a structured JSON line per request
