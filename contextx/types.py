"""Core data model shared by every stage of the pipeline.

Everything that moves through the engine is a `ContextItem`. Each stage reads
some fields and writes others, so a single item accumulates provenance and
scores as it travels Collect -> Retrieve -> Rank -> Filter -> ... -> Build.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Source(str, Enum):
    """Where a candidate piece of context came from (stage 1 records this)."""

    USER_MESSAGE = "user_message"
    CURRENT_CONVERSATION = "current_conversation"
    PREVIOUS_CONVERSATION = "previous_conversation"
    LONG_TERM_MEMORY = "long_term_memory"
    KNOWLEDGE_BASE = "knowledge_base"
    DOCUMENT = "document"
    DATABASE = "database"
    API_RESPONSE = "api_response"
    TOOL_OUTPUT = "tool_output"
    WEB_SEARCH = "web_search"
    CACHE = "cache"
    SYSTEM = "system"


# Rough ordering used when we need a tiebreak or a default importance prior.
# Higher = more inherently trustworthy/relevant, all else equal.
SOURCE_PRIOR: dict[Source, float] = {
    Source.SYSTEM: 1.0,
    Source.USER_MESSAGE: 1.0,
    Source.CURRENT_CONVERSATION: 0.9,
    Source.LONG_TERM_MEMORY: 0.8,
    Source.PREVIOUS_CONVERSATION: 0.7,
    Source.KNOWLEDGE_BASE: 0.65,
    Source.DOCUMENT: 0.6,
    Source.DATABASE: 0.6,
    Source.API_RESPONSE: 0.5,
    Source.TOOL_OUTPUT: 0.5,
    Source.WEB_SEARCH: 0.45,
    Source.CACHE: 0.4,
}


@dataclass
class ContextItem:
    """One normalized candidate piece of context.

    Fields fall into three groups:
      * set at collection: text, source, timestamp, metadata
      * filled by retrieval/ranking: embedding, similarity, score, importance
      * filled by budgeting/build: tokens, included
    """

    text: str
    source: Source
    timestamp: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)

    # populated as the item flows through the pipeline
    embedding: Any = None          # np.ndarray | None
    similarity: float = 0.0        # cosine to the query (stage 2)
    rerank_score: float = 0.0      # cross-encoder relevance, normalized 0..1 (stage 2b)
    raw_rerank_score: float = 0.0  # cross-encoder logit, un-normalized (for abstention)
    rrf_score: float = 0.0         # reciprocal-rank-fusion score (hybrid recall)
    score: float = 0.0             # composite rank score (stage 3)
    importance: float = 0.0        # pinned / critical weight (stage 3/5)
    tokens: int = 0                # counted in stage 7
    included: bool = True          # flipped off if trimmed by the budget

    # untrusted content (retrieved docs, web, tool output) can carry injection;
    # the builder delimits it and the validator scans it.
    trusted: bool = True

    item_id: str = ""

    def __post_init__(self) -> None:
        if not self.item_id:
            digest = hashlib.sha1(
                f"{self.source.value}:{self.text}".encode("utf-8")
            ).hexdigest()
            self.item_id = digest[:12]

    @property
    def age_seconds(self) -> float:
        return max(0.0, time.time() - self.timestamp)

    def preview(self, n: int = 60) -> str:
        flat = " ".join(self.text.split())
        return flat if len(flat) <= n else flat[: n - 3] + "..."


# Sources whose content is externally authored and therefore untrusted: it may
# contain prompt-injection payloads and must be delimited + scanned.
UNTRUSTED_SOURCES = {
    Source.KNOWLEDGE_BASE,
    Source.DOCUMENT,
    Source.WEB_SEARCH,
    Source.API_RESPONSE,
    Source.TOOL_OUTPUT,
    Source.PREVIOUS_CONVERSATION,
}


@dataclass
class Document:
    """A durable source document to be ingested into the vector store.

    `tenant_id` isolates namespaces; `acl` is a list of principals (users/groups)
    allowed to read it — empty means readable by anyone in the tenant.
    """

    text: str
    doc_id: str = ""
    source: Source = Source.KNOWLEDGE_BASE
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    tenant_id: str = "default"
    acl: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.doc_id:
            self.doc_id = hashlib.sha1(self.text.encode("utf-8")).hexdigest()[:16]


@dataclass
class Request:
    """The inbound request the pipeline is optimizing context for."""

    user_message: str
    conversation: list[str] = field(default_factory=list)  # prior turns, oldest first
    preferences: dict[str, Any] = field(default_factory=dict)
    model: str = "claude-sonnet-5"
    max_context_tokens: int = 32_000
    reserve_output_tokens: int = 4_000
    metadata_filter: dict[str, Any] = field(default_factory=dict)  # index prefilter
    tenant_id: str = "default"                       # namespace isolation
    principals: list[str] = field(default_factory=list)  # requester's user + groups
