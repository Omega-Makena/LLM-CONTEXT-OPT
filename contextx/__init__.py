"""contextx — a context-optimization engine.

A staged pipeline that turns a raw user request + scattered sources into one
tight, validated, in-budget prompt for an LLM, with an ingest/query split so
retrieval cost is independent of corpus size:

    ingest:  documents -> chunk -> embed -> persistent ANN index
    query:   Collect -> Retrieve -> Rerank -> Rank -> Filter -> Compress
             -> Budget -> Build -> Validate -> LLM   (Memory + Cache wrap it)

Quick start:

    from contextx import ContextEngine, Request, Document

    engine = ContextEngine()
    engine.ingest([Document(text="A refresh token is a long-lived credential ...")])
    result = engine.run(Request(user_message="Explain JWT refresh tokens"))
    print(result.answer)
    print(result.trace.report())
"""

from .config import Config
from .llm import Tool
from .pipeline import ContextEngine, PipelineResult, StreamResult
from .types import ContextItem, Document, Request, Source

__all__ = [
    "ContextEngine",
    "PipelineResult",
    "StreamResult",
    "Config",
    "Request",
    "Document",
    "ContextItem",
    "Source",
    "Tool",
]
__version__ = "0.3.0"
