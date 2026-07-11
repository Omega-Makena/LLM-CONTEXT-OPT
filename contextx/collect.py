"""Stage 1 — Context Collection (ephemeral sources).

Gather the per-request, non-durable context into a `ContextItem` pool: the
user message, this turn's conversation, freshly-fetched tool/API outputs, and
prior-conversation snippets. Durable knowledge (KB, documents) is NOT collected
here — it lives in the pre-built vector index and is pulled at retrieval time.
That separation is the ingest/query split.

Real deployments wire the source args to live fetchers (async, with timeouts
and partial-failure degradation); here they accept already-fetched data so the
stage stays pure and testable.
"""

from __future__ import annotations

import time

from .types import ContextItem, Request, Source, UNTRUSTED_SOURCES


class Collector:
    def collect(
        self,
        request: Request,
        previous_conversations: list[str] | None = None,
        tool_outputs: list[str] | None = None,
        api_responses: list[str] | None = None,
    ) -> list[ContextItem]:
        items: list[ContextItem] = []

        items.append(ContextItem(text=request.user_message, source=Source.USER_MESSAGE))

        for i, turn in enumerate(request.conversation):
            items.append(
                ContextItem(
                    text=turn,
                    source=Source.CURRENT_CONVERSATION,
                    metadata={"turn": i},
                    timestamp=time.time() - (len(request.conversation) - i),
                )
            )

        for turn in previous_conversations or []:
            items.append(
                ContextItem(
                    text=turn,
                    source=Source.PREVIOUS_CONVERSATION,
                    trusted=Source.PREVIOUS_CONVERSATION not in UNTRUSTED_SOURCES,
                )
            )

        for out in tool_outputs or []:
            items.append(ContextItem(text=out, source=Source.TOOL_OUTPUT, trusted=False))

        for out in api_responses or []:
            items.append(ContextItem(text=out, source=Source.API_RESPONSE, trusted=False))

        return items
