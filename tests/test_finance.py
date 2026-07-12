"""Tests for the finance domain pack."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from contextx import Config, ContextEngine, Request  # noqa: E402
from contextx.domains.finance import (  # noqa: E402
    FINANCE_SYSTEM_PROMPT, extract_entities, finance_config, redact_financial,
)
from contextx.domains.finance_data import FINANCE_CORPUS, FINANCE_QUERIES  # noqa: E402


def test_finance_config_preset():
    cfg = finance_config()
    assert cfg.redact_pii is True
    assert cfg.w_recency > Config().w_recency        # freshness weighted higher
    assert cfg.enable_hybrid is True


def test_finance_entity_extraction():
    ents = extract_entities("Buy 100 AAPL and MSFT; ISIN US0378331005")
    assert "AAPL" in ents.get("tickers", [])
    assert "US0378331005" in ents.get("isins", [])


def test_redact_financial_identifiers():
    text = "IBAN GB29NWBK60161331926819 SWIFT NWBKGB2L routing 021000021"
    red, counts = redact_financial(text)
    assert "GB29NWBK60161331926819" not in red
    assert "IBAN" in counts


def test_finance_eval_labels_are_valid():
    corpus_ids = {d.doc_id for d in FINANCE_CORPUS}
    for ex in FINANCE_QUERIES:
        assert ex.relevant, f"empty label for: {ex.query}"
        for did in ex.relevant:
            assert did in corpus_ids, f"label {did} not in corpus"


def test_finance_retrieval_disambiguates_cusip_vs_isin(tmp_path):
    cfg = finance_config(index_dir=str(tmp_path / "idx"),
                         memory_db_path=str(tmp_path / "m.db"),
                         max_context_tokens=3000, reserve_output_tokens=800)
    engine = ContextEngine(config=cfg, system_prompt=FINANCE_SYSTEM_PROMPT)
    engine.ingest(FINANCE_CORPUS)
    # the 9-character NA identifier is CUSIP, not ISIN
    hits = engine.retriever.retrieve(
        "what nine-character code identifies a North American security?", [], None)
    from contextx.rerank import Reranker
    ranked = Reranker(cfg).rerank(
        "what nine-character code identifies a North American security?", hits)
    top_ids = [h.metadata.get("doc_id") for h in ranked[:3]]
    assert "cusip" in top_ids
