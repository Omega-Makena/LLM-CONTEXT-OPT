"""Tests for API authentication + server-side tenant enforcement."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest  # noqa: E402

from contextx import Config, ContextEngine  # noqa: E402
from contextx.auth import APIKeyAuth  # noqa: E402

ACME = {"Authorization": "Bearer sk-acme"}
GLOBEX = {"Authorization": "Bearer sk-globex"}


def test_apikeyauth_unit():
    a = APIKeyAuth({"sk1": {"tenant_id": "t", "principals": ["execs"]}})
    ctx = a.authenticate("sk1")
    assert ctx.tenant_id == "t" and ctx.principals == ["execs"]
    assert a.authenticate("wrong") is None
    assert a.authenticate(None) is None
    assert a.configured
    # keys are stored hashed, not in plaintext
    assert "sk1" not in repr(a.__dict__)


@pytest.fixture(scope="module")
def client():
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    from contextx.service import create_app
    d = tempfile.mkdtemp(prefix="contextx_auth_")
    cfg = Config(index_dir=d + "/idx", memory_db_path=d + "/m.db")
    auth = APIKeyAuth({
        "sk-acme": {"tenant_id": "acme", "principals": ["execs"]},
        "sk-globex": {"tenant_id": "globex"},
    })
    return TestClient(create_app(ContextEngine(config=cfg), auth=auth))


def test_missing_key_is_rejected(client):
    assert client.post("/query", json={"user_message": "hi"}).status_code == 401
    assert client.post("/ingest", json={"documents": [{"text": "x"}]}).status_code == 401
    assert client.get("/health").status_code == 200  # health stays open


def test_ingest_binds_to_authenticated_tenant(client):
    r = client.post("/ingest", json={"documents": [
        {"text": "Acme revenue was 5 million last year.", "doc_id": "a"}]}, headers=ACME)
    assert r.status_code == 200 and r.json()["tenant_id"] == "acme"


def test_query_isolated_by_authenticated_tenant(client):
    # globex cannot see acme's document, even asking the same question
    g = client.post("/query", json={"user_message": "what was the revenue?"},
                    headers=GLOBEX).json()
    assert g["tenant_id"] == "globex"
    assert all(s.get("doc_id") != "a" for s in g["sources"])
    # acme sees its own
    a = client.post("/query", json={"user_message": "what was the revenue?"},
                    headers=ACME).json()
    assert any(s.get("doc_id") == "a" for s in a["sources"])


def test_delete_scoped_to_tenant(client):
    client.post("/ingest", json={"documents": [
        {"text": "confidential acme memo", "doc_id": "sec"}]}, headers=ACME)
    # globex cannot delete acme's document by id
    assert client.delete("/documents/sec", headers=GLOBEX).status_code == 404
    # acme can
    assert client.delete("/documents/sec", headers=ACME).status_code == 200


def test_body_cannot_override_tenant(client):
    # even if a caller stuffs tenant_id in the body, the schema ignores it and
    # the authenticated identity wins (globex can't write into acme)
    r = client.post("/ingest", json={"documents": [
        {"text": "sneaky", "doc_id": "sneak"}], "tenant_id": "acme"}, headers=GLOBEX)
    assert r.json()["tenant_id"] == "globex"
