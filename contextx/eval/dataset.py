"""Labeled evaluation dataset — the golden set.

A retrieval system you can't measure is a toy. This is a small but real
benchmark: a corpus with stable doc_ids and a set of queries each annotated
with the doc_ids that are *relevant* (ground truth). Metrics are computed by
comparing what retrieval returns against these labels.

The corpus mixes closely-related docs, paraphrases, and irrelevant distractors
so the metrics have to discriminate — precision is punished by returning
distractors, recall by missing paraphrases.

Bring your own set via `load_jsonl` (one {"query", "relevant"} object per line)
to evaluate on your real domain data.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from ..types import Document, Source


@dataclass
class EvalExample:
    query: str
    relevant: list[str]  # doc_ids that should be retrieved
    note: str = ""


# --- corpus (explicit, stable doc_ids) ------------------------------------
def _d(doc_id: str, text: str, source: Source = Source.KNOWLEDGE_BASE) -> Document:
    return Document(text=text, doc_id=doc_id, source=source)


GOLDEN_CORPUS: list[Document] = [
    _d("jwt_refresh", "A JWT refresh token is a long-lived credential used to obtain "
                      "new short-lived access tokens without re-authenticating."),
    _d("jwt_storage", "Refresh tokens should be stored in an httpOnly, Secure cookie, "
                      "rotated on each use, and be revocable server-side."),
    _d("jwt_access", "Access tokens are short-lived and carry the user's claims; they "
                     "are sent on every API request in the Authorization header."),
    _d("jwt_rotation", "Token rotation: when a refresh token is used, issue a new one "
                       "and invalidate the old; reuse of an invalidated token signals "
                       "theft and should revoke the whole token family."),
    _d("oauth_grants", "OAuth2 defines grant types including authorization_code for web "
                       "apps and client_credentials for machine-to-machine access."),
    _d("session_cookies", "Server-side sessions store a session id in a cookie and keep "
                          "session state in a store such as Redis."),
    _d("pg_intro", "PostgreSQL is an open-source relational database with strong SQL "
                   "compliance, MVCC concurrency, and rich indexing."),
    _d("pg_index", "PostgreSQL supports B-tree, GIN, and GiST indexes; GIN is used for "
                   "full-text search and JSONB containment queries."),
    _d("pg_vector", "The pgvector extension adds vector similarity search to PostgreSQL, "
                    "enabling nearest-neighbour queries over embeddings."),
    _d("mysql_intro", "MySQL is a widely used relational database; InnoDB is its default "
                      "transactional storage engine with row-level locking."),
    _d("fastapi_intro", "FastAPI is a Python web framework built on Starlette and Pydantic "
                        "with async support and automatic OpenAPI documentation."),
    _d("fastapi_auth", "In FastAPI, OAuth2PasswordBearer and dependency injection are used "
                       "to protect routes with bearer tokens."),
    _d("pydantic", "Pydantic validates and parses data using Python type hints, powering "
                   "FastAPI request and response models."),
    _d("docker_intro", "Docker packages an application and its dependencies into a "
                       "container image for reproducible deployment."),
    _d("k8s", "Kubernetes orchestrates containers across a cluster, handling scheduling, "
              "autoscaling, and self-healing."),
    _d("distractor_coffee", "The office coffee machine is a Jura E8 and needs monthly "
                            "descaling.", Source.DOCUMENT),
    _d("distractor_weather", "The weather forecast predicts rain on Tuesday with highs "
                             "near 15 degrees.", Source.DOCUMENT),
    _d("distractor_parking", "Visitor parking is on level 2; register the license plate "
                             "at reception.", Source.DOCUMENT),
    # --- hard negatives: semantically adjacent, lexically overlapping, but NOT
    # answers to any golden query. These create real selection pressure so the
    # embedding and rerank stages have to discriminate, not just separate topics.
    _d("jwt_signing", "A JWT is signed (JWS) with HMAC or RSA so its claims can be "
                      "verified; JWE additionally encrypts the payload."),
    _d("csrf_token", "A CSRF token is a per-form random value that prevents cross-site "
                     "request forgery; it is unrelated to bearer authentication."),
    _d("api_keys", "API keys are static secrets identifying a client application; unlike "
                   "tokens they typically do not expire or carry user claims."),
    _d("saml_sso", "SAML is an XML-based single sign-on standard used by enterprises, an "
                   "alternative to OAuth2/OIDC token flows."),
    _d("password_hash", "Passwords should be stored as salted bcrypt or Argon2 hashes, "
                        "never in plaintext or reversible encryption."),
    _d("mongodb", "MongoDB is a document database storing BSON; it scales horizontally "
                  "with sharding and has no fixed schema."),
    _d("redis", "Redis is an in-memory key-value store used for caching, rate limiting, "
                "and pub/sub messaging."),
    _d("sqlite", "SQLite is a serverless, file-based relational database embedded "
                 "directly into the application process."),
    _d("db_sharding", "Sharding partitions a database horizontally across nodes by a "
                      "shard key to scale writes beyond a single machine."),
    _d("flask", "Flask is a minimal synchronous Python web framework based on Werkzeug "
                "and Jinja2, without built-in async or data validation."),
    _d("django", "Django is a batteries-included Python web framework with an ORM, admin "
                 "site, and templating."),
    _d("elasticsearch", "Elasticsearch is a distributed search engine built on Lucene, "
                        "using BM25 and inverted indexes for full-text search."),
    _d("grpc", "gRPC is a high-performance RPC framework using HTTP/2 and protocol "
               "buffers for service-to-service communication."),
    _d("terraform", "Terraform provisions cloud infrastructure declaratively; it manages "
                    "state but does not build or run container images."),
    # --- more clusters, each with its own decoys ------------------------------
    _d("memcached", "Memcached is a bare in-memory cache for strings and objects; unlike "
                    "Redis it has no persistence and no data structures."),
    _d("rabbitmq", "RabbitMQ is a message broker implementing AMQP for reliable, "
                   "queue-based delivery of tasks between services."),
    _d("kafka", "Apache Kafka is a distributed append-only log for high-throughput event "
                "streaming, letting consumers replay past events."),
    _d("cdn", "A CDN caches static assets at edge locations close to users to reduce "
              "latency; it does not run application logic."),
    _d("prometheus", "Prometheus scrapes and stores time-series metrics and evaluates "
                     "alerting rules against them."),
    _d("opentelemetry", "OpenTelemetry is a vendor-neutral API and SDK for distributed "
                        "tracing, metrics, and logs across services."),
    _d("logging_struct", "Structured logging emits JSON key-value fields per event so logs "
                         "are queryable, unlike plain-text log lines."),
    _d("tls", "TLS encrypts data in transit and authenticates the server to the client via "
              "a certificate during the handshake."),
    _d("mtls", "Mutual TLS authenticates both the client and the server with certificates, "
               "commonly used for service-to-service auth."),
    _d("cors", "CORS is a browser mechanism that lets a server permit cross-origin requests "
               "through response headers."),
    _d("rate_limit", "Rate limiting caps how many requests a client may make in a time "
                     "window, often using a token-bucket algorithm."),
    _d("waf", "A web application firewall inspects and filters malicious HTTP traffic such "
              "as SQL injection and cross-site scripting."),
    _d("oidc", "OpenID Connect is an identity layer on top of OAuth2 that adds an ID token "
               "describing the authenticated user."),
    _d("db_replication", "Read replicas copy a primary database to serve read traffic and "
                         "provide failover if the primary dies."),
    _d("db_transaction", "ACID transactions guarantee atomicity, consistency, isolation, "
                         "and durability for a group of writes."),
    _d("orm", "An object-relational mapper maps database rows to objects; SQLAlchemy is the "
              "common Python ORM."),
    _d("celery", "Celery runs background tasks in Python worker processes, backed by a "
                 "broker such as RabbitMQ or Redis."),
    _d("asgi", "ASGI is the asynchronous server interface Python web frameworks use; "
               "Uvicorn is a common ASGI server."),
    _d("helm", "Helm packages Kubernetes manifests into versioned, parameterized charts for "
               "repeatable deploys."),
    _d("github_actions", "GitHub Actions runs CI/CD workflows triggered by repository "
                         "events such as pushes and pull requests."),
]


GOLDEN_QUERIES: list[EvalExample] = [
    EvalExample("how do JWT refresh tokens work?", ["jwt_refresh", "jwt_rotation", "jwt_access"]),
    EvalExample("where should I store refresh tokens safely?", ["jwt_storage", "jwt_rotation"]),
    EvalExample("difference between access and refresh tokens", ["jwt_access", "jwt_refresh"]),
    EvalExample("OAuth2 machine to machine authentication", ["oauth_grants"]),
    EvalExample("protect FastAPI routes with bearer tokens", ["fastapi_auth", "jwt_access"]),
    EvalExample("which postgres index for JSONB and full text search?", ["pg_index"]),
    EvalExample("vector similarity search in postgres", ["pg_vector"]),
    EvalExample("what storage engine does MySQL use for transactions?", ["mysql_intro"]),
    EvalExample("what is FastAPI built on?", ["fastapi_intro", "pydantic"]),
    EvalExample("how does pydantic validate data?", ["pydantic"]),
    EvalExample("deploy an application in a container", ["docker_intro"]),
    EvalExample("how does kubernetes scale containers?", ["k8s"]),
]


# --- HARD set: lexical traps + multi-hop over the SAME corpus --------------
# Each query has a decoy doc with HIGH surface/embedding similarity that is
# wrong, while the true answer uses different vocabulary or must be inferred.
# Designed to push bi-encoder recall@k below 1.0 so the cross-encoder reranker
# has room to demonstrate (or fail to demonstrate) value. `note` names the trap.
def _q(query: str, relevant: list[str], note: str = "") -> EvalExample:
    return EvalExample(query, relevant, note)


HARD_QUERIES: list[EvalExample] = [
    # --- auth / tokens (negation, vocab-mismatch, mechanism indirection) ------
    _q("keep users signed in for weeks without making them re-enter their password",
       ["jwt_refresh"], "vocab-mismatch; decoys password_hash, session_cookies"),
    _q("issue new short-lived credentials without sending the user back to the login form",
       ["jwt_refresh"], "decoy jwt_access"),
    _q("the short-lived credential sent on every API call that carries who the user is",
       ["jwt_access"], "decoy jwt_refresh"),
    _q("stop a leaked long-lived token from being replayed forever",
       ["jwt_rotation"], "decoys jwt_storage, jwt_signing"),
    _q("detect that a refresh token was stolen and revoke the whole family",
       ["jwt_rotation"], "decoy jwt_refresh"),
    _q("where to keep a refresh token in the browser so JavaScript cannot read it",
       ["jwt_storage"], "decoy session_cookies; localStorage myth"),
    _q("verify a token's claims are authentic without decrypting a hidden payload",
       ["jwt_signing"], "JWS vs JWE; decoy jwt_access"),
    _q("which JOSE object signs the payload but does not encrypt it",
       ["jwt_signing"], "decoy jwt_storage"),
    _q("a static secret identifying an application, not a user, that never expires",
       ["api_keys"], "negation; decoy jwt_access"),
    _q("credential for a cron job to call our API carrying no user identity",
       ["api_keys"], "decoys oauth_grants, jwt_access"),
    _q("random per-form value that blocks forged requests from another website",
       ["csrf_token"], "decoy: bearer-token docs"),
    _q("protect a form POST against cross-site request forgery",
       ["csrf_token"], "decoy cors"),
    _q("grant type for machine-to-machine access with no user present",
       ["oauth_grants"], "decoy api_keys"),
    _q("the flow a web app uses to exchange an authorization code for tokens",
       ["oauth_grants"], "decoy oidc"),
    _q("add a verifiable ID token describing the user on top of OAuth2",
       ["oidc"], "decoy oauth_grants"),
    _q("standard that returns the user's identity, not just access, layered on OAuth",
       ["oidc"], "decoy saml_sso"),
    _q("enterprise XML single sign-on standard, not an OAuth token flow",
       ["saml_sso"], "negation; decoy oauth_grants"),
    _q("legacy XML-based single sign-on used by large enterprises",
       ["saml_sso"], "decoy oidc"),
    _q("keep server-side login state with an id in a cookie and state in a store",
       ["session_cookies"], "decoy jwt_refresh"),
    _q("the correct, irreversible way to persist user login passwords",
       ["password_hash"], "decoy jwt_storage"),
    _q("store credentials so a database leak cannot reveal them",
       ["password_hash"], "decoy tls/mtls encryption"),
    _q("authenticate both client and server with certificates between microservices",
       ["mtls"], "decoy tls"),
    _q("encrypt traffic in transit and authenticate the server with a certificate",
       ["tls"], "decoy mtls"),
    _q("let a browser call our API from a different origin via response headers",
       ["cors"], "decoy csrf_token"),
    _q("filter malicious HTTP like SQL injection and XSS before it reaches the app",
       ["waf"], "decoy rate_limit"),
    _q("cap how many requests a client can make per minute with a token bucket",
       ["rate_limit"], "decoy redis (also mentions rate limiting)"),
    _q("throttle abusive clients to a fixed number of requests per window",
       ["rate_limit"], "decoy waf"),
    # --- databases (lexical traps across engines) -----------------------------
    _q("speed up slow full-text search inside a relational database",
       ["pg_index"], "decoy elasticsearch ('full-text search')"),
    _q("which postgres index handles JSONB containment and full text",
       ["pg_index"], "decoy pg_vector"),
    _q("nearest-neighbour search over embeddings inside postgres",
       ["pg_vector"], "decoy elasticsearch"),
    _q("store and query vector embeddings in my SQL database",
       ["pg_vector"], "decoy mongodb"),
    _q("distributed engine using BM25 and inverted indexes for search",
       ["elasticsearch"], "decoy pg_index"),
    _q("which storage engine gives MySQL row-level locking and transactions",
       ["mysql_intro"], "decoy pg_intro"),
    _q("schemaless document store that scales horizontally by sharding",
       ["mongodb"], "decoy sqlite"),
    _q("serverless file-based SQL database embedded in the application process",
       ["sqlite"], "decoy mysql_intro"),
    _q("open-source relational database with MVCC and strong SQL compliance",
       ["pg_intro"], "decoy mysql_intro"),
    _q("serve read traffic from copies of the primary and fail over if it dies",
       ["db_replication"], "decoy db_sharding"),
    _q("partition writes across nodes by a key to scale beyond one machine",
       ["db_sharding"], "decoy db_replication"),
    _q("guarantee a group of writes is atomic, consistent, isolated, and durable",
       ["db_transaction"], "decoy db_replication"),
    _q("the Python library that maps database rows to objects",
       ["orm"], "decoy pydantic (also maps data to objects)"),
    # --- caching / messaging --------------------------------------------------
    _q("share ephemeral counters across servers in memory for throttling",
       ["redis"], "decoy memcached"),
    _q("in-memory store with data structures and persistence for caching and pub/sub",
       ["redis"], "decoy memcached"),
    _q("a bare in-memory cache with no persistence and no data structures",
       ["memcached"], "decoy redis"),
    _q("reliable AMQP message broker with queue-based delivery of tasks",
       ["rabbitmq"], "decoy kafka"),
    _q("distributed append-only log for high-throughput events that consumers can replay",
       ["kafka"], "decoy rabbitmq"),
    _q("replay a stream of past events from an immutable log",
       ["kafka"], "decoy rabbitmq"),
    _q("cache static assets at edge locations near users to cut latency",
       ["cdn"], "decoys redis, memcached"),
    # --- python / frameworks --------------------------------------------------
    _q("Python API framework with built-in async and automatic request validation",
       ["fastapi_intro", "pydantic"], "decoy flask (lacks both)"),
    _q("async Python web framework with automatic OpenAPI documentation",
       ["fastapi_intro"], "decoy flask"),
    _q("validate and parse data from Python type hints",
       ["pydantic"], "decoy orm"),
    _q("minimal synchronous Python web framework with no built-in async",
       ["flask"], "decoy fastapi_intro"),
    _q("batteries-included Python framework with an ORM and an admin site",
       ["django"], "decoy flask"),
    _q("protect FastAPI routes with bearer tokens using dependencies",
       ["fastapi_auth"], "decoy jwt_access"),
    _q("run background jobs in Python worker processes behind a broker",
       ["celery"], "decoy rabbitmq"),
    _q("the async server interface Python uses, served by Uvicorn",
       ["asgi"], "decoy fastapi_intro"),
    # --- infra / deploy -------------------------------------------------------
    _q("package an app and its dependencies into an image for reproducible runs",
       ["docker_intro"], "decoy k8s"),
    _q("orchestrate containers across a cluster with autoscaling and self-healing",
       ["k8s"], "decoy docker_intro"),
    _q("declaratively provision cloud servers, but it must not build container images",
       ["terraform"], "negation; decoy docker_intro"),
    _q("package Kubernetes manifests into versioned, parameterized charts",
       ["helm"], "decoy k8s"),
    _q("run CI/CD pipelines triggered by repository pushes and pull requests",
       ["github_actions"], "decoy terraform"),
    _q("high-performance service-to-service RPC over HTTP/2 with schema-defined messages",
       ["grpc"], "decoy rabbitmq"),
    # --- observability --------------------------------------------------------
    _q("scrape and store time-series metrics and evaluate alerting rules",
       ["prometheus"], "decoy opentelemetry"),
    _q("vendor-neutral API for distributed tracing across services",
       ["opentelemetry"], "decoy prometheus"),
    _q("emit JSON key-value log fields so logs can be queried",
       ["logging_struct"], "decoy opentelemetry"),
    _q("a time-series datastore for metrics with a built-in alerting engine",
       ["prometheus"], "decoy elasticsearch"),
    # --- multi-hop / cross-cluster --------------------------------------------
    _q("run async Python background tasks using an in-memory broker",
       ["celery"], "multi-hop celery+redis broker; decoy rabbitmq"),
    _q("why does an HTTPS handshake require a certificate",
       ["tls"], "decoy mtls"),
    _q("route tasks to worker processes over a durable message queue",
       ["rabbitmq"], "decoy celery"),
]


def all_doc_ids() -> list[str]:
    return [d.doc_id for d in GOLDEN_CORPUS]


def load_jsonl(path: str) -> list[EvalExample]:
    out: list[EvalExample] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        out.append(EvalExample(obj["query"], list(obj["relevant"]), obj.get("note", "")))
    return out


def save_jsonl(examples: list[EvalExample], path: str) -> None:
    lines = [json.dumps({"query": e.query, "relevant": e.relevant}) for e in examples]
    Path(path).write_text("\n".join(lines), encoding="utf-8")
