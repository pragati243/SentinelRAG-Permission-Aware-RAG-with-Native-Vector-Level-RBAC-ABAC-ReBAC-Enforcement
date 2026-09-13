# 🛡️ SentinelRAG — Permission-Aware RAG with Native Vector RBAC/ABAC/ReBAC Enforcement

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![Qdrant](https://img.shields.io/badge/Qdrant-Vector%20DB-red.svg)](https://qdrant.tech/)
[![Security Gate](https://img.shields.io/badge/Security%20Gate-0%25%20Leak%20Passed-brightgreen.svg)](#-red-team-evaluation-suite--security-kpis)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**SentinelRAG** is an industrial-standard Retrieval-Augmented Generation (RAG) system engineered to enforce enterprise authorization policies **natively inside vector search graph traversal**. By denormalizing security metadata onto vector payloads and filtering candidates during HNSW graph traversal in Qdrant, SentinelRAG eliminates post-retrieval data leaks, quality degradation, compute waste, and existence-confirmation side channels — hardened with verified JWT identity, LLM guardrails, RAGAS-style evals, and end-to-end tracing.

📖 **For a complete, interview-depth walkthrough of every component and design decision, see [ARCHITECTURE.md](ARCHITECTURE.md).**

---

## 📋 Table of Contents

- [💡 Problem Definition \& Core Architecture Insight](#-problem-definition--core-architecture-insight)
- [🏗️ System Architecture](#️-system-architecture)
- [📁 Repository Structure](#-repository-structure)
- [🔒 Security Principles \& Access Control Models](#-security-principles--access-control-models)
- [🛡️ Guardrails: Defense-in-Depth Beyond RBAC](#️-guardrails-defense-in-depth-beyond-rbac)
- [🧪 Red-Team Evaluation Suite \& Security KPIs](#-red-team-evaluation-suite--security-kpis)
- [📊 Observability: End-to-End Tracing](#-observability-end-to-end-tracing)
- [⚡ Side-by-Side Naive vs. SentinelRAG Comparison](#-side-by-side-naive-vs-sentinelrag-comparison)
- [🛠️ Tech Stack \& Technical Decisions](#️-tech-stack--technical-decisions)
- [🚀 Quickstart \& Setup](#-quickstart--setup)
- [🔌 API Endpoint Documentation](#-api-endpoint-documentation)
- [🎯 Interview Talking Points](#-interview-talking-points--system-design-insights)
- [🩹 Known Limitations](#-known-limitations)

---

## 💡 Problem Definition & Core Architecture Insight

### The Flaw in Naive RAG & Post-Filtering

Standard enterprise RAG applications retrieve the top-$k$ semantically closest document chunks and feed them to an LLM. When sensitive documents (e.g., HR executive severance, confidential finance forecasts, client NDAs) are indexed into a unified vector database:

1. **Naive RAG**: Returns unauthorized chunks directly to the LLM context window, causing catastrophic data leaks.
2. **Naive Post-Filtering** (Retrieve top-$k$, then filter permissions in app code):
   - **Quality Degradation**: If the top 5 semantic matches are unauthorized, post-filtering returns zero chunks or degrades to low-score, irrelevant fallbacks.
   - **Compute Waste**: Wastes vector search compute and network bandwidth fetching documents that are subsequently discarded.
   - **Security Vulnerability**: Application-layer security controls are "one skipped filter bug away" from a security breach.

### The SentinelRAG Solution (Filtered ANN Search)

SentinelRAG denormalizes authorization metadata onto vector payloads during ingestion and executes permission predicates **inside Qdrant's Approximate Nearest Neighbor (ANN) search graph traversal**:

- Unauthorized vectors are **never evaluated as candidates** during vector graph traversal.
- Security enforcement mirrors **Row-Level Security (RLS)** at the vector engine layer.
- A minimum relevance-score cutoff (`MIN_RELEVANCE_SCORE`) then prevents a *permitted-but-irrelevant* chunk from being confidently narrated as an answer — see [Known Limitations](#-known-limitations) for the residual edge case this doesn't fully solve.

---

## 🏗️ System Architecture

```
                              ┌───────────────────────────┐
                              │      Document Ingestion      │
                              │  (HR / Legal / Finance / Eng)│
                              └─────────────┬─────────────┘
                                            │
                     ┌───────────────────────▼────────────────────────┐
                     │        Authorization Metadata Tagger              │
                     │  sensitivity_tier / owning_department /           │
                     │  required_clearance_level / allowed_roles[] /     │
                     │  project_scope (nullable, for ReBAC)              │
                     └───────────────┬────────────────────────────────┘
                                     │
                     ┌───────────────▼────────────────────────────────┐
                     │     Section-Aware Chunking + Embedding            │
                     │  all-MiniLM-L6-v2 (384-d), denormalized auth      │
                     │  metadata baked onto every chunk payload          │
                     └───────────────┬────────────────────────────────┘
                                     │
                          ┌──────────▼───────────┐
                          │   Qdrant Vector Store │
                          │  Filterable Metadata  │
                          └──────────┬───────────┘
                                     │
   ┌─────────────────┐   ┌──────────▼──────────────────────────────────┐
   │ Client + Bearer   │──►│  JWT Verification (auth_service.py)          │
   │ Token             │   │  signature + expiry check → verified user_id │
   └─────────────────┘   └──────────┬──────────────────────────────────┘
                                     │  user_id NEVER trusted from client input past this point
                     ┌───────────────▼────────────────────────────────┐
                     │  Permission Resolution (RBAC + ABAC + ReBAC)      │
                     │  resolved FRESH from DB every request — never     │
                     │  cached in the token, so a role change is live    │
                     │  immediately                                      │
                     └───────────────┬────────────────────────────────┘
                                     │  (effective permission set)
                     ┌───────────────▼────────────────────────────────┐
                     │  🛡️ Input Guardrail — Llama Prompt Guard 2         │
                     │  (jailbreak / prompt-injection classifier)        │
                     └───────────────┬────────────────────────────────┘
                                     │ not flagged
                     ┌───────────────▼────────────────────────────────┐
                     │       Filtered ANN Search (single query)          │
                     │  Qdrant HNSW payload filter (tier/dept/project)   │
                     │  + minimum relevance-score cutoff                 │
                     └───────────────┬────────────────────────────────┘
                                     │
                          ┌──────────▼───────────┐
                          │ Zero permitted or      │── yes ──► "I don't have access
                          │ input flagged?         │           to information that
                          └──────────┬───────────┘           answers this."
                                     │ no
                     ┌───────────────▼────────────────────────────────┐
                     │  LLM Generation (OpenAI → Groq → local template)  │
                     │  sees ONLY permitted chunks                       │
                     └───────────────┬────────────────────────────────┘
                                     │
                     ┌───────────────▼────────────────────────────────┐
                     │  🛡️ Output Guardrails — LLM-judged Groundedness   │
                     │  + regex PII scan                                 │
                     └───────────────┬────────────────────────────────┘
                                     │
                     ┌───────────────▼────────────────────────────────┐
                     │  Response + SHA-256 Hash-Chained Audit Log        │
                     │  (guardrail verdicts folded into the hash)        │
                     └────────────────────────────────────────────────┘

   Every box above is traced as a nested Langfuse span (backend/core/observability.py)
   when LANGFUSE_PUBLIC_KEY/SECRET_KEY are set — a safe no-op otherwise.
```

---

## 📁 Repository Structure

```text
SentinelRAG/
├── backend/
│   ├── api/
│   │   └── main.py               # FastAPI endpoints, JWT-gated, static frontend serving
│   ├── core/
│   │   ├── embedding.py          # all-MiniLM-L6-v2 (with deterministic hash fallback)
│   │   ├── ingestion.py          # Document ingestion pipeline & payload tagging
│   │   ├── llm_client.py         # Answer generation: OpenAI → Groq → local template
│   │   ├── groq_client.py        # Shared, traced low-level Groq API call plumbing
│   │   ├── observability.py      # No-op-safe Langfuse tracing wrapper
│   │   └── vector_store.py       # Qdrant client, HNSW filter predicates, relevance cutoff
│   ├── database/
│   │   ├── db.py                 # SQLAlchemy engine/session setup
│   │   ├── models.py             # User, staffing, document, audit log ORM models
│   │   └── seed_data.py          # Demo users, ReBAC staffing, multi-department corpus
│   ├── eval/
│   │   ├── metrics.py            # RAGAS-style LLM-judged faithfulness/relevancy/leak metrics
│   │   └── red_team.py           # Adversarial test-case battery using those metrics
│   ├── services/
│   │   ├── auth_service.py       # JWT issuance + verification (identity trust boundary)
│   │   ├── audit_service.py      # SHA-256 hash-chained audit logger (lock-serialized)
│   │   ├── guardrail_service.py  # Input safety, groundedness, PII guardrails
│   │   ├── permission_service.py # Hybrid RBAC/ABAC/ReBAC permission resolver
│   │   └── rag_engine.py         # Core orchestrator: the full traced request pipeline
│   └── config.py                 # Centralized settings (.env-driven)
├── frontend/
│   ├── index.html                # Enterprise dark-mode dashboard
│   ├── app.js                    # Persona login flow, Bearer-token API calls
│   └── style.css                 # CSS variables & glassmorphism styling
├── tests/
│   ├── test_permission_rag.py    # Core RBAC/ABAC/ReBAC + audit chain tests
│   ├── test_auth.py              # JWT auth boundary regression tests
│   ├── test_guardrails.py        # Guardrail unit + engine-integration tests
│   └── test_metrics.py           # RAGAS-style metric unit tests
├── .env.example                  # Documented env vars (.env itself is gitignored)
├── requirements.txt              # Python package dependencies
├── ARCHITECTURE.md               # Complete architecture deep-dive & interview prep
└── README.md                     # This file
```

---

## 🔒 Security Principles & Access Control Models

### 1. Verified Identity (JWT), Never Client-Supplied

`POST /auth/token` issues a signed JWT for a known user (a stand-in for a real IdP/SSO exchange). Every other endpoint requires `Authorization: Bearer <token>`, verified via `backend/services/auth_service.py`. **Only `sub` (the user id) is embedded in the token** — role, department, and clearance are deliberately *not* baked in, so they're re-resolved fresh from the database on every request instead of going stale until the token expires.

### 2. Hybrid Authorization Model (ABAC + RBAC + ReBAC)

SentinelRAG combines three security paradigms into a unified vector search predicate:

- **RBAC (Role-Based Access Control)**: Enforces role boundaries (`HR_Manager`, `Finance_Analyst`, `Legal_Paralegal`, `VP`, `Employee`).
- **ABAC (Attribute-Based Access Control)**: Evaluates departmental alignment (`owning_department == user.department`) and numeric clearance levels:
  - `Level 1`: Public
  - `Level 2`: Internal / Confidential
  - `Level 3`: Restricted
  - `Level 4`: Executive (full access)
- **ReBAC (Relationship-Based Access Control)**: Validates project staffing relationships (`project_scope IN user.staffed_projects`).

### 3. Fail-Closed Defaulting (and Fail-Loud)

If permission resolution encounters an invalid identity, missing database record, or service exception, SentinelRAG defaults to `Anonymous` mode with `Level 1 (Public)` clearance — it **never** falls back to permissive open access. Every fail-closed exception is also logged (`logger.exception(...)`), so a genuine outage isn't silently indistinguishable from a malicious probe.

### 4. Zero Existence Confirmation Leak Policy

When an unauthorized user queries content for which they lack clearance, SentinelRAG returns:

> *"I don't have access to information that answers this."*

The **exact same message** is also returned when the input guardrail blocks a prompt-injection attempt or the groundedness guardrail rejects a mis-grounded answer — a distinct message per defense would itself leak which control fired.

### 5. Cryptographic Hash-Chained Audit Trail

Every query execution records an immutable, SHA-256 hash-chained log entry, now including guardrail verdicts:

$$\text{Hash}_i = \text{SHA256}(\text{Hash}_{i-1} \parallel \text{Timestamp} \parallel \text{User ID} \parallel \text{Query} \parallel \text{Retrieved Chunks} \parallel \text{Answer} \parallel \text{Guardrail Report})$$

Appends are serialized with a process-local lock to prevent concurrent requests from forking the chain. This structure allows instant detection of any log tampering for compliance audits (SOC 2, HIPAA, ISO 27001).

---

## 🛡️ Guardrails: Defense-in-Depth Beyond RBAC

RBAC/ABAC/ReBAC filtering decides **what content the model is allowed to see**. Guardrails (`backend/services/guardrail_service.py`) decide **what the model is allowed to say**, even given permitted content — a different, complementary threat surface (`backend/eval/metrics.py` covers the *measurement* side of the same concern; see [Evals](#-red-team-evaluation-suite--security-kpis)).

| Guardrail | How | Fails... |
| :--- | :--- | :--- |
| **Input safety** | Llama Prompt Guard 2 (via Groq) classifies the query for jailbreak/prompt-injection attempts before search runs | **Open** — a judge outage doesn't block legitimate traffic; RBAC/ABAC is the real security boundary and doesn't depend on this |
| **Groundedness** | An LLM judge checks the generated answer is actually supported by, and relevant to, the retrieved context | **Open** — flags replace the answer with the standard refusal; judge unavailability just skips the check |
| **PII leak** | Local regex scan (email/phone/SSN/card-like patterns) over the generated answer, no LLM call | N/A — deterministic, always available |

All three verdicts are folded into `guardrail_report` on every response and audit log entry, and into the hash chain itself.

---

## 🧪 Red-Team Evaluation Suite & Security KPIs

SentinelRAG includes an automated Red-Team evaluation harness (`backend/eval/red_team.py`) that runs adversarial query scenarios across multiple user profiles and scores them with **RAGAS-style LLM-judged metrics** (`backend/eval/metrics.py`) instead of brittle keyword matching:

- **Leak Rate Target**: `0.0%` — union of literal substring matching **and** an LLM-judged semantic check, so a rephrased leak ("twelve months of pay" instead of "12 months salary continuation") can't slip through just because the exact words differ.
- **Fail-Closed Compliance**: `100.0%`
- **ReBAC Staffing Precision**: `100.0%`
- **Existence Leak Rate**: `0.0%`
- **Avg. Faithfulness**: is the answer's content actually supported by the retrieved context?
- **Avg. Answer Relevancy**: does the answer actually address the query?

```bash
==================================================
🛡️ SENTINEL RAG — RED-TEAM SECURITY EVALUATION
==================================================
Evaluated 11 adversarial test vectors across 5 identity personas.
Security Gate Status: PASSED
Data Leakage Rate: 0.0% (keyword + semantic judge)
Fail-Closed Compliance: 100.0%
Avg. Faithfulness: 1.00 | Avg. Answer Relevancy: 0.97
==================================================
```

---

## 📊 Observability: End-to-End Tracing

`backend/core/observability.py` wraps [Langfuse](https://langfuse.com) into a thin, no-op-safe `trace_span()` helper. When `LANGFUSE_PUBLIC_KEY`/`LANGFUSE_SECRET_KEY` are set, every `/ask` request produces a nested trace tree:

```
sentinel_rag.ask
├── permission_resolution
├── groq:meta-llama/llama-prompt-guard-2-86m   (input guardrail)
├── vector_search.permissioned
├── vector_search.naive
├── generation                                  (model + token usage)
├── groq:openai/gpt-oss-20b                     (groundedness judge)
└── (pii guardrail — local, not traced; no LLM call)
```

Every Groq call anywhere in the app (guardrails, red-team eval judges) is traced automatically because they all route through the same instrumented `backend/core/groq_client.py` helper — no per-call-site tracing code needed. Without Langfuse configured, `trace_span()` is a genuine no-op (verified via the SDK's own "disabled client" mode), so the app behaves identically either way.

---

## ⚡ Side-by-Side Naive vs. SentinelRAG Comparison

| Feature | Naive RAG | SentinelRAG (This System) |
| :--- | :--- | :--- |
| **Identity** | Often trusts a client-supplied user id | ❌→✅ Verified signed JWT only |
| **Vector Filtering** | Post-Filter (or None) | **Native Qdrant HNSW Graph Payload Filtering** |
| **Data Leak Prevention** | ❌ Fails (Leaks sensitive data) | **✅ 0.0% Leak Rate Gate (keyword + semantic judge)** |
| **Prompt Injection** | Usually unguarded | **✅ Dedicated classifier (Llama Prompt Guard 2)** |
| **Answer Groundedness** | Rarely checked | **✅ LLM-judged, non-blocking-on-outage** |
| **Existence Side-Channel** | ❌ Confirms restricted doc existence | **✅ Zero existence confirmation leakage** |
| **Auditability** | ❌ Unstructured or absent | **✅ SHA-256 Hash-Chained Trail incl. guardrail verdicts** |
| **Observability** | ❌ Usually none | **✅ Full request trace tree (Langfuse)** |
| **Failure Mode** | ❌ Undefined / Fail-Open | **✅ Hard Fail-Closed on RBAC; guardrails fail open by design** |

---

## 🛠️ Tech Stack & Technical Decisions

| Component | Technology | Rationale |
| :--- | :--- | :--- |
| **API Framework** | FastAPI (Python 3.10+) | High-performance asynchronous REST endpoints with OpenAPI schema generation. |
| **Vector Database** | Qdrant | Supports fast in-memory filtering during vector graph traversal without secondary joins. |
| **Relational Store** | SQLite / SQLAlchemy 2.0 | Lightweight user RBAC/ABAC policy, staffing metadata, and audit log persistence. |
| **Embeddings** | `all-MiniLM-L6-v2` (sentence-transformers) | Real semantic embeddings, with a deterministic hash-projection fallback if unavailable. |
| **Identity** | PyJWT (HS256) | Signed, expiry-checked tokens; the trust boundary the whole authorization model sits behind. |
| **Guardrail / Eval Judge** | Groq (Llama Prompt Guard 2, `openai/gpt-oss-20b`) | Fast, cheap inference for a purpose-built injection classifier plus a general reasoning judge. |
| **Answer Generation** | OpenAI (if configured) → Groq → local template | Graceful multi-provider fallback; never hard-fails to no answer. |
| **Observability** | Langfuse (Cloud or self-hosted) | No-op-safe tracing; every LLM call and pipeline stage as a nested span. |
| **Audit Chain** | SHA-256 Hash Chaining | Tamper-evident logging for enterprise security compliance. |
| **Frontend** | Vanilla JS + Glassmorphic CSS | Lightweight, dependency-free interactive dashboard with side-by-side comparison mode. |

---

## 🚀 Quickstart & Setup

### 1. Environment Setup

```bash
git clone <this-repo-url>
cd SentinelRAG

python -m venv venv
.\venv\Scripts\activate      # Windows
# source venv/bin/activate   # Linux/macOS

pip install -r requirements.txt
```

### 2. Configure Environment Variables

```bash
cp .env.example .env
```

Fill in `.env`:
- `JWT_SECRET_KEY` — required for anything beyond local dev.
- `GROQ_API_KEY` — optional but recommended; unlocks the input-safety, groundedness, and eval judges, and is used as the answer-generation fallback when no `OPENAI_API_KEY` is set. Get one free at [console.groq.com](https://console.groq.com).
- `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` — optional; unlocks request tracing. Get free-tier keys at [cloud.langfuse.com](https://cloud.langfuse.com).

Everything above degrades gracefully when unset — the app runs fine without any of them, just without the corresponding capability.

### 3. Running Automated Tests

```bash
pytest tests/ -v
```

### 4. Running the Red-Team Security Suite

```bash
python -m backend.eval.red_team
```

### 5. Launching the Application & UI Dashboard

```bash
uvicorn backend.api.main:app --reload --port 8000
```

- **Interactive UI Dashboard**: [http://127.0.0.1:8000](http://127.0.0.1:8000)
- **FastAPI Interactive Docs**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

---

## 🔌 API Endpoint Documentation

| Method | Endpoint | Auth Required | Description |
| :--- | :--- | :--- | :--- |
| `GET` | `/` | None | Serves the interactive frontend web dashboard. |
| `POST` | `/auth/token` | None | Demo IdP simulation — issues a signed JWT for a known persona. |
| `POST` | `/ask` | Bearer token | Main permission-filtered RAG query endpoint. |
| `GET` | `/debug/naive-vs-permissioned` | Bearer token | Side-by-side naive vs. filtered comparison for the caller's own identity. |
| `GET` | `/debug/resolved-permissions` | Bearer token | Inspect the caller's own resolved RBAC/ABAC/ReBAC grants. |
| `POST` | `/ingest` | None (see [Known Limitations](#-known-limitations)) | Ingests a document with authorization metadata tagging. |
| `GET` | `/audit-log` | Bearer token + VP/Security_Auditor/Admin role | Hash-chained audit log + integrity verification. |
| `GET` | `/eval/run` | Bearer token | Runs the red-team evaluation battery. |
| `GET` | `/admin/users` | None (intentional — demo persona picker) | Lists demo personas for the UI's login dropdown. |

---

## 🎯 Interview Talking Points & System Design Insights

1. **Why Filtered ANN Search?** Post-filtering top-k vector results breaks down in enterprise environments because if the most relevant chunks are restricted, post-filtering leaves you with zero context or low-quality fallbacks. SentinelRAG enforces filters during HNSW graph traversal in Qdrant, so retrieved chunks are both semantically relevant *and* authorized by construction.

2. **Fail-Closed Security, Fail-Open Guardrails — and why that's not a contradiction.** RBAC/ABAC filtering at the vector layer is the hard security boundary and fails closed. Guardrails (prompt-injection detection, groundedness) are defense-in-depth *on top of* that boundary — if the guardrail judge is unavailable, the request still can't leak restricted data, so it's safe to let it through unflagged rather than take down the whole app over an auxiliary check.

3. **Zero Existence Confirmation Leakage.** A permission denial, a blocked prompt-injection attempt, and a groundedness-rejected answer all produce the *identical* refusal string. Differentiating them in the response would itself be a side channel — an attacker could use distinct error messages to fingerprint which defense caught them.

4. **Identity Is a Verified Token, Never a Request Field.** Only `sub` (user id) is signed into the JWT; role/department/clearance are re-resolved from the database on every request. This avoids the common mistake of baking authorization into a long-lived token, which would make a permission change (e.g., an employee's access being revoked) not take effect until the token expires.

5. **LLM-as-Judge, Not String Matching.** Both the guardrails and the red-team eval suite use LLM judges instead of keyword lists, specifically because keyword matching has a straightforward blind spot: paraphrase. `judge_semantic_leak` and `check_groundedness` catch the two shapes that keyword matching structurally cannot.

6. **Cryptographic Tamper-Evident Auditability.** Every retrieval, denial, and guardrail decision is chained using SHA-256 hashes, including the guardrail verdicts themselves. Any modification anywhere in that history breaks the chain, which `verify_chain_integrity` detects deterministically.

**→ For the full request-by-request walkthrough, every design tradeoff, and a much longer bank of likely interview questions with model answers, see [ARCHITECTURE.md](ARCHITECTURE.md).**

---

## 🩹 Known Limitations

Documented honestly, because a reviewer will find these anyway:

- **Groundedness judge is non-deterministic.** Observed an occasional (~1-in-5) false-positive flag on a genuinely correct, well-grounded answer, forcing an unnecessary refusal. Fails in the safe direction (over-refusal, not a leak), but is a real UX cost.
- **`GROQ_JUDGE_MODEL` and `GROQ_GENERATION_MODEL` default to the same model.** A model judging its own (or a sibling call's) output has a known self-preference bias in the LLM-eval literature; they're separate settings specifically so they can diverge later.
- **SQLite + a process-local lock** serializes the audit hash chain correctly for a single process, but a multi-worker deployment needs a DB-native equivalent (e.g., Postgres `pg_advisory_xact_lock`).
- **`/ingest` has no authorization gate** — anyone can tag and ingest a document into any department/tier today.
- **PII detection is regex-based**, not a proper NER model (e.g., Microsoft Presidio) — adequate for a demo, not for production-grade redaction.

---

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.
