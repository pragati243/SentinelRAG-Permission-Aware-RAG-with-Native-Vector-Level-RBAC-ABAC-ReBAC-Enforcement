# 🛡️ SentinelRAG — Permission-Aware RAG with Native Vector RBAC/ABAC/ReBAC Enforcement

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![Qdrant](https://img.shields.io/badge/Qdrant-Vector%20DB-red.svg)](https://qdrant.tech/)
[![Security Gate](https://img.shields.io/badge/Security%20Gate-0%25%20Leak%20Passed-brightgreen.svg)](#-red-team-evaluation-suite--security-kpis)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**SentinelRAG** is an industrial-standard Retrieval-Augmented Generation (RAG) system engineered to enforce enterprise authorization policies **natively inside vector search graph traversal**. By denormalizing security metadata onto vector payloads and filtering candidates during HNSW graph traversal in Qdrant, SentinelRAG eliminates post-retrieval data leaks, quality degradation, compute waste, and existence-confirmation side channels.

---

## 📋 Table of Contents

- [💡 Problem Definition \& Core Architecture Insight](#-problem-definition--core-architecture-insight)
  - [The Flaw in Naive RAG \& Post-Filtering](#the-flaw-in-naive-rag--post-filtering)
  - [The SentinelRAG Solution (Filtered ANN Search)](#the-sentinelrag-solution-filtered-ann-search)
- [🏗️ System Architecture](#️-system-architecture)
- [📁 Repository Structure](#-repository-structure)
- [🔒 Security Principles \& Access Control Models](#-security-principles--access-control-models)
  - [1. Hybrid Authorization Model (ABAC + RBAC + ReBAC)](#1-hybrid-authorization-model-abac--rbac--rebac)
  - [2. Fail-Closed Defaulting](#2-fail-closed-defaulting)
  - [3. Zero Existence Confirmation Leak Policy](#3-zero-existence-confirmation-leak-policy)
  - [4. Cryptographic Hash-Chained Audit Trail](#4-cryptographic-hash-chained-audit-trail)
- [⚡ Side-by-Side Naive vs. SentinelRAG Comparison](#-side-by-side-naive-vs-sentinelrag-comparison)
- [🧪 Red-Team Evaluation Suite \& Security KPIs](#-red-team-evaluation-suite--security-kpis)
- [🛠️ Tech Stack \& Technical Decisions](#️-tech-stack--technical-decisions)
- [🚀 Quickstart \& Setup](#-quickstart--setup)
  - [1. Environment Setup](#1-environment-setup)
  - [2. Running Automated Tests](#2-running-automated-tests)
  - [3. Running Red-Team Security Suite](#3-running-red-team-security-suite)
  - [4. Launching Application \& UI Dashboard](#4-launching-application--ui-dashboard)
- [🔌 API Endpoint Documentation](#-api-endpoint-documentation)
- [🎯 Interview Talking Points \& System Design Insights](#-interview-talking-points--system-design-insights)

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
                     │  - sensitivity_tier (public/internal/confidential/│
                     │    restricted)                                    │
                     │  - owning_department                              │
                     │  - required_clearance_level                       │
                     │  - allowed_roles[] (explicit override)            │
                     │  - project_scope (nullable, for ReBAC)            │
                     └───────────────┬────────────────────────────────┘
                                     │
                     ┌───────────────▼────────────────────────────────┐
                     │     Section-Aware Chunking + Embedding            │
                     │  - Denormalizes auth metadata onto every chunk    │
                     │    payload for zero-join vector filtering         │
                     └───────────────┬────────────────────────────────┘
                                     │
                          ┌──────────▼───────────┐
                          │    Qdrant Store with │
                          │  Filterable Metadata │
                          └──────────┬───────────┘
                                     │
   ┌─────────────────┐   ┌──────────▼───────────┐   ┌──────────────────────┐
   │ User Query +      │──►│  Permission Resolution │──►│  Relational DB / IdP │
   │ Identity Token    │   │  Service                │   │  (users, roles, dept,│
   │                   │   │  - resolve role/clearance│   │   clearance, ReBAC   │
   │                   │   │  - resolve ReBAC projects│   │   staffing tables)   │
   └─────────────────┘   └──────────┬───────────┘   └──────────────────────┘
                                     │  (effective permission set)
                     ┌───────────────▼────────────────────────────────┐
                     │       Filtered ANN Search (single query)          │
                     │  Qdrant HNSW Payload Filter                       │
                     │  - tier <= user.clearance                         │
                     │  - (dept = user.dept OR tier = 'public')          │
                     │  - (project IS NULL OR project IN user.projects)  │
                     └───────────────┬────────────────────────────────┘
                                     │
                          ┌──────────▼───────────┐
                          │  Zero permitted        │──── yes ───► "I don't have access
                          │  results?              │              to information that
                          └──────────┬───────────┘              answers this."
                                     │ no
                     ┌───────────────▼────────────────────────────────┐
                     │   LLM Answer Generation (sees ONLY permitted      │
                     │   chunks — never sees restricted content)         │
                     └───────────────┬────────────────────────────────┘
                                     │
                     ┌───────────────▼────────────────────────────────┐
                     │   Response + Hash-Chained Access Audit Log        │
                     │  (SHA256 chained, immutable access trail)         │
                     └────────────────────────────────────────────────┘
```

---

## 📁 Repository Structure

```text
SentinelRAG/
├── backend/
│   ├── api/
│   │   └── main.py              # FastAPI endpoints & static frontend serving
│   ├── core/
│   │   ├── embedding.py         # Mock & Deterministic Vector Embeddings Generator
│   │   ├── ingestion.py         # Document Ingestion Pipeline & Payload Tagging
│   │   ├── llm_client.py        # LLM Answer Generation Engine
│   │   └── vector_store.py      # Qdrant Client with HNSW Filter Predicates
│   ├── database/
│   │   ├── db.py                # SQLAlchemy Database Engine Setup
│   │   ├── models.py            # User, Role, Department & Audit Log Models
│   │   └── seed_data.py         # Enterprise Seed Data & Test Document Corpuses
│   ├── eval/
│   │   └── red_team.py          # Automated Red-Team Security Benchmarks
│   ├── services/
│   │   ├── audit_service.py     # Cryptographic SHA-256 Hash-Chained Audit Logger
│   │   ├── permission_service.py# Hybrid RBAC/ABAC/ReBAC Permission Resolver
│   │   └── rag_engine.py        # Core SentinelRAG Engine Orchestrator
│   └── config.py                # Global System Configurations
├── frontend/
│   ├── index.html               # Enterprise Dark-Mode Dashboard
│   ├── app.js                   # Interactive UI State & Query Controller
│   └── style.css                # CSS Variables & Glassmorphism Styling
├── tests/
│   └── test_permission_rag.py   # Pytest Automated Test Suite
├── requirements.txt             # Python Package Dependencies
└── README.md                    # System Documentation & Architecture Guide
```

---

## 🔒 Security Principles & Access Control Models

### 1. Hybrid Authorization Model (ABAC + RBAC + ReBAC)

SentinelRAG combines three security paradigms into a unified vector search predicate:

- **RBAC (Role-Based Access Control)**: Enforces role boundaries (`HR_Manager`, `Finance_Analyst`, `Legal_Paralegal`, `VP`, `Employee`).
- **ABAC (Attribute-Based Access Control)**: Evaluates departmental alignment (`owning_department == user.department`) and numeric clearance levels:
  - `Level 1`: Public
  - `Level 2`: Internal / Confidential
  - `Level 3`: Restricted
  - `Level 4`: Executive
- **ReBAC (Relationship-Based Access Control)**: Validates project staffing relationships (`project_scope IN user.staffed_projects`).

### 2. Fail-Closed Defaulting

If permission resolution encounters an invalid identity token, missing database record, or service exception, SentinelRAG defaults to `Anonymous` mode with `Level 1 (Public)` clearance. It **never** falls back to permissive open access.

### 3. Zero Existence Confirmation Leak Policy

When an unauthorized user queries content for which they lack clearance, SentinelRAG returns:

> *"I don't have access to information that answers this."*

The system **never** reveals whether restricted matching documents exist in the vector index, preventing side-channel reconnaissance.

### 4. Cryptographic Hash-Chained Audit Trail

Every query execution records an immutable, SHA-256 hash-chained log entry:

$$\text{Hash}_i = \text{SHA256}(\text{Hash}_{i-1} \parallel \text{Timestamp} \parallel \text{User ID} \parallel \text{Query} \parallel \text{Retrieved Chunks} \parallel \text{Denied Count})$$

This structure allows instant detection of any log tampering or unauthorized back-dating for compliance audits (SOC 2, HIPAA, ISO 27001).

---

## ⚡ Side-by-Side Naive vs. SentinelRAG Comparison

| Feature | Naive RAG | SentinelRAG (This System) |
| :--- | :--- | :--- |
| **Vector Filtering** | Post-Filter (or None) | **Native Qdrant HNSW Graph Payload Filtering** |
| **Data Leak Prevention** | ❌ Fails (Leaks sensitive data) | **✅ 0.0% Leak Rate Gate** |
| **Search Efficiency** | ❌ Searches unauthorized vectors | **✅ Excludes unauthorized vectors prior to distance evaluation** |
| **Existence Side-Channel** | ❌ Confirms restricted doc existence | **✅ Zero existence confirmation leakage** |
| **Auditability** | ❌ Unstructured or absent | **✅ Cryptographic SHA-256 Hash-Chained Trail** |
| **Failure Mode** | ❌ Undefined / Fail-Open | **✅ Hard Fail-Closed Default** |

---

## 🧪 Red-Team Evaluation Suite & Security KPIs

SentinelRAG includes an automated Red-Team evaluation harness (`backend/eval/red_team.py`) that tests adversarial query scenarios across multiple user profiles:

- **Leak Rate Target**: `0.0%` (Hard Security Gate)
- **Fail-Closed Compliance**: `100.0%`
- **ReBAC Staffing Precision**: `100.0%`
- **Existence Leak Rate**: `0.0%`

```bash
==================================================
🛡️ SENTINEL RAG — RED-TEAM SECURITY EVALUATION
==================================================
Evaluated 12 adversarial test vectors across 5 identity personas.
Security Gate Status: PASSED
Data Leakage Rate: 0.0%
Fail-Closed Compliance: 100.0%
==================================================
```

---

## 🛠️ Tech Stack & Technical Decisions

| Component | Technology | Rationale |
| :--- | :--- | :--- |
| **API Framework** | FastAPI (Python 3.10+) | High-performance asynchronous REST endpoints with OpenAPI schema generation. |
| **Vector Database** | Qdrant | Supports fast in-memory filtering during vector graph traversal without secondary joins. |
| **Relational Store** | SQLite / SQLAlchemy 2.0 | Lightweight user RBAC/ABAC policy, staffing metadata, and audit log persistence. |
| **Embeddings** | Deterministic 384-d Vector Engine | Rapid local execution with full vector similarity preservation for testing. |
| **Audit Chain** | SHA-256 Hash Chaining | Tamper-evident logging for enterprise security compliance. |
| **Frontend** | Vanilla JS + Glassmorphic CSS | Lightweight, dependency-free interactive dashboard with side-by-side comparison mode. |

---

## 🚀 Quickstart & Setup

### 1. Environment Setup

Clone the repository and install the dependencies:

```bash
# Clone the repository
git clone https://github.com/your-username/SentinelRAG.git
cd SentinelRAG

# Create virtual environment
python -m venv venv

# Activate virtual environment (Windows)
.\venv\Scripts\activate

# Activate virtual environment (Linux/macOS)
# source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Running Automated Tests

Execute the Pytest test suite:

```bash
pytest tests/ -v
```

### 3. Running Red-Team Security Suite

Run the security evaluation harness to verify the zero-leak guarantee:

```bash
python -m backend.eval.red_team
```

### 4. Launching Application & UI Dashboard

Start the FastAPI server:

```bash
uvicorn backend.api.main:app --reload --port 8000
```

Access the application in your browser:
- **Interactive UI Dashboard**: [http://127.0.0.1:8000](http://127.0.0.1:8000)
- **FastAPI Interactive Docs**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

---

## 🔌 API Endpoint Documentation

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/` | Serves the interactive frontend web dashboard. |
| `GET` | `/api/users` | Lists available test user personas with roles and department metadata. |
| `POST` | `/api/query` | Executes a permission-filtered RAG query for a selected user persona. |
| `POST` | `/api/compare` | Runs a side-by-side query comparing Naive RAG vs. SentinelRAG. |
| `GET` | `/api/audit-log` | Retrieves the hash-chained audit log and verifies cryptographic chain integrity. |

---

## 🎯 Interview Talking Points & System Design Insights

1. **Why Filtered ANN Search?**
   > *"Post-filtering top-k vector results breaks down in enterprise environments because if the most relevant chunks are restricted, post-filtering leaves you with zero context or low-quality fallbacks. SentinelRAG enforces filters during HNSW graph traversal in Qdrant, ensuring that 100% of retrieved chunks are both semantically relevant and authorized."*

2. **Fail-Closed Security Design**
   > *"Security architectures must fail closed. If the permission resolver receives an invalid user token or experiences a database lookup failure, SentinelRAG defaults to public-level clearance rather than throwing an unhandled exception or granting permissive access."*

3. **Zero Existence Confirmation Leakage**
   > *"Confirming that a user lacks access to a document implicitly confirms the document's existence. SentinelRAG responds with a generic 'I don't have access to information that answers this' message for both non-existent and unauthorized queries, eliminating side-channel leakage."*

4. **Cryptographic Tamper-Evident Auditability**
   > *"To meet SOC 2 and HIPAA compliance requirements, every retrieval query and denial decision is chained using SHA-256 hashes. Any back-dated entry or modification breaks the hash chain, enabling instant validation of audit log integrity."*

---

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.
