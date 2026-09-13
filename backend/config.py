import os
from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()

class Settings(BaseModel):
    PROJECT_NAME: str = "SentinelRAG — Permission-Aware RAG"
    VERSION: str = "1.0.0"
    
    # Database settings
    DB_URL: str = os.getenv("DB_URL", "sqlite:///./sentinel_rag.db")
    
    # Qdrant settings
    QDRANT_COLLECTION_NAME: str = "sentinel_chunks"
    QDRANT_LOCATION: str = os.getenv("QDRANT_LOCATION", ":memory:")
    
    # Embedding settings
    EMBEDDING_DIM: int = 384
    
    # Security Defaults
    FAIL_CLOSED_TIER: str = "public"
    DEFAULT_PUBLIC_CLEARANCE: int = 1

    # Minimum cosine-similarity score a chunk must clear to be treated as a genuine
    # match. Without this, a permitted-but-irrelevant chunk (e.g. the only public-tier
    # doc a low-clearance user can see) gets returned for literally any query just
    # because it's the best available candidate within the permission filter, instead
    # of correctly falling through to the "no access" refusal. Empirically, real
    # matches in this corpus score ~0.4-0.75 and irrelevant fallbacks score ~-0.06-0.21
    # with the all-MiniLM-L6-v2 embeddings; 0.3 is a deliberately conservative cut
    # between those bands and should be re-validated if the embedding model changes.
    MIN_RELEVANCE_SCORE: float = float(os.getenv("MIN_RELEVANCE_SCORE", "0.3"))

    # Auth / JWT settings
    # NOTE: the dev default below is intentionally obvious and MUST be overridden
    # via the JWT_SECRET_KEY env var in any non-local deployment.
    JWT_SECRET_KEY: str = os.getenv("JWT_SECRET_KEY", "dev-insecure-secret-change-in-production")
    JWT_EXPIRY_MINUTES: int = int(os.getenv("JWT_EXPIRY_MINUTES", "30"))

    # Guardrail LLM-judge settings (backend/services/guardrail_service.py).
    # Guardrails fail OPEN when GROQ_API_KEY is unset -- they're defense-in-depth
    # on top of the RBAC/ABAC vector filter, which is the actual fail-closed
    # security boundary and doesn't depend on any of this.
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    GROQ_BASE_URL: str = "https://api.groq.com/openai/v1"
    # Dedicated prompt-injection/jailbreak classifier (outputs a raw 0-1 score,
    # not chat JSON) -- purpose-built for this task rather than asking a general
    # chat model to self-report via a system prompt.
    GROQ_PROMPT_GUARD_MODEL: str = os.getenv("GROQ_PROMPT_GUARD_MODEL", "meta-llama/llama-prompt-guard-2-86m")
    GROQ_PROMPT_GUARD_THRESHOLD: float = float(os.getenv("GROQ_PROMPT_GUARD_THRESHOLD", "0.5"))
    # General-purpose reasoning model used as the groundedness judge.
    GROQ_JUDGE_MODEL: str = os.getenv("GROQ_JUDGE_MODEL", "openai/gpt-oss-20b")

settings = Settings()
