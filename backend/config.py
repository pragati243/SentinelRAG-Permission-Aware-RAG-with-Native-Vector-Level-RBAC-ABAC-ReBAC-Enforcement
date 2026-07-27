import os
from pydantic import BaseModel

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

settings = Settings()
