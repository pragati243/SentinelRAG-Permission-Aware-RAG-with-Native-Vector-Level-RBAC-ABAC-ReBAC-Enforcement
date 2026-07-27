import hashlib
import numpy as np
from typing import List
from backend.config import settings

class EmbeddingService:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.dim = settings.EMBEDDING_DIM
        self._st_model = None

    def _get_st_model(self):
        if self._st_model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._st_model = SentenceTransformer(self.model_name)
            except Exception:
                self._st_model = False
        return self._st_model

    def embed_text(self, text: str) -> List[float]:
        """
        Embeds a single string into a vector.
        Uses SentenceTransformers if available, otherwise a deterministic hash-projection fallback.
        """
        st_model = self._get_st_model()
        if st_model:
            vector = st_model.encode(text, convert_to_numpy=True).tolist()
            return vector
        
        # Deterministic lightweight semantic fallback
        return self._fallback_embed(text)

    def _fallback_embed(self, text: str) -> List[float]:
        # Hash words to generate a pseudo-random yet deterministic dense vector
        words = text.lower().split()
        vec = np.zeros(self.dim, dtype=np.float32)
        for w in words:
            seed = int(hashlib.md5(w.encode("utf-8")).hexdigest(), 16) % (2**32 - 1)
            rng = np.random.RandomState(seed)
            vec += rng.randn(self.dim).astype(np.float32)
        
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec.tolist()

embedding_service = EmbeddingService()
