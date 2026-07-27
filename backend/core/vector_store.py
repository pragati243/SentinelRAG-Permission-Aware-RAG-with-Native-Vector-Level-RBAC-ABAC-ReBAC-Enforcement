from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams, Distance, PointStruct, Filter, FieldCondition, MatchValue, MatchAny, Range, IsNullCondition
)
from backend.config import settings
from backend.services.permission_service import ResolvedPermissionSet
from backend.core.embedding import embedding_service

class QdrantVectorStore:
    def __init__(self, location: str = settings.QDRANT_LOCATION, collection_name: str = settings.QDRANT_COLLECTION_NAME):
        self.collection_name = collection_name
        self.client = QdrantClient(location=location)
        self._ensure_collection()

    def _ensure_collection(self):
        collections = [c.name for c in self.client.get_collections().collections]
        if self.collection_name not in collections:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=settings.EMBEDDING_DIM, distance=Distance.COSINE)
            )

    def upsert_chunks(self, chunks: List[Dict[str, Any]]):
        """
        Upserts chunk payloads with denormalized permission metadata into Qdrant.
        """
        points = []
        for chunk in chunks:
            vector = chunk.get("vector") or embedding_service.embed_text(chunk["text"])
            
            payload = {
                "chunk_id": chunk["chunk_id"],
                "doc_id": chunk["doc_id"],
                "section": chunk.get("section", ""),
                "text": chunk["text"],
                "sensitivity_tier": chunk["sensitivity_tier"],
                "owning_department": chunk["owning_department"],
                "required_clearance_level": int(chunk["required_clearance_level"]),
                "allowed_roles": chunk.get("allowed_roles") or [],
                "project_scope": chunk.get("project_scope") or "none",
            }

            points.append(
                PointStruct(
                    id=chunk["chunk_id"],
                    vector=vector,
                    payload=payload
                )
            )

        if points:
            self.client.upsert(collection_name=self.collection_name, points=points)

    def build_qdrant_filter(self, perms: ResolvedPermissionSet) -> Filter:
        """
        Translates a ResolvedPermissionSet into a native Qdrant Payload Filter.
        Combines ABAC (clearance, department), RBAC (roles), and ReBAC (staffed projects).
        """
        must_conditions = []

        # 1. ABAC: Clearance level requirement
        must_conditions.append(
            FieldCondition(
                key="required_clearance_level",
                range=Range(lte=perms.clearance_level)
            )
        )

        # 2. ABAC: Allowed sensitivity tiers
        must_conditions.append(
            FieldCondition(
                key="sensitivity_tier",
                match=MatchAny(any=perms.allowed_tiers)
            )
        )

        # 3. ABAC: Department Scoping (owning_dept == user.dept OR public OR VP override)
        dept_conditions = [
            FieldCondition(key="owning_department", match=MatchValue(value=perms.department)),
            FieldCondition(key="owning_department", match=MatchValue(value="Public")),
            FieldCondition(key="sensitivity_tier", match=MatchValue(value="public")),
        ]
        # VP or Executive role gets cross-department access override
        if perms.role in ["VP", "Executive", "Admin"]:
            dept_conditions.append(FieldCondition(key="required_clearance_level", range=Range(lte=perms.clearance_level)))

        must_conditions.append(Filter(should=dept_conditions))

        # 4. ReBAC: Project Scope filter
        project_conditions = [
            FieldCondition(key="project_scope", match=MatchValue(value="none")),
            FieldCondition(key="project_scope", match=MatchValue(value="")),
        ]
        if perms.staffed_projects:
            project_conditions.append(
                FieldCondition(key="project_scope", match=MatchAny(any=perms.staffed_projects))
            )
        
        must_conditions.append(Filter(should=project_conditions))

        return Filter(must=must_conditions)

    def search_permissioned(self, query_text: str, perms: ResolvedPermissionSet, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Executes FILTERED ANN Search inside Qdrant.
        The permission filter is applied AS PART OF the HNSW vector traversal.
        """
        query_vector = embedding_service.embed_text(query_text)
        qdrant_filter = self.build_qdrant_filter(perms)

        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            query_filter=qdrant_filter,
            limit=top_k
        )

        return [
            {
                "chunk_id": res.payload["chunk_id"],
                "doc_id": res.payload["doc_id"],
                "section": res.payload["section"],
                "text": res.payload["text"],
                "sensitivity_tier": res.payload["sensitivity_tier"],
                "owning_department": res.payload["owning_department"],
                "required_clearance_level": res.payload["required_clearance_level"],
                "project_scope": res.payload["project_scope"],
                "score": float(res.score)
            }
            for res in results
        ]

    def search_naive(self, query_text: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Unfiltered naive similarity search (simulates traditional non-permissioned RAG).
        Returns top-k chunks regardless of sensitivity or authorization.
        """
        query_vector = embedding_service.embed_text(query_text)

        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=top_k
        )

        return [
            {
                "chunk_id": res.payload["chunk_id"],
                "doc_id": res.payload["doc_id"],
                "section": res.payload["section"],
                "text": res.payload["text"],
                "sensitivity_tier": res.payload["sensitivity_tier"],
                "owning_department": res.payload["owning_department"],
                "required_clearance_level": res.payload["required_clearance_level"],
                "project_scope": res.payload["project_scope"],
                "score": float(res.score)
            }
            for res in results
        ]

    def clear(self):
        """Resets collection for clean test runs."""
        self.client.delete_collection(collection_name=self.collection_name)
        self._ensure_collection()
