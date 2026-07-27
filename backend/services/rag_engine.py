from typing import Dict, Any, List
from sqlalchemy.orm import Session
from backend.services.permission_service import PermissionResolver, ResolvedPermissionSet
from backend.core.vector_store import QdrantVectorStore
from backend.core.llm_client import llm_client
from backend.services.audit_service import AuditLogger

class SentinelRAGEngine:
    def __init__(self, vector_store: QdrantVectorStore):
        self.vector_store = vector_store

    def ask(self, db: Session, user_id: str, query: str, top_k: int = 5) -> Dict[str, Any]:
        """
        Main execution flow: Permission Resolution -> Filtered ANN Search -> Refusal/Answer -> Hash Audit Log.
        """
        # Step 1: Resolve Permission Set (Fail-closed fallback on missing/invalid user)
        perms: ResolvedPermissionSet = PermissionResolver.resolve_permissions(db, user_id)

        # Step 2: Execute Filtered ANN Search inside Qdrant
        permitted_chunks = self.vector_store.search_permissioned(query, perms, top_k=top_k)

        # Calculate denied chunks count by comparing with naive search
        naive_chunks = self.vector_store.search_naive(query, top_k=top_k)
        permitted_chunk_ids = {c["chunk_id"] for c in permitted_chunks}
        chunks_denied_count = sum(1 for c in naive_chunks if c["chunk_id"] not in permitted_chunk_ids)

        # Step 3: Zero-Result Refusal (Zero Existence Confirmation Leak)
        if not permitted_chunks:
            answer = "I don't have access to information that answers this."
        else:
            # Step 4: Grounded LLM Generation using permitted chunks ONLY
            answer = llm_client.generate_answer(
                query=query,
                context_chunks=permitted_chunks,
                user_role=perms.role,
                user_dept=perms.department
            )

        # Step 5: Hash-Chained Audit Logging
        retrieved_ids = [c["chunk_id"] for c in permitted_chunks]
        audit_entry = AuditLogger.log_access(
            db=db,
            user_id=perms.user_id,
            query=query,
            resolved_permissions=perms.model_dump(),
            chunks_retrieved=retrieved_ids,
            chunks_denied_count=chunks_denied_count,
            answer=answer
        )

        return {
            "query": query,
            "answer": answer,
            "resolved_permissions": perms.model_dump(),
            "retrieved_chunks": permitted_chunks,
            "chunks_denied_count": chunks_denied_count,
            "audit_log_id": audit_entry.log_id,
            "audit_hash": audit_entry.this_hash
        }

    def compare_naive_vs_permissioned(self, db: Session, user_id: str, query: str, top_k: int = 5) -> Dict[str, Any]:
        """
        Debug endpoint helper: side-by-side comparison of Naive RAG vs SentinelRAG.
        Explicitly demonstrates which sensitive chunks naive RAG would have leaked!
        """
        perms = PermissionResolver.resolve_permissions(db, user_id)
        
        # Filtered retrieval
        permissioned_chunks = self.vector_store.search_permissioned(query, perms, top_k=top_k)
        permissioned_ids = {c["chunk_id"] for c in permissioned_chunks}

        # Naive retrieval (Unfiltered)
        naive_chunks = self.vector_store.search_naive(query, top_k=top_k)

        leaked_chunks = [c for c in naive_chunks if c["chunk_id"] not in permissioned_ids]

        naive_answer = llm_client.generate_answer(query, naive_chunks, perms.role, perms.department)
        sentinel_answer = "I don't have access to information that answers this." if not permissioned_chunks else llm_client.generate_answer(query, permissioned_chunks, perms.role, perms.department)

        return {
            "user_id": user_id,
            "query": query,
            "resolved_permissions": perms.model_dump(),
            "naive_rag": {
                "retrieved_chunks": naive_chunks,
                "answer": naive_answer,
                "leaked_chunks_count": len(leaked_chunks),
                "leaked_chunks": leaked_chunks
            },
            "sentinel_rag": {
                "retrieved_chunks": permissioned_chunks,
                "answer": sentinel_answer,
                "security_status": "SECURE — Filtered ANN Enforced"
            }
        }
