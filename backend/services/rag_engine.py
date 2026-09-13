from typing import Dict, Any, List
from sqlalchemy.orm import Session
from backend.services.permission_service import PermissionResolver, ResolvedPermissionSet
from backend.core.vector_store import QdrantVectorStore
from backend.core.llm_client import llm_client
from backend.services.audit_service import AuditLogger
from backend.services.guardrail_service import check_input_safety, check_groundedness, check_pii_leak
from backend.core.observability import trace_span

REFUSAL_MESSAGE = "I don't have access to information that answers this."

class SentinelRAGEngine:
    def __init__(self, vector_store: QdrantVectorStore):
        self.vector_store = vector_store

    def ask(self, db: Session, user_id: str, query: str, top_k: int = 5) -> Dict[str, Any]:
        """
        Main execution flow: Permission Resolution -> Input Guardrail -> Filtered
        ANN Search -> Refusal/Generation -> Output Guardrails -> Hash Audit Log.
        Traced end-to-end as a Langfuse span tree (a no-op when Langfuse isn't
        configured) so retrieval/generation/guardrail latency and LLM token
        cost are all visible under one root trace per request.
        """
        with trace_span("sentinel_rag.ask", input={"user_id": user_id, "query": query}) as root:
            # Step 1: Resolve Permission Set (Fail-closed fallback on missing/invalid user)
            with trace_span("permission_resolution") as span:
                perms: ResolvedPermissionSet = PermissionResolver.resolve_permissions(db, user_id)
                span.update(output=perms.model_dump())

            guardrail_report: Dict[str, Any] = {}

            # Step 2: Input guardrail (defense-in-depth -- RBAC/ABAC filtering below
            # is the actual security boundary and holds regardless of this check's
            # outcome). Uses the SAME refusal text as a permission denial: giving an
            # attacker a distinct message for "blocked injection attempt" vs. "no
            # permission" would itself leak which defense fired. The real reason is
            # only visible internally, in guardrail_report.
            input_verdict = check_input_safety(query)
            guardrail_report["input_safety"] = input_verdict.model_dump()

            if input_verdict.flagged:
                audit_entry = AuditLogger.log_access(
                    db=db,
                    user_id=perms.user_id,
                    query=query,
                    resolved_permissions=perms.model_dump(),
                    chunks_retrieved=[],
                    chunks_denied_count=0,
                    answer=REFUSAL_MESSAGE,
                    guardrail_report=guardrail_report
                )
                result = {
                    "query": query,
                    "answer": REFUSAL_MESSAGE,
                    "resolved_permissions": perms.model_dump(),
                    "retrieved_chunks": [],
                    "chunks_denied_count": 0,
                    "audit_log_id": audit_entry.log_id,
                    "audit_hash": audit_entry.this_hash,
                    "guardrail_report": guardrail_report
                }
                root.update(output={"answer": REFUSAL_MESSAGE, "guardrail_report": guardrail_report})
                return result

            # Step 3: Execute Filtered ANN Search inside Qdrant
            with trace_span("vector_search.permissioned") as span:
                permitted_chunks = self.vector_store.search_permissioned(query, perms, top_k=top_k)
                span.update(output={"chunk_ids": [c["chunk_id"] for c in permitted_chunks], "count": len(permitted_chunks)})

            # Calculate denied chunks count by comparing with naive search
            with trace_span("vector_search.naive") as span:
                naive_chunks = self.vector_store.search_naive(query, top_k=top_k)
                span.update(output={"count": len(naive_chunks)})

            permitted_chunk_ids = {c["chunk_id"] for c in permitted_chunks}
            chunks_denied_count = sum(1 for c in naive_chunks if c["chunk_id"] not in permitted_chunk_ids)

            # Step 4: Zero-Result Refusal (Zero Existence Confirmation Leak)
            if not permitted_chunks:
                answer = REFUSAL_MESSAGE
            else:
                # Step 5: Grounded LLM Generation using permitted chunks ONLY
                with trace_span(
                    "generation", as_type="generation",
                    input={"query": query, "context_chunk_ids": [c["chunk_id"] for c in permitted_chunks]}
                ) as gen:
                    answer = llm_client.generate_answer(
                        query=query,
                        context_chunks=permitted_chunks,
                        user_role=perms.role,
                        user_dept=perms.department
                    )
                    # Read back which provider actually answered instead of
                    # re-deriving llm_client's OpenAI -> Groq -> local
                    # precedence a second time here.
                    gen.update(output=answer, model=llm_client.last_model_used)

                # Step 6: Output guardrails. Groundedness catches both hallucination
                # and "right permission, wrong document" (a permitted-but-irrelevant
                # chunk being confidently narrated). PII is a local regex scan, no
                # LLM call needed.
                groundedness_verdict = check_groundedness(query, answer, permitted_chunks)
                guardrail_report["groundedness"] = groundedness_verdict.model_dump()
                if groundedness_verdict.flagged:
                    answer = REFUSAL_MESSAGE

                if answer != REFUSAL_MESSAGE:
                    pii_verdict = check_pii_leak(answer)
                    guardrail_report["pii_leak"] = pii_verdict.model_dump()
                    if pii_verdict.flagged:
                        answer = "The generated answer was withheld because it may contain personally identifiable information."

            # Step 7: Hash-Chained Audit Logging
            retrieved_ids = [c["chunk_id"] for c in permitted_chunks]
            audit_entry = AuditLogger.log_access(
                db=db,
                user_id=perms.user_id,
                query=query,
                resolved_permissions=perms.model_dump(),
                chunks_retrieved=retrieved_ids,
                chunks_denied_count=chunks_denied_count,
                answer=answer,
                guardrail_report=guardrail_report
            )

            result = {
                "query": query,
                "answer": answer,
                "resolved_permissions": perms.model_dump(),
                "retrieved_chunks": permitted_chunks,
                "chunks_denied_count": chunks_denied_count,
                "audit_log_id": audit_entry.log_id,
                "audit_hash": audit_entry.this_hash,
                "guardrail_report": guardrail_report
            }
            root.update(output={"answer": answer, "chunks_denied_count": chunks_denied_count, "guardrail_report": guardrail_report})
            return result

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
