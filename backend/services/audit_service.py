import hashlib
import json
import threading
import uuid
import datetime
from typing import List, Dict, Any
from sqlalchemy.orm import Session
from backend.database.models import AccessAuditLogModel

# Serializes the read-last-hash -> compute -> append critical section so concurrent
# requests (FastAPI runs sync endpoints in a threadpool) can't both read the same
# prev_hash and fork the chain. This only guarantees correctness within a single
# process — a multi-worker deployment needs a DB-native equivalent instead, e.g.
# Postgres `pg_advisory_xact_lock` or `SELECT ... FOR UPDATE` against a dedicated
# chain-head row.
_chain_lock = threading.Lock()

class AuditLogger:
    @staticmethod
    def _compute_hash(
        prev_hash: str,
        user_id: str,
        query: str,
        chunks_retrieved: List[str],
        answer: str,
        timestamp_str: str,
        guardrail_report: Dict[str, Any] = None
    ) -> str:
        data_str = (
            f"{prev_hash}|{user_id}|{query}|{json.dumps(sorted(chunks_retrieved))}|{answer}|"
            f"{timestamp_str}|{json.dumps(guardrail_report or {}, sort_keys=True)}"
        )
        return hashlib.sha256(data_str.encode("utf-8")).hexdigest()

    @classmethod
    def log_access(
        cls,
        db: Session,
        user_id: str,
        query: str,
        resolved_permissions: Dict[str, Any],
        chunks_retrieved: List[str],
        chunks_denied_count: int,
        answer: str,
        guardrail_report: Dict[str, Any] = None
    ) -> AccessAuditLogModel:
        """
        Creates an append-only, cryptographic hash-chained audit log entry.
        guardrail_report (input_safety / groundedness / pii_leak verdicts) is
        folded into the hash so a tampered guardrail record breaks the chain
        exactly like a tampered answer or chunk list would.
        """
        guardrail_report = guardrail_report or {}
        with _chain_lock:
            # Fetch the most recent audit entry to get prev_hash
            last_log = db.query(AccessAuditLogModel).order_by(AccessAuditLogModel.timestamp.desc()).first()
            prev_hash = last_log.this_hash if last_log else "GENESIS_HASH_SENTINEL_RAG_0000000000000000"

            log_id = f"log_{uuid.uuid4().hex[:12]}"
            now = datetime.datetime.utcnow()
            now_str = now.isoformat()

            this_hash = cls._compute_hash(
                prev_hash=prev_hash,
                user_id=user_id,
                query=query,
                chunks_retrieved=chunks_retrieved,
                answer=answer,
                timestamp_str=now_str,
                guardrail_report=guardrail_report
            )

            audit_entry = AccessAuditLogModel(
                log_id=log_id,
                user_id=user_id,
                query=query,
                resolved_permission_set=resolved_permissions,
                chunks_retrieved=chunks_retrieved,
                chunks_denied_count=chunks_denied_count,
                answer=answer,
                guardrail_report=guardrail_report,
                timestamp=now,
                prev_hash=prev_hash,
                this_hash=this_hash
            )

            db.add(audit_entry)
            db.commit()
            db.refresh(audit_entry)

        return audit_entry

    @classmethod
    def verify_chain_integrity(cls, db: Session) -> Dict[str, Any]:
        """
        Verifies cryptographic integrity of the entire access audit log chain.
        """
        logs = db.query(AccessAuditLogModel).order_by(AccessAuditLogModel.timestamp.asc()).all()
        if not logs:
            return {"valid": True, "entries_checked": 0, "message": "Log chain empty."}

        expected_prev = "GENESIS_HASH_SENTINEL_RAG_0000000000000000"
        for idx, entry in enumerate(logs):
            if entry.prev_hash != expected_prev:
                return {
                    "valid": False,
                    "tampered_index": idx,
                    "log_id": entry.log_id,
                    "reason": f"prev_hash mismatch at index {idx}. Expected {expected_prev}, got {entry.prev_hash}"
                }
            
            recomputed = cls._compute_hash(
                prev_hash=entry.prev_hash,
                user_id=entry.user_id,
                query=entry.query,
                chunks_retrieved=entry.chunks_retrieved,
                answer=entry.answer,
                timestamp_str=entry.timestamp.isoformat(),
                guardrail_report=entry.guardrail_report
            )

            if recomputed != entry.this_hash:
                return {
                    "valid": False,
                    "tampered_index": idx,
                    "log_id": entry.log_id,
                    "reason": f"this_hash tampering detected at index {idx}."
                }

            expected_prev = entry.this_hash

        return {"valid": True, "entries_checked": len(logs), "message": "All log hashes cryptographically verified."}
