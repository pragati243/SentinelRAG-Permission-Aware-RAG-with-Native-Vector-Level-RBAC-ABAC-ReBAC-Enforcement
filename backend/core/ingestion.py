import uuid
import datetime
from typing import List, Dict, Any, Optional
from sqlalchemy.orm import Session
from backend.database.models import DocumentModel, DocumentChunkModel
from backend.core.vector_store import QdrantVectorStore
from backend.core.embedding import embedding_service

class DocumentIngestor:
    def __init__(self, vector_store: QdrantVectorStore):
        self.vector_store = vector_store

    def ingest_document(
        self,
        db: Session,
        title: str,
        content: str,
        doc_type: str,
        owning_department: str,
        sensitivity_tier: str,
        required_clearance_level: int,
        allowed_roles: Optional[List[str]] = None,
        project_scope: Optional[str] = None,
        doc_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Ingests a document, breaks it into section-aware chunks, denormalizes permissions onto chunks,
        persists to SQLite database, and indexes in Qdrant Vector Store.
        """
        doc_id = doc_id or f"doc_{uuid.uuid4().hex[:8]}"
        allowed_roles = allowed_roles or []
        project_scope = project_scope or "none"

        # 1. Create and save DocumentModel in relational DB
        doc_model = DocumentModel(
            doc_id=doc_id,
            title=title,
            doc_type=doc_type,
            owning_department=owning_department,
            sensitivity_tier=sensitivity_tier,
            required_clearance_level=required_clearance_level,
            allowed_roles=allowed_roles,
            project_scope=project_scope,
            ingested_at=datetime.datetime.utcnow()
        )
        db.add(doc_model)

        # 2. Section-Aware Chunking
        raw_sections = content.split("\n\n")
        chunk_payloads = []

        for idx, sec_text in enumerate(raw_sections):
            sec_text = sec_text.strip()
            if not sec_text:
                continue

            chunk_id = f"{doc_id}_c{idx+1}"
            section_title = f"Section {idx+1}"

            # Create chunk DB model with denormalized permissions
            chunk_model = DocumentChunkModel(
                chunk_id=chunk_id,
                doc_id=doc_id,
                section=section_title,
                text=sec_text,
                sensitivity_tier=sensitivity_tier,
                owning_department=owning_department,
                required_clearance_level=required_clearance_level,
                allowed_roles=allowed_roles,
                project_scope=project_scope
            )
            db.add(chunk_model)

            # Prepare Qdrant vector payload
            chunk_payloads.append({
                "chunk_id": chunk_id,
                "doc_id": doc_id,
                "section": section_title,
                "text": sec_text,
                "sensitivity_tier": sensitivity_tier,
                "owning_department": owning_department,
                "required_clearance_level": required_clearance_level,
                "allowed_roles": allowed_roles,
                "project_scope": project_scope,
                "vector": embedding_service.embed_text(sec_text)
            })

        db.commit()

        # 3. Upsert payloads into Qdrant Vector Store
        self.vector_store.upsert_chunks(chunk_payloads)

        return {
            "doc_id": doc_id,
            "title": title,
            "chunks_count": len(chunk_payloads),
            "owning_department": owning_department,
            "sensitivity_tier": sensitivity_tier,
            "required_clearance_level": required_clearance_level,
            "project_scope": project_scope
        }
