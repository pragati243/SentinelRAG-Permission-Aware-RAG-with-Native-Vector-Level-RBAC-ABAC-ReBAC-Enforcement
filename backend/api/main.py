import os
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, Depends, HTTPException, Query, Header
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from backend.config import settings
from backend.database.db import get_db, init_db
from backend.database.models import UserModel, ProjectStaffingModel, AccessAuditLogModel
from backend.core.vector_store import QdrantVectorStore
from backend.database.seed_data import seed_database
from backend.services.permission_service import PermissionResolver
from backend.core.ingestion import DocumentIngestor
from backend.services.rag_engine import SentinelRAGEngine
from backend.services.audit_service import AuditLogger
from backend.eval.red_team import RedTeamEvaluator

# Initialize core services
vector_store = QdrantVectorStore()
rag_engine = SentinelRAGEngine(vector_store)
red_team_evaluator = RedTeamEvaluator(rag_engine)

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="SentinelRAG — Permission-Aware RAG with RBAC/ABAC/ReBAC Native Vector Search Enforcement"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def on_startup():
    init_db()
    # Seed database and vector store with multi-department corpus on startup
    seed_database(vector_store)

# Request / Response Schemas
class AskRequest(BaseModel):
    user_id: str
    query: str
    top_k: int = Field(default=5, ge=1, le=20)

class IngestRequest(BaseModel):
    title: str
    content: str
    doc_type: str = "policy"
    owning_department: str = "HR"
    sensitivity_tier: str = "internal"
    required_clearance_level: int = 2
    allowed_roles: Optional[List[str]] = None
    project_scope: Optional[str] = None

# API Endpoints
@app.post("/ask", summary="Main Permission-Aware RAG Query Endpoint")
def ask_question(req: AskRequest, db: Session = Depends(get_db)):
    """
    Main RAG query endpoint enforcing native vector-level payload filtering.
    """
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query text cannot be empty.")
    
    result = rag_engine.ask(db=db, user_id=req.user_id, query=req.query, top_k=req.top_k)
    return result

@app.get("/debug/naive-vs-permissioned", summary="Side-by-Side Data Leakage Comparison Debugger")
def compare_naive_vs_permissioned(
    user_id: str = Query(..., description="Identity user_id to test"),
    query: str = Query(..., description="Search query string"),
    top_k: int = Query(5, ge=1, le=20),
    db: Session = Depends(get_db)
):
    """
    Side-by-side comparative endpoint comparing naive top-k vector search vs SentinelRAG filtered ANN.
    Demonstrates exact sensitive chunks naive RAG would have leaked!
    """
    return rag_engine.compare_naive_vs_permissioned(db=db, user_id=user_id, query=query, top_k=top_k)

@app.get("/debug/resolved-permissions/{user_id}", summary="Inspect Resolved Effective Authorization Grants")
def get_resolved_permissions(user_id: str, db: Session = Depends(get_db)):
    """
    Inspects fully resolved RBAC, ABAC, and ReBAC effective permissions for a given user.
    """
    perms = PermissionResolver.resolve_permissions(db, user_id)
    return perms.model_dump()

@app.post("/ingest", summary="Ingest Document with Metadata Tagging")
def ingest_document(req: IngestRequest, db: Session = Depends(get_db)):
    """
    Ingests a document with denormalized authorization metadata tagged onto vector payloads.
    """
    ingestor = DocumentIngestor(vector_store)
    result = ingestor.ingest_document(
        db=db,
        title=req.title,
        content=req.content,
        doc_type=req.doc_type,
        owning_department=req.owning_department,
        sensitivity_tier=req.sensitivity_tier,
        required_clearance_level=req.required_clearance_level,
        allowed_roles=req.allowed_roles,
        project_scope=req.project_scope
    )
    return result

@app.get("/audit-log", summary="Access-Controlled Immutable Audit Log Trail")
def get_audit_log(
    x_user_id: str = Header("user_vp_ops_01", alias="X-User-Id"),
    limit: int = Query(50, ge=1, le=200),
    db: Session = Depends(get_db)
):
    """
    Returns cryptographic hash-chained audit logs.
    Restricted to VP / Security_Auditor role.
    """
    perms = PermissionResolver.resolve_permissions(db, x_user_id)
    if perms.role not in ["VP", "Security_Auditor", "Admin"]:
        raise HTTPException(status_code=403, detail="Access Denied: Security Auditor role required to view access audit logs.")

    logs = db.query(AccessAuditLogModel).order_by(AccessAuditLogModel.timestamp.desc()).limit(limit).all()
    integrity = AuditLogger.verify_chain_integrity(db)

    return {
        "integrity_check": integrity,
        "total_logs": len(logs),
        "audit_logs": [
            {
                "log_id": l.log_id,
                "user_id": l.user_id,
                "query": l.query,
                "resolved_permission_set": l.resolved_permission_set,
                "chunks_retrieved_count": len(l.chunks_retrieved),
                "chunks_denied_count": l.chunks_denied_count,
                "answer": l.answer,
                "timestamp": l.timestamp.isoformat(),
                "prev_hash": l.prev_hash,
                "this_hash": l.this_hash
            }
            for l in logs
        ]
    }

@app.get("/eval/run", summary="Trigger Red-Team Security Evaluation Battery")
def run_red_team_eval(db: Session = Depends(get_db)):
    """
    Executes Red-Team evaluation suite asserting 0% leak rate and 100% fail-closed compliance.
    """
    return red_team_evaluator.run_eval_suite(db)

@app.get("/admin/users", summary="List All Demo Personas and Identities")
def list_users(db: Session = Depends(get_db)):
    users = db.query(UserModel).all()
    out = []
    for u in users:
        staffings = db.query(ProjectStaffingModel).filter(ProjectStaffingModel.user_id == u.user_id).all()
        out.append({
            "user_id": u.user_id,
            "name": u.name,
            "role": u.role,
            "department": u.department,
            "clearance_level": u.clearance_level,
            "staffed_projects": [s.project_or_account_id for s in staffings]
        })
    return out

# Mount Static Frontend
frontend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../frontend"))
if os.path.exists(frontend_dir):
    app.mount("/static", StaticFiles(directory=frontend_dir), name="static")

    @app.get("/")
    def read_root():
        return FileResponse(os.path.join(frontend_dir, "index.html"))
