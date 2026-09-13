import logging
import os
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, Depends, HTTPException, Query
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
from backend.services.auth_service import AuthenticatedIdentity, create_access_token, get_current_identity

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

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
    query: str
    top_k: int = Field(default=5, ge=1, le=20)

class LoginRequest(BaseModel):
    user_id: str

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
@app.post("/auth/token", summary="Issue Signed Identity Token (Demo IdP Simulation)")
def login(req: LoginRequest, db: Session = Depends(get_db)):
    """
    Demo stand-in for a real login/SSO exchange. A production deployment would
    put a real IdP (Auth0, Keycloak, Azure AD/OIDC) here and verify a password
    or SSO assertion; this endpoint only verifies the persona exists so the demo
    UI can "log in" as any seeded test user.

    What this genuinely fixes: after this point, `user_id` is NEVER read from a
    client-supplied body/header field again — every downstream endpoint derives
    it exclusively from a signed, expiry-checked JWT via get_current_identity.
    """
    user = db.query(UserModel).filter(UserModel.user_id == req.user_id).first()
    if not user:
        raise HTTPException(status_code=401, detail="Unknown identity.")

    token = create_access_token(user_id=user.user_id)
    return {
        "access_token": token,
        "token_type": "bearer",
        "expires_in_minutes": settings.JWT_EXPIRY_MINUTES
    }

@app.post("/ask", summary="Main Permission-Aware RAG Query Endpoint")
def ask_question(
    req: AskRequest,
    identity: AuthenticatedIdentity = Depends(get_current_identity),
    db: Session = Depends(get_db)
):
    """
    Main RAG query endpoint enforcing native vector-level payload filtering.
    Identity comes exclusively from the verified bearer token — a caller can no
    longer claim to be another user by passing a different `user_id` in the body.
    """
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query text cannot be empty.")

    result = rag_engine.ask(db=db, user_id=identity.user_id, query=req.query, top_k=req.top_k)
    return result

@app.get("/debug/naive-vs-permissioned", summary="Side-by-Side Data Leakage Comparison Debugger")
def compare_naive_vs_permissioned(
    query: str = Query(..., description="Search query string"),
    top_k: int = Query(5, ge=1, le=20),
    identity: AuthenticatedIdentity = Depends(get_current_identity),
    db: Session = Depends(get_db)
):
    """
    Side-by-side comparative endpoint comparing naive top-k vector search vs SentinelRAG filtered ANN
    for the authenticated caller's own identity. Demonstrates exact sensitive chunks naive RAG would
    have leaked!
    """
    return rag_engine.compare_naive_vs_permissioned(db=db, user_id=identity.user_id, query=query, top_k=top_k)

@app.get("/debug/resolved-permissions", summary="Inspect Resolved Effective Authorization Grants")
def get_resolved_permissions(
    identity: AuthenticatedIdentity = Depends(get_current_identity),
    db: Session = Depends(get_db)
):
    """
    Inspects fully resolved RBAC, ABAC, and ReBAC effective permissions for the
    authenticated caller. No longer accepts an arbitrary user_id path param —
    that would let any caller inspect (and, worse, implicitly confirm details
    about) another identity's grants without ever proving they are that user.
    """
    perms = PermissionResolver.resolve_permissions(db, identity.user_id)
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
    limit: int = Query(50, ge=1, le=200),
    identity: AuthenticatedIdentity = Depends(get_current_identity),
    db: Session = Depends(get_db)
):
    """
    Returns cryptographic hash-chained audit logs.
    Restricted to VP / Security_Auditor role, resolved from the verified caller's
    own identity — previously this trusted a client-supplied X-User-Id header,
    which meant ANY caller could self-declare themselves as the VP persona.
    """
    perms = PermissionResolver.resolve_permissions(db, identity.user_id)
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
                "guardrail_report": l.guardrail_report,
                "timestamp": l.timestamp.isoformat(),
                "prev_hash": l.prev_hash,
                "this_hash": l.this_hash
            }
            for l in logs
        ]
    }

@app.get("/eval/run", summary="Trigger Red-Team Security Evaluation Battery")
def run_red_team_eval(
    identity: AuthenticatedIdentity = Depends(get_current_identity),
    db: Session = Depends(get_db)
):
    """
    Executes Red-Team evaluation suite asserting 0% leak rate and 100% fail-closed compliance.
    Requires any authenticated caller (prevents anonymous/external abuse of this
    diagnostic battery); it does not read or expose the caller's own data, only
    a fixed internal battery of test personas, so no further role gate is applied
    here. A stricter deployment should additionally restrict this to an
    Admin/Security_Auditor role.
    """
    return red_team_evaluator.run_eval_suite(db)

@app.get("/admin/users", summary="List All Demo Personas and Identities")
def list_users(db: Session = Depends(get_db)):
    # Intentionally public: this is the "pick a demo account to log in as" listing
    # for the persona-switcher UI (analogous to a sandbox environment's account
    # picker) and exposes only demo persona metadata already visible in the UI —
    # never documents, tokens, or audit data. Do not add sensitive fields here.
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
