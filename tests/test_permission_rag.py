import pytest
from sqlalchemy.orm import Session
from backend.database.db import SessionLocal, init_db, engine
from backend.database.models import Base
from backend.core.vector_store import QdrantVectorStore
from backend.database.seed_data import seed_database
from backend.services.permission_service import PermissionResolver
from backend.services.rag_engine import SentinelRAGEngine
from backend.services.audit_service import AuditLogger
from backend.eval.red_team import RedTeamEvaluator

@pytest.fixture(scope="module")
def setup_test_env():
    init_db()
    vstore = QdrantVectorStore(location=":memory:", collection_name="test_sentinel_chunks")
    seed_database(vstore)
    db = SessionLocal()
    engine_inst = SentinelRAGEngine(vstore)
    yield db, engine_inst, vstore
    db.close()

def test_permission_resolver(setup_test_env):
    db, _, _ = setup_test_env
    
    # 1. Test HR Manager Resolution
    hr_perms = PermissionResolver.resolve_permissions(db, "user_hr_mgr_01")
    assert hr_perms.role == "HR_Manager"
    assert hr_perms.department == "HR"
    assert hr_perms.clearance_level == 3
    assert "confidential" in hr_perms.allowed_tiers
    assert hr_perms.is_fallback is False

    # 2. Test ReBAC Staffing for Paralegal Alpha vs Beta
    alpha_perms = PermissionResolver.resolve_permissions(db, "user_paralegal_alpha")
    assert "Project_Alpha" in alpha_perms.staffed_projects
    assert "Project_Beta" not in alpha_perms.staffed_projects

    # 3. Fail-Closed Fallback on Unknown User
    invalid_perms = PermissionResolver.resolve_permissions(db, "unknown_ghost_user")
    assert invalid_perms.role == "Anonymous"
    assert invalid_perms.clearance_level == 1
    assert invalid_perms.allowed_tiers == ["public"]
    assert invalid_perms.is_fallback is True

def test_sentinel_rag_access_enforcement(setup_test_env):
    db, engine_inst, _ = setup_test_env

    # Support Agent asking about Executive Severance (RESTRICTED)
    res_deny = engine_inst.ask(db, "user_support_01", "What is the executive severance package?")
    assert res_deny["answer"] == "I don't have access to information that answers this."
    assert len(res_deny["retrieved_chunks"]) == 0
    assert res_deny["chunks_denied_count"] > 0

    # HR Manager asking about Executive Severance (ALLOWED)
    res_allow = engine_inst.ask(db, "user_hr_mgr_01", "What is the executive severance package?")
    assert res_allow["answer"] != "I don't have access to information that answers this."
    assert len(res_allow["retrieved_chunks"]) > 0
    assert "salary continuation" in res_allow["answer"].lower() or "executive severance" in res_allow["answer"].lower()

def test_side_by_side_comparison(setup_test_env):
    db, engine_inst, _ = setup_test_env

    comp = engine_inst.compare_naive_vs_permissioned(db, "user_support_01", "Show executive compensation and severance details")
    
    # Naive RAG leaks sensitive restricted chunks
    assert comp["naive_rag"]["leaked_chunks_count"] > 0
    assert len(comp["naive_rag"]["retrieved_chunks"]) > 0

    # SentinelRAG safely prevents any leakage inside Qdrant vector store
    assert comp["sentinel_rag"]["answer"] == "I don't have access to information that answers this."
    assert len(comp["sentinel_rag"]["retrieved_chunks"]) == 0

def test_audit_hash_chain_integrity(setup_test_env):
    db, engine_inst, _ = setup_test_env

    # Run queries to populate audit log entries
    engine_inst.ask(db, "user_support_01", "What is parental leave?")
    engine_inst.ask(db, "user_fin_analyst_01", "What is Q3 margin forecast?")

    integrity = AuditLogger.verify_chain_integrity(db)
    assert integrity["valid"] is True
    assert integrity["entries_checked"] >= 2

def test_red_team_eval_zero_leak_gate(setup_test_env):
    db, engine_inst, _ = setup_test_env

    evaluator = RedTeamEvaluator(engine_inst)
    report = evaluator.run_eval_suite(db)

    summary = report["summary"]
    assert summary["leak_rate_percent"] == 0.0, f"Leak rate failed! Detected {summary['leak_rate_percent']}%"
    assert summary["fail_closed_compliance_percent"] == 100.0
    assert summary["security_gate_status"] == "PASSED"
