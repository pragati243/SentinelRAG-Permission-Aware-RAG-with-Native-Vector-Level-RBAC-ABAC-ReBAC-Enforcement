from sqlalchemy.orm import Session
from backend.database.models import UserModel, ProjectStaffingModel, Base
from backend.database.db import engine, SessionLocal
from backend.core.vector_store import QdrantVectorStore
from backend.core.ingestion import DocumentIngestor

def seed_database(vector_store: QdrantVectorStore):
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)
    vector_store.clear()

    db: Session = SessionLocal()
    try:
        # 1. Seed Users
        users = [
            UserModel(user_id="user_support_01", name="Sam Support", role="Employee", department="Support", clearance_level=1),
            UserModel(user_id="user_hr_mgr_01", name="Hannah HR Manager", role="HR_Manager", department="HR", clearance_level=3),
            UserModel(user_id="user_fin_analyst_01", name="Frank Finance", role="Finance_Analyst", department="Finance", clearance_level=2),
            UserModel(user_id="user_vp_ops_01", name="Victoria VP", role="VP", department="Executive", clearance_level=4),
            UserModel(user_id="user_paralegal_alpha", name="Alice Paralegal (Alpha Staffed)", role="Legal_Paralegal", department="Legal", clearance_level=2),
            UserModel(user_id="user_paralegal_beta", name="Bob Paralegal (Beta Staffed)", role="Legal_Paralegal", department="Legal", clearance_level=2),
        ]
        db.add_all(users)

        # 2. Seed ReBAC Project Staffings
        staffings = [
            ProjectStaffingModel(user_id="user_paralegal_alpha", project_or_account_id="Project_Alpha"),
            ProjectStaffingModel(user_id="user_paralegal_beta", project_or_account_id="Project_Beta"),
            ProjectStaffingModel(user_id="user_vp_ops_01", project_or_account_id="Project_Alpha"),
            ProjectStaffingModel(user_id="user_vp_ops_01", project_or_account_id="Project_Beta"),
        ]
        db.add_all(staffings)
        db.commit()

        # 3. Ingest Multi-Department Test Corpus with Metadata
        ingestor = DocumentIngestor(vector_store)

        corpus_docs = [
            {
                "doc_id": "doc_hr_exec_comp",
                "title": "HR Executive Compensation & Severance Policy",
                "content": "Executive Severance Package Details:\nExecutive severance grants 12 months salary continuation plus immediate unvested equity acceleration for VP level and above.\n\nExecutive Bonus Structure:\nExecutive annual bonuses are capped at 45% of base salary tied to corporate EBITDA milestones. HR Manager approval required.",
                "doc_type": "policy",
                "owning_department": "HR",
                "sensitivity_tier": "restricted",
                "required_clearance_level": 3,
                "allowed_roles": ["HR_Manager", "VP"],
                "project_scope": "none"
            },
            {
                "doc_id": "doc_fin_q3_forecast",
                "title": "Finance Q3 Customer Churn & Margin Forecast",
                "content": "Q3 Churn Analysis:\nQ3 customer churn increased by 3.2% in the enterprise tier primarily due to competitor pricing.\n\nFinancial Margin Forecast:\nNet operating margin forecast is projected at $4.2M for fiscal year end with strict spending freezes in place.",
                "doc_type": "report",
                "owning_department": "Finance",
                "sensitivity_tier": "confidential",
                "required_clearance_level": 2,
                "allowed_roles": ["Finance_Analyst", "VP"],
                "project_scope": "none"
            },
            {
                "doc_id": "doc_legal_nda_alpha",
                "title": "Legal Confidential Client NDA — Project Alpha",
                "content": "Project Alpha Client Terms:\nProject Alpha NDA terms stipulate strict non-disclosure of patent application claims with client ACME Corp until Q4 product launch.\n\nExclusivity Covenant:\nACME Corp maintains exclusive marketing rights across North American jurisdictions.",
                "doc_type": "contract",
                "owning_department": "Legal",
                "sensitivity_tier": "confidential",
                "required_clearance_level": 2,
                "allowed_roles": ["Legal_Paralegal", "VP"],
                "project_scope": "Project_Alpha"
            },
            {
                "doc_id": "doc_legal_nda_beta",
                "title": "Legal Confidential Client NDA — Project Beta",
                "content": "Project Beta Settlement Terms:\nProject Beta settlement contract binds both parties to confidential arbitration with maximum liability capped at $500,000.\n\nDispute Resolution Clause:\nAll claims must be submitted to binding arbitration in Delaware within 30 days.",
                "doc_type": "contract",
                "owning_department": "Legal",
                "sensitivity_tier": "confidential",
                "required_clearance_level": 2,
                "allowed_roles": ["Legal_Paralegal", "VP"],
                "project_scope": "Project_Beta"
            },
            {
                "doc_id": "doc_eng_arch_guide",
                "title": "Engineering Vector Architecture Guide",
                "content": "SentinelRAG Microservice Architecture:\nSentinelRAG vector search microservice runs on Qdrant using payload filtering for HNSW graph traversal and Redis caching for session state.\n\nPayload Filter Specification:\nPayload filters are denormalized onto every chunk record to execute permission predicates inside the vector index traversal.",
                "doc_type": "technical",
                "owning_department": "Engineering",
                "sensitivity_tier": "internal",
                "required_clearance_level": 2,
                "allowed_roles": [],
                "project_scope": "none"
            },
            {
                "doc_id": "doc_public_handbook",
                "title": "General Employee Handbook & Paid Time Off",
                "content": "Parental Leave Policy:\nStandard parental leave covers 12 weeks of paid leave for primary caregivers.\n\nOffice Hours & Flexible Work:\nStandard office hours are 9 AM to 5 PM local time with flexible hybrid remote work options available upon manager approval.",
                "doc_type": "policy",
                "owning_department": "HR",
                "sensitivity_tier": "public",
                "required_clearance_level": 1,
                "allowed_roles": [],
                "project_scope": "none"
            }
        ]

        for doc in corpus_docs:
            ingestor.ingest_document(
                db=db,
                doc_id=doc["doc_id"],
                title=doc["title"],
                content=doc["content"],
                doc_type=doc["doc_type"],
                owning_department=doc["owning_department"],
                sensitivity_tier=doc["sensitivity_tier"],
                required_clearance_level=doc["required_clearance_level"],
                allowed_roles=doc["allowed_roles"],
                project_scope=doc["project_scope"]
            )

        print("Database and Vector Store successfully seeded with multi-department corpus!")
    finally:
        db.close()
