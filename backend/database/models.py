import datetime
from typing import Optional, List
from sqlalchemy import Column, String, Integer, DateTime, Text, ForeignKey, JSON
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()

class UserModel(Base):
    __tablename__ = "users"
    
    user_id = Column(String, primary_key=True)
    name = Column(String, nullable=False)
    role = Column(String, nullable=False)  # e.g., Employee, HR_Manager, Finance_Analyst, VP, Legal_Paralegal
    department = Column(String, nullable=False)  # e.g., HR, Finance, Legal, Engineering, Support
    clearance_level = Column(Integer, nullable=False, default=1)  # 1: Public, 2: Internal/Confidential, 3: Restricted, 4: Executive
    manager_id = Column(String, nullable=True)

    staffings = relationship("ProjectStaffingModel", back_populates="user", cascade="all, delete-orphan")

class ProjectStaffingModel(Base):
    __tablename__ = "project_staffing"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String, ForeignKey("users.user_id"), nullable=False)
    project_or_account_id = Column(String, nullable=False)
    staffed_from = Column(DateTime, default=datetime.datetime.utcnow)
    staffed_until = Column(DateTime, nullable=True)

    user = relationship("UserModel", back_populates="staffings")

class DocumentModel(Base):
    __tablename__ = "documents"
    
    doc_id = Column(String, primary_key=True)
    title = Column(String, nullable=False)
    doc_type = Column(String, nullable=False)  # e.g., policy, contract, report, technical
    owning_department = Column(String, nullable=False)
    sensitivity_tier = Column(String, nullable=False)  # public, internal, confidential, restricted
    required_clearance_level = Column(Integer, nullable=False, default=1)
    allowed_roles = Column(JSON, nullable=True)  # List[str] override allowed roles
    project_scope = Column(String, nullable=True)  # ReBAC project scope (e.g., Project Alpha)
    ingested_at = Column(DateTime, default=datetime.datetime.utcnow)

    chunks = relationship("DocumentChunkModel", back_populates="document", cascade="all, delete-orphan")

class DocumentChunkModel(Base):
    __tablename__ = "document_chunks"
    
    chunk_id = Column(String, primary_key=True)
    doc_id = Column(String, ForeignKey("documents.doc_id"), nullable=False)
    section = Column(String, nullable=True)
    text = Column(Text, nullable=False)
    
    # Denormalized payload fields for Vector Store alignment
    sensitivity_tier = Column(String, nullable=False)
    owning_department = Column(String, nullable=False)
    required_clearance_level = Column(Integer, nullable=False)
    allowed_roles = Column(JSON, nullable=True)
    project_scope = Column(String, nullable=True)

    document = relationship("DocumentModel", back_populates="chunks")

class AccessAuditLogModel(Base):
    __tablename__ = "access_audit_log"
    
    log_id = Column(String, primary_key=True)
    user_id = Column(String, nullable=False)
    query = Column(Text, nullable=False)
    resolved_permission_set = Column(JSON, nullable=False)  # Snapshot of RBAC/ABAC/ReBAC fields
    chunks_retrieved = Column(JSON, nullable=False)  # List of chunk IDs returned
    chunks_denied_count = Column(Integer, nullable=False, default=0)
    answer = Column(Text, nullable=False)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)
    prev_hash = Column(String, nullable=False)
    this_hash = Column(String, nullable=False)
