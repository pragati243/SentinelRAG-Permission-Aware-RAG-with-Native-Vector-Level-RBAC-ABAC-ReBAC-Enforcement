import datetime
from typing import List, Optional
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
from backend.database.models import UserModel, ProjectStaffingModel
from backend.config import settings

TIER_MAP = {
    1: ["public"],
    2: ["public", "internal"],
    3: ["public", "internal", "confidential"],
    4: ["public", "internal", "confidential", "restricted"]
}

class ResolvedPermissionSet(BaseModel):
    user_id: str
    user_name: str
    role: str
    department: str
    clearance_level: int
    staffed_projects: List[str] = Field(default_factory=list)
    allowed_tiers: List[str] = Field(default_factory=list)
    is_fallback: bool = False

class PermissionResolver:
    @staticmethod
    def get_fail_closed_permissions(user_id: str = "unknown") -> ResolvedPermissionSet:
        """
        Fail-closed default permission set when identity resolution fails or is unauthenticated.
        Enforces strict 'public' tier access only.
        """
        return ResolvedPermissionSet(
            user_id=user_id,
            user_name="Unauthenticated/Anonymous User",
            role="Anonymous",
            department="Public",
            clearance_level=settings.DEFAULT_PUBLIC_CLEARANCE,
            staffed_projects=[],
            allowed_tiers=TIER_MAP[settings.DEFAULT_PUBLIC_CLEARANCE],
            is_fallback=True
        )

    @classmethod
    def resolve_permissions(cls, db: Session, user_id: str) -> ResolvedPermissionSet:
        """
        Resolves effective RBAC, ABAC, and ReBAC permissions for a given user identity.
        """
        if not user_id:
            return cls.get_fail_closed_permissions("missing_id")

        try:
            user = db.query(UserModel).filter(UserModel.user_id == user_id).first()
            if not user:
                return cls.get_fail_closed_permissions(user_id)

            # ReBAC: Fetch active project/account staffings
            now = datetime.datetime.utcnow()
            staffings = db.query(ProjectStaffingModel).filter(
                ProjectStaffingModel.user_id == user_id,
                (ProjectStaffingModel.staffed_until == None) | (ProjectStaffingModel.staffed_until >= now)
            ).all()

            staffed_projects = [s.project_or_account_id for s in staffings]
            
            # Map clearance level (capped between 1 and 4)
            cleared_lvl = max(1, min(user.clearance_level, 4))
            allowed_tiers = TIER_MAP.get(cleared_lvl, ["public"])

            return ResolvedPermissionSet(
                user_id=user.user_id,
                user_name=user.name,
                role=user.role,
                department=user.department,
                clearance_level=user.clearance_level,
                staffed_projects=staffed_projects,
                allowed_tiers=allowed_tiers,
                is_fallback=False
            )
        except Exception as e:
            # Fail closed on system or query errors
            return cls.get_fail_closed_permissions(user_id)
