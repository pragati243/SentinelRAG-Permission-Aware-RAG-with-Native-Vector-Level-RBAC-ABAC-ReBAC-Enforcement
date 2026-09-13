import datetime
import logging
from typing import Optional

import jwt
from fastapi import Header, HTTPException
from pydantic import BaseModel

from backend.config import settings

logger = logging.getLogger(__name__)

JWT_ALGORITHM = "HS256"


class AuthenticatedIdentity(BaseModel):
    """
    Verified request identity. The `user_id` here is derived exclusively from a
    signature- and expiry-checked JWT — never from a client-supplied body/header
    field — so it cannot be spoofed by an unauthenticated caller.
    """
    user_id: str


def create_access_token(user_id: str) -> str:
    """
    Issues a signed identity token for `user_id`.

    Only `sub` (subject) is embedded — role/department/clearance are deliberately
    NOT baked into the token. Authorization is re-resolved fresh from the database
    via PermissionResolver on every request, so a role change takes effect
    immediately instead of waiting for token expiry.
    """
    now = datetime.datetime.utcnow()
    payload = {
        "sub": user_id,
        "iat": now,
        "exp": now + datetime.timedelta(minutes=settings.JWT_EXPIRY_MINUTES),
    }
    return jwt.encode(payload, settings.JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)


def get_current_identity(authorization: Optional[str] = Header(default=None)) -> AuthenticatedIdentity:
    """
    FastAPI dependency enforcing the authentication trust boundary.

    Raises 401 on a missing, malformed, expired, or tampered token. This is what
    replaces trusting a raw client-supplied `user_id` field: nothing downstream
    of this dependency should ever read identity from anywhere else.
    """
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=401,
            detail="Missing bearer token. Call POST /auth/token to obtain one."
        )

    token = authorization[len("Bearer "):].strip()
    try:
        claims = jwt.decode(token, settings.JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired.")
    except jwt.InvalidTokenError:
        logger.warning("Rejected invalid or tampered bearer token.")
        raise HTTPException(status_code=401, detail="Invalid authentication token.")

    user_id = claims.get("sub")
    if not user_id:
        raise HTTPException(status_code=401, detail="Malformed token: missing subject claim.")

    return AuthenticatedIdentity(user_id=user_id)
