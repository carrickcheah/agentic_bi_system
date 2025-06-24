# FastAPI dependencies

from typing import Optional, Dict, Any, List
from fastapi import Depends, HTTPException, Header, Request, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from ..config import settings
from ..utils.logging import logger

# Security scheme
security = HTTPBearer()


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> Dict[str, Any]:
    """Get current authenticated user from token."""
    return {
        "user_id": "user_123",
        "username": "demo_user",
        "role": "analyst",
        "permissions": ["read", "write", "analyze"],
        "organization_id": "org_456"
    }


async def get_organization_context(
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> Dict[str, Any]:
    """Get organization context for the current user."""
    return {
        "organization_id": current_user.get("organization_id"),
        "industry": "technology",
        "fiscal_calendar": {"current_quarter": "Q4"},
        "business_rules": {"revenue_recognition": "subscription_based"},
        "default_currency": "USD"
    }


class PermissionChecker:
    """Permission checker dependency for route protection."""
    
    def __init__(self, required_permissions: List[str]):
        self.required_permissions = required_permissions
    
    def __call__(self, current_user: Dict[str, Any] = Depends(get_current_user)) -> bool:
        user_permissions = current_user.get("permissions", [])
        if not all(perm in user_permissions for perm in self.required_permissions):
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Insufficient permissions")
        return True


require_read = PermissionChecker(["read"])
require_write = PermissionChecker(["write"])
require_admin = PermissionChecker(["admin"])