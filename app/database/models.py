"""
Database Models and Data Classes

Defines data structures for investigations, sessions, and query results.
"""

from datetime import datetime
from enum import Enum
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from pydantic import BaseModel


class InvestigationStatus(str, Enum):
    """Investigation status enumeration."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class Investigation:
    """Investigation data model."""
    id: str
    query: str
    user_id: Optional[str]
    status: InvestigationStatus
    context: Dict[str, Any]
    progress: Optional[Dict[str, Any]] = None
    results: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    created_at: datetime = None
    updated_at: datetime = None


@dataclass
class QueryResult:
    """SQL query execution result."""
    success: bool
    data: Optional[List[Dict[str, Any]]] = None
    columns: Optional[List[str]] = None
    row_count: int = 0
    execution_time: float = 0.0
    error: Optional[str] = None


@dataclass
class TableSchema:
    """Database table schema information."""
    name: str
    columns: List['ColumnInfo']
    primary_keys: List[str] = None
    foreign_keys: List['ForeignKeyInfo'] = None
    indexes: List[str] = None


@dataclass
class ColumnInfo:
    """Database column information."""
    name: str
    data_type: str
    is_nullable: bool
    default_value: Optional[str] = None
    max_length: Optional[int] = None
    precision: Optional[int] = None
    scale: Optional[int] = None


@dataclass
class ForeignKeyInfo:
    """Foreign key relationship information."""
    column: str
    referenced_table: str
    referenced_column: str


class ValidationResult(BaseModel):
    """SQL query validation result."""
    is_safe: bool
    syntax_valid: bool
    security_issues: List[str] = []
    suggestions: List[str] = []
    reason: Optional[str] = None


@dataclass
class SessionData:
    """User session data."""
    session_id: str
    user_id: str
    context: Dict[str, Any]
    created_at: datetime
    updated_at: datetime


@dataclass
class CachedResult:
    """Cached query result."""
    key: str
    data: Dict[str, Any]
    created_at: datetime
    expires_at: datetime
    session_id: str