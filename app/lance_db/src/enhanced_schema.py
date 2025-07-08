"""
Enhanced SQL Query Schema for LanceDB
Provides data structures for storing enhanced SQL queries with business context.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Any, Optional
from enum import Enum
import uuid


class QueryType(Enum):
    """Types of SQL queries."""
    SELECT = "select"
    INSERT = "insert"
    UPDATE = "update"
    DELETE = "delete"
    CREATE = "create"
    ALTER = "alter"
    DROP = "drop"
    UNKNOWN = "unknown"


class BusinessDomain(Enum):
    """Business domains for queries."""
    SALES = "sales"
    FINANCE = "finance"
    OPERATIONS = "operations"
    MARKETING = "marketing"
    HR = "hr"
    PRODUCTION = "production"
    QUALITY = "quality"
    SUPPLY_CHAIN = "supply_chain"
    CUSTOMER = "customer"
    STRATEGIC = "strategic"
    UNKNOWN = "unknown"


@dataclass
class QueryContent:
    """SQL query content."""
    sql_query: str
    query_type: QueryType = QueryType.SELECT
    business_question: Optional[str] = None
    readable_description: Optional[str] = None


@dataclass
class SemanticContext:
    """Semantic context for the query."""
    business_domain: BusinessDomain = BusinessDomain.UNKNOWN
    intent: Optional[str] = None
    entities: List[str] = field(default_factory=list)
    relationships: List[str] = field(default_factory=list)


@dataclass
class TechnicalMetadata:
    """Technical metadata for the query."""
    database_type: str = "mariadb"
    database_name: Optional[str] = None
    tables_accessed: List[str] = field(default_factory=list)
    complexity_score: float = 0.0


@dataclass
class UserContext:
    """User context information."""
    user_id: str = "system"
    user_role: Optional[str] = None
    department: Optional[str] = None
    access_level: Optional[str] = None


@dataclass
class InvestigationContext:
    """Investigation context."""
    investigation_id: Optional[str] = None
    parent_query_id: Optional[str] = None
    investigation_phase: Optional[str] = None
    hypothesis: Optional[str] = None


@dataclass
class ExecutionResults:
    """Query execution results."""
    execution_status: str = "pending"
    execution_time_ms: Optional[float] = None
    rows_returned: Optional[int] = None
    error_message: Optional[str] = None
    success: bool = False
    execution_count: int = 0
    first_executed_at: Optional[datetime] = None
    last_executed_at: Optional[datetime] = None


@dataclass
class LearningMetadata:
    """Learning and improvement metadata."""
    usage_frequency: int = 0
    user_ratings: List[int] = field(default_factory=list)
    average_rating: Optional[float] = None
    improvement_suggestions: List[str] = field(default_factory=list)
    learned_patterns: List[str] = field(default_factory=list)
    first_used_at: Optional[datetime] = None
    last_used_at: Optional[datetime] = None


@dataclass
class BusinessIntelligence:
    """Business intelligence insights."""
    insights: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    kpis_affected: List[str] = field(default_factory=list)
    business_impact: Optional[str] = None
    strategic_value: Optional[float] = None


@dataclass
class Embeddings:
    """Vector embeddings for the query."""
    query_embedding: Optional[List[float]] = None
    description_embedding: Optional[List[float]] = None
    business_context_embedding: Optional[List[float]] = None
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_created_at: Optional[datetime] = None


@dataclass
class EnhancedSQLQuery:
    """Enhanced SQL query with full context and metadata."""
    _id: str = field(default_factory=lambda: str(uuid.uuid4()))
    query_content: QueryContent = field(default_factory=QueryContent)
    semantic_context: SemanticContext = field(default_factory=SemanticContext)
    technical_metadata: TechnicalMetadata = field(default_factory=TechnicalMetadata)
    user_context: UserContext = field(default_factory=UserContext)
    investigation_context: InvestigationContext = field(default_factory=InvestigationContext)
    execution_results: ExecutionResults = field(default_factory=ExecutionResults)
    learning_metadata: LearningMetadata = field(default_factory=LearningMetadata)
    business_intelligence: BusinessIntelligence = field(default_factory=BusinessIntelligence)
    embeddings: Embeddings = field(default_factory=Embeddings)
    created_at: datetime = field(default_factory=lambda: datetime.now())
    updated_at: datetime = field(default_factory=lambda: datetime.now())
    
    def to_lancedb_record(self) -> Dict[str, Any]:
        """Convert to LanceDB record format."""
        return {
            "_id": self._id,
            "sql_query": self.query_content.sql_query,
            "query_type": self.query_content.query_type.value,
            "business_question": self.query_content.business_question,
            "readable_description": self.query_content.readable_description,
            "business_domain": self.semantic_context.business_domain.value,
            "database": self.technical_metadata.database_name or self.technical_metadata.database_type,
            "user_id": self.user_context.user_id,
            "investigation_id": self.investigation_context.investigation_id,
            "execution_time_ms": self.execution_results.execution_time_ms,
            "rows_returned": self.execution_results.rows_returned,
            "success": self.execution_results.success,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            # Embeddings will be added separately
        }


def create_enhanced_query_from_simple(
    sql_query: str,
    database: str = "mariadb",
    user_id: str = "system",
    business_context: Optional[Dict[str, Any]] = None
) -> EnhancedSQLQuery:
    """Create an enhanced query from simple parameters."""
    enhanced = EnhancedSQLQuery()
    
    # Set query content
    enhanced.query_content.sql_query = sql_query
    enhanced.query_content.query_type = _detect_query_type(sql_query)
    
    if business_context:
        enhanced.query_content.business_question = business_context.get("original_question")
        enhanced.semantic_context.business_domain = _detect_business_domain(
            business_context.get("original_question", "")
        )
    
    # Set technical metadata
    enhanced.technical_metadata.database_type = database
    enhanced.technical_metadata.database_name = database
    
    # Set user context
    enhanced.user_context.user_id = user_id
    
    return enhanced


def validate_enhanced_query(query: EnhancedSQLQuery) -> List[str]:
    """Validate an enhanced query and return list of errors."""
    errors = []
    
    if not query.query_content.sql_query:
        errors.append("SQL query is required")
    
    if not query._id:
        errors.append("Query ID is required")
    
    return errors


def _detect_query_type(sql_query: str) -> QueryType:
    """Detect the type of SQL query."""
    query_lower = sql_query.lower().strip()
    
    if query_lower.startswith("select"):
        return QueryType.SELECT
    elif query_lower.startswith("insert"):
        return QueryType.INSERT
    elif query_lower.startswith("update"):
        return QueryType.UPDATE
    elif query_lower.startswith("delete"):
        return QueryType.DELETE
    elif query_lower.startswith("create"):
        return QueryType.CREATE
    elif query_lower.startswith("alter"):
        return QueryType.ALTER
    elif query_lower.startswith("drop"):
        return QueryType.DROP
    else:
        return QueryType.UNKNOWN


def _detect_business_domain(text: str) -> BusinessDomain:
    """Detect business domain from text."""
    text_lower = text.lower()
    
    domain_keywords = {
        BusinessDomain.SALES: ["sales", "revenue", "customer", "order", "purchase"],
        BusinessDomain.FINANCE: ["finance", "budget", "cost", "expense", "profit"],
        BusinessDomain.OPERATIONS: ["operation", "process", "efficiency", "workflow"],
        BusinessDomain.MARKETING: ["marketing", "campaign", "promotion", "brand"],
        BusinessDomain.HR: ["employee", "staff", "hr", "human resource", "talent"],
        BusinessDomain.PRODUCTION: ["production", "manufacturing", "output", "yield"],
        BusinessDomain.QUALITY: ["quality", "defect", "compliance", "standard"],
        BusinessDomain.SUPPLY_CHAIN: ["supply", "inventory", "logistics", "warehouse"],
        BusinessDomain.CUSTOMER: ["customer", "client", "satisfaction", "service"],
        BusinessDomain.STRATEGIC: ["strategic", "planning", "forecast", "analysis"]
    }
    
    for domain, keywords in domain_keywords.items():
        if any(keyword in text_lower for keyword in keywords):
            return domain
    
    return BusinessDomain.UNKNOWN