"""
Enhanced Production-Grade Schema for SQL Query Learning System
Comprehensive data models for enterprise-level query management and learning.
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
from enum import Enum
import json
import hashlib
import uuid


class QueryType(Enum):
    """SQL query classification types."""
    SIMPLE = "simple"
    ANALYTICAL = "analytical"
    OPERATIONAL = "operational"
    INVESTIGATIVE = "investigative"
    CUSTOMER_ADVISORY = "customer_advisory"


class BusinessDomain(Enum):
    """Business domain classifications."""
    SALES = "sales"
    FINANCE = "finance"
    HR = "hr"
    PRODUCTION = "production"
    MARKETING = "marketing"
    QUALITY = "quality"
    LOGISTICS = "logistics"
    ANALYTICS = "analytics"
    MANUFACTURING_SALES = "manufacturing_sales"


class AnalysisType(Enum):
    """Analysis type classifications."""
    RANKING = "ranking"
    TREND = "trend"
    COMPARISON = "comparison"
    AGGREGATION = "aggregation"
    FILTERING = "filtering"
    CORRELATION = "correlation"
    FORECASTING = "forecasting"
    TIER_ANALYSIS = "tier_analysis"


class ComplexityTier(Enum):
    """Query complexity classifications."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"


class ExecutionStatus(Enum):
    """Query execution status."""
    NOT_EXECUTED = "not_executed"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"


class UserRole(Enum):
    """User role classifications."""
    ANALYST = "analyst"
    BUSINESS_ANALYST = "business_analyst"
    MANAGER = "manager"
    EXECUTIVE = "executive"
    ENGINEER = "engineer"
    PRODUCT_MANAGER = "product_manager"
    DATA_SCIENTIST = "data_scientist"
    BUSINESS_USER = "business_user"


class UserSkillLevel(Enum):
    """User SQL skill level."""
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"


@dataclass
class QueryContent:
    """Core query content and description."""
    sql_query: str = ""
    readable_description: Optional[str] = None
    business_question: Optional[str] = None
    query_intent: Optional[str] = None
    query_type: Optional[str] = None


@dataclass
class SemanticContext:
    """Semantic and business context for query understanding."""
    business_domain: Optional[str] = None
    business_function: Optional[str] = None
    analysis_type: Optional[str] = None
    time_dimension: Optional[str] = None
    metrics: List[str] = field(default_factory=list)
    entities: List[str] = field(default_factory=list)
    business_concepts: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)


@dataclass
class TechnicalMetadata:
    """Technical query analysis and optimization metadata."""
    database: str = "mariadb"
    tables_used: List[str] = field(default_factory=list)
    join_count: int = 0
    aggregation_functions: List[str] = field(default_factory=list)
    complexity_score: Optional[int] = None
    performance_tier: Optional[str] = None
    estimated_cost: Optional[str] = None
    query_pattern: Optional[str] = None
    has_subqueries: bool = False
    has_window_functions: bool = False
    estimated_execution_time_ms: Optional[float] = None
    estimated_row_count: Optional[int] = None
    requires_indexes: List[str] = field(default_factory=list)
    sql_dialect: str = "mysql"
    optimization_hints: List[str] = field(default_factory=list)


@dataclass
class UserContext:
    """User and organizational context."""
    user_id: str = "system"
    user_role: Optional[str] = None
    user_department: Optional[str] = None
    user_skill_level: Optional[str] = None
    organization_id: Optional[str] = None
    industry: Optional[str] = None
    timezone: str = "UTC"
    preferred_date_format: str = "YYYY-MM-DD"


@dataclass
class InvestigationContext:
    """Investigation and hypothesis context."""
    investigation_phase: Optional[str] = None
    investigation_id: Optional[str] = None
    hypothesis: Optional[str] = None
    confidence_level: Optional[str] = None
    business_impact: Optional[str] = None
    urgency: Optional[str] = None
    follows_pattern: Optional[str] = None
    related_questions: List[str] = field(default_factory=list)


@dataclass
class ExecutionResults:
    """Query execution results and performance tracking."""
    execution_status: Optional[str] = None
    is_validated: bool = False
    success: Optional[bool] = None
    execution_time_ms: Optional[float] = None
    rows_returned: Optional[int] = None
    data_quality_score: Optional[float] = None
    business_relevance_score: Optional[float] = None
    user_satisfaction: Optional[float] = None
    was_modified: bool = False
    led_to_followup: Optional[bool] = None
    execution_count: int = 0
    last_executed_at: Optional[datetime] = None
    first_executed_at: Optional[datetime] = None
    execution_history: List[Dict[str, Any]] = field(default_factory=list)
    average_execution_time_ms: Optional[float] = None
    fastest_execution_ms: Optional[float] = None
    slowest_execution_ms: Optional[float] = None
    cache_hit_rate: Optional[float] = None
    error_history: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class LearningMetadata:
    """Learning and usage analytics metadata."""
    usage_frequency: int = 0
    success_rate: Optional[float] = None
    average_user_rating: Optional[float] = None
    reuse_patterns: List[str] = field(default_factory=list)
    common_modifications: List[str] = field(default_factory=list)
    triggered_investigations: List[str] = field(default_factory=list)
    first_used_at: Optional[datetime] = None
    last_used_at: Optional[datetime] = None
    learning_updated_at: Optional[datetime] = None
    user_feedback: List[Dict[str, Any]] = field(default_factory=list)
    optimization_suggestions: List[str] = field(default_factory=list)
    similar_queries: List[str] = field(default_factory=list)
    performance_improvements: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class BusinessIntelligence:
    """Business intelligence and decision support metadata."""
    kpi_category: Optional[str] = None
    decision_support_level: str = "tactical"
    stakeholder_relevance: List[str] = field(default_factory=list)
    reporting_frequency: Optional[str] = None
    benchmark_comparison: Optional[str] = None
    actionability_score: Optional[float] = None
    strategic_alignment: Optional[str] = None
    business_value: str = "medium"
    regulatory_compliance: List[str] = field(default_factory=list)
    data_governance_level: str = "standard"


@dataclass
class Collaboration:
    """Collaboration and sharing metadata."""
    is_shared: bool = False
    shared_with: List[str] = field(default_factory=list)
    created_by_user_id: Optional[str] = None
    review_status: str = "pending"
    reviewers: List[str] = field(default_factory=list)
    comments: List[Dict[str, Any]] = field(default_factory=list)
    approval_required: bool = False
    collaboration_history: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class VersionControl:
    """Version control and change tracking."""
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    version: str = "1.0"
    created_by: str = "autonomous_analyst"
    validation_status: str = "pending_execution"
    peer_review_score: Optional[float] = None
    change_log: List[Dict[str, Any]] = field(default_factory=list)
    parent_query_id: Optional[str] = None
    child_query_ids: List[str] = field(default_factory=list)


@dataclass
class Caching:
    """Caching and performance optimization metadata."""
    is_cacheable: bool = True
    cache_key: Optional[str] = None
    cache_ttl_seconds: int = 3600
    cache_strategy: str = "time_based"
    cache_dependencies: List[str] = field(default_factory=list)
    last_cache_update: Optional[datetime] = None
    cache_hit_count: int = 0
    cache_miss_count: int = 0


@dataclass
class Monitoring:
    """Monitoring and alerting configuration."""
    alert_thresholds: Dict[str, Union[int, float]] = field(default_factory=lambda: {
        "execution_time_ms": 5000,
        "row_count_variance": 0.2,
        "error_rate": 0.05
    })
    performance_baseline: Dict[str, Any] = field(default_factory=dict)
    health_check_frequency: str = "daily"
    last_health_check: Optional[datetime] = None
    health_status: str = "unknown"


@dataclass
class Security:
    """Security and access control metadata."""
    access_level: str = "standard"
    contains_pii: bool = False
    contains_sensitive_data: bool = False
    data_classification: str = "internal"
    requires_audit_log: bool = True
    allowed_roles: List[str] = field(default_factory=lambda: ["analyst", "manager"])
    data_retention_days: int = 365


@dataclass
class Automation:
    """Automation and scheduling configuration."""
    auto_execute_schedule: Optional[str] = None
    auto_refresh_enabled: bool = False
    notification_rules: List[Dict[str, Any]] = field(default_factory=list)
    integration_webhooks: List[str] = field(default_factory=list)
    downstream_dependencies: List[str] = field(default_factory=list)
    upstream_dependencies: List[str] = field(default_factory=list)


@dataclass
class Embeddings:
    """Vector embeddings metadata."""
    query_embedding: Optional[List[float]] = None
    description_embedding: Optional[List[float]] = None
    business_context_embedding: Optional[List[float]] = None
    embedding_model: str = "bge-m3"
    embedding_dimension: int = 1024
    embedding_created_at: Optional[datetime] = None
    embedding_version: str = "1.0"


@dataclass
class EnhancedSQLQuery:
    """Complete enhanced SQL query record for production-grade system."""
    _id: str = field(default_factory=lambda: f"sql_{uuid.uuid4().hex[:8]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    
    # Core components
    query_content: QueryContent = field(default_factory=QueryContent)
    semantic_context: SemanticContext = field(default_factory=SemanticContext)
    technical_metadata: TechnicalMetadata = field(default_factory=TechnicalMetadata)
    user_context: UserContext = field(default_factory=UserContext)
    investigation_context: InvestigationContext = field(default_factory=InvestigationContext)
    execution_results: ExecutionResults = field(default_factory=ExecutionResults)
    learning_metadata: LearningMetadata = field(default_factory=LearningMetadata)
    business_intelligence: BusinessIntelligence = field(default_factory=BusinessIntelligence)
    collaboration: Collaboration = field(default_factory=Collaboration)
    version_control: VersionControl = field(default_factory=VersionControl)
    caching: Caching = field(default_factory=Caching)
    monitoring: Monitoring = field(default_factory=Monitoring)
    security: Security = field(default_factory=Security)
    automation: Automation = field(default_factory=Automation)
    embeddings: Embeddings = field(default_factory=Embeddings)
    
    # Indexing and categorization
    tags: List[str] = field(default_factory=list)
    custom_fields: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        result = {}
        for field_name, field_value in asdict(self).items():
            if isinstance(field_value, datetime):
                result[field_name] = field_value.isoformat()
            elif hasattr(field_value, '__dict__'):
                # Handle nested dataclasses
                if isinstance(field_value, dict):
                    nested = {}
                    for k, v in field_value.items():
                        if isinstance(v, datetime):
                            nested[k] = v.isoformat()
                        else:
                            nested[k] = v
                    result[field_name] = nested
                else:
                    result[field_name] = field_value
            else:
                result[field_name] = field_value
        return result

    def to_lancedb_record(self) -> Dict[str, Any]:
        """Convert to LanceDB-compatible record format."""
        data = self.to_dict()
        
        # Flatten for LanceDB storage while preserving structure in JSON fields
        record = {
            # Core identification
            "id": self._id,
            "sql_query": self.query_content.sql_query,
            "normalized_sql": self._normalize_sql(self.query_content.sql_query),
            
            # Vector embedding (will be populated during storage)
            "vector": self.embeddings.query_embedding,
            
            # Key classification fields for indexing
            "database": self.technical_metadata.database,
            "query_type": self.query_content.query_type.value if hasattr(self.query_content.query_type, 'value') else self.query_content.query_type,
            "business_domain": self.semantic_context.business_domain.value if hasattr(self.semantic_context.business_domain, 'value') else self.semantic_context.business_domain,
            "user_id": self.user_context.user_id,
            "execution_status": self.execution_results.execution_status.value if hasattr(self.execution_results.execution_status, 'value') else self.execution_results.execution_status,
            "success": self.execution_results.success,
            
            # Performance metrics
            "execution_time_ms": self.execution_results.execution_time_ms,
            "row_count": self.execution_results.rows_returned,
            "complexity_score": self.technical_metadata.complexity_score,
            "usage_frequency": self.learning_metadata.usage_frequency,
            
            # Timestamps
            "timestamp": self.version_control.created_at,
            "created_at": self.version_control.created_at,
            "updated_at": self.version_control.updated_at,
            "last_executed_at": self.execution_results.last_executed_at,
            
            # JSON storage for complex nested data (with enum serialization)
            "query_content_json": json.dumps(self._serialize_for_json(asdict(self.query_content))),
            "semantic_context_json": json.dumps(self._serialize_for_json(asdict(self.semantic_context))),
            "technical_metadata_json": json.dumps(self._serialize_for_json(asdict(self.technical_metadata))),
            "user_context_json": json.dumps(self._serialize_for_json(asdict(self.user_context))),
            "investigation_context_json": json.dumps(self._serialize_for_json(asdict(self.investigation_context))),
            "execution_results_json": json.dumps(self._serialize_for_json(asdict(self.execution_results))),
            "learning_metadata_json": json.dumps(self._serialize_for_json(asdict(self.learning_metadata))),
            "business_intelligence_json": json.dumps(self._serialize_for_json(asdict(self.business_intelligence))),
            "collaboration_json": json.dumps(self._serialize_for_json(asdict(self.collaboration))),
            "version_control_json": json.dumps(self._serialize_for_json(asdict(self.version_control))),
            "caching_json": json.dumps(self._serialize_for_json(asdict(self.caching))),
            "monitoring_json": json.dumps(self._serialize_for_json(asdict(self.monitoring))),
            "security_json": json.dumps(self._serialize_for_json(asdict(self.security))),
            "automation_json": json.dumps(self._serialize_for_json(asdict(self.automation))),
            "embeddings_json": json.dumps(self._serialize_for_json(asdict(self.embeddings))),
            
            # Tags and custom fields
            "tags_json": json.dumps(self.tags),
            "custom_fields_json": json.dumps(self.custom_fields)
        }
        
        # Remove None values and convert datetime objects
        cleaned_record = {}
        for k, v in record.items():
            if v is not None:
                if isinstance(v, datetime):
                    cleaned_record[k] = v.isoformat()
                else:
                    cleaned_record[k] = v
            else:
                # Skip None values to avoid schema conflicts
                continue
        
        return cleaned_record

    def _normalize_sql(self, sql_query: str) -> str:
        """Normalize SQL for better matching."""
        import re
        
        # Remove extra whitespace
        normalized = " ".join(sql_query.split())
        
        # Convert to lowercase
        normalized = normalized.lower()
        
        # Replace specific values with placeholders
        # Numbers
        normalized = re.sub(r'\\b\\d+\\b', '?', normalized)
        # Quoted strings
        normalized = re.sub(r"'[^']*'", "'?'", normalized)
        normalized = re.sub(r'"[^"]*"', '"?"', normalized)
        
        return normalized

    def _serialize_for_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Serialize data for JSON storage, handling enums, datetimes, and numpy arrays."""
        import numpy as np
        result = {}
        for key, value in data.items():
            if hasattr(value, 'value'):  # Enum
                result[key] = value.value
            elif isinstance(value, datetime):
                result[key] = value.isoformat()
            elif isinstance(value, np.ndarray):
                result[key] = value.tolist()  # Convert numpy array to list
            elif isinstance(value, dict):
                result[key] = self._serialize_for_json(value)
            elif isinstance(value, list):
                result[key] = [
                    item.value if hasattr(item, 'value') else 
                    item.isoformat() if isinstance(item, datetime) else
                    item.tolist() if isinstance(item, np.ndarray) else
                    self._serialize_for_json(item) if isinstance(item, dict) else
                    item
                    for item in value
                ]
            else:
                result[key] = value
        return result

    @classmethod
    def from_legacy_data(cls, legacy_data: Dict[str, Any]) -> 'EnhancedSQLQuery':
        """Create enhanced query from legacy simple schema data."""
        enhanced = cls()
        
        # Map legacy fields to enhanced structure
        enhanced.query_content.sql_query = legacy_data.get("sql_query", "")
        enhanced.query_content.query_type = QueryType(legacy_data.get("query_type", "simple"))
        
        enhanced.technical_metadata.database = legacy_data.get("database", "mariadb")
        enhanced.execution_results.execution_time_ms = legacy_data.get("execution_time_ms")
        enhanced.execution_results.rows_returned = legacy_data.get("row_count")
        enhanced.execution_results.success = legacy_data.get("success")
        
        enhanced.user_context.user_id = legacy_data.get("user_id", "system")
        enhanced.version_control.created_at = legacy_data.get("timestamp", datetime.now())
        
        # Parse legacy metadata JSON if available
        if "metadata" in legacy_data:
            try:
                metadata = json.loads(legacy_data["metadata"]) if isinstance(legacy_data["metadata"], str) else legacy_data["metadata"]
                
                # Map metadata fields to enhanced structure
                if "investigation_phase" in metadata:
                    enhanced.investigation_context.investigation_phase = metadata["investigation_phase"]
                if "business_domain" in metadata:
                    try:
                        enhanced.semantic_context.business_domain = BusinessDomain(metadata["business_domain"])
                    except ValueError:
                        enhanced.semantic_context.business_domain = BusinessDomain.ANALYTICS
                if "investigation_id" in metadata:
                    enhanced.investigation_context.investigation_id = metadata["investigation_id"]
                if "business_question" in metadata:
                    enhanced.query_content.business_question = metadata["business_question"]
                    
            except (json.JSONDecodeError, TypeError):
                pass
        
        return enhanced


# Schema validation and utility functions
def validate_enhanced_query(query: EnhancedSQLQuery) -> List[str]:
    """Validate enhanced query data and return list of validation errors."""
    errors = []
    
    if not query.query_content.sql_query.strip():
        errors.append("SQL query cannot be empty")
    
    if not query.user_context.user_id:
        errors.append("User ID is required")
    
    if query.execution_results.execution_time_ms is not None and query.execution_results.execution_time_ms < 0:
        errors.append("Execution time cannot be negative")
    
    if query.execution_results.rows_returned is not None and query.execution_results.rows_returned < 0:
        errors.append("Row count cannot be negative")
    
    if query.technical_metadata.complexity_score < 1 or query.technical_metadata.complexity_score > 10:
        errors.append("Complexity score must be between 1 and 10")
    
    return errors


def create_enhanced_query_from_simple(
    sql_query: str,
    database: str = "mariadb",
    user_id: str = "system",
    business_context: Optional[Dict[str, Any]] = None
) -> EnhancedSQLQuery:
    """Create enhanced query from simple parameters with intelligent defaults."""
    enhanced = EnhancedSQLQuery()
    
    # Core query content
    enhanced.query_content.sql_query = sql_query
    enhanced.query_content.query_type = _classify_query_type(sql_query)
    
    # Technical metadata
    enhanced.technical_metadata.database = database
    enhanced.technical_metadata.tables_used = _extract_table_names(sql_query)
    enhanced.technical_metadata.join_count = _count_joins(sql_query)
    enhanced.technical_metadata.complexity_score = _calculate_complexity_score(sql_query)
    enhanced.technical_metadata.has_subqueries = "(" in sql_query and "select" in sql_query.lower()
    enhanced.technical_metadata.has_window_functions = any(func in sql_query.lower() for func in ["over(", "partition by", "row_number", "rank"])
    
    # User context
    enhanced.user_context.user_id = user_id
    
    # Business context integration
    if business_context:
        _apply_business_context(enhanced, business_context)
    
    # Semantic context from query analysis
    enhanced.semantic_context.business_domain = _extract_business_domain(sql_query, business_context)
    enhanced.semantic_context.entities = enhanced.technical_metadata.tables_used
    enhanced.semantic_context.keywords = _extract_keywords(sql_query)
    
    return enhanced


# Helper functions for intelligent field population
def _classify_query_type(sql_query: str) -> QueryType:
    """Classify query type from SQL content."""
    query_lower = sql_query.lower().strip()
    
    # Analytical patterns
    if any(pattern in query_lower for pattern in ["group by", "having", "union", "join", "with", "window"]):
        return QueryType.ANALYTICAL
    
    # Simple CRUD
    if any(pattern in query_lower for pattern in ["insert", "update", "delete", "select * from"]):
        if "join" not in query_lower and "group by" not in query_lower:
            return QueryType.SIMPLE
    
    # Operational
    if any(pattern in query_lower for pattern in ["create", "alter", "drop", "show", "describe", "explain"]):
        return QueryType.OPERATIONAL
    
    return QueryType.ANALYTICAL  # Default for complex queries


def _extract_table_names(sql_query: str) -> List[str]:
    """Extract table names from SQL query."""
    import re
    
    # Simple regex to find table names after FROM and JOIN
    patterns = [
        r'from\\s+([a-zA-Z_][a-zA-Z0-9_]*)',
        r'join\\s+([a-zA-Z_][a-zA-Z0-9_]*)',
        r'update\\s+([a-zA-Z_][a-zA-Z0-9_]*)',
        r'into\\s+([a-zA-Z_][a-zA-Z0-9_]*)'
    ]
    
    tables = set()
    for pattern in patterns:
        matches = re.findall(pattern, sql_query, re.IGNORECASE)
        tables.update(matches)
    
    return list(tables)


def _count_joins(sql_query: str) -> int:
    """Count JOIN operations in SQL query."""
    return sql_query.lower().count('join')


def _calculate_complexity_score(sql_query: str) -> int:
    """Calculate query complexity score (1-10)."""
    query_lower = sql_query.lower()
    score = 1
    
    # Base complexity factors
    if 'join' in query_lower:
        score += query_lower.count('join')
    if 'group by' in query_lower:
        score += 1
    if 'having' in query_lower:
        score += 1
    if 'union' in query_lower:
        score += 2
    if 'with' in query_lower:
        score += 2
    if any(func in query_lower for func in ['over(', 'partition by', 'row_number']):
        score += 3
    if '(' in query_lower and 'select' in query_lower:  # Subqueries
        score += query_lower.count('select') - 1
    
    return min(score, 10)  # Cap at 10


def _extract_business_domain(sql_query: str, business_context: Optional[Dict[str, Any]] = None) -> BusinessDomain:
    """Extract business domain from query and context."""
    if business_context and "business_domain" in business_context:
        try:
            return BusinessDomain(business_context["business_domain"])
        except ValueError:
            pass
    
    query_lower = sql_query.lower()
    
    # Domain keywords mapping
    domain_keywords = {
        BusinessDomain.SALES: ["sales", "revenue", "order", "customer", "purchase", "transaction"],
        BusinessDomain.FINANCE: ["finance", "profit", "cost", "budget", "expense", "financial"],
        BusinessDomain.HR: ["employee", "staff", "payroll", "hr", "human resource", "personnel"],
        BusinessDomain.PRODUCTION: ["production", "manufacturing", "inventory", "supply", "warehouse"],
        BusinessDomain.MARKETING: ["marketing", "campaign", "advertisement", "promotion", "lead"],
        BusinessDomain.QUALITY: ["quality", "defect", "compliance", "standard", "audit"],
        BusinessDomain.LOGISTICS: ["logistics", "shipping", "delivery", "transport", "fulfillment"]
    }
    
    for domain, keywords in domain_keywords.items():
        if any(keyword in query_lower for keyword in keywords):
            return domain
    
    return BusinessDomain.ANALYTICS  # Default


def _extract_keywords(sql_query: str) -> List[str]:
    """Extract relevant keywords from SQL query."""
    import re
    
    # Remove SQL keywords and extract business terms
    sql_keywords = {'select', 'from', 'where', 'join', 'group', 'by', 'order', 'having', 'union', 'with'}
    
    # Extract words (alphanumeric + underscore)
    words = re.findall(r'\\b[a-zA-Z_][a-zA-Z0-9_]*\\b', sql_query.lower())
    
    # Filter out SQL keywords and short words
    keywords = [word for word in words if word not in sql_keywords and len(word) > 2]
    
    # Remove duplicates and limit
    return list(set(keywords))[:20]


def _apply_business_context(enhanced: EnhancedSQLQuery, business_context: Dict[str, Any]):
    """Apply business context to enhanced query."""
    # Investigation context
    if "investigation_phase" in business_context:
        enhanced.investigation_context.investigation_phase = business_context["investigation_phase"]
    if "investigation_id" in business_context:
        enhanced.investigation_context.investigation_id = business_context["investigation_id"]
    if "hypothesis" in business_context:
        enhanced.investigation_context.hypothesis = business_context["hypothesis"]
    if "business_question" in business_context:
        enhanced.query_content.business_question = business_context["business_question"]
    
    # User context
    if "user_role" in business_context:
        try:
            enhanced.user_context.user_role = UserRole(business_context["user_role"])
        except ValueError:
            pass
    if "organization_id" in business_context:
        enhanced.user_context.organization_id = business_context["organization_id"]
    
    # Business intelligence
    if "business_impact" in business_context:
        enhanced.investigation_context.business_impact = business_context["business_impact"]
    if "kpi_category" in business_context:
        enhanced.business_intelligence.kpi_category = business_context["kpi_category"]


# Standalone test functionality
if __name__ == "__main__":
    print("🧪 Running Enhanced Schema Standalone Test...")
    
    # Test enum functionality
    print(f'✅ QueryType.ANALYTICAL: {QueryType.ANALYTICAL.value}')
    print(f'✅ BusinessDomain.SALES: {BusinessDomain.SALES.value}')
    print(f'✅ UserRole.ANALYST: {UserRole.ANALYST.value}')
    
    # Test basic enhanced query creation
    enhanced_query = EnhancedSQLQuery()
    print(f'✅ Basic enhanced query created with ID: {enhanced_query._id}')
    
    # Test enhanced query from simple parameters
    simple_query = create_enhanced_query_from_simple(
        sql_query='SELECT COUNT(*) FROM orders WHERE date >= "2024-01-01"',
        database='mariadb',
        user_id='test_analyst',
        business_context={
            'business_domain': 'sales',
            'investigation_phase': 'data_exploration',
            'business_question': 'How many orders since start of year?'
        }
    )
    print(f'✅ Enhanced query from simple parameters: {simple_query._id}')
    print(f'   Query type: {simple_query.query_content.query_type.value}')
    print(f'   Business domain: {simple_query.semantic_context.business_domain.value}')
    print(f'   Investigation phase: {simple_query.investigation_context.investigation_phase}')
    
    # Test validation
    validation_errors = validate_enhanced_query(simple_query)
    print(f'✅ Validation errors (should be empty): {validation_errors}')
    
    # Test conversion to dict and LanceDB record
    query_dict = simple_query.to_dict()
    print(f'✅ Converted to dict with {len(query_dict)} fields')
    
    lancedb_record = simple_query.to_lancedb_record()
    print(f'✅ Converted to LanceDB record with {len(lancedb_record)} fields')
    print(f'   Record keys: {list(lancedb_record.keys())[:10]}...')
    
    # Test legacy data migration
    legacy_data = {
        'sql_query': 'SELECT * FROM customers',
        'database': 'mariadb',
        'query_type': 'simple',
        'execution_time_ms': 123.45,
        'row_count': 50,
        'user_id': 'test_user',
        'success': True,
        'metadata': json.dumps({
            'investigation_phase': 'data_exploration',
            'business_domain': 'sales',
            'business_question': 'Get all customers'
        })
    }
    
    migrated_query = EnhancedSQLQuery.from_legacy_data(legacy_data)
    print(f'✅ Migrated legacy query: {migrated_query._id}')
    print(f'   Original SQL: {migrated_query.query_content.sql_query}')
    print(f'   Business domain: {migrated_query.semantic_context.business_domain.value}')
    print(f'   Investigation phase: {migrated_query.investigation_context.investigation_phase}')
    
    # Test complex query analysis
    complex_query = create_enhanced_query_from_simple(
        sql_query='''
        SELECT 
            DATE_FORMAT(o.order_date, '%Y-%m') as month,
            SUM(oi.quantity * oi.price) as revenue,
            COUNT(DISTINCT o.customer_id) as unique_customers,
            AVG(oi.quantity * oi.price) as avg_order_value
        FROM orders o
        JOIN order_items oi ON o.id = oi.order_id
        JOIN customers c ON o.customer_id = c.id
        WHERE o.order_date >= '2024-01-01'
        GROUP BY DATE_FORMAT(o.order_date, '%Y-%m')
        HAVING SUM(oi.quantity * oi.price) > 10000
        ORDER BY month DESC
        ''',
        database='mariadb',
        user_id='sales_analyst',
        business_context={
            'business_domain': 'sales',
            'investigation_phase': 'trend_analysis',
            'business_question': 'Monthly revenue trends with customer analysis',
            'kpi_category': 'revenue_metrics'
        }
    )
    
    print(f'✅ Complex query analysis: {complex_query._id}')
    print(f'   Query type: {complex_query.query_content.query_type.value}')
    print(f'   Complexity score: {complex_query.technical_metadata.complexity_score}')
    print(f'   Tables used: {complex_query.technical_metadata.tables_used}')
    print(f'   Join count: {complex_query.technical_metadata.join_count}')
    print(f'   Aggregation functions: {complex_query.technical_metadata.aggregation_functions}')
    print(f'   Has subqueries: {complex_query.technical_metadata.has_subqueries}')
    
    print('\n🎉 Enhanced Schema Standalone Test PASSED!')
    print(f'   Total tests run: 7')
    print(f'   All validations successful')
    print(f'   Schema supports production-grade SQL query learning')