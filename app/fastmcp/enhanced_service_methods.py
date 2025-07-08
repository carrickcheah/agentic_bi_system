"""
Enhanced service methods for production-grade SQL query learning.
This file contains the enhanced methods to be integrated into BusinessService.
"""

import re
from typing import Dict, List, Optional, Any
from datetime import datetime

# All the enhanced methods that will be added to BusinessService

def _record_query_for_learning_enhanced(
    self,
    query: str,
    database: str,
    execution_time_ms: float,
    row_count: int,
    user_id: str,
    success: bool,
    business_context: Optional[Dict[str, Any]] = None,
    error: Optional[str] = None
):
    """
    Record a query for learning using enhanced production-grade schema.
    """
    try:
        if not self.sql_embedding_service:
            return
        
        # Create enhanced query with comprehensive metadata
        enhanced_query = self._create_enhanced_query(
            query=query,
            database=database,
            execution_time_ms=execution_time_ms,
            row_count=row_count,
            user_id=user_id,
            success=success,
            business_context=business_context,
            error=error
        )
        
        # Store enhanced query in Qdrant
        query_id = await self.sql_embedding_service.store_enhanced_query(enhanced_query)
        
        logger.debug(
            f"Recorded enhanced query: {query_id} | "
            f"Domain: {enhanced_query.semantic_context.business_domain.value} | "
            f"Type: {enhanced_query.query_content.query_type.value} | "
            f"Phase: {enhanced_query.investigation_context.investigation_phase or 'none'}"
        )
        
    except Exception as e:
        # Don't let query recording errors affect main query execution
        logger.warning(f"Failed to record enhanced query for learning: {e}")


def _create_enhanced_query(
    self,
    query: str,
    database: str,
    execution_time_ms: float,
    row_count: int,
    user_id: str,
    success: bool,
    business_context: Optional[Dict[str, Any]] = None,
    error: Optional[str] = None
) -> EnhancedSQLQuery:
    """
    Create comprehensive enhanced query with production-grade metadata.
    """
    enhanced_query = EnhancedSQLQuery()
    
    # Core query content
    enhanced_query.query_content = QueryContent(
        sql_query=query,
        query_type=self._classify_query_type_enhanced(query, business_context),
        business_question=business_context.get("business_question") if business_context else None,
        query_intent=business_context.get("query_intent") if business_context else None,
        readable_description=self._generate_query_description(query, business_context)
    )
    
    # Semantic context
    enhanced_query.semantic_context = SemanticContext(
        business_domain=self._extract_business_domain_enhanced(query, business_context),
        business_function=business_context.get("business_function") if business_context else None,
        analysis_type=self._classify_analysis_type(query),
        time_dimension=self._extract_time_dimension(query),
        metrics=self._extract_metrics(query),
        entities=self._extract_entities(query),
        business_concepts=self._extract_business_concepts(query, business_context),
        keywords=self._extract_keywords_enhanced(query)
    )
    
    # Technical metadata
    enhanced_query.technical_metadata = TechnicalMetadata(
        database=database,
        tables_used=self._extract_table_names_enhanced(query),
        join_count=self._count_joins_enhanced(query),
        aggregation_functions=self._extract_aggregation_functions(query),
        complexity_score=self._calculate_complexity_score_enhanced(query),
        performance_tier=self._classify_performance_tier(execution_time_ms),
        query_pattern=self._identify_query_pattern(query),
        has_subqueries=self._has_subqueries(query),
        has_window_functions=self._has_window_functions(query),
        estimated_execution_time_ms=execution_time_ms,
        estimated_row_count=row_count,
        sql_dialect="mysql" if database == "mariadb" else "postgresql",
        optimization_hints=self._generate_optimization_hints(query)
    )
    
    # User context
    enhanced_query.user_context = UserContext(
        user_id=user_id,
        user_role=self._infer_user_role(user_id, business_context),
        user_department=business_context.get("user_department") if business_context else None,
        organization_id=business_context.get("organization_id") if business_context else None,
        industry=business_context.get("industry") if business_context else None
    )
    
    # Investigation context
    enhanced_query.investigation_context = InvestigationContext(
        investigation_phase=business_context.get("investigation_phase") if business_context else None,
        investigation_id=business_context.get("investigation_id") if business_context else None,
        hypothesis=business_context.get("hypothesis") if business_context else None,
        confidence_level=business_context.get("confidence_level") if business_context else None,
        business_impact=business_context.get("business_impact") if business_context else None
    )
    
    # Execution results
    enhanced_query.execution_results = ExecutionResults(
        execution_status=ExecutionStatus.COMPLETED if success else ExecutionStatus.FAILED,
        is_validated=True,
        success=success,
        execution_time_ms=execution_time_ms,
        rows_returned=row_count,
        execution_count=1,
        first_executed_at=datetime.now(),
        last_executed_at=datetime.now(),
        average_execution_time_ms=execution_time_ms
    )
    
    if error:
        enhanced_query.execution_results.error_history = [
            {
                "error": error,
                "timestamp": datetime.now().isoformat(),
                "context": "business_service_execution"
            }
        ]
    
    # Learning metadata
    enhanced_query.learning_metadata = LearningMetadata(
        usage_frequency=1,
        first_used_at=datetime.now(),
        last_used_at=datetime.now(),
        learning_updated_at=datetime.now()
    )
    
    # Business intelligence
    enhanced_query.business_intelligence = BusinessIntelligence(
        kpi_category=business_context.get("kpi_category") if business_context else None,
        decision_support_level=self._classify_decision_support_level(query, business_context),
        stakeholder_relevance=self._identify_stakeholder_relevance(enhanced_query.semantic_context.business_domain),
        actionability_score=self._calculate_actionability_score(query, business_context),
        business_value=self._assess_business_value(query, business_context)
    )
    
    # Version control
    enhanced_query.version_control.created_by = "business_service_auto"
    enhanced_query.version_control.validation_status = "auto_validated" if success else "execution_failed"
    
    # Tags
    enhanced_query.tags = self._generate_tags(enhanced_query)
    
    # Custom fields for business service
    enhanced_query.custom_fields = {
        "recorded_by_service": "business_service",
        "automatic_learning": True,
        "recording_method": "post_execution",
        "source_system": "agentic_sql"
    }
    
    if business_context:
        enhanced_query.custom_fields.update({
            k: v for k, v in business_context.items() 
            if k not in ["investigation_phase", "business_question", "hypothesis"]
        })
    
    return enhanced_query


# All helper methods for enhanced query creation

def _classify_query_type_enhanced(self, query: str, business_context: Optional[Dict[str, Any]] = None) -> QueryType:
    """Enhanced query type classification using production-grade enum."""
    query_lower = query.lower().strip()
    
    # Check for investigative context
    if business_context and business_context.get("investigation_phase"):
        return QueryType.INVESTIGATIVE
    
    # Analytical queries
    analytical_patterns = [
        "group by", "having", "union", "join", "subquery", 
        "with", "case when", "window", "over(", "partition by"
    ]
    if any(pattern in query_lower for pattern in analytical_patterns):
        return QueryType.ANALYTICAL
    
    # Simple CRUD operations
    simple_patterns = ["insert", "update", "delete", "select * from", "where id ="]
    if any(pattern in query_lower for pattern in simple_patterns):
        if "join" not in query_lower and "group by" not in query_lower:
            return QueryType.SIMPLE
    
    # Operational queries
    operational_patterns = [
        "create table", "alter table", "drop table", "create index",
        "show", "describe", "explain", "information_schema"
    ]
    if any(pattern in query_lower for pattern in operational_patterns):
        return QueryType.OPERATIONAL
    
    return QueryType.ANALYTICAL


def _extract_business_domain_enhanced(self, query: str, business_context: Optional[Dict[str, Any]] = None) -> BusinessDomain:
    """Enhanced business domain extraction using production-grade enum."""
    # Check business context first
    if business_context and "business_domain" in business_context:
        try:
            return BusinessDomain(business_context["business_domain"])
        except ValueError:
            pass
    
    # Extract from query and investigation request
    text_to_analyze = query.lower()
    if business_context and "business_question" in business_context:
        text_to_analyze += " " + business_context["business_question"].lower()
    
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
        if any(keyword in text_to_analyze for keyword in keywords):
            return domain
    
    return BusinessDomain.ANALYTICS


def _classify_analysis_type(self, query: str) -> Optional[AnalysisType]:
    """Classify the type of analysis being performed."""
    query_lower = query.lower()
    
    if any(pattern in query_lower for pattern in ["order by", "limit", "top", "rank"]):
        return AnalysisType.RANKING
    elif any(pattern in query_lower for pattern in ["date", "year", "month", "day", "time"]):
        return AnalysisType.TREND
    elif any(pattern in query_lower for pattern in ["group by", "sum", "count", "avg", "max", "min"]):
        return AnalysisType.AGGREGATION
    elif any(pattern in query_lower for pattern in ["where", "having", "filter"]):
        return AnalysisType.FILTERING
    elif any(pattern in query_lower for pattern in ["join", "union", "intersect"]):
        return AnalysisType.COMPARISON
    
    return None


def _extract_time_dimension(self, query: str) -> Optional[str]:
    """Extract time dimension from query."""
    query_lower = query.lower()
    
    time_patterns = {
        "daily": ["day", "daily"],
        "weekly": ["week", "weekly"],
        "monthly": ["month", "monthly"],
        "quarterly": ["quarter", "quarterly"],
        "yearly": ["year", "yearly", "annual"],
        "real_time": ["now()", "current_timestamp", "real time"]
    }
    
    for dimension, patterns in time_patterns.items():
        if any(pattern in query_lower for pattern in patterns):
            return dimension
    
    return None


def _extract_metrics(self, query: str) -> List[str]:
    """Extract metrics being calculated in the query."""
    metrics = []
    query_lower = query.lower()
    
    metric_patterns = {
        "count": ["count(", "count "],
        "sum": ["sum(", "sum "],
        "average": ["avg(", "average("],
        "maximum": ["max(", "maximum("],
        "minimum": ["min(", "minimum("],
        "revenue": ["revenue", "sales", "amount"],
        "profit": ["profit", "margin"],
        "cost": ["cost", "expense"],
        "percentage": ["percent", "%", "ratio"]
    }
    
    for metric, patterns in metric_patterns.items():
        if any(pattern in query_lower for pattern in patterns):
            metrics.append(metric)
    
    return metrics


def _extract_entities(self, query: str) -> List[str]:
    """Extract business entities from the query."""
    entities = []
    query_lower = query.lower()
    
    entity_patterns = [
        "customer", "user", "client", "product", "order", "sale", "transaction",
        "employee", "staff", "department", "supplier", "vendor", "campaign",
        "inventory", "stock", "warehouse", "shipment", "delivery", "invoice",
        "payment", "account", "budget", "cost", "expense", "revenue"
    ]
    
    for entity in entity_patterns:
        if entity in query_lower:
            entities.append(entity)
    
    return entities


def _extract_business_concepts(self, query: str, business_context: Optional[Dict[str, Any]] = None) -> List[str]:
    """Extract business concepts from query and context."""
    concepts = []
    
    # From query
    query_lower = query.lower()
    concept_patterns = [
        "performance", "analysis", "trend", "comparison", "ranking", "distribution",
        "growth", "decline", "efficiency", "productivity", "quality", "satisfaction",
        "conversion", "retention", "acquisition", "churn", "segmentation"
    ]
    
    for concept in concept_patterns:
        if concept in query_lower:
            concepts.append(concept)
    
    # From business context
    if business_context:
        if business_context.get("investigation_phase"):
            concepts.append(f"investigation_{business_context['investigation_phase']}")
        if business_context.get("hypothesis"):
            concepts.append("hypothesis_testing")
    
    return concepts


def _extract_keywords_enhanced(self, query: str) -> List[str]:
    """Extract relevant keywords from SQL query."""
    # Remove SQL keywords and extract business terms
    sql_keywords = {
        'select', 'from', 'where', 'join', 'group', 'by', 'order', 'having', 
        'union', 'with', 'as', 'on', 'and', 'or', 'not', 'null', 'is', 'in',
        'between', 'like', 'exists', 'case', 'when', 'then', 'else', 'end',
        'inner', 'left', 'right', 'outer', 'full', 'cross', 'distinct', 
        'all', 'any', 'some', 'limit', 'offset', 'top'
    }
    
    # Extract words (alphanumeric + underscore)
    words = re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', query.lower())
    
    # Filter out SQL keywords and short words
    keywords = [word for word in words if word not in sql_keywords and len(word) > 2]
    
    # Remove duplicates and limit
    return list(set(keywords))[:20]


def _extract_table_names_enhanced(self, query: str) -> List[str]:
    """Extract table names from SQL query."""
    # Enhanced regex patterns for table extraction
    patterns = [
        r'from\s+([a-zA-Z_][a-zA-Z0-9_]*)',
        r'join\s+([a-zA-Z_][a-zA-Z0-9_]*)',
        r'update\s+([a-zA-Z_][a-zA-Z0-9_]*)',
        r'into\s+([a-zA-Z_][a-zA-Z0-9_]*)',
        r'table\s+([a-zA-Z_][a-zA-Z0-9_]*)'
    ]
    
    tables = set()
    for pattern in patterns:
        matches = re.findall(pattern, query, re.IGNORECASE)
        tables.update(matches)
    
    return list(tables)


def _count_joins_enhanced(self, query: str) -> int:
    """Count JOIN operations in SQL query."""
    return len(re.findall(r'\bjoin\b', query, re.IGNORECASE))


def _extract_aggregation_functions(self, query: str) -> List[str]:
    """Extract aggregation functions used in the query."""
    functions = []
    query_lower = query.lower()
    
    agg_functions = ['count', 'sum', 'avg', 'max', 'min', 'group_concat', 'string_agg']
    
    for func in agg_functions:
        if f"{func}(" in query_lower:
            functions.append(func.upper())
    
    return functions


def _calculate_complexity_score_enhanced(self, query: str) -> int:
    """Calculate enhanced query complexity score (1-10)."""
    query_lower = query.lower()
    score = 1
    
    # Base complexity factors
    if 'join' in query_lower:
        score += len(re.findall(r'\bjoin\b', query_lower))
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
    
    # Advanced complexity factors
    if 'case when' in query_lower:
        score += 1
    if any(func in query_lower for func in ['rank(', 'dense_rank(', 'lag(', 'lead(']):
        score += 2
    
    return min(score, 10)  # Cap at 10


def _classify_performance_tier(self, execution_time_ms: float) -> ComplexityTier:
    """Classify performance tier based on execution time."""
    if execution_time_ms < 100:
        return ComplexityTier.LOW
    elif execution_time_ms < 1000:
        return ComplexityTier.MEDIUM
    elif execution_time_ms < 5000:
        return ComplexityTier.HIGH
    else:
        return ComplexityTier.VERY_HIGH


def _identify_query_pattern(self, query: str) -> Optional[str]:
    """Identify common query patterns."""
    query_lower = query.lower()
    
    patterns = {
        "top_n_analysis": ["order by", "limit"],
        "aggregation_analysis": ["group by", "sum(", "count("],
        "time_series_analysis": ["date", "timestamp", "year", "month"],
        "join_analysis": ["join", "inner join", "left join"],
        "filtering_analysis": ["where", "having"],
        "window_analysis": ["over(", "partition by"],
        "union_analysis": ["union", "union all"],
        "subquery_analysis": ["exists", "in (select"]
    }
    
    for pattern_name, keywords in patterns.items():
        if all(keyword in query_lower for keyword in keywords):
            return pattern_name
        elif any(keyword in query_lower for keyword in keywords):
            # Partial match for single keyword patterns
            if len(keywords) == 1:
                return pattern_name
    
    return None


def _has_subqueries(self, query: str) -> bool:
    """Check if query has subqueries."""
    return '(' in query and 'select' in query.lower()


def _has_window_functions(self, query: str) -> bool:
    """Check if query has window functions."""
    query_lower = query.lower()
    return any(func in query_lower for func in ['over(', 'partition by', 'row_number', 'rank', 'dense_rank'])


def _generate_optimization_hints(self, query: str) -> List[str]:
    """Generate optimization hints for the query."""
    hints = []
    query_lower = query.lower()
    
    if 'order by' in query_lower and 'limit' in query_lower:
        hints.append("consider_index_on_order_column")
    
    if query_lower.count('join') > 2:
        hints.append("review_join_order")
    
    if 'group by' in query_lower and 'having' in query_lower:
        hints.append("consider_where_instead_of_having")
    
    if 'select *' in query_lower:
        hints.append("specify_required_columns_only")
    
    if 'union' in query_lower:
        hints.append("consider_union_all_if_no_duplicates")
    
    return hints


def _infer_user_role(self, user_id: str, business_context: Optional[Dict[str, Any]] = None) -> Optional[UserRole]:
    """Infer user role from user ID and context."""
    if business_context and "user_role" in business_context:
        try:
            return UserRole(business_context["user_role"])
        except ValueError:
            pass
    
    user_id_lower = user_id.lower()
    
    role_patterns = {
        UserRole.ANALYST: ["analyst", "data", "bi"],
        UserRole.MANAGER: ["manager", "mgr", "lead"],
        UserRole.EXECUTIVE: ["ceo", "cto", "vp", "director", "exec"],
        UserRole.ENGINEER: ["engineer", "dev", "tech"],
        UserRole.PRODUCT_MANAGER: ["product", "pm"],
        UserRole.DATA_SCIENTIST: ["scientist", "ds", "ml"]
    }
    
    for role, patterns in role_patterns.items():
        if any(pattern in user_id_lower for pattern in patterns):
            return role
    
    return UserRole.BUSINESS_USER  # Default


def _classify_decision_support_level(self, query: str, business_context: Optional[Dict[str, Any]] = None) -> str:
    """Classify decision support level of the query."""
    query_lower = query.lower()
    
    # Strategic level indicators
    strategic_indicators = ["revenue", "profit", "growth", "market", "strategic", "forecast"]
    if any(indicator in query_lower for indicator in strategic_indicators):
        return "strategic"
    
    # Tactical level indicators
    tactical_indicators = ["efficiency", "performance", "optimization", "analysis"]
    if any(indicator in query_lower for indicator in tactical_indicators):
        return "tactical"
    
    # Operational level indicators
    operational_indicators = ["daily", "status", "count", "list", "current"]
    if any(indicator in query_lower for indicator in operational_indicators):
        return "operational"
    
    return "tactical"  # Default


def _identify_stakeholder_relevance(self, business_domain: BusinessDomain) -> List[str]:
    """Identify relevant stakeholders based on business domain."""
    stakeholder_mapping = {
        BusinessDomain.SALES: ["sales_team", "sales_manager", "revenue_team"],
        BusinessDomain.FINANCE: ["finance_team", "cfo", "accounting"],
        BusinessDomain.HR: ["hr_team", "hr_manager", "people_ops"],
        BusinessDomain.PRODUCTION: ["operations_team", "production_manager", "supply_chain"],
        BusinessDomain.MARKETING: ["marketing_team", "marketing_manager", "growth_team"],
        BusinessDomain.QUALITY: ["quality_team", "compliance_team", "audit"],
        BusinessDomain.LOGISTICS: ["logistics_team", "fulfillment", "supply_chain"],
        BusinessDomain.ANALYTICS: ["data_team", "analyst", "bi_team"]
    }
    
    return stakeholder_mapping.get(business_domain, ["analyst", "manager"])


def _calculate_actionability_score(self, query: str, business_context: Optional[Dict[str, Any]] = None) -> float:
    """Calculate how actionable the query results are."""
    score = 0.5  # Base score
    query_lower = query.lower()
    
    # Increase score for specific metrics
    if any(metric in query_lower for metric in ["count", "sum", "avg", "total"]):
        score += 0.2
    
    # Increase score for time-based analysis
    if any(time_word in query_lower for time_word in ["date", "month", "year", "day"]):
        score += 0.1
    
    # Increase score for business context
    if business_context:
        if business_context.get("business_impact") in ["high", "critical"]:
            score += 0.2
        if business_context.get("investigation_phase"):
            score += 0.1
    
    return min(score, 1.0)  # Cap at 1.0


def _assess_business_value(self, query: str, business_context: Optional[Dict[str, Any]] = None) -> str:
    """Assess the business value of the query."""
    query_lower = query.lower()
    
    # High value indicators
    high_value_indicators = ["revenue", "profit", "cost", "customer", "strategic"]
    if any(indicator in query_lower for indicator in high_value_indicators):
        return "high"
    
    # Medium value indicators
    medium_value_indicators = ["performance", "efficiency", "analysis", "trend"]
    if any(indicator in query_lower for indicator in medium_value_indicators):
        return "medium"
    
    # Check business context
    if business_context:
        if business_context.get("business_impact") == "high":
            return "high"
        elif business_context.get("investigation_phase"):
            return "medium"
    
    return "low"


def _generate_tags(self, enhanced_query: EnhancedSQLQuery) -> List[str]:
    """Generate tags for the enhanced query."""
    tags = []
    
    # Add domain tag
    tags.append(enhanced_query.semantic_context.business_domain.value)
    
    # Add query type tag
    tags.append(enhanced_query.query_content.query_type.value)
    
    # Add analysis type tag if available
    if enhanced_query.semantic_context.analysis_type:
        tags.append(enhanced_query.semantic_context.analysis_type.value)
    
    # Add investigation phase tag if available
    if enhanced_query.investigation_context.investigation_phase:
        tags.append(f"phase_{enhanced_query.investigation_context.investigation_phase}")
    
    # Add complexity tag
    complexity_score = enhanced_query.technical_metadata.complexity_score
    if complexity_score <= 3:
        tags.append("low_complexity")
    elif complexity_score <= 6:
        tags.append("medium_complexity")
    else:
        tags.append("high_complexity")
    
    # Add performance tag
    tags.append(f"performance_{enhanced_query.technical_metadata.performance_tier.value}")
    
    # Add automatic learning tag
    tags.append("auto_recorded")
    
    return tags


def _generate_query_description(self, query: str, business_context: Optional[Dict[str, Any]] = None) -> Optional[str]:
    """Generate a human-readable description of the query."""
    if business_context and business_context.get("business_question"):
        return business_context["business_question"]
    
    # Simple description generation based on query patterns
    query_lower = query.lower()
    
    if "count(*)" in query_lower:
        table_match = re.search(r'from\s+([a-zA-Z_][a-zA-Z0-9_]*)', query_lower)
        if table_match:
            return f"Count total records in {table_match.group(1)} table"
    
    if "group by" in query_lower and "sum(" in query_lower:
        return "Aggregate analysis with grouping and summation"
    
    if "join" in query_lower:
        return "Multi-table analysis with data joining"
    
    if "order by" in query_lower and "limit" in query_lower:
        return "Ranked analysis with top results"
    
    return None