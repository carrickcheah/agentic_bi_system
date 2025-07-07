"""
Business Service Layer - Database Operations via MCP Clients

This service layer provides the business logic interface for all database
operations. It uses MCP clients to connect to external database servers
and abstracts these operations into business methods.

Features:
- Business-focused API (not database-focused)
- Multi-database coordination via MCP clients
- Transaction management across databases
- Error handling and retry logic
- Performance optimization
- Semantic caching integration

Note: This service USES MCP clients to connect to external MCP servers.
It is NOT an MCP server itself.
"""

from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from dataclasses import dataclass
import asyncio
import hashlib
import json

try:
    from ..utils.logging import logger
    from ..utils.exceptions import BusinessLogicError, DatabaseOperationError
    from ..lance_db.runner import SQLEmbeddingService
    from ..lance_db.src.enhanced_schema import (
        EnhancedSQLQuery, QueryContent, SemanticContext, TechnicalMetadata,
        UserContext, InvestigationContext, ExecutionResults, LearningMetadata,
        BusinessIntelligence, create_enhanced_query_from_simple, validate_enhanced_query,
        QueryType, BusinessDomain, UserRole, ExecutionStatus, AnalysisType, ComplexityTier
    )
except ImportError:
    import logging
    logger = logging.getLogger(__name__)
    # Simple exception classes for standalone mode
    class BusinessLogicError(Exception):
        pass
    class DatabaseOperationError(Exception):
        pass
    # Placeholder for enhanced schema in standalone mode
    class SQLEmbeddingService:
        async def initialize(self): pass
        async def store_sql_query(self, data): return "placeholder"
        async def cleanup(self): pass
    class EnhancedSQLQuery:
        def __init__(self): pass
    def create_enhanced_query_from_simple(*args, **kwargs):
        return EnhancedSQLQuery()
from .client_manager import MCPClientManager


@dataclass
class QueryResult:
    """Result of a database query operation."""
    data: List[Dict[str, Any]]
    columns: List[str]
    row_count: int
    execution_time: float
    database: str
    success: bool = True
    error: Optional[str] = None


@dataclass
class InvestigationResult:
    """Result of an investigation operation."""
    investigation_id: str
    status: str
    findings: List[Dict[str, Any]]
    insights: List[str]
    confidence_score: float
    execution_time: float
    created_at: datetime


class BusinessService:
    """
    Business logic service layer for database operations via MCP clients.
    
    This service provides high-level business operations that abstract
    away the complexity of multiple databases and MCP clients.
    """
    
    def __init__(self, client_manager: MCPClientManager, enable_query_learning: bool = True):
        self.client_manager = client_manager
        self._initialized = False
        self.enable_query_learning = enable_query_learning
        self.sql_embedding_service: Optional[SQLEmbeddingService] = None
    
    async def initialize(self):
        """Initialize the service layer."""
        if self._initialized:
            return
        
        # Ensure client manager is initialized
        if not self.client_manager._initialized:
            await self.client_manager.initialize()
        
        # Initialize SQL embedding service for query learning
        if self.enable_query_learning:
            try:
                self.sql_embedding_service = SQLEmbeddingService()
                await self.sql_embedding_service.initialize()
                logger.info("SQL query learning enabled with LanceDB integration")
            except Exception as e:
                logger.warning(f"SQL query learning disabled due to initialization error: {e}")
                self.sql_embedding_service = None
        
        self._initialized = True
        logger.info("FastMCP service layer initialized")
    
    async def cleanup(self):
        """Cleanup service resources."""
        if self.sql_embedding_service:
            try:
                await self.sql_embedding_service.cleanup()
            except Exception as e:
                logger.warning(f"Error cleaning up SQL embedding service: {e}")
        
        self._initialized = False
        logger.info("FastMCP service layer cleaned up")
    
    def is_healthy(self) -> bool:
        """Check if the service is healthy."""
        return self._initialized and self.client_manager.is_healthy()
    
    # Database Operations
    
    async def execute_sql(
        self,
        query: str,
        database: str = "mariadb",
        max_rows: int = 1000,
        timeout: int = 30,
        user_id: str = "system",
        business_context: Optional[Dict[str, Any]] = None
    ) -> QueryResult:
        """
        Execute SQL query on specified database with automatic learning.
        
        Args:
            query: SQL query to execute
            database: Target database (mariadb, postgres, supabase)
            max_rows: Maximum rows to return
            timeout: Query timeout in seconds
            user_id: User executing the query
            business_context: Additional business context for learning
            
        Returns:
            QueryResult with data and metadata
        """
        start_time = datetime.utcnow()
        
        try:
            logger.info(f"Executing SQL on {database}: {query[:100]}...")
            
            # Get appropriate client
            client = self._get_database_client(database)
            if not client:
                raise DatabaseOperationError(f"Database client '{database}' not available")
            
            # Execute query with timeout
            result = await asyncio.wait_for(
                client.execute_query(query, max_rows=max_rows),
                timeout=timeout
            )
            
            execution_time = (datetime.utcnow() - start_time).total_seconds()
            
            query_result = QueryResult(
                data=result.get("data", []),
                columns=result.get("columns", []),
                row_count=result.get("row_count", 0),
                execution_time=execution_time,
                database=database,
                success=True
            )
            
            # Record successful query for learning (async, non-blocking)
            if self.sql_embedding_service and self._should_record_query(query, database, business_context):
                asyncio.create_task(self._record_query_for_learning(
                    query=query,
                    database=database,
                    execution_time_ms=execution_time * 1000,
                    row_count=query_result.row_count,
                    user_id=user_id,
                    success=True,
                    business_context=business_context
                ))
            
            return query_result
            
        except asyncio.TimeoutError:
            execution_time = (datetime.utcnow() - start_time).total_seconds()
            error = f"Query timeout after {timeout} seconds"
            logger.error(f"SQL execution timeout: {error}")
            
            query_result = QueryResult(
                data=[],
                columns=[],
                row_count=0,
                execution_time=execution_time,
                database=database,
                success=False,
                error=error
            )
            
            # Record failed query for learning (async, non-blocking)
            if self.sql_embedding_service and self._should_record_query(query, database, business_context):
                asyncio.create_task(self._record_query_for_learning(
                    query=query,
                    database=database,
                    execution_time_ms=execution_time * 1000,
                    row_count=0,
                    user_id=user_id,
                    success=False,
                    business_context=business_context,
                    error=error
                ))
            
            return query_result
            
        except Exception as e:
            execution_time = (datetime.utcnow() - start_time).total_seconds()
            error = str(e)
            logger.error(f"SQL execution failed: {error}")
            
            query_result = QueryResult(
                data=[],
                columns=[],
                row_count=0,
                execution_time=execution_time,
                database=database,
                success=False,
                error=error
            )
            
            # Record failed query for learning (async, non-blocking)
            if self.sql_embedding_service and self._should_record_query(query, database, business_context):
                asyncio.create_task(self._record_query_for_learning(
                    query=query,
                    database=database,
                    execution_time_ms=execution_time * 1000,
                    row_count=0,
                    user_id=user_id,
                    success=False,
                    business_context=business_context,
                    error=error
                ))
            
            return query_result
    
    async def get_database_schema(
        self,
        database: str = "mariadb",
        table_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get database schema information.
        
        Args:
            database: Target database
            table_name: Specific table (optional)
            
        Returns:
            Schema information dictionary
        """
        try:
            logger.info(f"Getting schema for {database}" + (f".{table_name}" if table_name else ""))
            
            client = self._get_database_client(database)
            if not client:
                raise DatabaseOperationError(f"Database client '{database}' not available")
            
            # Get schema information
            if table_name:
                schema_info = await client.get_table_schema(table_name)
            else:
                schema_info = await client.get_database_schema()
            
            return {
                "database": database,
                "schema": schema_info,
                "retrieved_at": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Schema retrieval failed for {database}: {e}")
            raise DatabaseOperationError(f"Failed to get schema: {e}")
    
    async def list_tables(self, database: str = "mariadb") -> List[str]:
        """
        List all tables in the database.
        
        Args:
            database: Target database
            
        Returns:
            List of table names
        """
        try:
            client = self._get_database_client(database)
            if not client:
                raise DatabaseOperationError(f"Database client '{database}' not available")
            
            tables = await client.list_tables()
            return tables
            
        except Exception as e:
            logger.error(f"Table listing failed for {database}: {e}")
            raise DatabaseOperationError(f"Failed to list tables: {e}")
    
    # Vector Operations (LanceDB)
    
    async def store_query_pattern(
        self,
        sql_query: str,
        description: str,
        result_summary: Optional[str] = None,
        execution_time: Optional[float] = None,
        success: bool = True,
        user_id: str = "system",
        business_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Store SQL query pattern for semantic search using LanceDB.
        
        Args:
            sql_query: The SQL query
            description: Human-readable description
            result_summary: Summary of results
            execution_time: Query execution time
            success: Whether query was successful
            user_id: User who executed the query
            business_context: Additional business context
            
        Returns:
            Storage result
        """
        try:
            if not self.sql_embedding_service:
                logger.warning("SQL embedding service not available for pattern storage")
                return {"status": "skipped", "reason": "SQL embedding service not available"}
            
            # Prepare query data for LanceDB storage
            query_data = {
                "sql_query": sql_query,
                "database": "mariadb",  # Default for business queries
                "query_type": self._classify_query_type(sql_query, business_context),
                "execution_time_ms": execution_time * 1000 if execution_time else 0.0,
                "row_count": 0,  # Unknown for manual storage
                "user_id": user_id,
                "success": success,
                "metadata": {
                    "description": description,
                    "result_summary": result_summary,
                    "business_context": business_context or {},
                    "storage_method": "manual"
                }
            }
            
            # Store in LanceDB
            query_id = await self.sql_embedding_service.store_sql_query(query_data)
            
            result = {
                "status": "stored",
                "pattern_id": query_id,
                "description": description,
                "stored_at": datetime.utcnow().isoformat()
            }
            
            logger.info(f"Stored query pattern in LanceDB: {description}")
            return result
            
        except Exception as e:
            logger.error(f"Failed to store query pattern: {e}")
            raise DatabaseOperationError(f"Vector storage failed: {e}")
    
    async def find_similar_queries(
        self,
        description: str,
        limit: int = 5,
        threshold: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        Find similar SQL queries based on description using LanceDB.
        
        Args:
            description: Query description to search for
            limit: Maximum number of results
            threshold: Similarity threshold (optional)
            
        Returns:
            List of similar queries with metadata
        """
        try:
            if not self.sql_embedding_service:
                logger.warning("SQL embedding service not available for similarity search")
                return []
            
            # Use LanceDB for similarity search
            results = await self.sql_embedding_service.find_similar_queries(
                query=description,
                threshold=threshold,
                limit=limit
            )
            
            logger.info(f"Found {len(results)} similar queries for: {description}")
            return results
            
        except Exception as e:
            logger.error(f"Similar query search failed: {e}")
            raise DatabaseOperationError(f"Vector search failed: {e}")
    
    # Investigation Operations
    
    async def create_investigation(
        self,
        query: str,
        user_id: str,
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Create a new investigation.
        
        Args:
            query: Investigation query
            user_id: User creating the investigation
            context: Additional context
            
        Returns:
            Investigation ID
        """
        try:
            if not self.client_manager.postgres:
                raise DatabaseOperationError("PostgreSQL client not available")
            
            investigation_id = await self.client_manager.postgres.create_investigation(
                query=query,
                user_id=user_id,
                context=context or {}
            )
            
            logger.info(f"Created investigation {investigation_id} for user {user_id}")
            return investigation_id
            
        except Exception as e:
            logger.error(f"Investigation creation failed: {e}")
            raise BusinessLogicError(f"Failed to create investigation: {e}")
    
    async def get_investigation(self, investigation_id: str) -> Optional[Dict[str, Any]]:
        """
        Get investigation by ID.
        
        Args:
            investigation_id: Investigation ID
            
        Returns:
            Investigation data or None if not found
        """
        try:
            if not self.client_manager.postgres:
                raise DatabaseOperationError("PostgreSQL client not available")
            
            investigation = await self.client_manager.postgres.get_investigation(investigation_id)
            return investigation
            
        except Exception as e:
            logger.error(f"Investigation retrieval failed: {e}")
            raise BusinessLogicError(f"Failed to get investigation: {e}")
    
    async def update_investigation_status(
        self,
        investigation_id: str,
        status: str,
        results: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Update investigation status and results.
        
        Args:
            investigation_id: Investigation ID
            status: New status
            results: Investigation results
            
        Returns:
            True if updated successfully
        """
        try:
            if not self.client_manager.postgres:
                raise DatabaseOperationError("PostgreSQL client not available")
            
            success = await self.client_manager.postgres.update_investigation_status(
                investigation_id=investigation_id,
                status=status,
                results=results
            )
            
            logger.info(f"Updated investigation {investigation_id} status to {status}")
            return success
            
        except Exception as e:
            logger.error(f"Investigation update failed: {e}")
            raise BusinessLogicError(f"Failed to update investigation: {e}")
    
    # Business Intelligence Operations
    
    async def analyze_business_query(
        self,
        query: str,
        domain: Optional[str] = None,
        user_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Analyze business query for patterns and insights.
        
        Args:
            query: Business query to analyze
            domain: Business domain (optional)
            user_context: User context for personalization
            
        Returns:
            Analysis results with recommendations
        """
        try:
            # This is a high-level business operation that coordinates
            # multiple services and databases
            
            analysis_result = {
                "query": query,
                "domain": domain,
                "analysis": {
                    "intent": "data_exploration",  # Would be determined by AI
                    "complexity": "medium",
                    "estimated_time": "2-5 minutes",
                    "required_databases": ["mariadb"],
                    "recommended_approach": "step_by_step_investigation"
                },
                "suggestions": [
                    "Start with schema exploration",
                    "Identify relevant tables",
                    "Execute targeted queries"
                ],
                "similar_queries": await self.find_similar_queries(query, limit=3),
                "timestamp": datetime.utcnow().isoformat()
            }
            
            logger.info(f"Analyzed business query: {query[:50]}...")
            return analysis_result
            
        except Exception as e:
            logger.error(f"Business query analysis failed: {e}")
            raise BusinessLogicError(f"Query analysis failed: {e}")
    
    # Helper Methods
    
    def _get_database_client(self, database: str):
        """Get the appropriate database client."""
        client_map = {
            "mariadb": self.client_manager.mariadb,
            "postgres": self.client_manager.postgres,
            "postgresql": self.client_manager.postgres
        }
        return client_map.get(database.lower())
    
    def _should_record_query(self, query: str, database: str, business_context: Optional[Dict[str, Any]] = None) -> bool:
        """
        Determine if a query should be recorded for learning.
        
        Args:
            query: The SQL query
            database: Target database
            business_context: Business context information
            
        Returns:
            True if query should be recorded for learning
        """
        # Don't record operational queries on PostgreSQL (agent memory, sessions)
        if database.lower() in ["postgres", "postgresql"]:
            return False
        
        # Don't record simple administrative queries
        query_lower = query.lower().strip()
        
        # Skip basic system queries
        skip_patterns = [
            "show tables",
            "describe ",
            "select version()",
            "select database()",
            "show databases",
            "information_schema",
            "pg_catalog",
            "select 1",
            "show status"
        ]
        
        for pattern in skip_patterns:
            if pattern in query_lower:
                return False
        
        # Record business queries on MariaDB
        if database.lower() == "mariadb":
            # Record analytical queries with business context
            if business_context and business_context.get("investigation_phase"):
                return True
            
            # Record complex queries (containing joins, aggregations, etc.)
            if any(keyword in query_lower for keyword in ["join", "group by", "having", "union", "with"]):
                return True
            
            # Record queries with reasonable length (not just "SELECT 1")
            if len(query.strip()) > 30:
                return True
        
        return False
    
    def _classify_query_type(self, query: str, business_context: Optional[Dict[str, Any]] = None) -> str:
        """
        Classify the type of SQL query for learning purposes.
        
        Args:
            query: The SQL query
            business_context: Business context information
            
        Returns:
            Query type: "simple", "analytical", "operational", or "investigative"
        """
        query_lower = query.lower().strip()
        
        # Check for investigative context
        if business_context:
            if business_context.get("investigation_phase"):
                return "investigative"
            if business_context.get("business_domain"):
                return "analytical"
        
        # Analytical queries
        analytical_patterns = [
            "group by", "having", "union", "join", "subquery", 
            "with", "case when", "window", "over(", "partition by"
        ]
        if any(pattern in query_lower for pattern in analytical_patterns):
            return "analytical"
        
        # Simple CRUD operations
        simple_patterns = ["insert", "update", "delete", "select * from", "where id ="]
        if any(pattern in query_lower for pattern in simple_patterns):
            if "join" not in query_lower and "group by" not in query_lower:
                return "simple"
        
        # Operational queries
        operational_patterns = [
            "create table", "alter table", "drop table", "create index",
            "show", "describe", "explain", "information_schema"
        ]
        if any(pattern in query_lower for pattern in operational_patterns):
            return "operational"
        
        # Default to analytical for complex queries
        return "analytical"
    
    async def _record_query_for_learning(
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
        
        Args:
            query: The SQL query
            database: Target database
            execution_time_ms: Execution time in milliseconds
            row_count: Number of rows returned
            user_id: User who executed the query
            success: Whether query was successful
            business_context: Rich business context information
            error: Error message if query failed
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
            
            # Store enhanced query in LanceDB
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
        """Create comprehensive enhanced query with production-grade metadata."""
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
    
    # Additional enhanced helper methods (adding abbreviated versions for space)
    def _classify_analysis_type(self, query: str) -> Optional[AnalysisType]:
        query_lower = query.lower()
        if any(p in query_lower for p in ["order by", "limit", "top", "rank"]):
            return AnalysisType.RANKING
        elif any(p in query_lower for p in ["group by", "sum", "count", "avg"]):
            return AnalysisType.AGGREGATION
        return None
    
    def _extract_time_dimension(self, query: str) -> Optional[str]:
        query_lower = query.lower()
        if any(p in query_lower for p in ["day", "daily"]): return "daily"
        elif any(p in query_lower for p in ["month", "monthly"]): return "monthly"
        elif any(p in query_lower for p in ["year", "yearly"]): return "yearly"
        return None
    
    def _extract_metrics(self, query: str) -> List[str]:
        metrics = []
        query_lower = query.lower()
        metric_patterns = ["count", "sum", "avg", "max", "min", "revenue", "profit", "cost"]
        for metric in metric_patterns:
            if metric in query_lower:
                metrics.append(metric)
        return metrics
    
    def _extract_entities(self, query: str) -> List[str]:
        entities = []
        query_lower = query.lower()
        entity_patterns = ["customer", "user", "product", "order", "employee", "department"]
        for entity in entity_patterns:
            if entity in query_lower:
                entities.append(entity)
        return entities
    
    def _extract_business_concepts(self, query: str, business_context: Optional[Dict[str, Any]] = None) -> List[str]:
        concepts = []
        query_lower = query.lower()
        concept_patterns = ["performance", "analysis", "trend", "ranking", "efficiency"]
        for concept in concept_patterns:
            if concept in query_lower:
                concepts.append(concept)
        if business_context and business_context.get("investigation_phase"):
            concepts.append(f"investigation_{business_context['investigation_phase']}")
        return concepts
    
    def _extract_keywords_enhanced(self, query: str) -> List[str]:
        import re
        sql_keywords = {'select', 'from', 'where', 'join', 'group', 'by', 'order', 'having'}
        words = re.findall(r'\\b[a-zA-Z_][a-zA-Z0-9_]*\\b', query.lower())
        keywords = [w for w in words if w not in sql_keywords and len(w) > 2]
        return list(set(keywords))[:20]
    
    def _extract_table_names_enhanced(self, query: str) -> List[str]:
        import re
        patterns = [r'from\\s+([a-zA-Z_][a-zA-Z0-9_]*)', r'join\\s+([a-zA-Z_][a-zA-Z0-9_]*)']
        tables = set()
        for pattern in patterns:
            tables.update(re.findall(pattern, query, re.IGNORECASE))
        return list(tables)
    
    def _count_joins_enhanced(self, query: str) -> int:
        import re
        return len(re.findall(r'\\bjoin\\b', query, re.IGNORECASE))
    
    def _extract_aggregation_functions(self, query: str) -> List[str]:
        functions = []
        query_lower = query.lower()
        agg_functions = ['count', 'sum', 'avg', 'max', 'min']
        for func in agg_functions:
            if f"{func}(" in query_lower:
                functions.append(func.upper())
        return functions
    
    def _calculate_complexity_score_enhanced(self, query: str) -> int:
        query_lower = query.lower()
        score = 1
        if 'join' in query_lower: score += query_lower.count('join')
        if 'group by' in query_lower: score += 1
        if 'having' in query_lower: score += 1
        if 'union' in query_lower: score += 2
        if 'with' in query_lower: score += 2
        return min(score, 10)
    
    def _classify_performance_tier(self, execution_time_ms: float) -> ComplexityTier:
        if execution_time_ms < 100: return ComplexityTier.LOW
        elif execution_time_ms < 1000: return ComplexityTier.MEDIUM
        elif execution_time_ms < 5000: return ComplexityTier.HIGH
        else: return ComplexityTier.VERY_HIGH
    
    def _identify_query_pattern(self, query: str) -> Optional[str]:
        query_lower = query.lower()
        if "order by" in query_lower and "limit" in query_lower: return "top_n_analysis"
        elif "group by" in query_lower: return "aggregation_analysis"
        elif "join" in query_lower: return "join_analysis"
        return None
    
    def _has_subqueries(self, query: str) -> bool:
        return '(' in query and 'select' in query.lower()
    
    def _has_window_functions(self, query: str) -> bool:
        query_lower = query.lower()
        return any(f in query_lower for f in ['over(', 'partition by', 'row_number'])
    
    def _generate_optimization_hints(self, query: str) -> List[str]:
        hints = []
        query_lower = query.lower()
        if 'order by' in query_lower and 'limit' in query_lower:
            hints.append("consider_index_on_order_column")
        if 'select *' in query_lower:
            hints.append("specify_required_columns_only")
        return hints
    
    def _infer_user_role(self, user_id: str, business_context: Optional[Dict[str, Any]] = None) -> Optional[UserRole]:
        if business_context and "user_role" in business_context:
            try:
                return UserRole(business_context["user_role"])
            except ValueError:
                pass
        user_id_lower = user_id.lower()
        if "analyst" in user_id_lower: return UserRole.ANALYST
        elif "manager" in user_id_lower: return UserRole.MANAGER
        elif any(r in user_id_lower for r in ["ceo", "cto", "vp"]): return UserRole.EXECUTIVE
        return UserRole.BUSINESS_USER
    
    def _classify_decision_support_level(self, query: str, business_context: Optional[Dict[str, Any]] = None) -> str:
        query_lower = query.lower()
        if any(i in query_lower for i in ["revenue", "profit", "strategic"]): return "strategic"
        elif any(i in query_lower for i in ["efficiency", "performance"]): return "tactical"
        return "operational"
    
    def _identify_stakeholder_relevance(self, business_domain: BusinessDomain) -> List[str]:
        mapping = {
            BusinessDomain.SALES: ["sales_team", "sales_manager"],
            BusinessDomain.FINANCE: ["finance_team", "cfo"],
            BusinessDomain.HR: ["hr_team", "hr_manager"],
            BusinessDomain.PRODUCTION: ["operations_team", "production_manager"],
            BusinessDomain.MARKETING: ["marketing_team", "marketing_manager"]
        }
        return mapping.get(business_domain, ["analyst", "manager"])
    
    def _calculate_actionability_score(self, query: str, business_context: Optional[Dict[str, Any]] = None) -> float:
        score = 0.5
        query_lower = query.lower()
        if any(m in query_lower for m in ["count", "sum", "avg"]): score += 0.2
        if business_context and business_context.get("business_impact") == "high": score += 0.2
        return min(score, 1.0)
    
    def _assess_business_value(self, query: str, business_context: Optional[Dict[str, Any]] = None) -> str:
        query_lower = query.lower()
        if any(i in query_lower for i in ["revenue", "profit", "customer"]): return "high"
        elif any(i in query_lower for i in ["performance", "analysis"]): return "medium"
        return "low"
    
    def _generate_tags(self, enhanced_query: EnhancedSQLQuery) -> List[str]:
        tags = [
            enhanced_query.semantic_context.business_domain.value,
            enhanced_query.query_content.query_type.value,
            f"complexity_{enhanced_query.technical_metadata.complexity_score}",
            f"performance_{enhanced_query.technical_metadata.performance_tier.value}",
            "auto_recorded"
        ]
        if enhanced_query.investigation_context.investigation_phase:
            tags.append(f"phase_{enhanced_query.investigation_context.investigation_phase}")
        return tags
    
    def _generate_query_description(self, query: str, business_context: Optional[Dict[str, Any]] = None) -> Optional[str]:
        if business_context and business_context.get("business_question"):
            return business_context["business_question"]
        query_lower = query.lower()
        if "count(*)" in query_lower: return "Count analysis"
        elif "group by" in query_lower and "sum(" in query_lower: return "Aggregation analysis"
        elif "join" in query_lower: return "Multi-table analysis"
        return None
    
    # Health and Status
    
    async def get_service_status(self) -> Dict[str, Any]:
        """
        Get comprehensive service status.
        
        Returns:
            Service status information
        """
        return {
            "service": {
                "name": "FastMCP Service",
                "status": "healthy" if self.is_healthy() else "unhealthy",
                "initialized": self._initialized
            },
            "databases": {
                "mariadb": bool(self.client_manager.mariadb),
                "postgres": bool(self.client_manager.postgres),
                "lancedb": "pending_integration",
                "graphrag": bool(self.client_manager.graphrag)
            },
            "client_manager": {
                "status": "healthy" if self.client_manager.is_healthy() else "unhealthy",
                "active_sessions": len(self.client_manager.sessions)
            },
            "timestamp": datetime.utcnow().isoformat()
        }