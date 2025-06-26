"""
Database Operation API Routes

Provides secure SQL execution and database introspection tools.
These endpoints are automatically exposed as MCP tools for the agent.
"""

from typing import List, Dict, Any, Optional

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field

from ...database.operations import DatabaseOperations
from ...database.models import QueryResult, TableSchema
from ...guardrails.query_guard import QueryGuard
from ..app_factory import get_database_manager


router = APIRouter()


class SQLExecuteRequest(BaseModel):
    """Request to execute SQL query."""
    query: str = Field(..., description="SQL query to execute")
    database: str = Field("mariadb", description="Target database (mariadb/postgres)")
    max_rows: Optional[int] = Field(1000, description="Maximum rows to return")
    timeout: Optional[int] = Field(30, description="Query timeout in seconds")


class SQLExecuteResponse(BaseModel):
    """Response from SQL execution."""
    success: bool
    data: Optional[List[Dict[str, Any]]] = None
    columns: Optional[List[str]] = None
    row_count: int = 0
    execution_time: float = 0.0
    error: Optional[str] = None


class SchemaRequest(BaseModel):
    """Request to get database schema."""
    database: str = Field("mariadb", description="Target database")
    table_name: Optional[str] = Field(None, description="Specific table (optional)")


class SchemaResponse(BaseModel):
    """Database schema information."""
    database: str
    tables: List[TableSchema]


async def get_database_ops() -> DatabaseOperations:
    """Dependency to get database operations."""
    db_manager = get_database_manager()
    if not db_manager:
        raise HTTPException(status_code=500, detail="Database not available")
    return DatabaseOperations(db_manager)


@router.post("/execute", response_model=SQLExecuteResponse)
async def execute_sql(
    request: SQLExecuteRequest,
    db_ops: DatabaseOperations = Depends(get_database_ops)
):
    """
    Execute SQL query with safety checks.
    
    This endpoint is automatically exposed as an MCP tool for the agent
    to use during autonomous investigations.
    """
    try:
        # Validate query safety
        guard = QueryGuard()
        validation_result = guard.validate_query(request.query)
        
        if not validation_result.is_safe:
            raise HTTPException(
                status_code=400,
                detail=f"Unsafe query: {validation_result.reason}"
            )
        
        # Execute query
        result = await db_ops.execute_query(
            query=request.query,
            database=request.database,
            max_rows=request.max_rows,
            timeout=request.timeout
        )
        
        return SQLExecuteResponse(
            success=True,
            data=result.data,
            columns=result.columns,
            row_count=result.row_count,
            execution_time=result.execution_time
        )
        
    except Exception as e:
        return SQLExecuteResponse(
            success=False,
            error=str(e)
        )


@router.get("/schema", response_model=SchemaResponse)
async def get_database_schema(
    database: str = "mariadb",
    table_name: Optional[str] = None,
    db_ops: DatabaseOperations = Depends(get_database_ops)
):
    """
    Get database schema information.
    
    Returns table structures, columns, and relationships.
    Exposed as MCP tool for agent schema discovery.
    """
    try:
        schema_info = await db_ops.get_schema(
            database=database,
            table_name=table_name
        )
        
        return SchemaResponse(
            database=database,
            tables=schema_info
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/validate", response_model=dict)
async def validate_sql(query: str):
    """
    Validate SQL query without executing it.
    
    Checks for syntax errors and security issues.
    """
    try:
        guard = QueryGuard()
        validation_result = guard.validate_query(query)
        
        return {
            "is_valid": validation_result.is_safe,
            "syntax_valid": validation_result.syntax_valid,
            "security_issues": validation_result.security_issues,
            "suggestions": validation_result.suggestions
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/tables", response_model=List[str])
async def list_tables(
    database: str = "mariadb",
    db_ops: DatabaseOperations = Depends(get_database_ops)
):
    """
    List all tables in the database.
    
    Quick reference for the agent to discover available data sources.
    """
    try:
        tables = await db_ops.list_tables(database)
        return tables
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/tables/{table_name}/sample", response_model=SQLExecuteResponse)
async def get_table_sample(
    table_name: str,
    database: str = "mariadb",
    limit: int = 10,
    db_ops: DatabaseOperations = Depends(get_database_ops)
):
    """
    Get sample data from a table.
    
    Helps the agent understand data structure and content.
    """
    try:
        # Construct safe sample query
        query = f"SELECT * FROM {table_name} LIMIT {limit}"
        
        result = await db_ops.execute_query(
            query=query,
            database=database,
            max_rows=limit
        )
        
        return SQLExecuteResponse(
            success=True,
            data=result.data,
            columns=result.columns,
            row_count=result.row_count,
            execution_time=result.execution_time
        )
        
    except Exception as e:
        return SQLExecuteResponse(
            success=False,
            error=str(e)
        )