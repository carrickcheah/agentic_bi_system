"""
Database Operation API Routes

Provides secure SQL execution and database introspection tools.
These endpoints are automatically exposed as MCP tools for the agent.
"""

from typing import List, Dict, Any, Optional

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field

from ...guardrails.sql_validator import SQLValidator
from ...bridge.service_bridge import get_service_bridge
from ...fastmcp.service import QueryResult as FastMCPQueryResult


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
    schema: Dict[str, Any]


async def get_service_bridge_dep():
    """Dependency to get service bridge instance."""
    bridge = get_service_bridge()
    if not await bridge.is_healthy():
        raise HTTPException(status_code=500, detail="FastMCP backend not available")
    return bridge


@router.post("/execute", response_model=SQLExecuteResponse)
async def execute_sql(
    request: SQLExecuteRequest,
    bridge = Depends(get_service_bridge_dep)
):
    """
    Execute SQL query with safety checks.
    
    This endpoint is automatically exposed as an MCP tool for the agent
    to use during autonomous investigations.
    """
    try:
        # Validate query safety
        validator = SQLValidator()
        validation_result = validator.validate_query(request.query)
        
        if not validation_result.is_safe:
            raise HTTPException(
                status_code=400,
                detail=f"Unsafe query: {validation_result.reason}"
            )
        
        # Execute query through service bridge
        result = await bridge.execute_sql(
            query=request.query,
            database=request.database,
            max_rows=request.max_rows,
            timeout=request.timeout
        )
        
        return SQLExecuteResponse(
            success=result.success,
            data=result.data,
            columns=result.columns,
            row_count=result.row_count,
            execution_time=result.execution_time,
            error=result.error
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
    bridge = Depends(get_service_bridge_dep)
):
    """
    Get database schema information.
    
    Returns table structures, columns, and relationships.
    Exposed as MCP tool for agent schema discovery.
    """
    try:
        schema_info = await bridge.get_database_schema(
            database=database,
            table_name=table_name
        )
        
        return SchemaResponse(
            database=database,
            schema=schema_info
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
        validator = SQLValidator()
        validation_result = validator.validate_query(query)
        
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
    bridge = Depends(get_service_bridge_dep)
):
    """
    List all tables in the database.
    
    Quick reference for the agent to discover available data sources.
    """
    try:
        tables = await bridge.list_tables(database)
        return tables
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/tables/{table_name}/sample", response_model=SQLExecuteResponse)
async def get_table_sample(
    table_name: str,
    database: str = "mariadb",
    limit: int = 10,
    bridge = Depends(get_service_bridge_dep)
):
    """
    Get sample data from a table.
    
    Helps the agent understand data structure and content.
    """
    try:
        # Construct safe sample query
        query = f"SELECT * FROM {table_name} LIMIT {limit}"
        
        result = await bridge.execute_sql(
            query=query,
            database=database,
            max_rows=limit
        )
        
        return SQLExecuteResponse(
            success=result.success,
            data=result.data,
            columns=result.columns,
            row_count=result.row_count,
            execution_time=result.execution_time,
            error=result.error
        )
        
    except Exception as e:
        return SQLExecuteResponse(
            success=False,
            error=str(e)
        )