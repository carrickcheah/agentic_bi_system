"""
MariaDB MCP Client

Provides MariaDB operations through MCP protocol.
"""

from typing import Dict, Any, List, Optional
from mcp.client.session import ClientSession

from ..database.models import QueryResult, TableSchema, ColumnInfo
from ..utils.logging import logger


class MariaDBClient:
    """MariaDB operations through MCP."""
    
    def __init__(self, session: ClientSession):
        self.session = session
    
    async def execute_query(
        self,
        query: str,
        parameters: Dict[str, Any] = None,
        max_rows: int = 1000,
        timeout: int = 30
    ) -> QueryResult:
        """Execute SQL query on MariaDB through MCP."""
        try:
            # Call MCP tool for query execution
            result = await self.session.call_tool(
                "execute_sql",
                {
                    "query": query,
                    "parameters": parameters or {},
                    "max_rows": max_rows,
                    "timeout": timeout
                }
            )
            
            # Convert MCP result to QueryResult model
            return QueryResult(
                columns=result.get("columns", []),
                rows=result.get("rows", []),
                row_count=result.get("row_count", 0),
                execution_time=result.get("execution_time", 0.0),
                query=query
            )
            
        except Exception as e:
            logger.error(f"MariaDB MCP query failed: {e}")
            raise
    
    async def get_table_schema(self, table_name: str) -> TableSchema:
        """Get table schema information through MCP."""
        try:
            result = await self.session.call_tool(
                "describe_table",
                {"table_name": table_name}
            )
            
            columns = []
            for col_info in result.get("columns", []):
                columns.append(ColumnInfo(
                    name=col_info["name"],
                    data_type=col_info["type"],
                    is_nullable=col_info.get("nullable", True),
                    is_primary_key=col_info.get("primary_key", False),
                    default_value=col_info.get("default"),
                    comment=col_info.get("comment")
                ))
            
            return TableSchema(
                table_name=table_name,
                columns=columns,
                primary_keys=result.get("primary_keys", []),
                foreign_keys=result.get("foreign_keys", []),
                indexes=result.get("indexes", [])
            )
            
        except Exception as e:
            logger.error(f"MariaDB schema retrieval failed: {e}")
            raise
    
    async def list_tables(self) -> List[str]:
        """List all tables in the database through MCP."""
        try:
            result = await self.session.call_tool("list_tables", {})
            return result.get("tables", [])
            
        except Exception as e:
            logger.error(f"MariaDB table listing failed: {e}")
            raise
    
    async def get_table_count(self, table_name: str) -> int:
        """Get row count for a table through MCP."""
        try:
            result = await self.execute_query(f"SELECT COUNT(*) as count FROM {table_name}")
            return result.rows[0]["count"] if result.rows else 0
            
        except Exception as e:
            logger.error(f"MariaDB table count failed: {e}")
            raise
    
    async def test_connection(self) -> bool:
        """Test MariaDB MCP connection."""
        try:
            await self.execute_query("SELECT 1 as test")
            return True
        except Exception as e:
            logger.error(f"MariaDB MCP connection test failed: {e}")
            return False