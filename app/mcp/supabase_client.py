"""
Supabase MCP Client

Provides Supabase operations through MCP protocol.
"""

from typing import Dict, Any, List, Optional
from mcp.client.session import ClientSession

from ..utils.logging import logger


class SupabaseClient:
    """Supabase operations through MCP."""
    
    def __init__(self, session: ClientSession):
        self.session = session
    
    async def execute_query(
        self,
        query: str,
        parameters: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Execute query on Supabase through MCP."""
        try:
            result = await self.session.call_tool(
                "execute_sql",
                {
                    "query": query,
                    "parameters": parameters or {}
                }
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Supabase MCP query failed: {e}")
            raise
    
    async def insert_data(
        self,
        table: str,
        data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Insert data into Supabase table through MCP."""
        try:
            result = await self.session.call_tool(
                "insert",
                {
                    "table": table,
                    "data": data
                }
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Supabase MCP insert failed: {e}")
            raise
    
    async def update_data(
        self,
        table: str,
        data: Dict[str, Any],
        filters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Update data in Supabase table through MCP."""
        try:
            result = await self.session.call_tool(
                "update",
                {
                    "table": table,
                    "data": data,
                    "filters": filters
                }
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Supabase MCP update failed: {e}")
            raise
    
    async def select_data(
        self,
        table: str,
        columns: List[str] = None,
        filters: Dict[str, Any] = None,
        limit: int = None,
        offset: int = None
    ) -> Dict[str, Any]:
        """Select data from Supabase table through MCP."""
        try:
            result = await self.session.call_tool(
                "select",
                {
                    "table": table,
                    "columns": columns or ["*"],
                    "filters": filters or {},
                    "limit": limit,
                    "offset": offset
                }
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Supabase MCP select failed: {e}")
            raise
    
    async def delete_data(
        self,
        table: str,
        filters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Delete data from Supabase table through MCP."""
        try:
            result = await self.session.call_tool(
                "delete",
                {
                    "table": table,
                    "filters": filters
                }
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Supabase MCP delete failed: {e}")
            raise
    
    async def call_function(
        self,
        function_name: str,
        parameters: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Call Supabase function through MCP."""
        try:
            result = await self.session.call_tool(
                "call_function",
                {
                    "function": function_name,
                    "parameters": parameters or {}
                }
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Supabase MCP function call failed: {e}")
            raise
    
    async def get_table_info(self, table: str) -> Dict[str, Any]:
        """Get table information through MCP."""
        try:
            result = await self.session.call_tool(
                "describe_table",
                {"table": table}
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Supabase MCP table info failed: {e}")
            raise
    
    async def test_connection(self) -> bool:
        """Test Supabase MCP connection."""
        try:
            await self.session.call_tool("ping", {})
            return True
        except Exception as e:
            logger.error(f"Supabase MCP connection test failed: {e}")
            return False