"""
Example of how Intelligence module would use MCP when needed.
"""
from typing import Optional
from main import get_mcp_client_manager

class IntelligenceWithDatabase:
    """Example intelligence module that uses database when needed."""
    
    def __init__(self):
        self.mcp_manager = None
    
    async def analyze_with_data(self, query: str):
        """Analyze query with database access."""
        
        # Initialize MCP only when we need database
        if not self.mcp_manager:
            self.mcp_manager = await get_mcp_client_manager()
        
        # Now we can query database
        schema = await self.mcp_manager.mariadb.get_database_schema()
        
        # Do analysis with schema information
        return f"Analysis of '{query}' with {len(schema)} tables"
    
    async def cleanup(self):
        """Clean up resources."""
        if self.mcp_manager:
            await self.mcp_manager.close()
            self.mcp_manager = None

# Usage pattern
async def main():
    intelligence = IntelligenceWithDatabase()
    
    # First call initializes MCP
    result = await intelligence.analyze_with_data("sales trends")
    print(result)
    
    # Subsequent calls reuse the connection
    result2 = await intelligence.analyze_with_data("customer patterns")
    print(result2)
    
    # Clean up when done
    await intelligence.cleanup()