"""
Vector Search Component for LanceDB
Handles similarity search operations on stored vectors.
"""

import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger("lance_db.search")


class VectorSearcher:
    """Handles vector similarity search operations."""
    
    def __init__(self, table):
        self.table = table
        self.logger = logger
        
    async def search(
        self, 
        query_vector: List[float], 
        limit: int = 10,
        where: Optional[str] = None,
        prefilter: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Search for similar vectors in the table.
        
        Args:
            query_vector: The query embedding vector
            limit: Maximum number of results
            where: Optional SQL WHERE clause for filtering
            prefilter: Whether to apply filters before similarity search
            
        Returns:
            List of similar items with similarity scores
        """
        try:
            # Build search query
            search = self.table.search(query_vector).limit(limit)
            
            if where:
                search = search.where(where, prefilter=prefilter)
            
            # Execute search
            results = await search.to_list()
            
            # Format results
            formatted_results = []
            for result in results:
                formatted_results.append({
                    **result,
                    "similarity": 1.0 - result.get("_distance", 0)  # Convert distance to similarity
                })
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            return []
    
    async def search_by_metadata(
        self,
        filters: Dict[str, Any],
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Search by metadata filters without vector similarity."""
        try:
            # Build WHERE clause from filters
            where_clauses = []
            for key, value in filters.items():
                if isinstance(value, str):
                    where_clauses.append(f"{key} = '{value}'")
                else:
                    where_clauses.append(f"{key} = {value}")
            
            where = " AND ".join(where_clauses)
            
            # Execute filtered search
            results = await self.table.search().where(where).limit(limit).to_list()
            
            return results
            
        except Exception as e:
            logger.error(f"Metadata search failed: {e}")
            return []