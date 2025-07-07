"""
Vector search component for LanceDB operations.
Handles similarity search and result filtering.
"""

import json
import time
from typing import List, Dict, Any, Optional
import numpy as np

try:
    from ..config import settings
    from .lance_logging import get_logger, log_operation, log_error, log_performance
except ImportError:
    # For standalone execution
    from config import settings
    from lance_logging import get_logger, log_operation, log_error, log_performance


class VectorSearcher:
    """Handles vector similarity search operations in LanceDB."""
    
    def __init__(self, table):
        self.table = table
        self.logger = get_logger("search")
    
    async def search_similar(
        self,
        query_vector: np.ndarray,
        threshold: float = 0.85,
        limit: int = 10,
        filter_conditions: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for similar vectors in the table.
        
        Args:
            query_vector: Query embedding vector
            threshold: Minimum similarity threshold (0-1)
            limit: Maximum number of results
            filter_conditions: Optional filters (e.g., {"database": "mariadb"})
        
        Returns:
            List of similar queries with metadata
        """
        try:
            start_time = time.time()
            
            # Build search query
            search_query = self.table.search(query_vector).metric("cosine")
            
            # Apply filters if provided
            if filter_conditions:
                where_clause = self._build_where_clause(filter_conditions)
                if where_clause:
                    search_query = search_query.where(where_clause)
            
            # Execute search with limit
            search_query = search_query.limit(limit)
            
            # Get results as pandas DataFrame
            results_df = search_query.to_pandas()
            
            # Process results
            similar_queries = []
            for _, row in results_df.iterrows():
                # Calculate similarity from distance (cosine distance to similarity)
                similarity = 1 - row.get("_distance", 0)
                
                # Filter by threshold
                if similarity >= threshold:
                    # Parse metadata
                    metadata = {}
                    try:
                        metadata = json.loads(row.get("metadata", "{}"))
                    except:
                        pass
                    
                    similar_queries.append({
                        "id": row["id"],
                        "sql_query": row["sql_query"],
                        "normalized_sql": row.get("normalized_sql", ""),
                        "similarity": float(similarity),
                        "database": row.get("database", ""),
                        "query_type": row.get("query_type", ""),
                        "execution_time_ms": float(row.get("execution_time_ms", 0)),
                        "row_count": int(row.get("row_count", 0)),
                        "user_id": row.get("user_id", ""),
                        "timestamp": row.get("timestamp"),
                        "success": bool(row.get("success", True)),
                        "metadata": metadata
                    })
            
            duration_ms = (time.time() - start_time) * 1000
            log_performance(f"Vector search (found {len(similar_queries)}/{limit})", duration_ms)
            
            return similar_queries
            
        except Exception as e:
            log_error("Vector search", e)
            raise
    
    async def search_by_text_pattern(
        self,
        pattern: str,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Search for queries containing specific text patterns.
        
        Args:
            pattern: Text pattern to search for
            limit: Maximum number of results
        
        Returns:
            List of matching queries
        """
        try:
            # LanceDB supports SQL-like WHERE clauses
            where_clause = f"sql_query LIKE '%{pattern}%'"
            
            results_df = (
                self.table.search()
                .where(where_clause)
                .limit(limit)
                .to_pandas()
            )
            
            results = []
            for _, row in results_df.iterrows():
                metadata = {}
                try:
                    metadata = json.loads(row.get("metadata", "{}"))
                except:
                    pass
                
                results.append({
                    "id": row["id"],
                    "sql_query": row["sql_query"],
                    "database": row.get("database", ""),
                    "query_type": row.get("query_type", ""),
                    "execution_time_ms": float(row.get("execution_time_ms", 0)),
                    "timestamp": row.get("timestamp"),
                    "metadata": metadata
                })
            
            return results
            
        except Exception as e:
            log_error("Text pattern search", e)
            raise
    
    async def get_recent_queries(
        self,
        limit: int = 10,
        user_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get most recent queries, optionally filtered by user.
        
        Args:
            limit: Maximum number of results
            user_id: Optional user ID filter
        
        Returns:
            List of recent queries
        """
        try:
            # Build query
            query = self.table.search()
            
            if user_id:
                query = query.where(f"user_id = '{user_id}'")
            
            # LanceDB doesn't have direct ORDER BY in search API
            # Get all results and sort in pandas
            results_df = await query.limit(limit * 2).to_pandas()
            
            # Sort by timestamp descending
            if "timestamp" in results_df.columns:
                results_df = results_df.sort_values("timestamp", ascending=False)
            
            # Take top N
            results_df = results_df.head(limit)
            
            results = []
            for _, row in results_df.iterrows():
                metadata = {}
                try:
                    metadata = json.loads(row.get("metadata", "{}"))
                except:
                    pass
                
                results.append({
                    "id": row["id"],
                    "sql_query": row["sql_query"],
                    "database": row.get("database", ""),
                    "execution_time_ms": float(row.get("execution_time_ms", 0)),
                    "row_count": int(row.get("row_count", 0)),
                    "timestamp": row.get("timestamp"),
                    "metadata": metadata
                })
            
            return results
            
        except Exception as e:
            log_error("Get recent queries", e)
            raise
    
    def _build_where_clause(self, conditions: Dict[str, Any]) -> str:
        """Build WHERE clause from filter conditions."""
        clauses = []
        
        for key, value in conditions.items():
            if isinstance(value, str):
                clauses.append(f"{key} = '{value}'")
            elif isinstance(value, (int, float)):
                clauses.append(f"{key} = {value}")
            elif isinstance(value, bool):
                clauses.append(f"{key} = {str(value).lower()}")
        
        return " AND ".join(clauses) if clauses else ""
    
    async def reindex_if_needed(self):
        """Check if reindexing is needed based on table size."""
        try:
            # Get table statistics
            stats = await self.table.count_rows()
            
            if stats > 100000 and settings.enable_hnsw_index:
                log_operation("Reindexing may improve performance", {"rows": stats})
                # LanceDB handles indexing automatically in most cases
                # Manual index creation would go here if needed
            
        except Exception as e:
            self.logger.warning(f"Could not check index status: {e}")