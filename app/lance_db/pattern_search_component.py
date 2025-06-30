"""
Business Pattern Search Component for LanceDB.
Provides semantic search and discovery capabilities for business intelligence patterns.
"""

import json
import time
from typing import List, Dict, Any, Optional, Union
import numpy as np

try:
    from .config import settings
    from .lance_logging import get_logger, log_operation, log_error, log_performance
except ImportError:
    # For standalone execution
    from config import settings
    from lance_logging import get_logger, log_operation, log_error, log_performance


class BusinessPatternSearcher:
    """
    Advanced search capabilities for business intelligence patterns.
    Enables semantic discovery with metadata filtering and ranking.
    """
    
    def __init__(self, patterns_table, embedding_generator):
        self.patterns_table = patterns_table
        self.embedding_generator = embedding_generator
        self.logger = get_logger("pattern_search")
        
        # Search configuration
        self.default_similarity_threshold = 0.7
        self.max_results = 50
    
    async def search_patterns(
        self,
        query: str,
        search_type: str = "semantic",
        domain_filter: Optional[str] = None,
        complexity_filter: Optional[str] = None,
        timeframe_filter: Optional[str] = None,
        user_role_filter: Optional[str] = None,
        similarity_threshold: Optional[float] = None,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Search business patterns using various strategies.
        
        Args:
            query: Search query text
            search_type: "semantic", "workflow", "hybrid"
            domain_filter: Filter by domain category
            complexity_filter: Filter by complexity level
            timeframe_filter: Filter by analysis timeframe
            user_role_filter: Filter by user role
            similarity_threshold: Minimum similarity score
            limit: Maximum number of results
        
        Returns:
            List of matching patterns with metadata
        """
        try:
            start_time = time.time()
            
            # Use configured threshold if not provided
            if similarity_threshold is None:
                similarity_threshold = self.default_similarity_threshold
            
            # Generate query embedding based on search type
            if search_type == "semantic":
                query_embedding = self.embedding_generator.generate_embedding(query)
                search_column = "information_vector"
            elif search_type == "workflow":
                query_embedding = self.embedding_generator.generate_embedding(query)
                search_column = "pattern_vector"
            elif search_type == "hybrid":
                # Use information vector for primary search
                query_embedding = self.embedding_generator.generate_embedding(query)
                search_column = "information_vector"
            else:
                raise ValueError(f"Invalid search_type: {search_type}")
            
            # Build search query
            search_query = self.patterns_table.search(query_embedding, vector_column_name=search_column)
            search_query = search_query.metric("cosine")
            
            # Apply metadata filters
            where_clauses = self._build_where_clauses(
                domain_filter, complexity_filter, timeframe_filter, user_role_filter
            )
            
            if where_clauses:
                search_query = search_query.where(" AND ".join(where_clauses))
            
            # Execute search
            search_query = search_query.limit(min(limit * 2, self.max_results))  # Get extra for filtering
            results_df = search_query.to_pandas()
            
            # Process and filter results
            patterns = []
            for _, row in results_df.iterrows():
                # Calculate similarity from distance
                similarity = 1 - row.get("_distance", 0)
                
                # Apply similarity threshold
                if similarity >= similarity_threshold:
                    pattern = self._format_pattern_result(row, similarity)
                    patterns.append(pattern)
            
            # Sort by similarity and limit results
            patterns.sort(key=lambda x: x["similarity"], reverse=True)
            patterns = patterns[:limit]
            
            # For hybrid search, re-rank using workflow similarity
            if search_type == "hybrid" and patterns:
                patterns = await self._hybrid_rerank(patterns, query)
            
            duration_ms = (time.time() - start_time) * 1000
            log_performance(f"Pattern search ({search_type}, found {len(patterns)})", duration_ms)
            
            return patterns
            
        except Exception as e:
            log_error("Pattern search", e)
            raise
    
    async def find_similar_patterns(
        self,
        pattern_id: str,
        limit: int = 10,
        include_source_pattern: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Find patterns similar to a given pattern.
        
        Args:
            pattern_id: ID of the source pattern
            limit: Maximum number of similar patterns
            include_source_pattern: Whether to include the source pattern
        
        Returns:
            List of similar patterns
        """
        try:
            # Get the source pattern
            source_results = (
                self.patterns_table.search()
                .where(f"id = '{pattern_id}'")
                .limit(1)
                .to_pandas()
            )
            
            if len(source_results) == 0:
                raise ValueError(f"Pattern not found: {pattern_id}")
            
            source_pattern = source_results.iloc[0]
            source_embedding = source_pattern["information_vector"]
            
            # Search for similar patterns
            search_query = self.patterns_table.search(source_embedding)
            search_query = search_query.metric("cosine").limit(limit + 1)  # +1 for source pattern
            
            results_df = search_query.to_pandas()
            
            # Process results
            similar_patterns = []
            for _, row in results_df.iterrows():
                # Skip source pattern unless requested
                if row["id"] == pattern_id and not include_source_pattern:
                    continue
                
                similarity = 1 - row.get("_distance", 0)
                pattern = self._format_pattern_result(row, similarity)
                similar_patterns.append(pattern)
            
            # Limit results
            similar_patterns = similar_patterns[:limit]
            
            return similar_patterns
            
        except Exception as e:
            log_error("Find similar patterns", e)
            raise
    
    async def get_patterns_by_domain(
        self,
        domain: str,
        complexity: Optional[str] = None,
        limit: int = 20
    ) -> List[Dict[str, Any]]:
        """
        Get all patterns for a specific business domain.
        
        Args:
            domain: Business domain category
            complexity: Optional complexity filter
            limit: Maximum number of patterns
        
        Returns:
            List of domain patterns
        """
        try:
            # Build query
            where_clauses = [f"domain_category = '{domain}'"]
            if complexity:
                where_clauses.append(f"complexity = '{complexity}'")
            
            results_df = (
                self.patterns_table.search()
                .where(" AND ".join(where_clauses))
                .limit(limit)
                .to_pandas()
            )
            
            # Process results (no similarity since this isn't vector search)
            patterns = []
            for _, row in results_df.iterrows():
                pattern = self._format_pattern_result(row, similarity=None)
                patterns.append(pattern)
            
            # Sort by success rate
            patterns.sort(key=lambda x: x["success_rate"], reverse=True)
            
            return patterns
            
        except Exception as e:
            log_error("Get patterns by domain", e)
            raise
    
    async def get_recommended_patterns(
        self,
        user_role: str,
        complexity_preference: str = "moderate",
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Get recommended patterns for a specific user role.
        
        Args:
            user_role: User's role (e.g., "sales_manager", "production_manager")
            complexity_preference: Preferred complexity level
            limit: Maximum number of recommendations
        
        Returns:
            List of recommended patterns
        """
        try:
            # Search for patterns that include this user role
            results_df = (
                self.patterns_table.search()
                .where(f"user_roles LIKE '%{user_role}%' AND complexity = '{complexity_preference}'")
                .limit(limit * 2)  # Get extra for filtering
                .to_pandas()
            )
            
            # Process and rank by success rate
            patterns = []
            for _, row in results_df.iterrows():
                # Verify user role is actually in the list (not just substring match)
                user_roles = json.loads(row.get("user_roles", "[]"))
                if user_role in user_roles:
                    pattern = self._format_pattern_result(row, similarity=None)
                    patterns.append(pattern)
            
            # Sort by success rate and return top results
            patterns.sort(key=lambda x: x["success_rate"], reverse=True)
            return patterns[:limit]
            
        except Exception as e:
            log_error("Get recommended patterns", e)
            raise
    
    async def search_by_deliverables(
        self,
        deliverable_query: str,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Search patterns by expected deliverables.
        
        Args:
            deliverable_query: Query for deliverable types
            limit: Maximum number of results
        
        Returns:
            List of matching patterns
        """
        try:
            # Search in expected_deliverables field
            results_df = (
                self.patterns_table.search()
                .where(f"expected_deliverables LIKE '%{deliverable_query}%'")
                .limit(limit)
                .to_pandas()
            )
            
            patterns = []
            for _, row in results_df.iterrows():
                pattern = self._format_pattern_result(row, similarity=None)
                patterns.append(pattern)
            
            return patterns
            
        except Exception as e:
            log_error("Search by deliverables", e)
            raise
    
    def _build_where_clauses(
        self,
        domain_filter: Optional[str],
        complexity_filter: Optional[str],
        timeframe_filter: Optional[str],
        user_role_filter: Optional[str]
    ) -> List[str]:
        """Build WHERE clauses for filtering."""
        clauses = []
        
        if domain_filter:
            clauses.append(f"domain_category = '{domain_filter}'")
        
        if complexity_filter:
            clauses.append(f"complexity = '{complexity_filter}'")
        
        if timeframe_filter:
            clauses.append(f"timeframe = '{timeframe_filter}'")
        
        if user_role_filter:
            clauses.append(f"user_roles LIKE '%{user_role_filter}%'")
        
        return clauses
    
    def _format_pattern_result(self, row: Any, similarity: Optional[float]) -> Dict[str, Any]:
        """Format a pattern result for return."""
        result = {
            "id": row["id"],
            "information": row["information"],
            "pattern_workflow": row["pattern_workflow"],
            "user_roles": json.loads(row.get("user_roles", "[]")),
            "business_domain": row["business_domain"],
            "timeframe": row["timeframe"],
            "complexity": row["complexity"],
            "success_rate": float(row["success_rate"]),
            "confidence_indicators": json.loads(row.get("confidence_indicators", "[]")),
            "expected_deliverables": json.loads(row.get("expected_deliverables", "[]")),
            "domain_category": row["domain_category"],
            "source_file": row["source_file"]
        }
        
        if similarity is not None:
            result["similarity"] = float(similarity)
        
        return result
    
    async def _hybrid_rerank(self, patterns: List[Dict[str, Any]], query: str) -> List[Dict[str, Any]]:
        """Re-rank patterns using workflow similarity for hybrid search."""
        try:
            # Generate workflow embedding for query
            workflow_embedding = self.embedding_generator.generate_embedding(query)
            
            # Get workflow embeddings for each pattern
            pattern_ids = [p["id"] for p in patterns]
            ids_joined = "', '".join(pattern_ids)
            workflow_results = (
                self.patterns_table.search()
                .where(f"id IN ('{ids_joined}')")
                .limit(len(patterns))
                .to_pandas()
            )
            
            # Calculate workflow similarities
            workflow_similarities = {}
            for _, row in workflow_results.iterrows():
                pattern_vector = row["pattern_vector"]
                if pattern_vector is not None:
                    # Calculate cosine similarity
                    similarity = np.dot(workflow_embedding, pattern_vector) / (
                        np.linalg.norm(workflow_embedding) * np.linalg.norm(pattern_vector)
                    )
                    workflow_similarities[row["id"]] = float(similarity)
            
            # Combine similarities (weighted average)
            for pattern in patterns:
                info_sim = pattern["similarity"]
                workflow_sim = workflow_similarities.get(pattern["id"], 0.0)
                
                # Weighted combination: 70% information, 30% workflow
                combined_similarity = (0.7 * info_sim) + (0.3 * workflow_sim)
                pattern["similarity"] = combined_similarity
                pattern["workflow_similarity"] = workflow_sim
            
            # Re-sort by combined similarity
            patterns.sort(key=lambda x: x["similarity"], reverse=True)
            
            return patterns
            
        except Exception as e:
            self.logger.warning(f"Hybrid re-ranking failed: {e}")
            return patterns  # Return original ranking if re-ranking fails