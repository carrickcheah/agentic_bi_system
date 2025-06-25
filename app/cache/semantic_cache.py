"""
Semantic Cache Client - Pattern-Based Intelligence Cache

Manages Qdrant vector database for semantic similarity matching and pattern recognition.

Features:
- Semantic similarity matching for business questions
- Cross-domain pattern recognition
- Organizational knowledge accumulation
- FAQ-style instant responses
- Business intelligence pattern learning
"""

import asyncio
import hashlib
import json
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime

from ..utils.logging import logger
from ..mcp.qdrant_client import QdrantClient
from .ttl_manager import TTLManager


class SemanticCacheClient:
    """
    Client for Qdrant-based semantic caching and pattern recognition.
    
    Stores business intelligence patterns and provides semantic similarity
    matching for instant responses to similar questions.
    """
    
    def __init__(self):
        self.qdrant_client: Optional[QdrantClient] = None
        self.ttl_manager = TTLManager()
        self.collection_name = "business_intelligence_patterns"
        self.similarity_threshold = 0.75
        self.max_results = 10
        
    async def initialize(self):
        """Initialize Qdrant semantic cache client."""
        try:
            # Initialize Qdrant client connection
            # This will be injected from the MCP client manager
            logger.info("Semantic cache client initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize semantic cache: {e}")
            raise
    
    def set_qdrant_client(self, client: QdrantClient):
        """Set the Qdrant client instance."""
        self.qdrant_client = client
        logger.info("Qdrant client set for semantic cache")
    
    async def find_similar_insights(
        self,
        semantic_intent: Dict[str, Any],
        organization_id: str,
        user_permissions: List[str],
        max_results: int = 5
    ) -> Optional[Dict[str, Any]]:
        """
        Find semantically similar insights using vector search.
        
        Args:
            semantic_intent: Processed business question intent
            organization_id: Organization identifier
            user_permissions: User's permissions for filtering
            max_results: Maximum number of results to return
            
        Returns:
            Similar insights if found, None otherwise
        """
        try:
            if not self.qdrant_client:
                return None
            
            # Create search vector from semantic intent
            search_vector = await self._create_search_vector(semantic_intent)
            if not search_vector:
                return None
            
            # Prepare search filters
            search_filters = {
                "must": [
                    {"key": "organization_id", "match": {"value": organization_id}},
                    {"key": "active", "match": {"value": True}}
                ]
            }
            
            # Add permission filters
            if user_permissions:
                search_filters["should"] = [
                    {"key": "required_permissions", "match": {"any": user_permissions}},
                    {"key": "public", "match": {"value": True}}
                ]
            
            # Perform vector search
            search_results = await self.qdrant_client.search_vectors(
                collection_name=self.collection_name,
                query_vector=search_vector,
                filter=search_filters,
                limit=max_results,
                score_threshold=self.similarity_threshold
            )
            
            if not search_results:
                return None
            
            # Process and rank results
            best_match = search_results[0]
            
            # Check if similarity score is high enough
            if best_match.get("score", 0) < self.similarity_threshold:
                return None
            
            # Return the best matching insights
            return {
                "insights": best_match.get("payload", {}).get("insights"),
                "business_domain": best_match.get("payload", {}).get("business_domain"),
                "similarity_score": best_match.get("score"),
                "pattern_id": best_match.get("id"),
                "original_question": best_match.get("payload", {}).get("original_question"),
                "cached_at": best_match.get("payload", {}).get("cached_at"),
                "usage_count": best_match.get("payload", {}).get("usage_count", 0),
                "cache_tier": "semantic"
            }
            
        except Exception as e:
            logger.warning(f"Semantic cache search failed: {e}")
            return None
    
    async def store_semantic_pattern(
        self,
        semantic_intent: Dict[str, Any],
        business_domain: str,
        organization_id: str,
        insights: Dict[str, Any],
        user_permissions: List[str],
        original_question: Optional[str] = None
    ):
        """
        Store business intelligence pattern for semantic matching.
        
        Args:
            semantic_intent: Processed business question intent
            business_domain: Business domain classification
            organization_id: Organization identifier
            insights: Generated business insights
            user_permissions: Required permissions to access this pattern
            original_question: Original natural language question
        """
        try:
            if not self.qdrant_client:
                return
            
            # Create embedding vector from semantic intent
            pattern_vector = await self._create_pattern_vector(semantic_intent, insights)
            if not pattern_vector:
                return
            
            # Generate pattern ID
            pattern_id = self._generate_pattern_id(semantic_intent, business_domain, organization_id)
            
            # Prepare pattern payload
            pattern_payload = {
                "pattern_id": pattern_id,
                "semantic_intent": semantic_intent,
                "business_domain": business_domain,
                "organization_id": organization_id,
                "insights": insights,
                "required_permissions": user_permissions,
                "original_question": original_question or "",
                "cached_at": datetime.utcnow().isoformat(),
                "usage_count": 0,
                "active": True,
                "public": len(user_permissions) == 0 or "public" in user_permissions,
                "metadata": {
                    "question_type": semantic_intent.get("business_intent", {}).get("question_type"),
                    "time_period": semantic_intent.get("business_intent", {}).get("time_period"),
                    "analysis_type": semantic_intent.get("analysis_type"),
                    "complexity_score": semantic_intent.get("complexity_score", 0)
                }
            }
            
            # Store pattern in Qdrant
            await self.qdrant_client.upsert_vectors(
                collection_name=self.collection_name,
                points=[{
                    "id": pattern_id,
                    "vector": pattern_vector,
                    "payload": pattern_payload
                }]
            )
            
            logger.debug(f"Stored semantic pattern: {pattern_id} for {business_domain}")
            
        except Exception as e:
            logger.error(f"Failed to store semantic pattern: {e}")
    
    async def update_pattern_usage(self, pattern_id: str):
        """Update usage count for a semantic pattern."""
        try:
            if not self.qdrant_client:
                return
            
            # Get current pattern
            pattern_data = await self.qdrant_client.get_vector(
                collection_name=self.collection_name,
                point_id=pattern_id
            )
            
            if not pattern_data:
                return
            
            # Increment usage count
            payload = pattern_data.get("payload", {})
            payload["usage_count"] = payload.get("usage_count", 0) + 1
            payload["last_used"] = datetime.utcnow().isoformat()
            
            # Update pattern
            await self.qdrant_client.upsert_vectors(
                collection_name=self.collection_name,
                points=[{
                    "id": pattern_id,
                    "vector": pattern_data["vector"],
                    "payload": payload
                }]
            )
            
        except Exception as e:
            logger.warning(f"Failed to update pattern usage: {e}")
    
    async def get_popular_patterns(
        self,
        organization_id: str,
        business_domain: Optional[str] = None,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Get most popular semantic patterns for an organization.
        
        Args:
            organization_id: Organization identifier
            business_domain: Optional business domain filter
            limit: Maximum number of patterns to return
            
        Returns:
            List of popular patterns
        """
        try:
            if not self.qdrant_client:
                return []
            
            # Prepare filters
            filters = {
                "must": [
                    {"key": "organization_id", "match": {"value": organization_id}},
                    {"key": "active", "match": {"value": True}}
                ]
            }
            
            if business_domain:
                filters["must"].append(
                    {"key": "business_domain", "match": {"value": business_domain}}
                )
            
            # Get patterns sorted by usage count
            patterns = await self.qdrant_client.scroll_vectors(
                collection_name=self.collection_name,
                filter=filters,
                limit=limit,
                with_payload=True,
                with_vectors=False
            )
            
            if not patterns:
                return []
            
            # Sort by usage count and return
            sorted_patterns = sorted(
                patterns,
                key=lambda x: x.get("payload", {}).get("usage_count", 0),
                reverse=True
            )
            
            return [
                {
                    "pattern_id": pattern.get("id"),
                    "business_domain": pattern.get("payload", {}).get("business_domain"),
                    "question_type": pattern.get("payload", {}).get("metadata", {}).get("question_type"),
                    "usage_count": pattern.get("payload", {}).get("usage_count", 0),
                    "last_used": pattern.get("payload", {}).get("last_used"),
                    "original_question": pattern.get("payload", {}).get("original_question")
                }
                for pattern in sorted_patterns[:limit]
            ]
            
        except Exception as e:
            logger.error(f"Failed to get popular patterns: {e}")
            return []
    
    async def invalidate_patterns(
        self,
        organization_id: str,
        business_domain: Optional[str] = None,
        pattern_ids: Optional[List[str]] = None
    ):
        """
        Invalidate semantic patterns.
        
        Args:
            organization_id: Organization identifier
            business_domain: Optional business domain filter
            pattern_ids: Optional specific pattern IDs to invalidate
        """
        try:
            if not self.qdrant_client:
                return
            
            if pattern_ids:
                # Invalidate specific patterns
                for pattern_id in pattern_ids:
                    await self._deactivate_pattern(pattern_id)
                logger.info(f"Invalidated {len(pattern_ids)} specific patterns")
            else:
                # Invalidate by organization/domain
                filters = {
                    "must": [
                        {"key": "organization_id", "match": {"value": organization_id}}
                    ]
                }
                
                if business_domain:
                    filters["must"].append(
                        {"key": "business_domain", "match": {"value": business_domain}}
                    )
                
                # Get patterns to invalidate
                patterns = await self.qdrant_client.scroll_vectors(
                    collection_name=self.collection_name,
                    filter=filters,
                    with_payload=False,
                    with_vectors=False
                )
                
                # Deactivate patterns
                for pattern in patterns:
                    await self._deactivate_pattern(pattern.get("id"))
                
                logger.info(f"Invalidated {len(patterns)} patterns for {organization_id}")
            
        except Exception as e:
            logger.error(f"Failed to invalidate patterns: {e}")
    
    async def _create_search_vector(self, semantic_intent: Dict[str, Any]) -> Optional[List[float]]:
        """Create search vector from semantic intent."""
        try:
            if not self.qdrant_client:
                return None
            
            # Create text representation of semantic intent
            search_text = self._semantic_intent_to_text(semantic_intent)
            
            # Generate embedding using Qdrant's embedding service
            embedding = await self.qdrant_client.create_embedding(search_text)
            return embedding
            
        except Exception as e:
            logger.warning(f"Failed to create search vector: {e}")
            return None
    
    async def _create_pattern_vector(
        self, 
        semantic_intent: Dict[str, Any], 
        insights: Dict[str, Any]
    ) -> Optional[List[float]]:
        """Create pattern vector from semantic intent and insights."""
        try:
            if not self.qdrant_client:
                return None
            
            # Combine semantic intent and insights for comprehensive embedding
            pattern_text = self._create_pattern_text(semantic_intent, insights)
            
            # Generate embedding
            embedding = await self.qdrant_client.create_embedding(pattern_text)
            return embedding
            
        except Exception as e:
            logger.warning(f"Failed to create pattern vector: {e}")
            return None
    
    def _semantic_intent_to_text(self, semantic_intent: Dict[str, Any]) -> str:
        """Convert semantic intent to text for embedding."""
        components = []
        
        # Business domain
        if "business_domain" in semantic_intent:
            components.append(f"Domain: {semantic_intent['business_domain']}")
        
        # Business intent details
        business_intent = semantic_intent.get("business_intent", {})
        if "question_type" in business_intent:
            components.append(f"Question type: {business_intent['question_type']}")
        if "time_period" in business_intent:
            components.append(f"Time period: {business_intent['time_period']}")
        if "metrics" in business_intent:
            components.append(f"Metrics: {', '.join(business_intent['metrics'])}")
        
        # Analysis type
        if "analysis_type" in semantic_intent:
            components.append(f"Analysis: {semantic_intent['analysis_type']}")
        
        return " | ".join(components)
    
    def _create_pattern_text(self, semantic_intent: Dict[str, Any], insights: Dict[str, Any]) -> str:
        """Create comprehensive pattern text for embedding."""
        components = []
        
        # Add semantic intent
        components.append(self._semantic_intent_to_text(semantic_intent))
        
        # Add key insights
        if "summary" in insights:
            components.append(f"Summary: {insights['summary']}")
        if "key_findings" in insights:
            findings = insights["key_findings"]
            if isinstance(findings, list):
                components.append(f"Findings: {' '.join(findings)}")
            elif isinstance(findings, str):
                components.append(f"Findings: {findings}")
        
        # Add recommendations
        if "recommendations" in insights:
            recommendations = insights["recommendations"]
            if isinstance(recommendations, list):
                components.append(f"Recommendations: {' '.join(recommendations)}")
            elif isinstance(recommendations, str):
                components.append(f"Recommendations: {recommendations}")
        
        return " | ".join(components)
    
    def _generate_pattern_id(self, semantic_intent: Dict[str, Any], business_domain: str, organization_id: str) -> str:
        """Generate unique pattern ID."""
        # Create hash from key components
        hash_components = [
            semantic_intent.get("business_domain", ""),
            business_domain,
            organization_id,
            str(semantic_intent.get("business_intent", {})),
            str(datetime.utcnow().date())  # Include date for daily uniqueness
        ]
        
        hash_string = "|".join(hash_components)
        return hashlib.sha256(hash_string.encode()).hexdigest()[:16]
    
    async def _deactivate_pattern(self, pattern_id: str):
        """Deactivate a semantic pattern."""
        try:
            if not self.qdrant_client:
                return
            
            # Get current pattern
            pattern_data = await self.qdrant_client.get_vector(
                collection_name=self.collection_name,
                point_id=pattern_id
            )
            
            if not pattern_data:
                return
            
            # Deactivate pattern
            payload = pattern_data.get("payload", {})
            payload["active"] = False
            payload["deactivated_at"] = datetime.utcnow().isoformat()
            
            # Update pattern
            await self.qdrant_client.upsert_vectors(
                collection_name=self.collection_name,
                points=[{
                    "id": pattern_id,
                    "vector": pattern_data["vector"],
                    "payload": payload
                }]
            )
            
        except Exception as e:
            logger.warning(f"Failed to deactivate pattern {pattern_id}: {e}")
    
    def get_semantic_statistics(self) -> Dict[str, Any]:
        """Get semantic cache statistics."""
        return {
            "collection_name": self.collection_name,
            "similarity_threshold": self.similarity_threshold,
            "max_results": self.max_results,
            "status": "active" if self.qdrant_client else "inactive"
        }