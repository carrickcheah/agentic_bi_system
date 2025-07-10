"""
AgenticBiFlow - High-level interface for the 5-phase investigation workflow.
"""

import logging

logger = logging.getLogger(__name__)


class AgenticBiFlow:
    """
    Simple interface that makes the entire business intelligence flow visible.
    This is the 'facade' that provides a clean API for the 5-phase investigation workflow.
    
    Example usage:
        flow = AgenticBiFlow()
        await flow.initialize()
        result = await flow.quick_query("What were yesterday's sales?")
    """
    
    def __init__(self):
        self.business_analyst = None
        self.cache = None
        self.is_initialized = False
    
    async def initialize(self):
        """
        Step 1: Initialize all services needed for business intelligence.
        This includes cache, MCP services, and the business analyst.
        """
        # Import cache manager directly
        from cache import CacheManager
        
        # Initialize cache manager
        self.cache = CacheManager()
        await self.cache.initialize()
        
        # Initialize business analyst
        from .business_analyst import AutonomousBusinessAnalyst
        self.business_analyst = AutonomousBusinessAnalyst()
        await self.business_analyst.initialize()
        
        self.is_initialized = True
        return self
    
    async def investigate_query(
        self, 
        question: str,
        user_context: dict = None,
        organization_context: dict = None,
        stream_progress: bool = True
    ):
        """
        Step 2: Main investigation flow - visible at high level.
        
        The 5-Phase Flow:
        1. Cache Check (50-100ms if hit)
        2. Parallel Analysis (intent, qdrant, complexity)
        3. Service Orchestration (MCP setup)
        4. Investigation Execution (data retrieval)
        5. Insight Synthesis (strategic recommendations)
        
        Args:
            question: Natural language business question
            user_context: User information (role, department, etc.)
            organization_context: Organization settings and constraints
            stream_progress: Whether to yield progress updates
            
        Yields:
            Progress updates and final investigation results
        """
        if not self.is_initialized:
            await self.initialize()
        
        # Delegate to business analyst but make flow visible here
        async for result in self.business_analyst.conduct_investigation(
            question,
            user_context or {},
            organization_context or {},
            stream_progress=stream_progress
        ):
            yield result
    
    async def quick_query(self, question: str) -> dict:
        """
        Convenience method for simple queries - returns just the final result.
        Perfect for when you don't need streaming updates.
        
        Args:
            question: Business question to investigate
            
        Returns:
            Final investigation results with insights and recommendations
        """
        results = []
        async for result in self.investigate_query(question, stream_progress=True):
            if result.get("type") == "investigation_completed":
                return result
            results.append(result)
        return results[-1] if results else None
    
    async def get_investigation_status(self, investigation_id: str) -> dict:
        """Check status of an ongoing investigation."""
        return await self.business_analyst.get_investigation_status(investigation_id)
    
    async def get_cache_stats(self) -> dict:
        """Get cache performance statistics."""
        if self.cache:
            return self.cache.cache_stats
        return {}