"""
Memory Manager - Session and Context Management

Manages all memory operations using PostgreSQL as the unified storage system.
Handles session state, user history, caching, and pattern learning.
"""

import json
import hashlib
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict

from ..utils.logging import logger
from ..utils.exceptions import MemoryError
from ..database.memory_manager import PostgreSQLMemoryManager


@dataclass
class UserPattern:
    """Represents a learned user pattern."""
    pattern_id: str
    user_id: str
    query_type: str
    frequency: int
    success_rate: float
    last_used: datetime
    pattern_data: Dict[str, Any]


@dataclass
class SessionContext:
    """Represents current session context."""
    session_id: str
    user_id: str
    started_at: datetime
    last_activity: datetime
    investigation_count: int
    current_focus: Optional[str]
    preferences: Dict[str, Any]
    workspace: Dict[str, Any]


class MemoryManager:
    """
    Unified memory management system using PostgreSQL.
    
    Provides three types of memory:
    1. Short-term: Current session state and workspace
    2. Medium-term: User patterns and preferences
    3. Long-term: Knowledge base and successful investigations
    """
    
    def __init__(self):
        self.db_manager: Optional[PostgreSQLMemoryManager] = None
        self.cache_ttl = 1800  # 30 minutes
        self.session_timeout = 3600  # 1 hour
        self.is_initialized = False
        
        logger.info("Memory Manager initialized")
    
    async def initialize(self):
        """Initialize the memory manager with database connection."""
        try:
            self.db_manager = PostgreSQLMemoryManager()
            await self.db_manager.initialize()
            self.is_initialized = True
            
            logger.info("Memory Manager successfully connected to PostgreSQL")
            
        except Exception as e:
            logger.error(f"Failed to initialize Memory Manager: {e}")
            raise MemoryError(f"Memory initialization failed: {e}")
    
    # Session Management
    
    async def create_session(
        self, 
        user_id: str, 
        session_id: Optional[str] = None
    ) -> str:
        """Create a new session or retrieve existing one."""
        if not self.is_initialized:
            await self.initialize()
        
        session_id = session_id or self._generate_session_id(user_id)
        
        try:
            # Check if session already exists
            existing_session = await self.get_session_context(session_id)
            
            if existing_session:
                # Update last activity
                await self.db_manager.update_session_activity(session_id)
                logger.info(f"Resumed existing session: {session_id}")
                return session_id
            
            # Create new session
            session_data = {
                "session_id": session_id,
                "user_id": user_id,
                "started_at": datetime.utcnow(),
                "last_activity": datetime.utcnow(),
                "investigation_count": 0,
                "current_focus": None,
                "preferences": await self._get_user_preferences(user_id),
                "workspace": {
                    "active_queries": [],
                    "recent_findings": [],
                    "investigation_history": []
                }
            }
            
            await self.db_manager.create_session(session_data)
            
            logger.info(f"Created new session: {session_id} for user: {user_id}")
            return session_id
            
        except Exception as e:
            logger.error(f"Failed to create session: {e}")
            raise MemoryError(f"Session creation failed: {e}")
    
    async def get_session_context(self, session_id: str) -> Optional[SessionContext]:
        """Retrieve current session context."""
        try:
            session_data = await self.db_manager.get_session_context(session_id)
            
            if not session_data:
                return None
            
            # Check if session has expired
            last_activity = session_data.get("last_activity")
            if last_activity and (datetime.utcnow() - last_activity).total_seconds() > self.session_timeout:
                await self.expire_session(session_id)
                return None
            
            return SessionContext(
                session_id=session_data["session_id"],
                user_id=session_data["user_id"],
                started_at=session_data["started_at"],
                last_activity=session_data["last_activity"],
                investigation_count=session_data.get("investigation_count", 0),
                current_focus=session_data.get("current_focus"),
                preferences=session_data.get("preferences", {}),
                workspace=session_data.get("workspace", {})
            )
            
        except Exception as e:
            logger.error(f"Failed to get session context: {e}")
            return None
    
    async def update_session_workspace(
        self, 
        session_id: str, 
        workspace_update: Dict[str, Any]
    ):
        """Update session workspace with new data."""
        try:
            await self.db_manager.update_session_workspace(session_id, workspace_update)
            logger.debug(f"Updated workspace for session: {session_id}")
            
        except Exception as e:
            logger.error(f"Failed to update session workspace: {e}")
            raise MemoryError(f"Workspace update failed: {e}")
    
    async def expire_session(self, session_id: str):
        """Expire and clean up a session."""
        try:
            await self.db_manager.expire_session(session_id)
            logger.info(f"Session expired: {session_id}")
            
        except Exception as e:
            logger.warning(f"Failed to expire session {session_id}: {e}")
    
    # User Pattern Management
    
    async def get_user_context(self, user_id: str) -> Dict[str, Any]:
        """Get comprehensive user context including patterns and preferences."""
        try:
            # Get user patterns
            patterns = await self.get_user_patterns(user_id)
            
            # Get recent successful investigations
            recent_investigations = await self.db_manager.get_user_recent_investigations(
                user_id, limit=10
            )
            
            # Calculate user proficiency metrics
            proficiency = await self._calculate_user_proficiency(user_id)
            
            return {
                "user_id": user_id,
                "patterns": patterns,
                "recent_investigations": recent_investigations,
                "proficiency": proficiency,
                "preferences": await self._get_user_preferences(user_id),
                "common_domains": await self._get_user_common_domains(user_id)
            }
            
        except Exception as e:
            logger.error(f"Failed to get user context: {e}")
            return {"user_id": user_id, "patterns": [], "preferences": {}}
    
    async def get_user_patterns(self, user_id: str, limit: int = 20) -> List[UserPattern]:
        """Get learned patterns for a specific user."""
        try:
            pattern_data = await self.db_manager.get_user_patterns(user_id, limit)
            
            patterns = []
            for data in pattern_data:
                patterns.append(UserPattern(
                    pattern_id=data["pattern_id"],
                    user_id=data["user_id"],
                    query_type=data["query_type"],
                    frequency=data["frequency"],
                    success_rate=data["success_rate"],
                    last_used=data["last_used"],
                    pattern_data=data.get("pattern_data", {})
                ))
            
            return patterns
            
        except Exception as e:
            logger.error(f"Failed to get user patterns: {e}")
            return []
    
    async def update_user_patterns(
        self, 
        user_id: str, 
        investigation: Dict[str, Any]
    ):
        """Update user patterns based on successful investigation."""
        try:
            query = investigation.get("query", "")
            success = investigation.get("final_results", {}).get("confidence_score", 0) > 0.7
            
            if success:
                # Extract pattern characteristics
                pattern_type = await self._classify_query_pattern(query)
                
                # Update or create pattern
                await self.db_manager.update_user_pattern(
                    user_id=user_id,
                    pattern_type=pattern_type,
                    query_data={
                        "query": query,
                        "steps_taken": len(investigation.get("steps", [])),
                        "strategy_used": investigation.get("plan", {}).get("strategy"),
                        "success_time": investigation.get("investigation_time", 0)
                    }
                )
                
                logger.debug(f"Updated user pattern for {user_id}: {pattern_type}")
            
        except Exception as e:
            logger.warning(f"Failed to update user patterns: {e}")
    
    # Knowledge Storage
    
    async def store_successful_pattern(
        self,
        query_pattern: str,
        investigation_steps: List[Dict],
        results: Dict[str, Any],
        user_id: str
    ):
        """Store a successful investigation pattern for future learning."""
        try:
            pattern_id = self._generate_pattern_id(query_pattern, investigation_steps)
            
            pattern_data = {
                "pattern_id": pattern_id,
                "query_pattern": query_pattern,
                "user_id": user_id,
                "investigation_steps": investigation_steps,
                "results_summary": {
                    "confidence": results.get("confidence_score", 0),
                    "insights_count": len(results.get("key_insights", [])),
                    "execution_time": results.get("investigation_time", 0)
                },
                "success_metrics": {
                    "steps_count": len(investigation_steps),
                    "final_confidence": results.get("confidence_score", 0),
                    "user_satisfaction": 1.0  # Assume success means satisfaction
                },
                "created_at": datetime.utcnow(),
                "usage_count": 1
            }
            
            await self.db_manager.store_successful_pattern(pattern_data)
            
            logger.info(f"Stored successful pattern: {pattern_id}")
            
        except Exception as e:
            logger.error(f"Failed to store successful pattern: {e}")
            raise MemoryError(f"Pattern storage failed: {e}")
    
    # Caching System
    
    async def get_cached_result(
        self, 
        cache_key: str, 
        cache_type: str = "query_result"
    ) -> Optional[Dict[str, Any]]:
        """Retrieve cached result if still valid."""
        try:
            cached_data = await self.db_manager.get_cached_data(cache_key, cache_type)
            
            if cached_data:
                # Check if cache is still valid
                cached_at = cached_data.get("created_at")
                if cached_at and (datetime.utcnow() - cached_at).total_seconds() < self.cache_ttl:
                    logger.debug(f"Cache hit for key: {cache_key}")
                    return cached_data.get("cache_data")
                else:
                    # Cache expired, clean it up
                    await self.db_manager.invalidate_cache(cache_key)
                    logger.debug(f"Cache expired for key: {cache_key}")
            
            return None
            
        except Exception as e:
            logger.warning(f"Failed to get cached result: {e}")
            return None
    
    async def cache_result(
        self,
        cache_key: str,
        data: Dict[str, Any],
        cache_type: str = "query_result",
        ttl_override: Optional[int] = None
    ):
        """Cache a result for future quick retrieval."""
        try:
            ttl = ttl_override or self.cache_ttl
            
            cache_data = {
                "cache_key": cache_key,
                "cache_type": cache_type,
                "cache_data": data,
                "created_at": datetime.utcnow(),
                "expires_at": datetime.utcnow() + timedelta(seconds=ttl)
            }
            
            await self.db_manager.store_cache_data(cache_data)
            
            logger.debug(f"Cached result with key: {cache_key}, TTL: {ttl}s")
            
        except Exception as e:
            logger.warning(f"Failed to cache result: {e}")
    
    # FAQ System Integration
    
    async def record_faq_usage(
        self,
        faq_id: str,
        investigation_id: str,
        success: bool
    ):
        """Record FAQ pattern usage for learning."""
        try:
            usage_data = {
                "faq_id": faq_id,
                "investigation_id": investigation_id,
                "success": success,
                "used_at": datetime.utcnow()
            }
            
            await self.db_manager.record_faq_usage(usage_data)
            
            logger.debug(f"Recorded FAQ usage: {faq_id}, success: {success}")
            
        except Exception as e:
            logger.warning(f"Failed to record FAQ usage: {e}")
    
    # Utility Methods
    
    def _generate_session_id(self, user_id: str) -> str:
        """Generate a unique session ID."""
        timestamp = datetime.utcnow().isoformat()
        data = f"{user_id}:{timestamp}"
        return hashlib.md5(data.encode()).hexdigest()
    
    def _generate_pattern_id(self, query: str, steps: List[Dict]) -> str:
        """Generate a unique pattern ID based on query and steps."""
        steps_signature = str(sorted([step.get("type", "") for step in steps]))
        data = f"{query}:{steps_signature}"
        return hashlib.sha256(data.encode()).hexdigest()[:16]
    
    async def _get_user_preferences(self, user_id: str) -> Dict[str, Any]:
        """Get user preferences with defaults."""
        try:
            prefs = await self.db_manager.get_user_preferences(user_id)
            return prefs or {
                "detail_level": "medium",
                "explanation_style": "business_focused",
                "preferred_visualizations": ["tables", "charts"],
                "investigation_depth": "standard"
            }
            
        except Exception as e:
            logger.warning(f"Failed to get user preferences: {e}")
            return {}
    
    async def _get_user_common_domains(self, user_id: str) -> List[str]:
        """Get domains/topics the user commonly investigates."""
        try:
            domains = await self.db_manager.get_user_common_domains(user_id)
            return domains or []
            
        except Exception as e:
            logger.warning(f"Failed to get user domains: {e}")
            return []
    
    async def _calculate_user_proficiency(self, user_id: str) -> Dict[str, float]:
        """Calculate user proficiency metrics."""
        try:
            metrics = await self.db_manager.get_user_proficiency_metrics(user_id)
            
            return {
                "sql_complexity_comfort": metrics.get("avg_complexity", 0.5),
                "investigation_efficiency": metrics.get("avg_efficiency", 0.5),
                "success_rate": metrics.get("success_rate", 0.8),
                "domain_expertise": metrics.get("domain_expertise", 0.5)
            }
            
        except Exception as e:
            logger.warning(f"Failed to calculate user proficiency: {e}")
            return {
                "sql_complexity_comfort": 0.5,
                "investigation_efficiency": 0.5,
                "success_rate": 0.8,
                "domain_expertise": 0.5
            }
    
    async def _classify_query_pattern(self, query: str) -> str:
        """Classify query into pattern types for learning."""
        query_lower = query.lower()
        
        if any(word in query_lower for word in ["trend", "over time", "growth"]):
            return "trend_analysis"
        elif any(word in query_lower for word in ["compare", "vs", "difference"]):
            return "comparison"
        elif any(word in query_lower for word in ["total", "sum", "count", "average"]):
            return "aggregation"
        elif any(word in query_lower for word in ["why", "reason", "cause"]):
            return "causal_analysis"
        elif any(word in query_lower for word in ["segment", "group", "category"]):
            return "segmentation"
        else:
            return "general_inquiry"
    
    # Cleanup and Maintenance
    
    async def cleanup_expired_data(self):
        """Clean up expired sessions and cache data."""
        try:
            cleanup_results = await self.db_manager.cleanup_expired_data(
                session_timeout=self.session_timeout,
                cache_ttl=self.cache_ttl
            )
            
            logger.info(f"Cleanup completed: {cleanup_results}")
            
        except Exception as e:
            logger.error(f"Failed to cleanup expired data: {e}")
    
    async def get_memory_stats(self) -> Dict[str, int]:
        """Get memory system statistics."""
        try:
            stats = await self.db_manager.get_memory_stats()
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get memory stats: {e}")
            return {}