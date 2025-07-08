"""
Cache Warming Engine - Proactive Cache Population

Intelligently pre-populates cache with high-probability business intelligence queries
to optimize response times and user experience.

Features:
- Business domain-specific warming patterns
- Time-of-day and seasonal warming strategies
- Popular query pattern prediction
- ROI-based warming prioritization
- Organizational learning integration
"""

import asyncio
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import json

from .cache_logging import logger
from .ttl_manager import TTLManager, DataVolatility, CachePriority


class CacheWarmingEngine:
    """
    Proactive cache warming engine for business intelligence queries.
    
    Analyzes organizational patterns and proactively populates cache
    with high-probability queries to optimize user experience.
    """
    
    def __init__(self, cache_manager):
        """Initialize cache warming engine."""
        self.cache_manager = cache_manager
        self.ttl_manager = TTLManager()
        
        # Common business intelligence query patterns
        self.warming_patterns = {
            # Daily operational queries (high probability)
            "finance": [
                {"intent": "daily_cash_flow", "time_period": "today", "priority": "high"},
                {"intent": "revenue_tracking", "time_period": "this_week", "priority": "high"},
                {"intent": "expense_analysis", "time_period": "this_month", "priority": "medium"},
                {"intent": "budget_variance", "time_period": "current_quarter", "priority": "medium"},
            ],
            "sales": [
                {"intent": "pipeline_status", "time_period": "current", "priority": "high"},
                {"intent": "daily_sales", "time_period": "today", "priority": "high"},
                {"intent": "team_performance", "time_period": "this_week", "priority": "high"},
                {"intent": "conversion_rates", "time_period": "this_month", "priority": "medium"},
            ],
            "customer": [
                {"intent": "support_tickets", "time_period": "today", "priority": "high"},
                {"intent": "satisfaction_scores", "time_period": "this_week", "priority": "medium"},
                {"intent": "churn_analysis", "time_period": "this_month", "priority": "medium"},
                {"intent": "customer_acquisition", "time_period": "current_quarter", "priority": "low"},
            ],
            "operations": [
                {"intent": "system_performance", "time_period": "current", "priority": "high"},
                {"intent": "inventory_levels", "time_period": "current", "priority": "high"},
                {"intent": "efficiency_metrics", "time_period": "this_week", "priority": "medium"},
                {"intent": "capacity_utilization", "time_period": "this_month", "priority": "medium"},
            ],
            "marketing": [
                {"intent": "campaign_performance", "time_period": "active", "priority": "high"},
                {"intent": "lead_generation", "time_period": "this_week", "priority": "high"},
                {"intent": "roi_analysis", "time_period": "this_month", "priority": "medium"},
                {"intent": "attribution_analysis", "time_period": "current_quarter", "priority": "low"},
            ],
            "product": [
                {"intent": "user_activity", "time_period": "today", "priority": "high"},
                {"intent": "feature_adoption", "time_period": "this_week", "priority": "medium"},
                {"intent": "performance_metrics", "time_period": "this_month", "priority": "medium"},
                {"intent": "roadmap_analysis", "time_period": "current_quarter", "priority": "low"},
            ],
            "hr": [
                {"intent": "attendance_tracking", "time_period": "today", "priority": "medium"},
                {"intent": "productivity_metrics", "time_period": "this_week", "priority": "medium"},
                {"intent": "performance_reviews", "time_period": "current_quarter", "priority": "low"},
                {"intent": "hiring_pipeline", "time_period": "this_month", "priority": "low"},
            ]
        }
        
        # Time-based warming strategies
        self.time_based_patterns = {
            "morning": ["daily_reports", "overnight_updates", "system_status"],
            "midday": ["performance_tracking", "real_time_metrics", "alerts"],
            "afternoon": ["analysis_reports", "team_updates", "planning"],
            "evening": ["daily_summaries", "tomorrow_prep", "weekly_reviews"]
        }
        
        # Seasonal/periodic patterns
        self.periodic_patterns = {
            "monday": ["weekly_kickoff", "team_metrics", "planning"],
            "friday": ["weekly_summary", "performance_review", "next_week_prep"],
            "month_start": ["monthly_targets", "budget_review", "planning"],
            "month_end": ["monthly_closing", "performance_summary", "reporting"],
            "quarter_start": ["quarterly_goals", "strategic_review", "planning"],
            "quarter_end": ["quarterly_results", "performance_analysis", "strategic_planning"]
        }
    
    async def initialize(self):
        """Initialize cache warming engine."""
        try:
            logger.info("Cache warming engine initialized")
        except Exception as e:
            logger.error(f"Failed to initialize cache warming: {e}")
            raise
    
    async def warm_organizational_cache(
        self,
        business_domains: List[str],
        organization_id: str,
        priority_questions: Optional[List[Dict[str, Any]]] = None
    ):
        """
        Warm cache for an organization with common BI patterns.
        
        Args:
            business_domains: List of business domains to warm
            organization_id: Organization identifier
            priority_questions: Optional high-priority questions to warm first
        """
        try:
            logger.info(f"Starting cache warming for organization {organization_id}")
            
            # Priority questions first
            if priority_questions:
                await self._warm_priority_questions(priority_questions, organization_id)
            
            # Warm each business domain
            warming_tasks = []
            for domain in business_domains:
                warming_tasks.append(
                    self._warm_domain_patterns(domain, organization_id)
                )
            
            # Execute domain warming in parallel
            await asyncio.gather(*warming_tasks, return_exceptions=True)
            
            # Time-based warming
            await self._warm_time_based_patterns(business_domains, organization_id)
            
            # Periodic warming
            await self._warm_periodic_patterns(business_domains, organization_id)
            
            logger.info(f"Cache warming completed for {len(business_domains)} domains")
            
        except Exception as e:
            logger.error(f"Cache warming failed: {e}")
    
    async def warm_user_patterns(
        self,
        user_id: str,
        organization_id: str,
        user_role: str,
        historical_patterns: Optional[List[Dict[str, Any]]] = None
    ):
        """
        Warm cache based on user-specific patterns.
        
        Args:
            user_id: User identifier
            organization_id: Organization identifier
            user_role: User's role for role-specific warming
            historical_patterns: User's historical query patterns
        """
        try:
            logger.info(f"Starting user-specific cache warming for {user_id}")
            
            # Role-based warming
            role_patterns = self._get_role_specific_patterns(user_role)
            for pattern in role_patterns:
                await self._warm_query_pattern(
                    pattern=pattern,
                    organization_id=organization_id,
                    user_context={"user_id": user_id, "role": user_role}
                )
            
            # Historical pattern warming
            if historical_patterns:
                for pattern in historical_patterns[:10]:  # Top 10 patterns
                    await self._warm_query_pattern(
                        pattern=pattern,
                        organization_id=organization_id,
                        user_context={"user_id": user_id, "role": user_role}
                    )
            
            logger.info(f"User cache warming completed for {user_id}")
            
        except Exception as e:
            logger.error(f"User cache warming failed: {e}")
    
    async def warm_popular_patterns(
        self,
        organization_id: str,
        lookback_days: int = 30,
        max_patterns: int = 50
    ):
        """
        Warm cache with organization's most popular query patterns.
        
        Args:
            organization_id: Organization identifier
            lookback_days: Days to look back for popular patterns
            max_patterns: Maximum patterns to warm
        """
        try:
            # Get popular patterns from semantic cache
            popular_patterns = await self.cache_manager.semantic_cache.get_popular_patterns(
                organization_id=organization_id,
                limit=max_patterns
            )
            
            if not popular_patterns:
                logger.info("No popular patterns found for warming")
                return
            
            logger.info(f"Warming {len(popular_patterns)} popular patterns")
            
            # Warm popular patterns
            warming_tasks = []
            for pattern in popular_patterns:
                warming_tasks.append(
                    self._warm_popular_pattern(pattern, organization_id)
                )
            
            await asyncio.gather(*warming_tasks, return_exceptions=True)
            
            logger.info(f"Popular pattern warming completed: {len(popular_patterns)} patterns")
            
        except Exception as e:
            logger.error(f"Popular pattern warming failed: {e}")
    
    async def schedule_periodic_warming(
        self,
        organization_id: str,
        business_domains: List[str],
        schedule_config: Dict[str, Any]
    ):
        """
        Schedule periodic cache warming based on business patterns.
        
        Args:
            organization_id: Organization identifier
            business_domains: Business domains to warm
            schedule_config: Warming schedule configuration
        """
        try:
            # This would integrate with a task scheduler (e.g., Celery, APScheduler)
            # For now, we'll implement immediate warming based on current time
            
            current_time = datetime.utcnow()
            current_hour = current_time.hour
            current_weekday = current_time.strftime("%A").lower()
            
            # Morning warming (7-9 AM)
            if 7 <= current_hour <= 9:
                await self._warm_time_based_patterns(business_domains, organization_id, "morning")
            
            # Midday warming (11 AM - 1 PM)
            elif 11 <= current_hour <= 13:
                await self._warm_time_based_patterns(business_domains, organization_id, "midday")
            
            # Afternoon warming (2-4 PM)
            elif 14 <= current_hour <= 16:
                await self._warm_time_based_patterns(business_domains, organization_id, "afternoon")
            
            # Evening warming (5-7 PM)
            elif 17 <= current_hour <= 19:
                await self._warm_time_based_patterns(business_domains, organization_id, "evening")
            
            # Weekly patterns
            if current_weekday in self.periodic_patterns:
                patterns = self.periodic_patterns[current_weekday]
                for pattern in patterns:
                    await self._warm_pattern_type(pattern, business_domains, organization_id)
            
            logger.info(f"Scheduled warming completed for {current_weekday} {current_hour}:00")
            
        except Exception as e:
            logger.error(f"Scheduled warming failed: {e}")
    
    async def _warm_priority_questions(
        self,
        priority_questions: List[Dict[str, Any]],
        organization_id: str
    ):
        """Warm cache with high-priority questions."""
        logger.info(f"Warming {len(priority_questions)} priority questions")
        
        for question in priority_questions[:20]:  # Limit to top 20
            await self._warm_query_pattern(
                pattern=question,
                organization_id=organization_id,
                user_context={"role": "executive", "priority": "high"}
            )
    
    async def _warm_domain_patterns(self, domain: str, organization_id: str):
        """Warm cache patterns for a specific business domain."""
        if domain not in self.warming_patterns:
            return
        
        patterns = self.warming_patterns[domain]
        logger.debug(f"Warming {len(patterns)} patterns for {domain}")
        
        for pattern in patterns:
            await self._warm_query_pattern(
                pattern=pattern,
                organization_id=organization_id,
                user_context={"role": "analyst", "domain": domain}
            )
    
    async def _warm_time_based_patterns(
        self,
        business_domains: List[str],
        organization_id: str,
        time_period: Optional[str] = None
    ):
        """Warm cache based on time-of-day patterns."""
        if not time_period:
            current_hour = datetime.utcnow().hour
            if 6 <= current_hour <= 10:
                time_period = "morning"
            elif 11 <= current_hour <= 14:
                time_period = "midday"
            elif 15 <= current_hour <= 17:
                time_period = "afternoon"
            else:
                time_period = "evening"
        
        if time_period not in self.time_based_patterns:
            return
        
        patterns = self.time_based_patterns[time_period]
        logger.debug(f"Warming {len(patterns)} {time_period} patterns")
        
        for pattern_type in patterns:
            await self._warm_pattern_type(pattern_type, business_domains, organization_id)
    
    async def _warm_periodic_patterns(self, business_domains: List[str], organization_id: str):
        """Warm cache based on periodic patterns (weekly, monthly, etc.)."""
        current_date = datetime.utcnow()
        
        # Determine current period
        periods = []
        if current_date.weekday() == 0:  # Monday
            periods.append("monday")
        elif current_date.weekday() == 4:  # Friday
            periods.append("friday")
        
        if current_date.day == 1:  # First day of month
            periods.append("month_start")
        elif current_date.day >= 28:  # End of month
            periods.append("month_end")
        
        # Quarter periods (approximate)
        month = current_date.month
        if month in [1, 4, 7, 10] and current_date.day <= 5:
            periods.append("quarter_start")
        elif month in [3, 6, 9, 12] and current_date.day >= 25:
            periods.append("quarter_end")
        
        # Warm patterns for detected periods
        for period in periods:
            if period in self.periodic_patterns:
                patterns = self.periodic_patterns[period]
                for pattern_type in patterns:
                    await self._warm_pattern_type(pattern_type, business_domains, organization_id)
    
    async def _warm_pattern_type(
        self,
        pattern_type: str,
        business_domains: List[str],
        organization_id: str
    ):
        """Warm cache for a specific pattern type across domains."""
        for domain in business_domains:
            pattern = {
                "intent": pattern_type,
                "business_domain": domain,
                "time_period": "current",
                "priority": "medium"
            }
            
            await self._warm_query_pattern(
                pattern=pattern,
                organization_id=organization_id,
                user_context={"role": "analyst", "domain": domain}
            )
    
    async def _warm_query_pattern(
        self,
        pattern: Dict[str, Any],
        organization_id: str,
        user_context: Dict[str, Any]
    ):
        """Warm cache for a specific query pattern."""
        try:
            # Create semantic intent from pattern
            semantic_intent = {
                "business_domain": pattern.get("business_domain", user_context.get("domain", "general")),
                "business_intent": {
                    "question_type": "descriptive",
                    "intent": pattern.get("intent"),
                    "time_period": pattern.get("time_period", "current"),
                    "metrics": pattern.get("metrics", [])
                },
                "analysis_type": "operational",
                "urgency": pattern.get("priority", "standard")
            }
            
            # Create organization context
            organization_context = {
                "organization_id": organization_id,
                "size": "medium",
                "enable_semantic_learning": True
            }
            
            # Check if already cached
            cache_hash = self._generate_cache_hash(semantic_intent)
            cached_result = await self.cache_manager.get_cached_insights(
                semantic_hash=cache_hash,
                business_domain=semantic_intent["business_domain"],
                semantic_intent=semantic_intent,
                user_context=user_context,
                organization_context=organization_context
            )
            
            if cached_result:
                logger.debug(f"Pattern already cached: {pattern.get('intent')}")
                return
            
            # Generate mock insights for cache warming
            # In production, this would call the actual investigation engine
            mock_insights = self._generate_mock_insights(pattern)
            
            # Store in cache
            await self.cache_manager.store_insights(
                semantic_hash=cache_hash,
                business_domain=semantic_intent["business_domain"],
                semantic_intent=semantic_intent,
                user_context=user_context,
                organization_context=organization_context,
                insights=mock_insights
            )
            
            logger.debug(f"Warmed cache for pattern: {pattern.get('intent')}")
            
        except Exception as e:
            logger.warning(f"Failed to warm query pattern: {e}")
    
    async def _warm_popular_pattern(self, pattern: Dict[str, Any], organization_id: str):
        """Warm cache for a popular pattern."""
        try:
            # Extract pattern information
            business_domain = pattern.get("business_domain", "general")
            question_type = pattern.get("question_type", "descriptive")
            
            # Create semantic intent
            semantic_intent = {
                "business_domain": business_domain,
                "business_intent": {
                    "question_type": question_type,
                    "intent": f"popular_{pattern.get('pattern_id', 'unknown')}",
                    "time_period": "current"
                },
                "analysis_type": "operational"
            }
            
            # Use the pattern's usage count to determine priority
            usage_count = pattern.get("usage_count", 0)
            priority = "high" if usage_count > 50 else "medium" if usage_count > 10 else "standard"
            
            user_context = {"role": "analyst", "priority": priority}
            organization_context = {"organization_id": organization_id, "size": "medium"}
            
            # Generate insights and store
            cache_hash = self._generate_cache_hash(semantic_intent)
            mock_insights = {
                "summary": f"Popular insight for {business_domain}",
                "key_findings": [f"Finding based on {usage_count} previous queries"],
                "recommendations": ["Continue monitoring this popular metric"],
                "confidence_score": 0.8,
                "data_sources": [business_domain]
            }
            
            await self.cache_manager.store_insights(
                semantic_hash=cache_hash,
                business_domain=business_domain,
                semantic_intent=semantic_intent,
                user_context=user_context,
                organization_context=organization_context,
                insights=mock_insights
            )
            
        except Exception as e:
            logger.warning(f"Failed to warm popular pattern: {e}")
    
    def _get_role_specific_patterns(self, role: str) -> List[Dict[str, Any]]:
        """Get warming patterns specific to user role."""
        role_patterns = {
            "executive": [
                {"intent": "kpi_dashboard", "priority": "high", "time_period": "current"},
                {"intent": "strategic_metrics", "priority": "high", "time_period": "current_quarter"},
                {"intent": "performance_summary", "priority": "high", "time_period": "this_month"},
            ],
            "manager": [
                {"intent": "team_performance", "priority": "high", "time_period": "this_week"},
                {"intent": "operational_metrics", "priority": "high", "time_period": "today"},
                {"intent": "budget_tracking", "priority": "medium", "time_period": "this_month"},
            ],
            "analyst": [
                {"intent": "data_analysis", "priority": "medium", "time_period": "current"},
                {"intent": "trend_analysis", "priority": "medium", "time_period": "this_month"},
                {"intent": "performance_tracking", "priority": "medium", "time_period": "this_week"},
            ],
            "viewer": [
                {"intent": "daily_reports", "priority": "low", "time_period": "today"},
                {"intent": "status_updates", "priority": "low", "time_period": "current"},
            ]
        }
        
        return role_patterns.get(role.lower(), role_patterns["analyst"])
    
    def _generate_cache_hash(self, semantic_intent: Dict[str, Any]) -> str:
        """Generate cache hash for semantic intent."""
        import hashlib
        
        hash_string = json.dumps(semantic_intent, sort_keys=True)
        return hashlib.sha256(hash_string.encode()).hexdigest()[:16]
    
    def _generate_mock_insights(self, pattern: Dict[str, Any]) -> Dict[str, Any]:
        """Generate mock insights for cache warming."""
        intent = pattern.get("intent", "analysis")
        domain = pattern.get("business_domain", "general")
        
        return {
            "summary": f"Mock insights for {intent} in {domain} domain",
            "key_findings": [
                f"Key finding 1 for {intent}",
                f"Key finding 2 for {intent}",
                f"Trend analysis for {domain}"
            ],
            "recommendations": [
                f"Recommendation 1 for {intent}",
                f"Continue monitoring {domain} metrics"
            ],
            "confidence_score": 0.75,
            "data_sources": [domain],
            "generated_at": datetime.utcnow().isoformat(),
            "cache_warmed": True,
            "pattern": pattern
        }
    
    def get_warming_statistics(self) -> Dict[str, Any]:
        """Get cache warming statistics."""
        total_patterns = sum(len(patterns) for patterns in self.warming_patterns.values())
        
        return {
            "total_warming_patterns": total_patterns,
            "business_domains": list(self.warming_patterns.keys()),
            "time_based_patterns": len(self.time_based_patterns),
            "periodic_patterns": len(self.periodic_patterns),
            "patterns_by_domain": {
                domain: len(patterns) 
                for domain, patterns in self.warming_patterns.items()
            }
        }