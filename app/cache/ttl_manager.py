"""
TTL Manager - Dynamic Time-To-Live Optimization

Intelligently determines cache duration based on business context, data volatility,
and organizational patterns. Implements enterprise-grade caching strategies.

Features:
- Business domain-aware TTL calculation
- Data type sensitivity analysis
- Urgency-based cache duration
- User role-based optimization
- Organizational learning integration
"""

import json
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from enum import Enum

from ..utils.logging import logger


class DataVolatility(Enum):
    """Data volatility levels for TTL calculation."""
    CRITICAL = "critical"      # Changes every few minutes
    HIGH = "high"             # Changes hourly
    MEDIUM = "medium"         # Changes daily
    LOW = "low"              # Changes weekly
    STATIC = "static"        # Changes monthly/yearly


class CachePriority(Enum):
    """Cache priority levels."""
    EMERGENCY = "emergency"   # Mission critical
    HIGH = "high"            # Business critical  
    STANDARD = "standard"    # Normal operations
    LOW = "low"             # Background processes


class TTLManager:
    """
    Enterprise TTL Manager for intelligent cache duration optimization.
    
    Provides dynamic TTL calculation based on business context, data characteristics,
    user requirements, and organizational patterns.
    """
    
    def __init__(self):
        """Initialize TTL manager with enterprise configurations."""
        
        # Core TTL matrix - Real-world business scenarios
        self.ttl_matrix = {
            # === CRITICAL SYSTEMS (1-15 minutes) ===
            ("it", "security_alerts"): 300,           # 5 minutes - Security critical
            ("it", "system_outages"): 180,            # 3 minutes - Infrastructure critical
            ("finance", "fraud_detection"): 120,      # 2 minutes - Financial security
            ("operations", "emergency_alerts"): 300,  # 5 minutes - Operations critical
            ("trading", "market_data"): 60,           # 1 minute - Trading systems
            
            # === HIGH FREQUENCY (15 minutes - 2 hours) ===
            ("finance", "real_time_metrics"): 900,    # 15 minutes - Financial dashboards
            ("sales", "pipeline_updates"): 1800,      # 30 minutes - Sales tracking
            ("customer", "support_tickets"): 1200,    # 20 minutes - Customer service
            ("operations", "inventory_levels"): 1800, # 30 minutes - Inventory management
            ("marketing", "campaign_tracking"): 3600, # 1 hour - Campaign monitoring
            ("product", "user_activity"): 2700,       # 45 minutes - Product analytics
            
            # === BUSINESS OPERATIONS (2-6 hours) ===
            ("finance", "daily_cash_flow"): 7200,     # 2 hours - Financial operations
            ("sales", "performance_tracking"): 10800, # 3 hours - Sales performance
            ("marketing", "conversion_rates"): 14400, # 4 hours - Marketing effectiveness
            ("customer", "satisfaction_scores"): 18000, # 5 hours - Customer metrics
            ("operations", "efficiency_metrics"): 21600, # 6 hours - Operational KPIs
            ("product", "feature_adoption"): 18000,   # 5 hours - Product insights
            ("hr", "attendance_tracking"): 14400,     # 4 hours - HR operations
            
            # === DAILY REPORTING (6-24 hours) ===
            ("sales", "daily_reports"): 28800,        # 8 hours - Daily sales summary
            ("finance", "budget_variance"): 43200,    # 12 hours - Budget analysis
            ("customer", "churn_analysis"): 86400,    # 24 hours - Customer retention
            ("marketing", "roi_analysis"): 64800,     # 18 hours - Marketing ROI
            ("operations", "quality_metrics"): 43200, # 12 hours - Quality tracking
            ("product", "performance_metrics"): 32400, # 9 hours - Product performance
            ("hr", "productivity_metrics"): 86400,    # 24 hours - HR analytics
            ("it", "system_performance"): 21600,      # 6 hours - IT operations
            
            # === STRATEGIC ANALYSIS (1-7 days) ===
            ("sales", "monthly_analysis"): 259200,    # 3 days - Monthly sales review
            ("finance", "quarterly_statements"): 604800, # 7 days - Quarterly reports
            ("customer", "lifetime_value"): 432000,   # 5 days - Customer LTV analysis
            ("marketing", "attribution_analysis"): 345600, # 4 days - Attribution modeling
            ("operations", "capacity_planning"): 518400,  # 6 days - Capacity analysis
            ("product", "roadmap_planning"): 604800,  # 7 days - Product strategy
            ("hr", "performance_reviews"): 1209600,   # 14 days - Performance cycles
            ("executive", "kpi_dashboards"): 172800,  # 2 days - Executive reporting
            ("legal", "compliance_checks"): 432000,   # 5 days - Legal compliance
            ("research", "market_analysis"): 604800,  # 7 days - Market research
            
            # === LONG-TERM & HISTORICAL (7+ days) ===
            ("historical", "trend_analysis"): 1209600,    # 14 days - Historical trends
            ("compliance", "audit_trails"): 2592000,      # 30 days - Compliance records
            ("finance", "annual_reports"): 5184000,       # 60 days - Annual statements
            ("research", "competitive_intelligence"): 1814400, # 21 days - Competitive analysis
            ("legal", "contract_reviews"): 1209600,       # 14 days - Legal reviews
            ("executive", "strategic_planning"): 2592000, # 30 days - Strategic initiatives
            ("quality", "certification_audits"): 5184000, # 60 days - Quality certifications
            
            # === INDUSTRY-SPECIFIC ===
            # E-commerce
            ("ecommerce", "cart_abandonment"): 1800,      # 30 minutes
            ("ecommerce", "pricing_optimization"): 7200,  # 2 hours
            ("ecommerce", "inventory_alerts"): 900,       # 15 minutes
            ("ecommerce", "recommendation_performance"): 10800, # 3 hours
            
            # Healthcare
            ("healthcare", "patient_monitoring"): 300,    # 5 minutes
            ("healthcare", "medication_tracking"): 1800,  # 30 minutes
            ("healthcare", "compliance_reporting"): 86400, # 24 hours
            ("healthcare", "outcome_analysis"): 604800,   # 7 days
            
            # Financial Services
            ("fintech", "fraud_detection"): 60,           # 1 minute
            ("fintech", "trading_analytics"): 300,        # 5 minutes
            ("fintech", "risk_calculations"): 900,        # 15 minutes
            ("fintech", "regulatory_reporting"): 86400,   # 24 hours
            
            # Manufacturing
            ("manufacturing", "production_line"): 600,    # 10 minutes
            ("manufacturing", "quality_control"): 3600,   # 1 hour
            ("manufacturing", "supply_chain"): 14400,     # 4 hours
            ("manufacturing", "maintenance_schedules"): 86400, # 24 hours
        }
        
        # Domain-based default TTLs
        self.domain_defaults = {
            "finance": 3600,      # 1 hour - Financial data changes frequently
            "sales": 7200,        # 2 hours - Sales metrics need regular updates
            "customer": 10800,    # 3 hours - Customer data moderately dynamic
            "operations": 3600,   # 1 hour - Operations need current data
            "hr": 86400,          # 24 hours - HR data changes daily
            "legal": 259200,      # 3 days - Legal processes are slower
            "executive": 86400,   # 24 hours - Executive reports daily
            "it": 1800,           # 30 minutes - IT metrics change quickly
            "marketing": 7200,    # 2 hours - Marketing campaigns dynamic
            "product": 10800,     # 3 hours - Product metrics moderate frequency
            "quality": 21600,     # 6 hours - Quality metrics less frequent
            "research": 86400,    # 24 hours - Research data stable
            "compliance": 259200, # 3 days - Compliance checks periodic
            "historical": 604800, # 7 days - Historical data rarely changes
            "trading": 300,       # 5 minutes - Trading data very dynamic
            "ecommerce": 3600,    # 1 hour - E-commerce moderately dynamic
            "healthcare": 1800,   # 30 minutes - Healthcare data sensitive
            "fintech": 600,       # 10 minutes - Fintech data very dynamic
            "manufacturing": 7200, # 2 hours - Manufacturing moderately dynamic
        }
        
        # Volatility-based TTL multipliers
        self.volatility_multipliers = {
            DataVolatility.CRITICAL: 0.25,   # 25% of base TTL
            DataVolatility.HIGH: 0.5,        # 50% of base TTL
            DataVolatility.MEDIUM: 1.0,      # 100% of base TTL (baseline)
            DataVolatility.LOW: 2.0,         # 200% of base TTL
            DataVolatility.STATIC: 4.0,      # 400% of base TTL
        }
        
        # Priority-based TTL adjustments
        self.priority_multipliers = {
            CachePriority.EMERGENCY: 0.1,    # 10% of base TTL - Very short cache
            CachePriority.HIGH: 0.5,         # 50% of base TTL - Short cache
            CachePriority.STANDARD: 1.0,     # 100% of base TTL - Normal cache
            CachePriority.LOW: 2.0,          # 200% of base TTL - Long cache
        }
        
        # User role-based adjustments
        self.role_multipliers = {
            "executive": 0.5,     # Executives need fresh data
            "manager": 0.75,      # Managers need recent data
            "analyst": 1.0,       # Analysts use standard caching
            "viewer": 1.5,        # Viewers can use older cache
            "guest": 2.0,         # Guests get longest cache
        }
        
        # Time-of-day adjustments (business hours vs off-hours)
        self.time_multipliers = {
            "business_hours": 1.0,    # Standard TTL during business hours
            "off_hours": 2.0,         # Longer TTL during off hours
            "weekend": 3.0,           # Even longer TTL on weekends
            "holiday": 4.0,           # Longest TTL during holidays
        }
        
        # Minimum and maximum TTL bounds (seconds)
        self.min_ttl = 60         # 1 minute minimum
        self.max_ttl = 7776000    # 90 days maximum
        self.default_ttl = 3600   # 1 hour default
        
    def get_dynamic_ttl(
        self,
        business_domain: str,
        data_type: str,
        volatility: Optional[DataVolatility] = None,
        priority: Optional[CachePriority] = None,
        user_role: Optional[str] = None,
        urgency: Optional[str] = None,
        organization_context: Optional[Dict[str, Any]] = None
    ) -> int:
        """
        Calculate dynamic TTL based on multiple factors.
        
        Args:
            business_domain: Business domain (sales, finance, etc.)
            data_type: Type of data being cached
            volatility: Data volatility level
            priority: Cache priority level
            user_role: User's role in organization
            urgency: Urgency level (urgent, high, standard, low)
            organization_context: Additional organizational context
            
        Returns:
            TTL in seconds
        """
        try:
            # Start with base TTL
            base_ttl = self._get_base_ttl(business_domain, data_type)
            
            # Apply volatility adjustment
            if volatility:
                base_ttl = int(base_ttl * self.volatility_multipliers.get(volatility, 1.0))
            
            # Apply priority adjustment
            if priority:
                base_ttl = int(base_ttl * self.priority_multipliers.get(priority, 1.0))
            
            # Apply user role adjustment
            if user_role:
                base_ttl = int(base_ttl * self.role_multipliers.get(user_role.lower(), 1.0))
            
            # Apply urgency adjustment
            if urgency:
                urgency_multiplier = self._get_urgency_multiplier(urgency)
                base_ttl = int(base_ttl * urgency_multiplier)
            
            # Apply time-of-day adjustment
            time_multiplier = self._get_time_multiplier()
            base_ttl = int(base_ttl * time_multiplier)
            
            # Apply organizational adjustments
            if organization_context:
                org_multiplier = self._get_organization_multiplier(organization_context)
                base_ttl = int(base_ttl * org_multiplier)
            
            # Enforce bounds
            final_ttl = max(self.min_ttl, min(base_ttl, self.max_ttl))
            
            logger.debug(f"Dynamic TTL calculated: {business_domain}.{data_type} = {final_ttl}s")
            return final_ttl
            
        except Exception as e:
            logger.warning(f"TTL calculation failed, using default: {e}")
            return self.default_ttl
    
    def get_ttl_for_question(
        self,
        semantic_intent: Dict[str, Any],
        user_context: Optional[Dict[str, Any]] = None,
        organization_context: Optional[Dict[str, Any]] = None
    ) -> int:
        """
        Get TTL based on semantic intent and context.
        
        Args:
            semantic_intent: Processed business question intent
            user_context: User information and preferences
            organization_context: Organizational context and business rules
            
        Returns:
            TTL in seconds
        """
        try:
            # Extract business context
            business_domain = semantic_intent.get("business_domain", "general")
            question_type = semantic_intent.get("business_intent", {}).get("question_type", "descriptive")
            
            # Map question types to data types
            data_type_mapping = {
                "descriptive": "daily_reports",
                "analytical": "monthly_analysis",
                "predictive": "behavior_analysis",
                "comparative": "performance_tracking",
                "temporal": "real_time_metrics",
                "diagnostic": "efficiency_metrics"
            }
            
            data_type = data_type_mapping.get(question_type, "daily_reports")
            
            # Determine volatility based on question characteristics
            volatility = self._assess_data_volatility(semantic_intent)
            
            # Determine priority based on user and question context
            priority = self._assess_priority(semantic_intent, user_context)
            
            # Get user role
            user_role = user_context.get("role") if user_context else None
            
            # Get urgency
            urgency = semantic_intent.get("urgency", "standard")
            
            return self.get_dynamic_ttl(
                business_domain=business_domain,
                data_type=data_type,
                volatility=volatility,
                priority=priority,
                user_role=user_role,
                urgency=urgency,
                organization_context=organization_context
            )
            
        except Exception as e:
            logger.warning(f"Question TTL calculation failed: {e}")
            return self.default_ttl
    
    def get_recommended_ttl_strategy(
        self,
        business_domain: str,
        expected_usage_pattern: str = "standard"
    ) -> Dict[str, Any]:
        """
        Get recommended TTL strategy for a business domain.
        
        Args:
            business_domain: Business domain to analyze
            expected_usage_pattern: Expected usage pattern (low/standard/high/critical)
            
        Returns:
            TTL strategy recommendations
        """
        try:
            # Get relevant TTL configurations for the domain
            domain_ttls = {
                data_type: ttl for (domain, data_type), ttl in self.ttl_matrix.items()
                if domain == business_domain
            }
            
            if not domain_ttls:
                domain_ttls = {"default": self.domain_defaults.get(business_domain, self.default_ttl)}
            
            # Usage pattern adjustments
            pattern_multipliers = {
                "low": 2.0,      # Low usage - longer cache
                "standard": 1.0, # Standard usage - normal cache
                "high": 0.75,    # High usage - shorter cache
                "critical": 0.5  # Critical usage - very short cache
            }
            
            multiplier = pattern_multipliers.get(expected_usage_pattern, 1.0)
            
            # Apply pattern adjustment
            adjusted_ttls = {
                data_type: max(self.min_ttl, min(int(ttl * multiplier), self.max_ttl))
                for data_type, ttl in domain_ttls.items()
            }
            
            return {
                "business_domain": business_domain,
                "usage_pattern": expected_usage_pattern,
                "recommended_ttls": adjusted_ttls,
                "default_ttl": self.domain_defaults.get(business_domain, self.default_ttl),
                "min_ttl": min(adjusted_ttls.values()) if adjusted_ttls else self.min_ttl,
                "max_ttl": max(adjusted_ttls.values()) if adjusted_ttls else self.max_ttl,
                "avg_ttl": sum(adjusted_ttls.values()) // len(adjusted_ttls) if adjusted_ttls else self.default_ttl
            }
            
        except Exception as e:
            logger.error(f"TTL strategy calculation failed: {e}")
            return {
                "business_domain": business_domain,
                "error": str(e),
                "fallback_ttl": self.default_ttl
            }
    
    def _get_base_ttl(self, business_domain: str, data_type: str) -> int:
        """Get base TTL from matrix or domain default."""
        # Try exact match first
        exact_match = self.ttl_matrix.get((business_domain, data_type))
        if exact_match:
            return exact_match
        
        # Try partial matches (wildcard data types)
        for (domain, dtype), ttl in self.ttl_matrix.items():
            if domain == business_domain and dtype == "any":
                return ttl
        
        # Fall back to domain default
        return self.domain_defaults.get(business_domain, self.default_ttl)
    
    def _assess_data_volatility(self, semantic_intent: Dict[str, Any]) -> DataVolatility:
        """Assess data volatility based on semantic intent."""
        business_domain = semantic_intent.get("business_domain", "general")
        question_type = semantic_intent.get("business_intent", {}).get("question_type", "descriptive")
        time_period = semantic_intent.get("business_intent", {}).get("time_period")
        
        # Real-time and critical domains
        if business_domain in ["it", "trading", "fintech"] and "real_time" in str(semantic_intent):
            return DataVolatility.CRITICAL
        
        # High volatility indicators
        if question_type in ["predictive", "temporal"] or time_period in ["daily", "hourly"]:
            return DataVolatility.HIGH
        
        # Medium volatility (default)
        if question_type in ["descriptive", "comparative"]:
            return DataVolatility.MEDIUM
        
        # Low volatility
        if time_period in ["monthly", "quarterly", "yearly"]:
            return DataVolatility.LOW
        
        # Static volatility
        if business_domain == "historical" or "historical" in str(semantic_intent):
            return DataVolatility.STATIC
        
        return DataVolatility.MEDIUM
    
    def _assess_priority(
        self,
        semantic_intent: Dict[str, Any],
        user_context: Optional[Dict[str, Any]]
    ) -> CachePriority:
        """Assess cache priority based on context."""
        urgency = semantic_intent.get("urgency", "standard")
        business_domain = semantic_intent.get("business_domain", "general")
        user_role = user_context.get("role") if user_context else "analyst"
        
        # Emergency priority
        if urgency == "urgent" or "emergency" in str(semantic_intent):
            return CachePriority.EMERGENCY
        
        # High priority
        if (urgency == "high" or 
            user_role in ["executive", "director"] or 
            business_domain in ["finance", "security", "trading"]):
            return CachePriority.HIGH
        
        # Low priority
        if urgency == "low" or user_role in ["guest", "viewer"]:
            return CachePriority.LOW
        
        return CachePriority.STANDARD
    
    def _get_urgency_multiplier(self, urgency: str) -> float:
        """Get TTL multiplier based on urgency."""
        urgency_multipliers = {
            "urgent": 0.1,     # 10% of base TTL - Very urgent
            "high": 0.5,       # 50% of base TTL - High urgency
            "standard": 1.0,   # 100% of base TTL - Normal
            "low": 1.5,        # 150% of base TTL - Low urgency
        }
        return urgency_multipliers.get(urgency.lower(), 1.0)
    
    def _get_time_multiplier(self) -> float:
        """Get TTL multiplier based on current time."""
        now = datetime.utcnow()
        hour = now.hour
        weekday = now.weekday()  # 0=Monday, 6=Sunday
        
        # Weekend adjustment
        if weekday >= 5:  # Saturday or Sunday
            return self.time_multipliers["weekend"]
        
        # Business hours (9 AM - 5 PM UTC)
        if 9 <= hour <= 17:
            return self.time_multipliers["business_hours"]
        else:
            return self.time_multipliers["off_hours"]
    
    def _get_organization_multiplier(self, organization_context: Dict[str, Any]) -> float:
        """Get TTL multiplier based on organizational context."""
        # Organization size adjustment
        org_size = organization_context.get("size", "medium")
        size_multipliers = {
            "startup": 0.75,   # Startups need fresh data
            "small": 0.9,      # Small companies moderate caching
            "medium": 1.0,     # Medium companies standard caching
            "large": 1.25,     # Large companies can cache longer
            "enterprise": 1.5, # Enterprises benefit from longer caching
        }
        
        # Data classification adjustment
        classification = organization_context.get("data_classification", "standard")
        classification_multipliers = {
            "public": 2.0,       # Public data can be cached longer
            "internal": 1.0,     # Internal data standard caching
            "confidential": 0.75, # Confidential data shorter caching
            "restricted": 0.5,   # Restricted data very short caching
        }
        
        size_mult = size_multipliers.get(org_size, 1.0)
        class_mult = classification_multipliers.get(classification, 1.0)
        
        return size_mult * class_mult
    
    def get_ttl_statistics(self) -> Dict[str, Any]:
        """Get statistics about TTL configurations."""
        try:
            ttl_values = list(self.ttl_matrix.values())
            
            return {
                "total_configurations": len(self.ttl_matrix),
                "unique_domains": len(set(domain for domain, _ in self.ttl_matrix.keys())),
                "min_ttl_configured": min(ttl_values),
                "max_ttl_configured": max(ttl_values),
                "avg_ttl_configured": sum(ttl_values) // len(ttl_values),
                "ttl_distribution": {
                    "critical_1min_15min": len([t for t in ttl_values if t <= 900]),
                    "high_15min_2hrs": len([t for t in ttl_values if 900 < t <= 7200]),
                    "medium_2hrs_24hrs": len([t for t in ttl_values if 7200 < t <= 86400]),
                    "low_1day_7days": len([t for t in ttl_values if 86400 < t <= 604800]),
                    "static_7days_plus": len([t for t in ttl_values if t > 604800]),
                },
                "domain_coverage": list(self.domain_defaults.keys()),
                "supported_volatility_levels": [v.value for v in DataVolatility],
                "supported_priority_levels": [p.value for p in CachePriority],
            }
            
        except Exception as e:
            logger.error(f"TTL statistics calculation failed: {e}")
            return {"error": str(e)}
    
    def validate_ttl_configuration(self) -> Dict[str, Any]:
        """Validate TTL configuration for consistency and best practices."""
        issues = []
        warnings = []
        
        try:
            # Check for missing domain defaults
            matrix_domains = set(domain for domain, _ in self.ttl_matrix.keys())
            default_domains = set(self.domain_defaults.keys())
            
            missing_defaults = matrix_domains - default_domains
            if missing_defaults:
                warnings.append(f"Missing domain defaults: {missing_defaults}")
            
            # Check for TTL bounds violations
            for (domain, data_type), ttl in self.ttl_matrix.items():
                if ttl < self.min_ttl:
                    issues.append(f"TTL too low: {domain}.{data_type} = {ttl}s < {self.min_ttl}s")
                if ttl > self.max_ttl:
                    issues.append(f"TTL too high: {domain}.{data_type} = {ttl}s > {self.max_ttl}s")
            
            # Check for logical inconsistencies
            for domain in matrix_domains:
                domain_ttls = [ttl for (d, dt), ttl in self.ttl_matrix.items() if d == domain]
                if len(domain_ttls) > 1:
                    if max(domain_ttls) / min(domain_ttls) > 100:  # More than 100x difference
                        warnings.append(f"Large TTL variation in {domain}: {min(domain_ttls)}s - {max(domain_ttls)}s")
            
            return {
                "valid": len(issues) == 0,
                "issues": issues,
                "warnings": warnings,
                "summary": f"Configuration check: {len(issues)} issues, {len(warnings)} warnings"
            }
            
        except Exception as e:
            return {
                "valid": False,
                "error": str(e),
                "summary": "Configuration validation failed"
            }