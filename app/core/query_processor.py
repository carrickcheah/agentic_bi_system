"""
Enhanced Query Processor - Phase 1 of Five-Phase Workflow

Transforms natural language business questions into structured business intent
with semantic understanding, business context preservation, and pattern intelligence.

Key Features:
- Business intent extraction from natural language
- Pattern-aware semantic processing with 150+ manufacturing patterns
- Enhanced semantic hashing for cache optimization
- Business domain classification with pattern correlation
- User context integration with role-specific pattern matching
- Permission-aware processing
- Investigation methodology pre-selection from pattern library
"""

import hashlib
import json
from typing import Dict, List, Optional, Any
from datetime import datetime

from ..utils.logging import logger
from ..intelligence.pattern_library import PatternLibrary, PatternMatch
from ..fastmcp.qdrant_client import QdrantClient
from ..fastmcp.postgres_client import PostgresClient


class QueryProcessor:
    """
    Enhanced Phase 1: Query Processing with Pattern Intelligence
    
    Converts natural language business questions into structured semantic intent
    that preserves business context, enables intelligent cache lookup, and leverages
    150+ manufacturing business intelligence patterns for enhanced processing.
    """
    
    def __init__(
        self, 
        qdrant_client: Optional[QdrantClient] = None,
        postgres_client: Optional[PostgresClient] = None
    ):
        # Enhanced business domains aligned with manufacturing and business patterns
        self.business_domains = {
            # Manufacturing domains
            "production": ["production", "manufacturing", "output", "throughput", "efficiency", "oee", "yield"],
            "quality": ["quality", "defect", "inspection", "control", "compliance", "audit", "improvement"],
            "supply_chain": ["supplier", "inventory", "procurement", "logistics", "delivery", "lead_time", "stockout"],
            "cost_management": ["cost", "expense", "budget", "savings", "variance", "profitability", "roi"],
            "asset_management": ["equipment", "maintenance", "downtime", "reliability", "utilization", "lifecycle"],
            "safety": ["safety", "incident", "compliance", "risk", "hazard", "training", "culture"],
            "customer": ["customer", "demand", "satisfaction", "delivery", "service", "requirements"],
            "planning": ["planning", "scheduling", "capacity", "forecast", "demand", "resource", "allocation"],
            "hr": ["workforce", "productivity", "training", "skills", "retention", "performance"],
            # Business domains
            "sales": ["revenue", "sales", "income", "earnings", "profit", "funnel", "pipeline", "quota", "territory"],
            "product": ["product", "service", "feature", "usage", "adoption", "roadmap", "lifecycle", "onboarding"],
            "marketing": ["marketing", "campaign", "conversion", "acquisition", "leads", "attribution", "brand", "engagement"],
            "operations": ["operations", "efficiency", "cost", "process", "workflow", "automation", "optimization", "sla"],
            "finance": ["finance", "budget", "expense", "investment", "cash", "liquidity", "roas", "financial", "ratio"]
        }
        
        # Initialize pattern library for enhanced processing
        self.pattern_library = None
        if qdrant_client and postgres_client:
            self.pattern_library = PatternLibrary(qdrant_client, postgres_client)
        
        self.pattern_library_initialized = False
        
    async def process_business_question(
        self,
        business_question: str,
        user_context: Dict[str, Any],
        organization_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Process natural language business question into structured intent with pattern intelligence.
        
        Args:
            business_question: Natural language business question
            user_context: User information and permissions
            organization_context: Organizational context and business rules
            
        Returns:
            Structured semantic intent with business context and pattern intelligence
        """
        try:
            logger.info(f"= Processing business question: {business_question[:100]}...")
            
            # Initialize pattern library if available and not already initialized
            if self.pattern_library and not self.pattern_library_initialized:
                await self.pattern_library.initialize()
                self.pattern_library_initialized = True
                logger.info("✅ Pattern library initialized for enhanced processing")
            
            # Extract business intent
            business_intent = self._extract_business_intent(business_question)
            
            # Classify business domain
            business_domain = self._classify_business_domain(business_question)
            
            # Find matching patterns for enhanced processing
            pattern_matches = []
            suggested_methodologies = []
            pattern_confidence_boost = 0.0
            
            if self.pattern_library:
                try:
                    pattern_matches = await self.pattern_library.find_matching_patterns(
                        business_question, user_context, top_k=3
                    )
                    
                    if pattern_matches:
                        # Extract suggested investigation methodologies from top patterns
                        for match in pattern_matches[:2]:  # Top 2 matches
                            methodology = match.pattern_data["metadata"]["pattern"]
                            suggested_methodologies.append({
                                "methodology": methodology,
                                "confidence": match.total_score,
                                "success_rate": match.pattern_data["metadata"]["success_rate"],
                                "complexity": match.pattern_data["metadata"]["complexity"]
                            })
                        
                        # Boost confidence based on pattern matches
                        top_match_score = pattern_matches[0].total_score
                        pattern_confidence_boost = min(0.3, top_match_score * 0.2)  # Max 30% boost
                        
                        logger.info(f"🎯 Found {len(pattern_matches)} pattern matches (boost: +{pattern_confidence_boost:.3f})")
                
                except Exception as e:
                    logger.warning(f"Pattern matching failed, continuing without patterns: {e}")
            
            # Generate enhanced semantic hash with pattern context
            semantic_hash = self._generate_enhanced_semantic_hash(
                business_question, business_domain, user_context.get("role", ""), pattern_matches
            )
            
            # Determine complexity indicators
            complexity_indicators = self._analyze_complexity_indicators(business_question)
            
            # Calculate enhanced confidence score with pattern boost
            base_confidence = self._calculate_confidence_score(business_question)
            enhanced_confidence = min(1.0, base_confidence + pattern_confidence_boost)
            
            # Create structured intent with pattern intelligence
            semantic_intent = {
                "original_question": business_question,
                "business_intent": business_intent,
                "business_domain": business_domain,
                "semantic_hash": semantic_hash,
                "complexity_indicators": complexity_indicators,
                "pattern_intelligence": {
                    "matching_patterns": [
                        {
                            "pattern_id": match.pattern_id,
                            "information": match.pattern_data["information"],
                            "confidence": match.total_score,
                            "success_rate": match.pattern_data["metadata"]["success_rate"],
                            "expected_deliverables": match.pattern_data["metadata"]["expected_deliverables"]
                        } for match in pattern_matches[:3]
                    ],
                    "suggested_methodologies": suggested_methodologies,
                    "pattern_confidence_boost": pattern_confidence_boost,
                    "methodology_recommendations": [m["methodology"] for m in suggested_methodologies[:2]]
                },
                "user_context": {
                    "user_id": user_context.get("user_id"),
                    "role": user_context.get("role"),
                    "department": user_context.get("department"),
                    "permissions": user_context.get("permissions", [])
                },
                "organization_context": {
                    "organization_id": organization_context.get("organization_id"),
                    "business_rules": organization_context.get("business_rules", {}),
                    "fiscal_calendar": organization_context.get("fiscal_calendar", {}),
                    "data_classification": organization_context.get("data_classification", "standard")
                },
                "processing_metadata": {
                    "processed_at": datetime.utcnow().isoformat(),
                    "processor_version": "2.0",  # Updated version for pattern integration
                    "confidence_score": enhanced_confidence,
                    "base_confidence": base_confidence,
                    "pattern_enhanced": len(pattern_matches) > 0
                }
            }
            
            logger.info(f" Query processed - Domain: {business_domain}, Hash: {semantic_hash[:16]}...")
            return semantic_intent
            
        except Exception as e:
            logger.error(f"Query processing failed: {e}")
            raise
    
    def _extract_business_intent(self, business_question: str) -> Dict[str, Any]:
        """Extract structured business intent from natural language."""
        question_lower = business_question.lower()
        
        # Question type classification
        question_type = "unknown"
        if any(word in question_lower for word in ["what", "how much", "how many"]):
            question_type = "descriptive"
        elif any(word in question_lower for word in ["why", "what caused", "root cause"]):
            question_type = "analytical"
        elif any(word in question_lower for word in ["what if", "scenario", "predict"]):
            question_type = "predictive"
        elif any(word in question_lower for word in ["compare", "vs", "versus", "difference"]):
            question_type = "comparative"
        elif any(word in question_lower for word in ["trend", "over time", "change"]):
            question_type = "temporal"
        
        # Time period extraction
        time_period = None
        if any(word in question_lower for word in ["today", "yesterday", "daily"]):
            time_period = "daily"
        elif any(word in question_lower for word in ["week", "weekly"]):
            time_period = "weekly"
        elif any(word in question_lower for word in ["month", "monthly"]):
            time_period = "monthly"
        elif any(word in question_lower for word in ["quarter", "quarterly", "q1", "q2", "q3", "q4"]):
            time_period = "quarterly"
        elif any(word in question_lower for word in ["year", "yearly", "annual"]):
            time_period = "yearly"
        
        # Metric extraction
        potential_metrics = []
        metric_keywords = {
            "revenue": ["revenue", "sales", "income", "earnings"],
            "count": ["count", "number", "total", "amount"],
            "percentage": ["percentage", "percent", "rate", "ratio"],
            "average": ["average", "mean", "avg"],
            "growth": ["growth", "increase", "decrease", "change"]
        }
        
        for metric, keywords in metric_keywords.items():
            if any(keyword in question_lower for keyword in keywords):
                potential_metrics.append(metric)
        
        return {
            "question_type": question_type,
            "time_period": time_period,
            "potential_metrics": potential_metrics,
            "requires_comparison": "vs" in question_lower or "versus" in question_lower,
            "requires_aggregation": any(word in question_lower for word in ["total", "sum", "average", "max", "min"]),
            "requires_grouping": any(word in question_lower for word in ["by", "per", "each", "breakdown"])
        }
    
    def _classify_business_domain(self, business_question: str) -> str:
        """Classify the business domain of the question."""
        question_lower = business_question.lower()
        
        domain_scores = {}
        for domain, keywords in self.business_domains.items():
            score = sum(1 for keyword in keywords if keyword in question_lower)
            if score > 0:
                domain_scores[domain] = score
        
        if domain_scores:
            return max(domain_scores, key=domain_scores.get)
        else:
            return "general"
    
    def _generate_semantic_hash(
        self, 
        business_question: str, 
        business_domain: str, 
        user_role: str
    ) -> str:
        """Generate semantic hash for cache lookup and similarity matching."""
        # Normalize the question for semantic consistency
        normalized_question = self._normalize_question(business_question)
        
        # Create semantic components
        semantic_components = {
            "normalized_question": normalized_question,
            "business_domain": business_domain,
            "user_role": user_role
        }
        
        # Generate hash
        semantic_string = json.dumps(semantic_components, sort_keys=True)
        return hashlib.sha256(semantic_string.encode()).hexdigest()
    
    def _generate_enhanced_semantic_hash(
        self, 
        business_question: str, 
        business_domain: str, 
        user_role: str,
        pattern_matches: List[Any]
    ) -> str:
        """Generate enhanced semantic hash including pattern context for improved cache matching."""
        # Normalize the question for semantic consistency
        normalized_question = self._normalize_question(business_question)
        
        # Extract pattern signatures for hash enhancement
        pattern_signatures = []
        if pattern_matches:
            for match in pattern_matches[:2]:  # Top 2 patterns
                pattern_signatures.append({
                    "pattern_id": match.pattern_id,
                    "success_rate": match.pattern_data["metadata"]["success_rate"],
                    "complexity": match.pattern_data["metadata"]["complexity"]
                })
        
        # Create enhanced semantic components
        semantic_components = {
            "normalized_question": normalized_question,
            "business_domain": business_domain,
            "user_role": user_role,
            "pattern_signatures": pattern_signatures
        }
        
        # Generate enhanced hash
        semantic_string = json.dumps(semantic_components, sort_keys=True)
        return hashlib.sha256(semantic_string.encode()).hexdigest()
    
    def _normalize_question(self, question: str) -> str:
        """Normalize question for semantic consistency."""
        normalized = question.lower()
        
        # Standardize time references
        time_normalizations = {
            "yesterday": "previous day",
            "last week": "previous week", 
            "last month": "previous month",
            "last quarter": "previous quarter",
            "last year": "previous year",
            "this week": "current week",
            "this month": "current month",
            "this quarter": "current quarter",
            "this year": "current year"
        }
        
        for original, normalized_term in time_normalizations.items():
            normalized = normalized.replace(original, normalized_term)
        
        # Standardize metric terms
        metric_normalizations = {
            "sales": "revenue",
            "income": "revenue",
            "earnings": "revenue",
            "customers": "customer",
            "clients": "customer",
            "users": "customer"
        }
        
        for original, normalized_term in metric_normalizations.items():
            normalized = normalized.replace(original, normalized_term)
        
        return normalized
    
    def _analyze_complexity_indicators(self, business_question: str) -> Dict[str, Any]:
        """Analyze indicators of question complexity."""
        question_lower = business_question.lower()
        
        # Complexity indicators
        indicators = {
            "multi_domain": len([domain for domain in self.business_domains.keys() 
                               if any(keyword in question_lower for keyword in self.business_domains[domain])]) > 1,
            "temporal_analysis": any(word in question_lower for word in ["trend", "over time", "compare", "change"]),
            "causal_analysis": any(word in question_lower for word in ["why", "cause", "impact", "effect"]),
            "predictive_analysis": any(word in question_lower for word in ["predict", "forecast", "what if", "scenario"]),
            "multi_metric": len(question_lower.split("and")) > 1,
            "requires_context": any(word in question_lower for word in ["compared to", "vs", "relative to"]),
            "open_ended": question_lower.startswith(("why", "how", "what caused"))
        }
        
        # Calculate overall complexity score
        complexity_score = sum(indicators.values()) / len(indicators)
        
        # Determine complexity level
        if complexity_score < 0.3:
            complexity_level = "simple"
        elif complexity_score < 0.6:
            complexity_level = "moderate"
        else:
            complexity_level = "complex"
        
        return {
            "indicators": indicators,
            "complexity_score": complexity_score,
            "complexity_level": complexity_level
        }
    
    def _calculate_confidence_score(self, business_question: str) -> float:
        """Calculate confidence score for question processing."""
        # Simple confidence calculation based on clarity indicators
        clarity_indicators = [
            len(business_question.split()) > 3,  # Sufficient length
            any(char in business_question for char in "?"),  # Question format
            not any(word in business_question.lower() for word in ["maybe", "perhaps", "unclear"]),  # No uncertainty
            len(business_question) < 200  # Not too long
        ]
        
        return sum(clarity_indicators) / len(clarity_indicators)