"""
Query Processor - Phase 1 of Five-Phase Workflow

Transforms natural language business questions into structured business intent
with semantic understanding and business context preservation.

Key Features:
- Business intent extraction from natural language
- Semantic hashing for cache lookup
- Business domain classification
- User context integration
- Permission-aware processing
"""

import hashlib
import json
from typing import Dict, List, Optional, Any
from datetime import datetime

from ..utils.logging import logger


class QueryProcessor:
    """
    Phase 1: Query Processing
    
    Converts natural language business questions into structured semantic intent
    that preserves business context and enables intelligent cache lookup.
    """
    
    def __init__(self):
        self.business_domains = {
            "sales": ["revenue", "sales", "income", "earnings", "profit"],
            "customer": ["customer", "client", "user", "satisfaction", "retention"],
            "product": ["product", "service", "feature", "usage", "adoption"],
            "marketing": ["marketing", "campaign", "conversion", "acquisition", "leads"],
            "operations": ["operations", "efficiency", "cost", "process", "workflow"],
            "finance": ["finance", "budget", "expense", "investment", "cash"],
            "hr": ["employee", "staff", "team", "performance", "hiring"]
        }
        
    async def process_business_question(
        self,
        business_question: str,
        user_context: Dict[str, Any],
        organization_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Process natural language business question into structured intent.
        
        Args:
            business_question: Natural language business question
            user_context: User information and permissions
            organization_context: Organizational context and business rules
            
        Returns:
            Structured semantic intent with business context
        """
        try:
            logger.info(f"= Processing business question: {business_question[:100]}...")
            
            # Extract business intent
            business_intent = self._extract_business_intent(business_question)
            
            # Classify business domain
            business_domain = self._classify_business_domain(business_question)
            
            # Generate semantic hash for cache lookup
            semantic_hash = self._generate_semantic_hash(
                business_question, business_domain, user_context.get("role", "")
            )
            
            # Determine complexity indicators
            complexity_indicators = self._analyze_complexity_indicators(business_question)
            
            # Create structured intent
            semantic_intent = {
                "original_question": business_question,
                "business_intent": business_intent,
                "business_domain": business_domain,
                "semantic_hash": semantic_hash,
                "complexity_indicators": complexity_indicators,
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
                    "processor_version": "1.0",
                    "confidence_score": self._calculate_confidence_score(business_question)
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