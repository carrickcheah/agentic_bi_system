"""
Domain Expert - Business Domain Classification Component
Self-contained business intelligence domain classification for manufacturing contexts.
Zero external dependencies beyond module boundary.
"""

from enum import Enum
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import re
from pathlib import Path

try:
    from .config import settings
    from .intelligence_logging import setup_logger, performance_monitor
    from .query_intent_classifier import get_intent_classifier, QueryIntent
    from .response_templates import get_response_template
except ImportError:
    from config import settings
    from intelligence_logging import setup_logger, performance_monitor
    from query_intent_classifier import get_intent_classifier, QueryIntent
    from response_templates import get_response_template


class BusinessDomain(Enum):
    """Manufacturing business intelligence domains."""
    PRODUCTION = "production"
    QUALITY = "quality"
    SUPPLY_CHAIN = "supply_chain"
    COST = "cost"
    ASSETS = "assets"
    SAFETY = "safety"
    CUSTOMER = "customer"
    PLANNING = "planning"
    HUMAN_RESOURCES = "human_resources"
    SALES = "sales"
    FINANCE = "finance"
    MARKETING = "marketing"
    OPERATIONS = "operations"
    STRATEGIC = "strategic"


class AnalysisType(Enum):
    """Business analysis types aligned with investigation methodologies."""
    DESCRIPTIVE = "descriptive"      # What happened?
    DIAGNOSTIC = "diagnostic"        # Why did it happen?
    PREDICTIVE = "predictive"        # What will happen?
    PRESCRIPTIVE = "prescriptive"    # What should we do?


@dataclass
class NonBusinessResponse:
    """Response for non-business queries (greetings, help, casual)."""
    intent_type: str
    response_text: str
    confidence: float
    is_business: bool = False


@dataclass
class BusinessIntent:
    """Structured business intent classification."""
    primary_domain: BusinessDomain
    secondary_domains: List[BusinessDomain]
    analysis_type: AnalysisType
    confidence: float
    key_indicators: List[str]
    business_metrics: List[str]
    time_context: Optional[str] = None
    urgency_level: str = "normal"  # low, normal, high, critical


class DomainExpert:
    """
    Business domain classification expert using pattern-based analysis.
    Identifies business context and intent from natural language queries.
    """
    
    def __init__(self):
        self.logger = setup_logger("domain_expert")
        self._domain_patterns = self._load_domain_patterns()
        self._analysis_patterns = self._load_analysis_patterns()
        self._urgency_patterns = self._load_urgency_patterns()
        
        self.logger.info("Domain Expert initialized with pattern libraries")
    
    def _load_domain_patterns(self) -> Dict[BusinessDomain, Dict[str, List[str]]]:
        """Load domain classification patterns."""
        return {
            BusinessDomain.PRODUCTION: {
                "keywords": [
                    "production", "manufacturing", "output", "throughput", "efficiency",
                    "line", "equipment", "machine", "capacity", "utilization",
                    "downtime", "oee", "yield", "cycle time", "takt time",
                    "setup", "changeover", "bottleneck", "work center"
                ],
                "metrics": [
                    "units produced", "production rate", "efficiency %", "oee",
                    "downtime hours", "cycle time", "setup time", "yield rate",
                    "capacity utilization", "throughput", "work in progress"
                ],
                "processes": [
                    "assembly", "fabrication", "packaging", "testing",
                    "inspection", "material handling", "scheduling"
                ]
            },
            
            BusinessDomain.QUALITY: {
                "keywords": [
                    "quality", "defect", "reject", "scrap", "rework", "inspection",
                    "testing", "specification", "tolerance", "control", "assurance",
                    "compliance", "standard", "certification", "audit", "variance"
                ],
                "metrics": [
                    "defect rate", "first pass yield", "scrap rate", "rework cost",
                    "customer complaints", "quality score", "inspection results",
                    "control chart", "cpk", "sigma level"
                ],
                "processes": [
                    "incoming inspection", "in-process testing", "final inspection",
                    "quality control", "quality assurance", "corrective action"
                ]
            },
            
            BusinessDomain.SUPPLY_CHAIN: {
                "keywords": [
                    "supply", "supplier", "vendor", "procurement", "inventory",
                    "stock", "warehouse", "logistics", "shipping", "receiving",
                    "lead time", "delivery", "fulfillment", "distribution"
                ],
                "metrics": [
                    "inventory turnover", "stock level", "lead time", "fill rate",
                    "on-time delivery", "supplier performance", "carrying cost",
                    "stockout rate", "service level", "demand forecast accuracy"
                ],
                "processes": [
                    "purchasing", "receiving", "warehousing", "distribution",
                    "vendor management", "demand planning", "logistics"
                ]
            },
            
            BusinessDomain.COST: {
                "keywords": [
                    "cost", "expense", "budget", "spending", "price", "margin",
                    "profit", "variance", "overhead", "labor cost", "material cost",
                    "savings", "reduction", "optimization", "efficiency"
                ],
                "metrics": [
                    "cost per unit", "labor cost", "material cost", "overhead rate",
                    "gross margin", "cost variance", "budget variance",
                    "cost savings", "cost reduction", "roi", "payback period"
                ],
                "processes": [
                    "cost accounting", "budgeting", "cost analysis", "variance analysis",
                    "pricing", "profitability analysis", "cost control"
                ]
            },
            
            BusinessDomain.ASSETS: {
                "keywords": [
                    "equipment", "asset", "maintenance", "repair", "replacement",
                    "depreciation", "utilization", "availability", "reliability",
                    "pm", "preventive", "predictive", "breakdown", "mttr", "mtbf"
                ],
                "metrics": [
                    "equipment uptime", "mtbf", "mttr", "availability %",
                    "maintenance cost", "asset utilization", "oee",
                    "repair frequency", "pm compliance", "asset value"
                ],
                "processes": [
                    "preventive maintenance", "predictive maintenance", "repairs",
                    "asset management", "equipment planning", "condition monitoring"
                ]
            },
            
            BusinessDomain.SAFETY: {
                "keywords": [
                    "safety", "accident", "incident", "injury", "hazard", "risk",
                    "compliance", "regulation", "osha", "environmental", "ehs",
                    "training", "ppe", "ergonomic", "near miss"
                ],
                "metrics": [
                    "accident rate", "injury frequency", "lost time incidents",
                    "safety score", "near miss reports", "training compliance",
                    "ppe usage", "safety audit results", "workers comp cost"
                ],
                "processes": [
                    "safety training", "hazard identification", "risk assessment",
                    "incident investigation", "safety audits", "compliance monitoring"
                ]
            },
            
            BusinessDomain.CUSTOMER: {
                "keywords": [
                    "customer", "client", "satisfaction", "complaint", "service",
                    "delivery", "quality", "on-time", "relationship", "feedback",
                    "order", "fulfillment", "support", "experience"
                ],
                "metrics": [
                    "customer satisfaction", "on-time delivery", "order fill rate",
                    "complaint rate", "customer retention", "service level",
                    "response time", "resolution time", "customer score"
                ],
                "processes": [
                    "order management", "customer service", "complaint handling",
                    "delivery", "customer relationship management", "support"
                ]
            },
            
            BusinessDomain.PLANNING: {
                "keywords": [
                    "planning", "forecast", "schedule", "capacity", "demand",
                    "production plan", "master schedule", "mrp", "erp",
                    "resource", "allocation", "optimization", "scenario"
                ],
                "metrics": [
                    "forecast accuracy", "schedule adherence", "capacity utilization",
                    "plan efficiency", "resource utilization", "demand variance",
                    "planning cycle time", "inventory turns", "service level"
                ],
                "processes": [
                    "demand planning", "production planning", "capacity planning",
                    "resource allocation", "master scheduling", "mrp"
                ]
            },
            
            BusinessDomain.HUMAN_RESOURCES: {
                "keywords": [
                    "employee", "staff", "worker", "labor", "personnel", "hr",
                    "training", "skill", "performance", "productivity", "turnover",
                    "overtime", "attendance", "engagement", "safety"
                ],
                "metrics": [
                    "employee turnover", "productivity per worker", "overtime hours",
                    "training hours", "attendance rate", "engagement score",
                    "performance rating", "labor efficiency", "cost per employee"
                ],
                "processes": [
                    "recruitment", "training", "performance management",
                    "workforce planning", "employee development", "compensation"
                ]
            },
            
            BusinessDomain.SALES: {
                "keywords": [
                    "sales", "revenue", "order", "customer", "product", "market",
                    "quota", "pipeline", "conversion", "win rate", "territory",
                    "account", "opportunity", "forecast", "pricing"
                ],
                "metrics": [
                    "sales revenue", "order volume", "conversion rate", "win rate",
                    "quota attainment", "pipeline value", "average order value",
                    "customer acquisition cost", "sales cycle time"
                ],
                "processes": [
                    "lead generation", "opportunity management", "sales forecasting",
                    "account management", "pricing", "sales reporting"
                ]
            },
            
            BusinessDomain.FINANCE: {
                "keywords": [
                    "financial", "revenue", "profit", "cash flow", "budget",
                    "expense", "cost", "roi", "margin", "ebitda", "accounting",
                    "payable", "receivable", "balance sheet", "income"
                ],
                "metrics": [
                    "revenue", "profit margin", "cash flow", "roi", "ebitda",
                    "budget variance", "accounts receivable", "accounts payable",
                    "working capital", "debt ratio", "financial ratios"
                ],
                "processes": [
                    "financial planning", "budgeting", "accounting", "reporting",
                    "cash management", "financial analysis", "audit"
                ]
            },
            
            BusinessDomain.MARKETING: {
                "keywords": [
                    "marketing", "campaign", "promotion", "brand", "customer",
                    "market share", "lead", "conversion", "awareness", "engagement",
                    "channel", "segment", "target", "advertising"
                ],
                "metrics": [
                    "market share", "campaign roi", "lead conversion", "brand awareness",
                    "customer acquisition cost", "engagement rate", "reach",
                    "click-through rate", "cost per lead", "marketing spend"
                ],
                "processes": [
                    "campaign management", "lead generation", "market research",
                    "brand management", "digital marketing", "customer segmentation"
                ]
            },
            
            BusinessDomain.OPERATIONS: {
                "keywords": [
                    "operations", "process", "efficiency", "optimization", "workflow",
                    "continuous improvement", "lean", "six sigma", "kaizen",
                    "operational", "procedure", "standard", "best practice"
                ],
                "metrics": [
                    "operational efficiency", "process cycle time", "error rate",
                    "process capability", "improvement savings", "waste reduction",
                    "standard adherence", "process performance"
                ],
                "processes": [
                    "process improvement", "workflow optimization", "standard work",
                    "operational planning", "performance monitoring", "best practices"
                ]
            },
            
            BusinessDomain.STRATEGIC: {
                "keywords": [
                    "strategic", "strategy", "vision", "goal", "objective", "initiative",
                    "performance", "kpi", "scorecard", "dashboard", "executive",
                    "leadership", "transformation", "competitive", "market"
                ],
                "metrics": [
                    "kpi performance", "strategic goals", "initiative progress",
                    "competitive position", "market position", "transformation metrics",
                    "executive dashboard", "balanced scorecard", "strategic roi"
                ],
                "processes": [
                    "strategic planning", "performance management", "initiative tracking",
                    "competitive analysis", "market analysis", "executive reporting"
                ]
            }
        }
    
    def _load_analysis_patterns(self) -> Dict[AnalysisType, List[str]]:
        """Load analysis type classification patterns."""
        return {
            AnalysisType.DESCRIPTIVE: [
                "what", "how much", "how many", "show", "display", "list",
                "current", "status", "report", "summary", "overview",
                "trend", "historical", "dashboard", "metrics"
            ],
            AnalysisType.DIAGNOSTIC: [
                "why", "cause", "reason", "root cause", "analysis", "investigate",
                "problem", "issue", "decline", "increase", "variance",
                "compare", "correlation", "impact", "factor"
            ],
            AnalysisType.PREDICTIVE: [
                "predict", "forecast", "future", "will", "expect", "trend",
                "projection", "model", "estimate", "anticipate",
                "likely", "probability", "scenario", "outlook"
            ],
            AnalysisType.PRESCRIPTIVE: [
                "should", "recommend", "optimize", "improve", "action",
                "solution", "strategy", "plan", "best", "option",
                "decision", "choose", "suggest", "advice"
            ]
        }
    
    def _load_urgency_patterns(self) -> Dict[str, List[str]]:
        """Load urgency level classification patterns."""
        return {
            "critical": [
                "urgent", "critical", "emergency", "asap", "immediately",
                "crisis", "down", "stopped", "failure", "major"
            ],
            "high": [
                "priority", "important", "soon", "quickly", "fast",
                "high", "significant", "serious", "escalate"
            ],
            "low": [
                "when possible", "eventually", "low priority", "nice to have",
                "future", "planning", "research", "explore"
            ]
        }
    
    def classify_query_intent(self, query: str):
        """
        Pre-classify query intent before business domain analysis.
        Returns either NonBusinessResponse or BusinessIntent.
        
        Args:
            query: Natural language query
            
        Returns:
            NonBusinessResponse for greetings/help/casual, BusinessIntent for business queries
        """
        # Use intent classifier for pre-classification
        intent_classifier = get_intent_classifier()
        intent_classification = intent_classifier.classify_intent(query)
        
        # Handle non-business intents
        if intent_classification.intent != QueryIntent.BUSINESS:
            response_template = get_response_template()
            response_text = response_template.generate_response(
                intent_classification.intent, 
                query
            )
            
            self.logger.info(
                f"Non-business intent detected: {intent_classification.intent.value} "
                f"(confidence: {intent_classification.confidence:.3f})"
            )
            
            return NonBusinessResponse(
                intent_type=intent_classification.intent.value,
                response_text=response_text,
                confidence=intent_classification.confidence
            )
        
        # Business intent - proceed with domain classification
        return self.classify_business_intent(query)
    
    @performance_monitor("domain_classification")
    def classify_business_intent(self, query: str) -> BusinessIntent:
        """
        Classify business intent from natural language query.
        
        Args:
            query: Natural language business query
            
        Returns:
            BusinessIntent with domain classification and confidence
        """
        query_lower = query.lower()
        
        # Score each domain
        domain_scores = {}
        for domain in BusinessDomain:
            score = self._calculate_domain_score(query_lower, domain)
            if score > 0:
                domain_scores[domain] = score
        
        if not domain_scores:
            # Default to operations if no clear match
            primary_domain = BusinessDomain.OPERATIONS
            confidence = 0.3
            secondary_domains = []
        else:
            # Sort by score
            sorted_domains = sorted(domain_scores.items(), key=lambda x: x[1], reverse=True)
            primary_domain = sorted_domains[0][0]
            confidence = min(sorted_domains[0][1], 1.0)
            
            # Secondary domains with significant scores
            secondary_domains = [
                domain for domain, score in sorted_domains[1:3]
                if score >= 0.3 and score >= confidence * 0.5
            ]
        
        # Classify analysis type
        analysis_type = self._classify_analysis_type(query_lower)
        
        # Extract key indicators and metrics
        key_indicators = self._extract_key_indicators(query_lower, primary_domain)
        business_metrics = self._extract_business_metrics(query_lower, primary_domain)
        
        # Extract time context
        time_context = self._extract_time_context(query_lower)
        
        # Determine urgency
        urgency_level = self._determine_urgency(query_lower)
        
        self.logger.info(
            f"Business intent classified: {primary_domain.value} "
            f"({analysis_type.value}) confidence: {confidence:.3f}"
        )
        
        return BusinessIntent(
            primary_domain=primary_domain,
            secondary_domains=secondary_domains,
            analysis_type=analysis_type,
            confidence=confidence,
            key_indicators=key_indicators,
            business_metrics=business_metrics,
            time_context=time_context,
            urgency_level=urgency_level
        )
    
    def _calculate_domain_score(self, query: str, domain: BusinessDomain) -> float:
        """Calculate domain relevance score for query."""
        patterns = self._domain_patterns[domain]
        total_score = 0.0
        
        # Keywords scoring (40% weight)
        keyword_score = 0.0
        for keyword in patterns["keywords"]:
            if keyword in query:
                # Exact match gets full points
                keyword_score += 1.0
            elif any(word in query for word in keyword.split()):
                # Partial match gets reduced points
                keyword_score += 0.5
        
        # Normalize to prevent dilution with large keyword lists - boost scoring
        if keyword_score > 0:
            keyword_score = min(keyword_score * 5 / len(patterns["keywords"]), 1.0)
        total_score += keyword_score * 0.4
        
        # Metrics scoring (35% weight)
        metrics_score = 0.0
        for metric in patterns["metrics"]:
            if metric in query:
                metrics_score += 1.0
            elif any(word in query for word in metric.split()):
                metrics_score += 0.6
        
        # Normalize metrics scoring similarly
        if metrics_score > 0:
            metrics_score = min(metrics_score * 2 / len(patterns["metrics"]), 1.0)
        total_score += metrics_score * 0.35
        
        # Processes scoring (25% weight)
        process_score = 0.0
        for process in patterns["processes"]:
            if process in query:
                process_score += 1.0
            elif any(word in query for word in process.split()):
                process_score += 0.7
        
        # Normalize process scoring similarly
        if process_score > 0:
            process_score = min(process_score * 2 / len(patterns["processes"]), 1.0)
        total_score += process_score * 0.25
        
        return total_score
    
    def _classify_analysis_type(self, query: str) -> AnalysisType:
        """Classify the type of analysis requested."""
        type_scores = {}
        
        for analysis_type, patterns in self._analysis_patterns.items():
            score = sum(1 for pattern in patterns if pattern in query)
            if score > 0:
                type_scores[analysis_type] = score
        
        if not type_scores:
            return AnalysisType.DESCRIPTIVE
        
        return max(type_scores.items(), key=lambda x: x[1])[0]
    
    def _extract_key_indicators(self, query: str, domain: BusinessDomain) -> List[str]:
        """Extract key business indicators from query."""
        patterns = self._domain_patterns[domain]
        indicators = []
        
        # Find keywords and metrics mentioned in query
        all_terms = patterns["keywords"] + patterns["metrics"] + patterns["processes"]
        
        for term in all_terms:
            if term in query:
                indicators.append(term)
            elif any(word in query for word in term.split()):
                # Partial matches for compound terms
                matching_words = [word for word in term.split() if word in query]
                if len(matching_words) >= len(term.split()) // 2:
                    indicators.append(term)
        
        return indicators[:10]  # Limit to top 10
    
    def _extract_business_metrics(self, query: str, domain: BusinessDomain) -> List[str]:
        """Extract specific business metrics from query."""
        patterns = self._domain_patterns[domain]
        metrics = []
        
        for metric in patterns["metrics"]:
            if metric in query:
                metrics.append(metric)
        
        return metrics[:5]  # Limit to top 5
    
    def _extract_time_context(self, query: str) -> Optional[str]:
        """Extract time context from query."""
        time_patterns = {
            "today": r"\btoday\b|\bnow\b|\bcurrent\b",
            "yesterday": r"\byesterday\b|\blast day\b",
            "this_week": r"\bthis week\b|\bweekly\b",
            "last_week": r"\blast week\b|\bprevious week\b",
            "this_month": r"\bthis month\b|\bmonthly\b|\bmtd\b",
            "last_month": r"\blast month\b|\bprevious month\b",
            "this_quarter": r"\bthis quarter\b|\bq[1-4]\b|\bqtd\b",
            "last_quarter": r"\blast quarter\b|\bprevious quarter\b",
            "this_year": r"\bthis year\b|\byearly\b|\bytd\b",
            "last_year": r"\blast year\b|\bprevious year\b"
        }
        
        for context, pattern in time_patterns.items():
            if re.search(pattern, query, re.IGNORECASE):
                return context
        
        return None
    
    def _determine_urgency(self, query: str) -> str:
        """Determine urgency level from query language."""
        for urgency, patterns in self._urgency_patterns.items():
            if any(pattern in query for pattern in patterns):
                return urgency
        
        return "normal"
    
    def get_domain_context(self, domain: BusinessDomain) -> Dict[str, List[str]]:
        """Get detailed context for a specific business domain."""
        return self._domain_patterns.get(domain, {})
    
    def validate_business_intent(self, intent: BusinessIntent) -> bool:
        """
        Validate business intent classification meets confidence thresholds.
        
        Args:
            intent: Business intent to validate
            
        Returns:
            True if intent meets confidence requirements
        """
        meets_threshold = intent.confidence >= settings.domain_confidence_threshold
        
        if not meets_threshold:
            self.logger.warning(
                f"Business intent confidence {intent.confidence:.3f} "
                f"below threshold {settings.domain_confidence_threshold}"
            )
        
        return meets_threshold


# Standalone execution for testing
if __name__ == "__main__":
    expert = DomainExpert()
    
    # Test queries
    test_queries = [
        "Why did Line 2 efficiency drop 15% last week?",
        "Show me current inventory levels for raw materials",
        "Analyze Q4 revenue performance vs forecast",
        "What caused the increase in customer complaints?",
        "Optimize production schedule for next month"
    ]
    
    print("Domain Expert Classification Test")
    print("=" * 50)
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        intent = expert.classify_business_intent(query)
        print(f"Domain: {intent.primary_domain.value}")
        print(f"Analysis: {intent.analysis_type.value}")
        print(f"Confidence: {intent.confidence:.3f}")
        print(f"Indicators: {', '.join(intent.key_indicators[:3])}")
        if intent.time_context:
            print(f"Time Context: {intent.time_context}")
        print(f"Urgency: {intent.urgency_level}")