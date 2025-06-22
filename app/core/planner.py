"""
Task Planner - Strategic Investigation Planning

Handles high-level planning and strategy for autonomous investigations,
similar to how Claude Code plans its development approach.
"""

from typing import Dict, List, Optional, Any
from datetime import datetime
from enum import Enum

from ..utils.logging import logger


class InvestigationStrategy(Enum):
    """Different investigation approaches."""
    EXPLORATORY = "exploratory"  # Broad exploration, good for open questions
    TARGETED = "targeted"        # Focused on specific metrics/questions
    COMPARATIVE = "comparative"   # Comparing different time periods/segments
    DIAGNOSTIC = "diagnostic"     # Finding root causes of known issues
    PREDICTIVE = "predictive"     # Forecasting and trend analysis


class PlanningComplexity(Enum):
    """Investigation complexity levels."""
    SIMPLE = "simple"        # Single table, basic aggregations
    MODERATE = "moderate"    # Multiple tables, some joins
    COMPLEX = "complex"      # Complex analytics, multiple data sources
    ADVANCED = "advanced"    # Advanced statistics, ML-like patterns


class TaskPlanner:
    """
    Strategic planner for autonomous SQL investigations.
    
    Creates investigation plans similar to how Claude Code
    plans its development approach based on requirements.
    """
    
    def __init__(self):
        self.strategy_templates = {
            InvestigationStrategy.EXPLORATORY: self._create_exploratory_plan,
            InvestigationStrategy.TARGETED: self._create_targeted_plan,
            InvestigationStrategy.COMPARATIVE: self._create_comparative_plan,
            InvestigationStrategy.DIAGNOSTIC: self._create_diagnostic_plan,
            InvestigationStrategy.PREDICTIVE: self._create_predictive_plan,
        }
        
        logger.info("Task Planner initialized with strategy templates")
    
    async def create_investigation_plan(
        self,
        query: str,
        user_context: Optional[Dict] = None,
        session_context: Optional[Dict] = None,
        faq_hint: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Create a comprehensive investigation plan.
        
        Args:
            query: User's natural language query
            user_context: Historical patterns and preferences
            session_context: Current session information
            faq_hint: Hint from FAQ pattern matching
            
        Returns:
            Detailed investigation plan with strategy and steps
        """
        logger.info(f"Creating investigation plan for query: {query[:100]}...")
        
        try:
            # Analyze query characteristics
            query_analysis = await self._analyze_query_intent(query, user_context)
            
            # Determine investigation strategy
            strategy = await self._select_investigation_strategy(
                query_analysis, user_context, faq_hint
            )
            
            # Assess complexity level
            complexity = await self._assess_complexity(query_analysis, strategy)
            
            # Create strategy-specific plan
            plan_creator = self.strategy_templates[strategy]
            detailed_plan = await plan_creator(
                query_analysis, complexity, user_context, session_context
            )
            
            # Add meta information
            plan = {
                "plan_id": f"plan_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
                "created_at": datetime.utcnow().isoformat(),
                "query": query,
                "strategy": strategy.value,
                "complexity": complexity.value,
                "estimated_steps": detailed_plan["estimated_steps"],
                "estimated_duration": detailed_plan["estimated_duration"],
                "confidence": detailed_plan["confidence"],
                "approach": detailed_plan["approach"],
                "key_objectives": detailed_plan["objectives"],
                "step_sequence": detailed_plan["steps"],
                "fallback_strategies": detailed_plan.get("fallbacks", []),
                "success_criteria": detailed_plan["success_criteria"],
                "query_analysis": query_analysis
            }
            
            logger.info(f"Investigation plan created: {strategy.value} strategy, "
                       f"{complexity.value} complexity, {plan['estimated_steps']} steps")
            
            return plan
            
        except Exception as e:
            logger.error(f"Failed to create investigation plan: {e}")
            # Return basic exploratory plan as fallback
            return await self._create_fallback_plan(query)
    
    async def _analyze_query_intent(
        self, 
        query: str, 
        user_context: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """Analyze the user's query to understand intent and scope."""
        
        query_lower = query.lower()
        
        # Intent classification
        intent_keywords = {
            "trends": ["trend", "over time", "growth", "change", "increase", "decrease", "pattern"],
            "comparison": ["compare", "vs", "versus", "difference", "better", "worse", "than"],
            "aggregation": ["total", "sum", "count", "average", "mean", "max", "min", "top", "bottom"],
            "filtering": ["where", "filter", "specific", "only", "exclude", "include"],
            "causation": ["why", "reason", "cause", "because", "due to", "impact", "effect"],
            "forecasting": ["predict", "forecast", "future", "next", "will", "expect", "project"],
            "segmentation": ["segment", "group", "category", "type", "kind", "by region", "by age"],
            "anomaly": ["unusual", "strange", "anomaly", "outlier", "unexpected", "problem"]
        }
        
        detected_intents = []
        for intent, keywords in intent_keywords.items():
            if any(keyword in query_lower for keyword in keywords):
                detected_intents.append(intent)
        
        # Entity extraction (simple keyword-based)
        business_entities = {
            "revenue": ["revenue", "sales", "income", "earnings"],
            "customers": ["customer", "client", "user", "buyer"],
            "products": ["product", "item", "merchandise", "goods"],
            "orders": ["order", "purchase", "transaction", "sale"],
            "time": ["month", "quarter", "year", "week", "day", "period"],
            "geography": ["region", "country", "state", "city", "location"]
        }
        
        detected_entities = []
        for entity, keywords in business_entities.items():
            if any(keyword in query_lower for keyword in keywords):
                detected_entities.append(entity)
        
        # Complexity indicators
        complexity_indicators = {
            "joins_needed": any(word in query_lower for word in ["customer", "order", "product"]),
            "time_analysis": any(word in query_lower for word in ["trend", "over time", "growth"]),
            "aggregations": any(word in query_lower for word in ["total", "average", "count", "sum"]),
            "filtering": any(word in query_lower for word in ["where", "specific", "only"]),
            "multiple_metrics": len(detected_entities) > 2
        }
        
        return {
            "original_query": query,
            "detected_intents": detected_intents,
            "primary_intent": detected_intents[0] if detected_intents else "aggregation",
            "detected_entities": detected_entities,
            "complexity_indicators": complexity_indicators,
            "question_type": self._classify_question_type(query_lower),
            "scope": self._assess_query_scope(detected_entities, complexity_indicators),
            "user_preferences": user_context.get("preferences", {}) if user_context else {}
        }
    
    def _classify_question_type(self, query: str) -> str:
        """Classify the type of business question."""
        if any(word in query for word in ["what", "how much", "how many"]):
            return "descriptive"
        elif any(word in query for word in ["why", "how", "reason"]):
            return "explanatory"
        elif any(word in query for word in ["will", "predict", "forecast"]):
            return "predictive"
        elif any(word in query for word in ["should", "recommend", "suggest"]):
            return "prescriptive"
        else:
            return "descriptive"
    
    def _assess_query_scope(self, entities: List[str], complexity_indicators: Dict) -> str:
        """Assess the scope of the query."""
        if len(entities) <= 1 and not any(complexity_indicators.values()):
            return "narrow"
        elif len(entities) <= 2 and sum(complexity_indicators.values()) <= 2:
            return "moderate"
        else:
            return "broad"
    
    async def _select_investigation_strategy(
        self,
        query_analysis: Dict,
        user_context: Optional[Dict],
        faq_hint: Optional[Dict]
    ) -> InvestigationStrategy:
        """Select the most appropriate investigation strategy."""
        
        primary_intent = query_analysis["primary_intent"]
        question_type = query_analysis["question_type"]
        scope = query_analysis["scope"]
        
        # Strategy selection logic
        if faq_hint and faq_hint.get("confidence", 0) > 0.7:
            # If we have a strong FAQ hint, use targeted approach
            return InvestigationStrategy.TARGETED
        
        elif primary_intent == "comparison":
            return InvestigationStrategy.COMPARATIVE
        
        elif primary_intent == "causation" or question_type == "explanatory":
            return InvestigationStrategy.DIAGNOSTIC
        
        elif primary_intent == "forecasting" or question_type == "predictive":
            return InvestigationStrategy.PREDICTIVE
        
        elif scope == "narrow" and primary_intent in ["aggregation", "filtering"]:
            return InvestigationStrategy.TARGETED
        
        else:
            return InvestigationStrategy.EXPLORATORY
    
    async def _assess_complexity(
        self,
        query_analysis: Dict,
        strategy: InvestigationStrategy
    ) -> PlanningComplexity:
        """Assess the complexity level of the investigation."""
        
        complexity_score = 0
        indicators = query_analysis["complexity_indicators"]
        
        # Add points for complexity factors
        if indicators["joins_needed"]:
            complexity_score += 2
        if indicators["time_analysis"]:
            complexity_score += 2
        if indicators["multiple_metrics"]:
            complexity_score += 1
        if indicators["aggregations"]:
            complexity_score += 1
        if strategy in [InvestigationStrategy.DIAGNOSTIC, InvestigationStrategy.PREDICTIVE]:
            complexity_score += 2
        
        # Map score to complexity level
        if complexity_score <= 2:
            return PlanningComplexity.SIMPLE
        elif complexity_score <= 4:
            return PlanningComplexity.MODERATE
        elif complexity_score <= 6:
            return PlanningComplexity.COMPLEX
        else:
            return PlanningComplexity.ADVANCED
    
    # Strategy-specific plan creators
    
    async def _create_exploratory_plan(
        self,
        query_analysis: Dict,
        complexity: PlanningComplexity,
        user_context: Optional[Dict],
        session_context: Optional[Dict]
    ) -> Dict[str, Any]:
        """Create an exploratory investigation plan."""
        
        steps = [
            {
                "step": 1,
                "type": "schema_analysis",
                "description": "Analyze database schema for relevant tables",
                "estimated_time": 30,
                "config": {"focus": "data_discovery"}
            },
            {
                "step": 2,
                "type": "data_exploration",
                "description": "Explore sample data to understand patterns",
                "estimated_time": 60,
                "config": {"sample_size": 1000, "focus": "pattern_discovery"}
            },
            {
                "step": 3,
                "type": "pattern_discovery",
                "description": "Look for interesting patterns and trends",
                "estimated_time": 90,
                "config": {"focus_area": "trends_and_patterns"}
            }
        ]
        
        if complexity in [PlanningComplexity.COMPLEX, PlanningComplexity.ADVANCED]:
            steps.extend([
                {
                    "step": 4,
                    "type": "hypothesis_testing",
                    "description": "Test discovered patterns for significance",
                    "estimated_time": 120,
                    "config": {"confidence_threshold": 0.8}
                },
                {
                    "step": 5,
                    "type": "validation",
                    "description": "Validate findings with different approaches",
                    "estimated_time": 90,
                    "config": {"validation_methods": ["cross_validation", "alternative_metrics"]}
                }
            ])
        
        return {
            "approach": "Broad exploration to discover insights",
            "objectives": [
                "Understand available data structure",
                "Discover interesting patterns",
                "Generate actionable insights"
            ],
            "steps": steps,
            "estimated_steps": len(steps),
            "estimated_duration": sum(step["estimated_time"] for step in steps),
            "confidence": 0.8,
            "success_criteria": [
                "At least 3 significant patterns discovered",
                "Actionable business insights generated",
                "Data quality assessed"
            ]
        }
    
    async def _create_targeted_plan(
        self,
        query_analysis: Dict,
        complexity: PlanningComplexity,
        user_context: Optional[Dict],
        session_context: Optional[Dict]
    ) -> Dict[str, Any]:
        """Create a targeted investigation plan for specific metrics."""
        
        steps = [
            {
                "step": 1,
                "type": "schema_analysis",
                "description": "Identify tables relevant to specific query",
                "estimated_time": 20,
                "config": {"focus": "targeted_schema", "entities": query_analysis["detected_entities"]}
            },
            {
                "step": 2,
                "type": "data_exploration",
                "description": "Focus on specific metrics and dimensions",
                "estimated_time": 40,
                "config": {"targeted_exploration": True, "metrics": query_analysis["detected_entities"]}
            },
            {
                "step": 3,
                "type": "pattern_discovery",
                "description": "Analyze specific patterns in target metrics",
                "estimated_time": 60,
                "config": {"focus_area": query_analysis["primary_intent"]}
            }
        ]
        
        return {
            "approach": "Focused analysis of specific metrics",
            "objectives": [
                f"Analyze {', '.join(query_analysis['detected_entities'])}",
                "Provide precise answers to specific questions",
                "Optimize for speed and accuracy"
            ],
            "steps": steps,
            "estimated_steps": len(steps),
            "estimated_duration": sum(step["estimated_time"] for step in steps),
            "confidence": 0.9,
            "success_criteria": [
                "Specific metrics calculated accurately",
                "Clear answer to user's question",
                "Supporting data provided"
            ]
        }
    
    async def _create_comparative_plan(
        self,
        query_analysis: Dict,
        complexity: PlanningComplexity,
        user_context: Optional[Dict],
        session_context: Optional[Dict]
    ) -> Dict[str, Any]:
        """Create a comparative analysis plan."""
        
        steps = [
            {
                "step": 1,
                "type": "schema_analysis",
                "description": "Identify comparable dimensions and metrics",
                "estimated_time": 30,
                "config": {"focus": "comparative_schema"}
            },
            {
                "step": 2,
                "type": "data_exploration",
                "description": "Explore comparison groups",
                "estimated_time": 50,
                "config": {"comparison_focus": True}
            },
            {
                "step": 3,
                "type": "pattern_discovery",
                "description": "Identify differences and similarities",
                "estimated_time": 70,
                "config": {"focus_area": "comparative_patterns"}
            },
            {
                "step": 4,
                "type": "validation",
                "description": "Validate statistical significance of differences",
                "estimated_time": 60,
                "config": {"statistical_tests": True}
            }
        ]
        
        return {
            "approach": "Compare different segments, periods, or groups",
            "objectives": [
                "Identify key differences between comparison groups",
                "Quantify magnitude of differences",
                "Determine statistical significance"
            ],
            "steps": steps,
            "estimated_steps": len(steps),
            "estimated_duration": sum(step["estimated_time"] for step in steps),
            "confidence": 0.85,
            "success_criteria": [
                "Clear comparison results",
                "Statistical significance tested",
                "Business implications identified"
            ]
        }
    
    async def _create_diagnostic_plan(
        self,
        query_analysis: Dict,
        complexity: PlanningComplexity,
        user_context: Optional[Dict],
        session_context: Optional[Dict]
    ) -> Dict[str, Any]:
        """Create a diagnostic plan to find root causes."""
        
        steps = [
            {
                "step": 1,
                "type": "schema_analysis",
                "description": "Map relationships between potential cause and effect",
                "estimated_time": 40,
                "config": {"focus": "causal_relationships"}
            },
            {
                "step": 2,
                "type": "data_exploration",
                "description": "Examine potential causal factors",
                "estimated_time": 80,
                "config": {"causal_exploration": True}
            },
            {
                "step": 3,
                "type": "hypothesis_testing",
                "description": "Test causal hypotheses",
                "estimated_time": 100,
                "config": {"causal_testing": True}
            },
            {
                "step": 4,
                "type": "validation",
                "description": "Validate causal relationships",
                "estimated_time": 80,
                "config": {"causal_validation": True}
            }
        ]
        
        return {
            "approach": "Systematic root cause analysis",
            "objectives": [
                "Identify potential root causes",
                "Test causal relationships",
                "Prioritize causes by impact"
            ],
            "steps": steps,
            "estimated_steps": len(steps),
            "estimated_duration": sum(step["estimated_time"] for step in steps),
            "confidence": 0.75,
            "success_criteria": [
                "Root causes identified",
                "Causal relationships validated",
                "Impact quantified"
            ]
        }
    
    async def _create_predictive_plan(
        self,
        query_analysis: Dict,
        complexity: PlanningComplexity,
        user_context: Optional[Dict],
        session_context: Optional[Dict]
    ) -> Dict[str, Any]:
        """Create a predictive analysis plan."""
        
        steps = [
            {
                "step": 1,
                "type": "schema_analysis",
                "description": "Identify historical patterns and trends",
                "estimated_time": 35,
                "config": {"focus": "temporal_patterns"}
            },
            {
                "step": 2,
                "type": "data_exploration",
                "description": "Analyze historical trends",
                "estimated_time": 70,
                "config": {"trend_analysis": True}
            },
            {
                "step": 3,
                "type": "pattern_discovery",
                "description": "Discover predictive patterns",
                "estimated_time": 90,
                "config": {"focus_area": "predictive_patterns"}
            },
            {
                "step": 4,
                "type": "validation",
                "description": "Validate predictive accuracy",
                "estimated_time": 70,
                "config": {"predictive_validation": True}
            }
        ]
        
        return {
            "approach": "Historical pattern analysis for forecasting",
            "objectives": [
                "Identify predictive patterns",
                "Generate forecasts",
                "Assess prediction confidence"
            ],
            "steps": steps,
            "estimated_steps": len(steps),
            "estimated_duration": sum(step["estimated_time"] for step in steps),
            "confidence": 0.7,
            "success_criteria": [
                "Predictive patterns identified",
                "Forecasts generated",
                "Confidence intervals provided"
            ]
        }
    
    async def _create_fallback_plan(self, query: str) -> Dict[str, Any]:
        """Create a basic fallback plan when planning fails."""
        
        return {
            "plan_id": f"fallback_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
            "created_at": datetime.utcnow().isoformat(),
            "query": query,
            "strategy": "exploratory",
            "complexity": "simple",
            "approach": "Basic exploratory analysis",
            "objectives": ["Understand the data", "Find basic insights"],
            "steps": [
                {
                    "step": 1,
                    "type": "schema_analysis",
                    "description": "Basic schema exploration",
                    "estimated_time": 30
                },
                {
                    "step": 2,
                    "type": "data_exploration", 
                    "description": "Simple data exploration",
                    "estimated_time": 60
                }
            ],
            "estimated_steps": 2,
            "estimated_duration": 90,
            "confidence": 0.6,
            "success_criteria": ["Basic data understanding achieved"]
        }