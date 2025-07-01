"""
Intelligence Module Runner - Phase 1 & 2 Orchestrator
Orchestrates query processing and strategy planning through intelligence analysis.
"""

from typing import Dict, Any
from dataclasses import dataclass

from .domain_expert import DomainExpert, BusinessIntent
from .complexity_analyzer import ComplexityAnalyzer, ComplexityScore
from .business_context import BusinessContextAnalyzer, ContextualStrategy
from .hypothesis_generator import HypothesisGenerator, HypothesisSet
from .intelligence_logging import setup_logger


@dataclass
class IntelligencePlanningResult:
    """Result of intelligence planning (Phase 1 & 2)."""
    business_intent: BusinessIntent
    complexity_score: ComplexityScore
    contextual_strategy: ContextualStrategy
    hypothesis_set: HypothesisSet


class IntelligenceModuleRunner:
    """
    Runner for Intelligence Module - orchestrates Phase 1 & 2.
    Combines query processing and strategy planning.
    """
    
    def __init__(self):
        self.logger = setup_logger("intelligence_runner")
        
        # Initialize components
        self.domain_expert = DomainExpert()
        self.complexity_analyzer = ComplexityAnalyzer()
        self.context_analyzer = BusinessContextAnalyzer()
        self.hypothesis_generator = HypothesisGenerator()
        
        self.logger.info("Intelligence Module Runner initialized")
    
    async def plan_investigation_strategy(
        self,
        business_question: str,
        user_context: Dict[str, Any],
        organization_context: Dict[str, Any]
    ) -> IntelligencePlanningResult:
        """
        Execute Phase 1 & 2: Query processing and strategy planning.
        
        Args:
            business_question: Natural language business question
            user_context: User information and preferences
            organization_context: Organizational context and constraints
            
        Returns:
            IntelligencePlanningResult with strategy plan
        """
        try:
            # Phase 1: Query Processing - Business Intent Classification
            self.logger.info("Phase 1: Analyzing business intent...")
            business_intent = self.domain_expert.classify_business_intent(business_question)
            
            # Phase 1: Complexity Analysis
            self.logger.info("Phase 1: Analyzing investigation complexity...")
            complexity_score = self.complexity_analyzer.analyze_complexity(
                business_intent, business_question
            )
            
            # Phase 2: Strategy Planning - Context Analysis
            self.logger.info("Phase 2: Planning contextual strategy...")
            
            # Create user and org profiles if needed
            user_id = user_context.get("user_id", "unknown")
            org_id = organization_context.get("organization_id", "unknown")
            
            contextual_strategy = self.context_analyzer.analyze_context(
                business_intent=business_intent,
                complexity_level=complexity_score.level,
                base_methodology=complexity_score.methodology,
                user_id=user_id,
                organization_id=org_id
            )
            
            # Phase 2: Hypothesis Generation (for investigative queries)
            self.logger.info("Phase 2: Generating investigation hypotheses...")
            hypothesis_set = self.hypothesis_generator.generate_hypotheses(
                business_intent=business_intent,
                contextual_strategy=contextual_strategy,
                complexity_level=complexity_score.level
            )
            
            self.logger.info(
                f"Intelligence planning completed: {complexity_score.level.value} "
                f"complexity, {contextual_strategy.adapted_methodology.value} methodology"
            )
            
            return IntelligencePlanningResult(
                business_intent=business_intent,
                complexity_score=complexity_score,
                contextual_strategy=contextual_strategy,
                hypothesis_set=hypothesis_set
            )
            
        except Exception as e:
            self.logger.error(f"Intelligence planning failed: {e}")
            raise