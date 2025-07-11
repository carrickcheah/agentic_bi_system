"""
Intelligence Module Runner - Phase 1 & 2 Orchestrator
Orchestrates query processing and strategy planning through intelligence analysis.
"""

from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass

from .query_intent_classifier import QueryIntentClassifier
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
        self.intent_classifier = QueryIntentClassifier()  # This was missing!
        self.domain_expert = DomainExpert()
        self.complexity_analyzer = ComplexityAnalyzer()
        self.context_analyzer = BusinessContextAnalyzer()
        self.hypothesis_generator = HypothesisGenerator()
        
        self.logger.info("Intelligence Module Runner initialized")
    
    async def analyze_query_parallel(
        self,
        business_question: str,
        qdrant_service=None
    ) -> Tuple[Any, Optional[List[Dict[str, Any]]], Any, Any]:
        """
        Run intent classification, Qdrant search, and complexity analysis in parallel.
        This method should be in intelligence module, not business analyst!
        
        Args:
            business_question: Natural language business question
            qdrant_service: Optional Qdrant service for similarity search
            
        Returns:
            Tuple of (intent_classification, qdrant_results, complexity_score, business_intent)
        """
        import asyncio
        
        self.logger.info("Starting parallel query analysis...")
        start_time = asyncio.get_event_loop().time()
        
        # Create parallel tasks
        tasks = []
        
        # 1. Extract Intent (sync function, needs asyncio.to_thread)
        intent_task = asyncio.create_task(
            asyncio.to_thread(
                self.intent_classifier.classify_intent, 
                business_question
            ),
            name="intent_classification"
        )
        tasks.append(intent_task)
        
        # 2. Search Qdrant (if service provided)
        qdrant_task = None
        if qdrant_service:
            qdrant_task = asyncio.create_task(
                qdrant_service.search_similar_queries(
                    business_question, 
                    limit=5, 
                    threshold=0.70
                ),
                name="qdrant_search"
            )
            tasks.append(qdrant_task)
        
        # 3. Business Intent Classification
        business_intent_task = asyncio.create_task(
            asyncio.to_thread(
                self.domain_expert.classify_business_intent,
                business_question
            ),
            name="business_intent"
        )
        
        # Get business intent first as complexity depends on it
        business_intent = await business_intent_task
        
        # 4. Analyze Complexity
        complexity_task = asyncio.create_task(
            asyncio.to_thread(
                self.complexity_analyzer.analyze_complexity,
                business_intent,
                business_question
            ),
            name="complexity_analysis"
        )
        tasks.append(complexity_task)
        
        # Wait for remaining tasks
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Extract results
        intent_classification = results[0] if not isinstance(results[0], Exception) else None
        qdrant_results = None
        complexity_score = None
        
        if qdrant_task:
            qdrant_idx = 1
            qdrant_results = results[qdrant_idx] if not isinstance(results[qdrant_idx], Exception) else None
            complexity_score = results[-1] if not isinstance(results[-1], Exception) else None
        else:
            complexity_score = results[-1] if not isinstance(results[-1], Exception) else None
        
        # Log performance
        elapsed_time = asyncio.get_event_loop().time() - start_time
        self.logger.info(f"Parallel query analysis completed in {elapsed_time:.3f}s")
        
        return intent_classification, qdrant_results, complexity_score, business_intent

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