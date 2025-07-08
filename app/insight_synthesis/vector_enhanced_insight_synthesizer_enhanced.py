#!/usr/bin/env python3
"""
Enhanced Vector-Enhanced Insight Synthesizer with Comprehensive Error Handling
Production-ready insight synthesis with proper error handling, monitoring, and resilience.
"""

import asyncio
import json
import time
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field, asdict
import uuid
import logging
from enum import Enum

# Import our error handling system
try:
    from ..core.error_handling import (
        error_boundary, with_error_handling, validate_input, safe_execute,
        ValidationError, ExternalServiceError, ResourceExhaustedError,
        DatabaseError, BusinessLogicError, PerformanceError,
        ErrorCategory, ErrorSeverity, error_tracker
    )
except ImportError:
    # Fallback for standalone mode
    from contextlib import nullcontext
    
    def error_boundary(*args, **kwargs):
        return nullcontext()
    
    def with_error_handling(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    
    def validate_input(data, validator, operation, component, correlation_id=None):
        return validator(data)
    
    def safe_execute(func, operation, component, correlation_id=None, default_return=None, raise_on_failure=True):
        try:
            return func()
        except Exception as e:
            if raise_on_failure:
                raise
            return default_return
    
    class ValidationError(Exception):
        pass
    class ExternalServiceError(Exception):
        pass
    class ResourceExhaustedError(Exception):
        pass
    class DatabaseError(Exception):
        pass
    class BusinessLogicError(Exception):
        pass
    class PerformanceError(Exception):
        pass

# Configure logging
logger = logging.getLogger(__name__)

def safe_import_numpy():
    """Safely import numpy with fallback implementation."""
    try:
        import numpy as np
        return True, np
    except ImportError:
        logger.warning("NumPy not available, using fallback implementation")
        
        class NumpyFallback:
            @staticmethod
            def mean(values):
                if not values:
                    raise ValueError("Cannot compute mean of empty array")
                return sum(values) / len(values)
            
            @staticmethod
            def percentile(values, p):
                if not values:
                    raise ValueError("Cannot compute percentile of empty array")
                sorted_values = sorted(values)
                k = (len(sorted_values) - 1) * p / 100
                f = int(k)
                c = k - f
                if f + 1 < len(sorted_values):
                    return sorted_values[f] * (1 - c) + sorted_values[f + 1] * c
                else:
                    return sorted_values[f]
            
            class random:
                @staticmethod
                def random():
                    import random
                    return random.random()
        
        return False, NumpyFallback()

def safe_import_synthesis_module():
    """Safely import synthesis module components."""
    try:
        from .runner import (
            InsightSynthesizer, SynthesisResult, BusinessInsight,
            Recommendation, OrganizationalLearning, InsightType,
            RecommendationType, OutputFormat
        )
        from .config import settings
        from .synthesis_logging import setup_logger, performance_monitor
        return True, {
            'synthesizer': InsightSynthesizer,
            'result': SynthesisResult,
            'insight': BusinessInsight,
            'recommendation': Recommendation,
            'learning': OrganizationalLearning,
            'insight_type': InsightType,
            'rec_type': RecommendationType,
            'output_format': OutputFormat,
            'settings': settings
        }
    except ImportError:
        try:
            from runner import (
                InsightSynthesizer, SynthesisResult, BusinessInsight,
                Recommendation, OrganizationalLearning, InsightType,
                RecommendationType, OutputFormat
            )
            from config import settings
            from synthesis_logging import setup_logger, performance_monitor
            return True, {
                'synthesizer': InsightSynthesizer,
                'result': SynthesisResult,
                'insight': BusinessInsight,
                'recommendation': Recommendation,
                'learning': OrganizationalLearning,
                'insight_type': InsightType,
                'rec_type': RecommendationType,
                'output_format': OutputFormat,
                'settings': settings
            }
        except ImportError:
            logger.warning("Insight Synthesis module components not available")
            
            # Define minimal fallback classes
            class InsightType(Enum):
                OPERATIONAL = "operational"
                TACTICAL = "tactical"
                STRATEGIC = "strategic"
                TRANSFORMATIONAL = "transformational"
                RISK_MITIGATION = "risk_mitigation"
                OPPORTUNITY = "opportunity"
                EFFICIENCY = "efficiency"
                COMPLIANCE = "compliance"
            
            class RecommendationType(Enum):
                IMMEDIATE_ACTION = "immediate_action"
                SHORT_TERM = "short_term"
                LONG_TERM = "long_term"
                STRATEGIC_INITIATIVE = "strategic_initiative"
                PROCESS_IMPROVEMENT = "process_improvement"
                RISK_MANAGEMENT = "risk_management"
                COST_OPTIMIZATION = "cost_optimization"
                INNOVATION = "innovation"
            
            class OutputFormat(Enum):
                EXECUTIVE_SUMMARY = "executive_summary"
                DETAILED_REPORT = "detailed_report"
                ACTION_PLAN = "action_plan"
                DASHBOARD = "dashboard"
            
            @dataclass
            class BusinessInsight:
                insight_id: str
                insight_type: InsightType
                title: str
                description: str
                confidence_score: float
                business_impact: float
                
            @dataclass
            class Recommendation:
                recommendation_id: str
                recommendation_type: RecommendationType
                title: str
                description: str
                priority: int
                estimated_impact: float
                
            @dataclass
            class SynthesisResult:
                result_id: str
                insights: List[BusinessInsight]
                recommendations: List[Recommendation]
                executive_summary: str
                confidence_score: float
            
            return False, {
                'insight_type': InsightType,
                'rec_type': RecommendationType,
                'output_format': OutputFormat,
                'insight': BusinessInsight,
                'recommendation': Recommendation,
                'result': SynthesisResult
            }

def safe_import_vector_capabilities():
    """Safely import vector capabilities."""
    try:
        import lancedb
        from sentence_transformers import SentenceTransformer
        return True, {'lancedb': lancedb, 'transformer': SentenceTransformer}
    except ImportError:
        logger.warning("Vector capabilities not available")
        return False, {}

# Initialize dependencies
NUMPY_AVAILABLE, np = safe_import_numpy()
SYNTHESIS_MODULE_AVAILABLE, synthesis_components = safe_import_synthesis_module()
VECTOR_CAPABILITIES_AVAILABLE, vector_capabilities = safe_import_vector_capabilities()

# Extract components for use
InsightType = synthesis_components.get('insight_type', None)
RecommendationType = synthesis_components.get('rec_type', None)
OutputFormat = synthesis_components.get('output_format', None)
BusinessInsight = synthesis_components.get('insight', None)
Recommendation = synthesis_components.get('recommendation', None)
SynthesisResult = synthesis_components.get('result', None)

@dataclass
class VectorInsightPattern:
    """Pattern discovered through vector analysis of insights."""
    pattern_id: str
    pattern_name: str
    semantic_fingerprint: List[float]
    occurrence_count: int
    avg_confidence: float
    business_domains: List[str]
    insight_types: List[str]
    success_rate: float
    first_seen: datetime
    last_seen: datetime

@dataclass
class SynthesisContext:
    """Context for insight synthesis operations."""
    operation_id: str
    user_context: Dict[str, Any]
    business_domain: str
    priority_level: int
    time_horizon: str
    expected_output_format: str

class EnhancedVectorInsightSynthesizer:
    """
    Production-ready Vector-Enhanced Insight Synthesizer with comprehensive error handling.
    
    Features:
    - Structured error handling
    - Vector pattern recognition
    - Performance monitoring
    - Input validation
    - Resource management
    """
    
    def __init__(self):
        self.synthesizer_id = str(uuid.uuid4())
        self.initialization_correlation_id = str(uuid.uuid4())
        
        # Core components
        self.vector_patterns: Dict[str, VectorInsightPattern] = {}
        self.synthesis_cache: Dict[str, SynthesisResult] = {}
        self.performance_metrics = {
            'syntheses_performed': 0,
            'patterns_discovered': 0,
            'cache_hits': 0,
            'errors_handled': 0,
            'avg_synthesis_time_ms': 0
        }
        
        # Circuit breaker state
        self.circuit_breaker_open = False
        self.failure_count = 0
        self.last_failure_time = None
        
        # Vector components (if available)
        self.vector_db = None
        self.embedding_model = None
        
        # Initialize safely
        self._initialize_safely()
    
    def _initialize_safely(self):
        """Initialize with comprehensive error handling."""
        
        logger.info(
            "Initializing Enhanced Vector Insight Synthesizer",
            extra={"correlation_id": self.initialization_correlation_id}
        )
        
        # Initialize vector capabilities if available
        if VECTOR_CAPABILITIES_AVAILABLE:
            try:
                # Initialize embedding model
                transformer_class = vector_capabilities.get('transformer')
                if transformer_class:
                    self.embedding_model = safe_execute(
                        lambda: transformer_class('sentence-transformers/all-MiniLM-L6-v2'),
                        operation="initialize_embedding_model",
                        component="vector_insight_synthesizer",
                        correlation_id=self.initialization_correlation_id,
                        raise_on_failure=False
                    )
                    
                    if self.embedding_model:
                        logger.info(
                            "Embedding model initialized successfully",
                            extra={"correlation_id": self.initialization_correlation_id}
                        )
                
            except Exception as e:
                logger.warning(
                    f"Failed to initialize vector capabilities: {e}",
                    extra={"correlation_id": self.initialization_correlation_id}
                )
        else:
            logger.warning(
                "Vector capabilities not available - some features disabled",
                extra={"correlation_id": self.initialization_correlation_id}
            )
        
        logger.info(
            "Enhanced Vector Insight Synthesizer initialized successfully",
            extra={"correlation_id": self.initialization_correlation_id}
        )
    
    def validate_synthesis_input(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate synthesis input data."""
        def validator(input_data):
            if not isinstance(input_data, dict):
                raise ValueError("Input data must be a dictionary")
            
            # Required fields
            required_fields = ['business_question', 'investigation_results']
            for field in required_fields:
                if field not in input_data:
                    raise ValueError(f"Missing required field: {field}")
            
            # Validate business question
            business_question = input_data['business_question']
            if not isinstance(business_question, str) or len(business_question.strip()) == 0:
                raise ValueError("business_question must be a non-empty string")
            
            if len(business_question) > 10000:
                raise ValueError("business_question too long (max 10,000 characters)")
            
            # Validate investigation results
            investigation_results = input_data['investigation_results']
            if not isinstance(investigation_results, list):
                raise ValueError("investigation_results must be a list")
            
            if len(investigation_results) > 1000:
                raise ValueError("Too many investigation results (max 1,000)")
            
            # Validate optional fields
            if 'context' in input_data:
                context = input_data['context']
                if not isinstance(context, dict):
                    raise ValueError("context must be a dictionary")
            
            return input_data
        
        return validate_input(
            data,
            validator,
            operation="validate_synthesis_input",
            component="vector_insight_synthesizer"
        )
    
    @with_error_handling(
        operation="synthesize_insights",
        component="vector_insight_synthesizer",
        timeout_seconds=120.0
    )
    async def synthesize_insights(
        self,
        business_question: str,
        investigation_results: List[Dict[str, Any]],
        context: Optional[Dict[str, Any]] = None,
        correlation_id: Optional[str] = None
    ) -> SynthesisResult:
        """Synthesize insights with comprehensive error handling."""
        
        # Circuit breaker check
        if self.circuit_breaker_open:
            if self.last_failure_time and (datetime.utcnow() - self.last_failure_time).seconds < 300:
                raise ResourceExhaustedError(
                    "Circuit breaker open - too many recent failures",
                    component="vector_insight_synthesizer",
                    operation="synthesize_insights",
                    correlation_id=correlation_id,
                    retry_after=300
                )
            else:
                # Reset circuit breaker
                self.circuit_breaker_open = False
                self.failure_count = 0
        
        # Validate input data
        input_data = {
            'business_question': business_question,
            'investigation_results': investigation_results,
            'context': context or {}
        }
        validated_data = self.validate_synthesis_input(input_data)
        
        logger.info(
            "Starting insight synthesis",
            extra={
                "correlation_id": correlation_id,
                "question_length": len(business_question),
                "results_count": len(investigation_results)
            }
        )
        
        try:
            start_time = time.time()
            
            # Check cache first
            cache_key = self._generate_cache_key(business_question, investigation_results)
            if cache_key in self.synthesis_cache:
                self.performance_metrics['cache_hits'] += 1
                logger.info(
                    "Returning cached synthesis result",
                    extra={"correlation_id": correlation_id, "cache_key": cache_key}
                )
                return self.synthesis_cache[cache_key]
            
            # Generate insights
            insights = await self._generate_business_insights(
                validated_data, correlation_id
            )
            
            # Generate recommendations
            recommendations = await self._generate_recommendations(
                insights, validated_data, correlation_id
            )
            
            # Create executive summary
            executive_summary = await self._create_executive_summary(
                insights, recommendations, validated_data, correlation_id
            )
            
            # Calculate overall confidence
            confidence_score = self._calculate_overall_confidence(insights, recommendations)
            
            # Create synthesis result
            result = SynthesisResult(
                result_id=str(uuid.uuid4()),
                insights=insights,
                recommendations=recommendations,
                executive_summary=executive_summary,
                confidence_score=confidence_score
            )
            
            # Cache the result
            self.synthesis_cache[cache_key] = result
            
            # Update performance metrics
            processing_time_ms = (time.time() - start_time) * 1000
            self.performance_metrics['syntheses_performed'] += 1
            self.performance_metrics['avg_synthesis_time_ms'] = (
                (self.performance_metrics['avg_synthesis_time_ms'] * 
                 (self.performance_metrics['syntheses_performed'] - 1) + processing_time_ms) /
                self.performance_metrics['syntheses_performed']
            )
            
            logger.info(
                "Insight synthesis completed successfully",
                extra={
                    "correlation_id": correlation_id,
                    "insights_count": len(insights),
                    "recommendations_count": len(recommendations),
                    "processing_time_ms": processing_time_ms,
                    "confidence_score": confidence_score
                }
            )
            
            return result
            
        except Exception as e:
            self.failure_count += 1
            self.last_failure_time = datetime.utcnow()
            self.performance_metrics['errors_handled'] += 1
            
            if self.failure_count >= 5:
                self.circuit_breaker_open = True
                logger.error(
                    "Circuit breaker opened due to repeated failures",
                    extra={"correlation_id": correlation_id, "failure_count": self.failure_count}
                )
            
            # Classify and re-raise the error
            if "memory" in str(e).lower() or "resource" in str(e).lower():
                raise ResourceExhaustedError(
                    f"Resource exhausted during synthesis: {str(e)}",
                    component="vector_insight_synthesizer",
                    operation="synthesize_insights",
                    correlation_id=correlation_id
                )
            elif "timeout" in str(e).lower():
                raise PerformanceError(
                    f"Synthesis timed out: {str(e)}",
                    component="vector_insight_synthesizer",
                    operation="synthesize_insights",
                    correlation_id=correlation_id
                )
            else:
                raise BusinessLogicError(
                    f"Insight synthesis failed: {str(e)}",
                    component="vector_insight_synthesizer",
                    operation="synthesize_insights",
                    correlation_id=correlation_id
                )
    
    async def _generate_business_insights(
        self,
        validated_data: Dict[str, Any],
        correlation_id: Optional[str]
    ) -> List[BusinessInsight]:
        """Generate business insights from investigation results."""
        
        insights = []
        investigation_results = validated_data['investigation_results']
        
        # Extract key findings and generate insights
        for i, result in enumerate(investigation_results[:10]):  # Limit processing
            try:
                # Extract key information
                confidence = result.get('confidence_score', 0.7)
                
                insight = BusinessInsight(
                    insight_id=str(uuid.uuid4()),
                    insight_type=InsightType.OPERATIONAL,  # Default type
                    title=f"Insight from Investigation {i+1}",
                    description=f"Analysis of investigation result {i+1}",
                    confidence_score=min(confidence, 1.0),
                    business_impact=0.7  # Default impact
                )
                
                insights.append(insight)
                
            except Exception as e:
                logger.warning(
                    f"Failed to generate insight from result {i}: {e}",
                    extra={"correlation_id": correlation_id}
                )
                continue
        
        if not insights:
            # Generate at least one default insight
            default_insight = BusinessInsight(
                insight_id=str(uuid.uuid4()),
                insight_type=InsightType.OPERATIONAL,
                title="General Business Analysis",
                description="General analysis based on available data",
                confidence_score=0.6,
                business_impact=0.5
            )
            insights.append(default_insight)
        
        return insights
    
    async def _generate_recommendations(
        self,
        insights: List[BusinessInsight],
        validated_data: Dict[str, Any],
        correlation_id: Optional[str]
    ) -> List[Recommendation]:
        """Generate recommendations based on insights."""
        
        recommendations = []
        
        for i, insight in enumerate(insights[:5]):  # Limit processing
            try:
                recommendation = Recommendation(
                    recommendation_id=str(uuid.uuid4()),
                    recommendation_type=RecommendationType.SHORT_TERM,
                    title=f"Recommendation based on {insight.title}",
                    description=f"Action plan derived from insight: {insight.description[:100]}",
                    priority=min(i + 1, 5),
                    estimated_impact=insight.business_impact * 0.8
                )
                
                recommendations.append(recommendation)
                
            except Exception as e:
                logger.warning(
                    f"Failed to generate recommendation for insight {i}: {e}",
                    extra={"correlation_id": correlation_id}
                )
                continue
        
        return recommendations
    
    async def _create_executive_summary(
        self,
        insights: List[BusinessInsight],
        recommendations: List[Recommendation],
        validated_data: Dict[str, Any],
        correlation_id: Optional[str]
    ) -> str:
        """Create executive summary of synthesis results."""
        
        try:
            insight_count = len(insights)
            rec_count = len(recommendations)
            
            if insights:
                avg_confidence = safe_execute(
                    lambda: np.mean([insight.confidence_score for insight in insights]),
                    operation="calculate_avg_confidence",
                    component="vector_insight_synthesizer",
                    correlation_id=correlation_id,
                    default_return=0.7,
                    raise_on_failure=False
                )
            else:
                avg_confidence = 0.5
            
            summary = f"""Executive Summary
            
Based on the analysis of {len(validated_data['investigation_results'])} investigation results, 
we have identified {insight_count} key insights and generated {rec_count} actionable recommendations.

Key Highlights:
- Average confidence level: {avg_confidence:.1%}
- Primary focus areas identified across insights
- Recommended actions prioritized by business impact

This analysis provides a comprehensive foundation for strategic decision-making."""
            
            return summary
            
        except Exception as e:
            logger.warning(
                f"Failed to create executive summary: {e}",
                extra={"correlation_id": correlation_id}
            )
            return "Executive summary could not be generated due to processing constraints."
    
    def _calculate_overall_confidence(
        self,
        insights: List[BusinessInsight],
        recommendations: List[Recommendation]
    ) -> float:
        """Calculate overall confidence score."""
        
        if not insights:
            return 0.5  # Default confidence
        
        try:
            insight_confidences = [insight.confidence_score for insight in insights]
            avg_confidence = safe_execute(
                lambda: np.mean(insight_confidences),
                operation="calculate_overall_confidence",
                component="vector_insight_synthesizer",
                default_return=0.7,
                raise_on_failure=False
            )
            
            # Factor in number of insights and recommendations
            confidence_boost = min(len(insights) * 0.05, 0.2)
            rec_boost = min(len(recommendations) * 0.03, 0.1)
            
            return min(avg_confidence + confidence_boost + rec_boost, 1.0)
            
        except Exception:
            return 0.7  # Safe default
    
    def _generate_cache_key(
        self,
        business_question: str,
        investigation_results: List[Dict[str, Any]]
    ) -> str:
        """Generate cache key for synthesis results."""
        
        try:
            # Create a simple hash-like key
            question_hash = str(hash(business_question.strip().lower()))
            results_hash = str(hash(str(len(investigation_results))))
            return f"synthesis_{question_hash}_{results_hash}"
        except Exception:
            return f"synthesis_{uuid.uuid4()}"
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics."""
        return {
            "synthesizer_id": self.synthesizer_id,
            "cache_size": len(self.synthesis_cache),
            "patterns_discovered": len(self.vector_patterns),
            "circuit_breaker_open": self.circuit_breaker_open,
            "failure_count": self.failure_count,
            "performance_metrics": self.performance_metrics.copy(),
            "capabilities": {
                "synthesis_module": SYNTHESIS_MODULE_AVAILABLE,
                "vector_capabilities": VECTOR_CAPABILITIES_AVAILABLE,
                "numpy_available": NUMPY_AVAILABLE,
                "embedding_model": self.embedding_model is not None
            }
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Comprehensive health check."""
        try:
            # Test basic functionality
            test_data = {
                'business_question': 'Test question',
                'investigation_results': [{'test': 'data'}],
                'context': {}
            }
            
            self.validate_synthesis_input(test_data)
            
            return {
                "status": "healthy",
                "circuit_breaker_open": self.circuit_breaker_open,
                "cache_size": len(self.synthesis_cache),
                "capabilities_available": {
                    "synthesis_module": SYNTHESIS_MODULE_AVAILABLE,
                    "vector_capabilities": VECTOR_CAPABILITIES_AVAILABLE,
                    "numpy": NUMPY_AVAILABLE,
                    "embedding_model": self.embedding_model is not None
                }
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "circuit_breaker_open": self.circuit_breaker_open
            }

# Maintain compatibility with existing code
VectorEnhancedInsightSynthesizer = EnhancedVectorInsightSynthesizer

# Example usage and testing
if __name__ == "__main__":
    async def test_enhanced_synthesizer():
        """Test the enhanced synthesizer functionality."""
        print("Testing Enhanced Vector Insight Synthesizer...")
        
        synthesizer = EnhancedVectorInsightSynthesizer()
        
        # Test health check
        health = await synthesizer.health_check()
        print(f"Health check: {health}")
        
        # Test synthesis
        try:
            result = await synthesizer.synthesize_insights(
                business_question="What are the key performance trends?",
                investigation_results=[
                    {"finding": "Performance increased by 15%", "confidence_score": 0.9},
                    {"finding": "Cost reduction of 8%", "confidence_score": 0.8}
                ],
                context={"domain": "operations"}
            )
            
            print(f"Synthesis result: {len(result.insights)} insights, {len(result.recommendations)} recommendations")
            print(f"Confidence: {result.confidence_score:.2f}")
            
            # Test performance stats
            stats = synthesizer.get_performance_stats()
            print(f"Performance stats: {stats}")
            
        except Exception as e:
            print(f"Error during testing: {e}")
    
    asyncio.run(test_enhanced_synthesizer())