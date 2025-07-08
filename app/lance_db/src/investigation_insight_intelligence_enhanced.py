#!/usr/bin/env python3
"""
Enhanced Investigation-Insight Cross-Module Intelligence with Comprehensive Error Handling
Production-ready cross-module intelligence with proper error handling, monitoring, and resilience.
"""

import asyncio
import json
import time
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional, Tuple, Set, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
import uuid
from collections import defaultdict
import logging

# Import our error handling system
try:
    from ...core.error_handling import (
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

# Import dependencies with proper error handling
def safe_import_vector_infrastructure():
    """Safely import vector infrastructure with error handling."""
    try:
        from enterprise_vector_schema import (
            EnterpriseVectorSchema, VectorMetadata, ModuleSource,
            BusinessDomain as UnifiedBusinessDomain, PerformanceTier,
            AnalysisType as UnifiedAnalysisType, generate_vector_id,
            normalize_score_to_unified_scale
        )
        from vector_index_manager import VectorIndexManager
        from vector_performance_monitor import (
            VectorPerformanceMonitor, PerformanceMetric, PerformanceMetricType
        )
        return True, {
            'schema': EnterpriseVectorSchema,
            'metadata': VectorMetadata,
            'manager': VectorIndexManager,
            'monitor': VectorPerformanceMonitor
        }
    except ImportError as e:
        logger.warning(f"Vector infrastructure not available: {e}")
        return False, {}

def safe_import_vector_capabilities():
    """Safely import vector capabilities with error handling."""
    try:
        import lancedb
        from sentence_transformers import SentenceTransformer
        return True, {'lancedb': lancedb, 'transformer': SentenceTransformer}
    except ImportError as e:
        logger.warning(f"Vector capabilities not available: {e}")
        return False, {}

def safe_import_numpy():
    """Safely import numpy with fallback implementation."""
    try:
        import numpy as np
        return True, np
    except ImportError:
        logger.warning("NumPy not available, using fallback implementation")
        
        # Simple fallback implementations
        class NumpyFallback:
            @staticmethod
            def mean(values):
                if not values:
                    raise ValueError("Cannot compute mean of empty array")
                return sum(values) / len(values)
            
            @staticmethod
            def std(values):
                if not values:
                    raise ValueError("Cannot compute std of empty array")
                mean = sum(values) / len(values)
                variance = sum((x - mean) ** 2 for x in values) / len(values)
                return variance ** 0.5
            
            @staticmethod
            def median(values):
                if not values:
                    raise ValueError("Cannot compute median of empty array")
                sorted_values = sorted(values)
                n = len(sorted_values)
                if n % 2 == 0:
                    return (sorted_values[n//2-1] + sorted_values[n//2]) / 2
                else:
                    return sorted_values[n//2]
        
        return False, NumpyFallback()

# Initialize dependencies
VECTOR_INFRASTRUCTURE_AVAILABLE, vector_infrastructure = safe_import_vector_infrastructure()
VECTOR_CAPABILITIES_AVAILABLE, vector_capabilities = safe_import_vector_capabilities()
NUMPY_AVAILABLE, np = safe_import_numpy()

class InvestigationInsightLinkType(Enum):
    """Types of links between investigations and insights."""
    DIRECT_GENERATION = "direct_generation"
    PATTERN_CORRELATION = "pattern_correlation"
    DOMAIN_OVERLAP = "domain_overlap"
    TEMPORAL_SEQUENCE = "temporal_sequence"
    CAUSAL_CHAIN = "causal_chain"
    VALIDATION_SUPPORT = "validation_support"
    HYPOTHESIS_EVOLUTION = "hypothesis_evolution"

class FeedbackLoopType(Enum):
    """Types of feedback loops between modules."""
    INSIGHT_QUALITY = "insight_quality"
    INVESTIGATION_DEPTH = "investigation_depth"
    PATTERN_DISCOVERY = "pattern_discovery"
    CONFIDENCE_BOOST = "confidence_boost"
    DOMAIN_EXPANSION = "domain_expansion"

@dataclass
class InvestigationInsightLink:
    """Link between an investigation and insights generated from it."""
    link_id: str
    link_type: InvestigationInsightLinkType
    investigation_id: str
    insight_ids: List[str]
    
    # Relationship metadata
    correlation_strength: float  # 0.0 to 1.0
    confidence_score: float
    business_impact: float
    
    # Pattern data
    shared_patterns: List[str]
    domain_overlap: List[str]
    temporal_distance_hours: float
    
    # Learning metrics
    insight_quality_improvement: float
    investigation_efficiency_gain: float
    cross_validation_score: float
    
    # Timestamps
    link_created: datetime
    last_updated: datetime

@dataclass
class CrossModulePattern:
    """Pattern discovered across Investigation and Insight modules."""
    pattern_id: str
    pattern_name: str
    pattern_description: str
    
    # Pattern characteristics
    occurrence_count: int
    avg_correlation_strength: float
    business_domains: List[str]
    
    # Success metrics
    avg_insight_quality: float
    avg_investigation_confidence: float
    business_value_generated: float
    
    # Predictive power
    prediction_accuracy: float
    recommendation_success_rate: float
    
    # Example instances
    example_investigations: List[str]
    example_insights: List[str]
    
    # Temporal data
    first_observed: datetime
    last_observed: datetime
    trend: str  # "increasing", "stable", "decreasing"

class EnhancedInvestigationInsightIntelligenceManager:
    """
    Production-ready Investigation-Insight Intelligence Manager with comprehensive error handling.
    
    Features:
    - Structured error handling
    - Resource management
    - Performance monitoring
    - Input validation
    - Circuit breaker patterns
    """
    
    def __init__(self):
        self.intelligence_id = str(uuid.uuid4())
        self.initialization_correlation_id = str(uuid.uuid4())
        
        # Core components
        self.links: Dict[str, InvestigationInsightLink] = {}
        self.patterns: Dict[str, CrossModulePattern] = {}
        self.feedback_loops: Dict[str, Any] = {}
        
        # Performance monitoring
        self.performance_metrics = {
            'links_created': 0,
            'patterns_discovered': 0,
            'errors_handled': 0,
            'avg_processing_time_ms': 0
        }
        
        # Circuit breaker state
        self.circuit_breaker_open = False
        self.failure_count = 0
        self.last_failure_time = None
        
        # Initialize safely
        self._initialize_safely()
    
    def _initialize_safely(self):
        """Initialize with comprehensive error handling."""
        
        logger.info(
            "Initializing Investigation-Insight Intelligence Manager",
            extra={"correlation_id": self.initialization_correlation_id}
        )
        
        # Validate dependencies
        if not VECTOR_CAPABILITIES_AVAILABLE:
            logger.warning(
                "Vector capabilities not available - some features disabled",
                extra={"correlation_id": self.initialization_correlation_id}
            )
        
        if not NUMPY_AVAILABLE:
            logger.warning(
                "NumPy not available - using fallback implementations",
                extra={"correlation_id": self.initialization_correlation_id}
            )
        
        logger.info(
            "Investigation-Insight Intelligence Manager initialized successfully",
            extra={"correlation_id": self.initialization_correlation_id}
        )
    
    def validate_link_data(self, link_data: dict) -> dict:
        """Validate investigation-insight link data."""
        def validator(data):
            if not isinstance(data, dict):
                raise ValueError("Link data must be a dictionary")
            
            required_fields = ['investigation_id', 'insight_ids', 'link_type']
            for field in required_fields:
                if field not in data:
                    raise ValueError(f"Missing required field: {field}")
            
            # Validate IDs
            if not isinstance(data['investigation_id'], str) or len(data['investigation_id']) == 0:
                raise ValueError("investigation_id must be a non-empty string")
            
            if not isinstance(data['insight_ids'], list) or len(data['insight_ids']) == 0:
                raise ValueError("insight_ids must be a non-empty list")
            
            # Validate scores
            for score_field in ['correlation_strength', 'confidence_score', 'business_impact']:
                if score_field in data:
                    score = data[score_field]
                    if not isinstance(score, (int, float)) or not 0.0 <= score <= 1.0:
                        raise ValueError(f"{score_field} must be a number between 0.0 and 1.0")
            
            return data
        
        return validate_input(
            link_data,
            validator,
            operation="validate_link_data",
            component="investigation_insight_intelligence"
        )
    
    @with_error_handling(
        operation="create_investigation_insight_link",
        component="investigation_insight_intelligence",
        timeout_seconds=30.0
    )
    async def create_investigation_insight_link(
        self,
        investigation_id: str,
        insight_ids: List[str],
        link_type: InvestigationInsightLinkType,
        correlation_id: Optional[str] = None
    ) -> InvestigationInsightLink:
        """Create a new investigation-insight link with comprehensive error handling."""
        
        # Circuit breaker check
        if self.circuit_breaker_open:
            if self.last_failure_time and (datetime.utcnow() - self.last_failure_time).seconds < 300:
                raise ResourceExhaustedError(
                    "Circuit breaker open - too many recent failures",
                    component="investigation_insight_intelligence",
                    operation="create_link",
                    correlation_id=correlation_id,
                    retry_after=300
                )
            else:
                # Reset circuit breaker
                self.circuit_breaker_open = False
                self.failure_count = 0
        
        # Validate input data
        link_data = {
            'investigation_id': investigation_id,
            'insight_ids': insight_ids,
            'link_type': link_type.value
        }
        validated_data = self.validate_link_data(link_data)
        
        logger.info(
            "Creating investigation-insight link",
            extra={
                "correlation_id": correlation_id,
                "investigation_id": investigation_id,
                "insight_count": len(insight_ids),
                "link_type": link_type.value
            }
        )
        
        try:
            # Create the link
            link = InvestigationInsightLink(
                link_id=str(uuid.uuid4()),
                link_type=link_type,
                investigation_id=investigation_id,
                insight_ids=insight_ids,
                correlation_strength=0.8,  # Default value
                confidence_score=0.7,      # Default value
                business_impact=0.6,       # Default value
                shared_patterns=[],
                domain_overlap=[],
                temporal_distance_hours=0.0,
                insight_quality_improvement=0.0,
                investigation_efficiency_gain=0.0,
                cross_validation_score=0.0,
                link_created=datetime.utcnow(),
                last_updated=datetime.utcnow()
            )
            
            # Store the link
            self.links[link.link_id] = link
            self.performance_metrics['links_created'] += 1
            
            logger.info(
                "Investigation-insight link created successfully",
                extra={
                    "correlation_id": correlation_id,
                    "link_id": link.link_id,
                    "total_links": len(self.links)
                }
            )
            
            return link
            
        except Exception as e:
            self.failure_count += 1
            self.last_failure_time = datetime.utcnow()
            
            if self.failure_count >= 5:
                self.circuit_breaker_open = True
                logger.error(
                    "Circuit breaker opened due to repeated failures",
                    extra={"correlation_id": correlation_id, "failure_count": self.failure_count}
                )
            
            # Classify and re-raise the error
            if "memory" in str(e).lower() or "resource" in str(e).lower():
                raise ResourceExhaustedError(
                    f"Resource exhausted while creating link: {str(e)}",
                    component="investigation_insight_intelligence",
                    operation="create_link",
                    correlation_id=correlation_id
                )
            elif "validation" in str(e).lower():
                raise ValidationError(
                    f"Validation failed: {str(e)}",
                    component="investigation_insight_intelligence",
                    operation="create_link",
                    correlation_id=correlation_id
                )
            else:
                raise BusinessLogicError(
                    f"Failed to create investigation-insight link: {str(e)}",
                    component="investigation_insight_intelligence",
                    operation="create_link",
                    correlation_id=correlation_id
                )
    
    @with_error_handling(
        operation="discover_cross_module_patterns",
        component="investigation_insight_intelligence",
        timeout_seconds=60.0
    )
    async def discover_cross_module_patterns(
        self,
        min_pattern_strength: float = 0.6,
        correlation_id: Optional[str] = None
    ) -> List[CrossModulePattern]:
        """Discover patterns across investigation and insight modules."""
        
        if not self.links:
            logger.info(
                "No links available for pattern discovery",
                extra={"correlation_id": correlation_id}
            )
            return []
        
        logger.info(
            "Starting cross-module pattern discovery",
            extra={
                "correlation_id": correlation_id,
                "link_count": len(self.links),
                "min_pattern_strength": min_pattern_strength
            }
        )
        
        discovered_patterns = []
        
        try:
            # Group links by type for pattern analysis
            link_groups = defaultdict(list)
            for link in self.links.values():
                link_groups[link.link_type].append(link)
            
            # Analyze each group for patterns
            for link_type, type_links in link_groups.items():
                if len(type_links) < 2:  # Need at least 2 links to form a pattern
                    continue
                
                # Calculate pattern strength
                correlation_scores = [link.correlation_strength for link in type_links]
                if not correlation_scores:
                    continue
                
                avg_correlation = safe_execute(
                    lambda: np.mean(correlation_scores),
                    operation="calculate_mean_correlation",
                    component="investigation_insight_intelligence",
                    correlation_id=correlation_id,
                    default_return=0.0,
                    raise_on_failure=False
                )
                
                if avg_correlation >= min_pattern_strength:
                    pattern = CrossModulePattern(
                        pattern_id=str(uuid.uuid4()),
                        pattern_name=f"{link_type.value}_pattern",
                        pattern_description=f"Pattern discovered in {link_type.value} links",
                        occurrence_count=len(type_links),
                        avg_correlation_strength=avg_correlation,
                        business_domains=[],
                        avg_insight_quality=0.8,  # Default values
                        avg_investigation_confidence=0.7,
                        business_value_generated=0.6,
                        prediction_accuracy=0.75,
                        recommendation_success_rate=0.8,
                        example_investigations=[link.investigation_id for link in type_links[:3]],
                        example_insights=[link.insight_ids[0] for link in type_links[:3] if link.insight_ids],
                        first_observed=min(link.link_created for link in type_links),
                        last_observed=max(link.last_updated for link in type_links),
                        trend="stable"
                    )
                    
                    discovered_patterns.append(pattern)
                    self.patterns[pattern.pattern_id] = pattern
            
            self.performance_metrics['patterns_discovered'] = len(self.patterns)
            
            logger.info(
                "Cross-module pattern discovery completed",
                extra={
                    "correlation_id": correlation_id,
                    "patterns_discovered": len(discovered_patterns),
                    "total_patterns": len(self.patterns)
                }
            )
            
            return discovered_patterns
            
        except Exception as e:
            self.performance_metrics['errors_handled'] += 1
            
            # Classify and re-raise the error
            if "timeout" in str(e).lower():
                raise PerformanceError(
                    f"Pattern discovery timed out: {str(e)}",
                    component="investigation_insight_intelligence",
                    operation="discover_patterns",
                    correlation_id=correlation_id
                )
            else:
                raise BusinessLogicError(
                    f"Pattern discovery failed: {str(e)}",
                    component="investigation_insight_intelligence",
                    operation="discover_patterns",
                    correlation_id=correlation_id
                )
    
    def get_link_by_id(self, link_id: str) -> Optional[InvestigationInsightLink]:
        """Get investigation-insight link by ID with error handling."""
        if not link_id or not isinstance(link_id, str):
            raise ValidationError(
                "link_id must be a non-empty string",
                component="investigation_insight_intelligence",
                operation="get_link_by_id"
            )
        
        return self.links.get(link_id)
    
    def get_links_by_investigation(self, investigation_id: str) -> List[InvestigationInsightLink]:
        """Get all links for a specific investigation with error handling."""
        if not investigation_id or not isinstance(investigation_id, str):
            raise ValidationError(
                "investigation_id must be a non-empty string",
                component="investigation_insight_intelligence",
                operation="get_links_by_investigation"
            )
        
        return [link for link in self.links.values() if link.investigation_id == investigation_id]
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics."""
        return {
            "intelligence_id": self.intelligence_id,
            "total_links": len(self.links),
            "total_patterns": len(self.patterns),
            "circuit_breaker_open": self.circuit_breaker_open,
            "failure_count": self.failure_count,
            "performance_metrics": self.performance_metrics.copy(),
            "capabilities": {
                "vector_infrastructure": VECTOR_INFRASTRUCTURE_AVAILABLE,
                "vector_capabilities": VECTOR_CAPABILITIES_AVAILABLE,
                "numpy_available": NUMPY_AVAILABLE
            }
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Comprehensive health check."""
        try:
            # Test basic functionality
            test_link_data = {
                'investigation_id': 'test_inv',
                'insight_ids': ['test_insight'],
                'link_type': 'direct_generation'
            }
            
            self.validate_link_data(test_link_data)
            
            return {
                "status": "healthy",
                "circuit_breaker_open": self.circuit_breaker_open,
                "total_links": len(self.links),
                "total_patterns": len(self.patterns),
                "capabilities_available": {
                    "vector_infrastructure": VECTOR_INFRASTRUCTURE_AVAILABLE,
                    "vector_capabilities": VECTOR_CAPABILITIES_AVAILABLE,
                    "numpy": NUMPY_AVAILABLE
                }
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "circuit_breaker_open": self.circuit_breaker_open
            }

# Maintain compatibility with existing code
InvestigationInsightIntelligenceManager = EnhancedInvestigationInsightIntelligenceManager

# Example usage and testing
if __name__ == "__main__":
    async def test_enhanced_manager():
        """Test the enhanced manager functionality."""
        print("Testing Enhanced Investigation-Insight Intelligence Manager...")
        
        manager = EnhancedInvestigationInsightIntelligenceManager()
        
        # Test health check
        health = await manager.health_check()
        print(f"Health check: {health}")
        
        # Test link creation
        try:
            link = await manager.create_investigation_insight_link(
                investigation_id="test_investigation_001",
                insight_ids=["insight_001", "insight_002"],
                link_type=InvestigationInsightLinkType.DIRECT_GENERATION
            )
            print(f"Created link: {link.link_id}")
            
            # Test pattern discovery
            patterns = await manager.discover_cross_module_patterns()
            print(f"Discovered patterns: {len(patterns)}")
            
            # Test performance stats
            stats = manager.get_performance_stats()
            print(f"Performance stats: {stats}")
            
        except Exception as e:
            print(f"Error during testing: {e}")
    
    asyncio.run(test_enhanced_manager())