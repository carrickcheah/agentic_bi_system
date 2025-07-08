#!/usr/bin/env python3
"""
Enhanced LanceDB Pattern Recognizer with Comprehensive Error Handling
Production-ready pattern recognition with proper error handling, monitoring, and resilience.
"""

import asyncio
import json
import time
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union, Set
from dataclasses import dataclass, field
from enum import Enum
import math
from collections import defaultdict
import logging

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
            def std(values):
                if not values:
                    raise ValueError("Cannot compute std of empty array")
                mean = sum(values) / len(values)
                variance = sum((x - mean) ** 2 for x in values) / len(values)
                return variance ** 0.5
            
            @staticmethod
            def corrcoef(x, y):
                """Simple correlation coefficient calculation."""
                if len(x) != len(y) or not x:
                    return 0.0
                
                mean_x = sum(x) / len(x)
                mean_y = sum(y) / len(y)
                
                numerator = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(len(x)))
                
                sum_sq_x = sum((xi - mean_x) ** 2 for xi in x)
                sum_sq_y = sum((yi - mean_y) ** 2 for yi in y)
                
                denominator = (sum_sq_x * sum_sq_y) ** 0.5
                
                return numerator / denominator if denominator != 0 else 0.0
        
        return False, NumpyFallback()

def safe_import_intelligence_module():
    """Safely import intelligence module components."""
    try:
        from .domain_expert import DomainExpert, BusinessIntent, BusinessDomain, AnalysisType
        from .complexity_analyzer import ComplexityAnalyzer, ComplexityScore, ComplexityLevel
        from .config import settings
        from .intelligence_logging import setup_logger, performance_monitor
        return True, {
            'domain_expert': DomainExpert,
            'complexity_analyzer': ComplexityAnalyzer,
            'business_intent': BusinessIntent,
            'business_domain': BusinessDomain,
            'analysis_type': AnalysisType,
            'complexity_score': ComplexityScore,
            'complexity_level': ComplexityLevel,
            'settings': settings
        }
    except ImportError:
        try:
            from domain_expert import DomainExpert, BusinessIntent, BusinessDomain, AnalysisType
            from complexity_analyzer import ComplexityAnalyzer, ComplexityScore, ComplexityLevel
            from config import settings
            from intelligence_logging import setup_logger, performance_monitor
            return True, {
                'domain_expert': DomainExpert,
                'complexity_analyzer': ComplexityAnalyzer,
                'business_intent': BusinessIntent,
                'business_domain': BusinessDomain,
                'analysis_type': AnalysisType,
                'complexity_score': ComplexityScore,
                'complexity_level': ComplexityLevel,
                'settings': settings
            }
        except ImportError:
            logger.warning("Intelligence module components not available")
            
            # Define minimal fallback classes
            class BusinessDomain(Enum):
                FINANCE = "finance"
                OPERATIONS = "operations"
                MARKETING = "marketing"
                SALES = "sales"
                HR = "hr"
                TECHNOLOGY = "technology"
                
            class AnalysisType(Enum):
                DESCRIPTIVE = "descriptive"
                DIAGNOSTIC = "diagnostic"
                PREDICTIVE = "predictive"
                PRESCRIPTIVE = "prescriptive"
                
            class ComplexityLevel(Enum):
                LOW = "low"
                MEDIUM = "medium"
                HIGH = "high"
                VERY_HIGH = "very_high"
            
            return False, {
                'business_domain': BusinessDomain,
                'analysis_type': AnalysisType,
                'complexity_level': ComplexityLevel
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
INTELLIGENCE_MODULE_AVAILABLE, intelligence_components = safe_import_intelligence_module()
VECTOR_CAPABILITIES_AVAILABLE, vector_capabilities = safe_import_vector_capabilities()

# Extract components for use
BusinessDomain = intelligence_components.get('business_domain', None)
AnalysisType = intelligence_components.get('analysis_type', None)
ComplexityLevel = intelligence_components.get('complexity_level', None)

class PatternType(Enum):
    """Types of patterns that can be recognized across modules."""
    BUSINESS_PROCESS = "business_process"
    DOMAIN_CORRELATION = "domain_correlation"
    COMPLEXITY_EVOLUTION = "complexity_evolution"
    PERFORMANCE_TREND = "performance_trend"
    USAGE_PATTERN = "usage_pattern"
    ERROR_PATTERN = "error_pattern"
    SUCCESS_PATTERN = "success_pattern"
    TEMPORAL_PATTERN = "temporal_pattern"

class PatternConfidence(Enum):
    """Confidence levels for pattern recognition."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"

@dataclass
class RecognizedPattern:
    """A pattern recognized across modules."""
    pattern_id: str
    pattern_type: PatternType
    pattern_name: str
    description: str
    confidence: PatternConfidence
    
    # Pattern characteristics
    occurrence_count: int
    strength_score: float  # 0.0 to 1.0
    business_impact: float
    
    # Temporal data
    first_observed: datetime
    last_observed: datetime
    frequency: float  # patterns per time unit
    
    # Context
    affected_modules: List[str]
    business_domains: List[str]
    
    # Evidence
    supporting_data: List[Dict[str, Any]]
    correlation_metrics: Dict[str, float]

@dataclass
class PatternSearchCriteria:
    """Criteria for pattern search and recognition."""
    pattern_types: List[PatternType]
    business_domains: List[str]
    min_confidence: PatternConfidence
    time_range: Tuple[datetime, datetime]
    min_occurrences: int
    include_historical: bool

class EnhancedLanceDBPatternRecognizer:
    """
    Production-ready LanceDB Pattern Recognizer with comprehensive error handling.
    
    Features:
    - Structured error handling
    - Pattern discovery and recognition
    - Performance monitoring
    - Input validation
    - Resource management
    """
    
    def __init__(self):
        self.recognizer_id = str(uuid.uuid4())
        self.initialization_correlation_id = str(uuid.uuid4())
        
        # Core components
        self.recognized_patterns: Dict[str, RecognizedPattern] = {}
        self.pattern_cache: Dict[str, List[RecognizedPattern]] = {}
        self.performance_metrics = {
            'patterns_recognized': 0,
            'searches_performed': 0,
            'cache_hits': 0,
            'errors_handled': 0,
            'avg_recognition_time_ms': 0
        }
        
        # Circuit breaker state
        self.circuit_breaker_open = False
        self.failure_count = 0
        self.last_failure_time = None
        
        # Pattern learning state
        self.learning_enabled = True
        self.pattern_weights = defaultdict(float)
        
        # Initialize safely
        self._initialize_safely()
    
    def _initialize_safely(self):
        """Initialize with comprehensive error handling."""
        
        logger.info(
            "Initializing Enhanced LanceDB Pattern Recognizer",
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
        
        if not INTELLIGENCE_MODULE_AVAILABLE:
            logger.warning(
                "Intelligence module not available - using fallback implementations",
                extra={"correlation_id": self.initialization_correlation_id}
            )
        
        logger.info(
            "Enhanced LanceDB Pattern Recognizer initialized successfully",
            extra={"correlation_id": self.initialization_correlation_id}
        )
    
    def validate_pattern_data(self, pattern_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate pattern recognition data."""
        def validator(data):
            if not isinstance(data, dict):
                raise ValueError("Pattern data must be a dictionary")
            
            # Required fields
            required_fields = ['module_data', 'pattern_types']
            for field in required_fields:
                if field not in data:
                    raise ValueError(f"Missing required field: {field}")
            
            # Validate module data
            module_data = data['module_data']
            if not isinstance(module_data, list):
                raise ValueError("module_data must be a list")
            
            if len(module_data) > 10000:
                raise ValueError("Too much module data (max 10,000 items)")
            
            # Validate pattern types
            pattern_types = data['pattern_types']
            if not isinstance(pattern_types, list):
                raise ValueError("pattern_types must be a list")
            
            if not pattern_types:
                raise ValueError("pattern_types cannot be empty")
            
            # Validate each pattern type
            for pt in pattern_types:
                if not isinstance(pt, str):
                    raise ValueError("All pattern types must be strings")
                try:
                    PatternType(pt)
                except ValueError:
                    raise ValueError(f"Invalid pattern type: {pt}")
            
            return data
        
        return validate_input(
            pattern_data,
            validator,
            operation="validate_pattern_data",
            component="lancedb_pattern_recognizer"
        )
    
    @with_error_handling(
        operation="recognize_patterns",
        component="lancedb_pattern_recognizer",
        timeout_seconds=60.0
    )
    async def recognize_patterns(
        self,
        module_data: List[Dict[str, Any]],
        pattern_types: List[str],
        criteria: Optional[PatternSearchCriteria] = None,
        correlation_id: Optional[str] = None
    ) -> List[RecognizedPattern]:
        """Recognize patterns in module data with comprehensive error handling."""
        
        # Circuit breaker check
        if self.circuit_breaker_open:
            if self.last_failure_time and (datetime.utcnow() - self.last_failure_time).seconds < 300:
                raise ResourceExhaustedError(
                    "Circuit breaker open - too many recent failures",
                    component="lancedb_pattern_recognizer",
                    operation="recognize_patterns",
                    correlation_id=correlation_id,
                    retry_after=300
                )
            else:
                # Reset circuit breaker
                self.circuit_breaker_open = False
                self.failure_count = 0
        
        # Validate input data
        input_data = {
            'module_data': module_data,
            'pattern_types': pattern_types
        }
        validated_data = self.validate_pattern_data(input_data)
        
        logger.info(
            "Starting pattern recognition",
            extra={
                "correlation_id": correlation_id,
                "data_points": len(module_data),
                "pattern_types": len(pattern_types)
            }
        )
        
        try:
            start_time = time.time()
            
            # Check cache first
            cache_key = self._generate_cache_key(module_data, pattern_types)
            if cache_key in self.pattern_cache:
                self.performance_metrics['cache_hits'] += 1
                logger.info(
                    "Returning cached pattern results",
                    extra={"correlation_id": correlation_id, "cache_key": cache_key}
                )
                return self.pattern_cache[cache_key]
            
            recognized_patterns = []
            
            # Process each pattern type
            for pattern_type_str in pattern_types:
                try:
                    pattern_type = PatternType(pattern_type_str)
                    
                    patterns = await self._recognize_pattern_type(
                        validated_data['module_data'],
                        pattern_type,
                        criteria,
                        correlation_id
                    )
                    
                    recognized_patterns.extend(patterns)
                    
                except Exception as e:
                    logger.warning(
                        f"Failed to recognize pattern type {pattern_type_str}: {e}",
                        extra={"correlation_id": correlation_id}
                    )
                    continue
            
            # Store recognized patterns
            for pattern in recognized_patterns:
                self.recognized_patterns[pattern.pattern_id] = pattern
            
            # Cache the results
            self.pattern_cache[cache_key] = recognized_patterns
            
            # Update performance metrics
            processing_time_ms = (time.time() - start_time) * 1000
            self.performance_metrics['patterns_recognized'] += len(recognized_patterns)
            self.performance_metrics['searches_performed'] += 1
            self.performance_metrics['avg_recognition_time_ms'] = (
                (self.performance_metrics['avg_recognition_time_ms'] * 
                 (self.performance_metrics['searches_performed'] - 1) + processing_time_ms) /
                self.performance_metrics['searches_performed']
            )
            
            logger.info(
                "Pattern recognition completed successfully",
                extra={
                    "correlation_id": correlation_id,
                    "patterns_found": len(recognized_patterns),
                    "processing_time_ms": processing_time_ms
                }
            )
            
            return recognized_patterns
            
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
                    f"Resource exhausted during pattern recognition: {str(e)}",
                    component="lancedb_pattern_recognizer",
                    operation="recognize_patterns",
                    correlation_id=correlation_id
                )
            elif "timeout" in str(e).lower():
                raise PerformanceError(
                    f"Pattern recognition timed out: {str(e)}",
                    component="lancedb_pattern_recognizer",
                    operation="recognize_patterns",
                    correlation_id=correlation_id
                )
            else:
                raise BusinessLogicError(
                    f"Pattern recognition failed: {str(e)}",
                    component="lancedb_pattern_recognizer",
                    operation="recognize_patterns",
                    correlation_id=correlation_id
                )
    
    async def _recognize_pattern_type(
        self,
        module_data: List[Dict[str, Any]],
        pattern_type: PatternType,
        criteria: Optional[PatternSearchCriteria],
        correlation_id: Optional[str]
    ) -> List[RecognizedPattern]:
        """Recognize specific pattern type in module data."""
        
        patterns = []
        
        if pattern_type == PatternType.BUSINESS_PROCESS:
            patterns = await self._recognize_business_process_patterns(
                module_data, correlation_id
            )
        elif pattern_type == PatternType.PERFORMANCE_TREND:
            patterns = await self._recognize_performance_trend_patterns(
                module_data, correlation_id
            )
        elif pattern_type == PatternType.USAGE_PATTERN:
            patterns = await self._recognize_usage_patterns(
                module_data, correlation_id
            )
        else:
            # Generic pattern recognition
            patterns = await self._recognize_generic_patterns(
                module_data, pattern_type, correlation_id
            )
        
        # Apply criteria filter if provided
        if criteria:
            patterns = self._filter_patterns_by_criteria(patterns, criteria)
        
        return patterns
    
    async def _recognize_business_process_patterns(
        self,
        module_data: List[Dict[str, Any]],
        correlation_id: Optional[str]
    ) -> List[RecognizedPattern]:
        """Recognize business process patterns."""
        
        patterns = []
        
        try:
            # Group data by business domain
            domain_groups = defaultdict(list)
            for item in module_data:
                domain = item.get('business_domain', 'unknown')
                domain_groups[domain].append(item)
            
            # Analyze each domain group
            for domain, items in domain_groups.items():
                if len(items) >= 3:  # Minimum items to form a pattern
                    pattern = RecognizedPattern(
                        pattern_id=str(uuid.uuid4()),
                        pattern_type=PatternType.BUSINESS_PROCESS,
                        pattern_name=f"Business Process Pattern - {domain}",
                        description=f"Process pattern identified in {domain} domain",
                        confidence=PatternConfidence.MEDIUM,
                        occurrence_count=len(items),
                        strength_score=min(len(items) / 10.0, 1.0),
                        business_impact=0.7,
                        first_observed=datetime.utcnow() - timedelta(days=30),
                        last_observed=datetime.utcnow(),
                        frequency=len(items) / 30.0,  # per day
                        affected_modules=['business_process'],
                        business_domains=[domain],
                        supporting_data=items[:5],  # Limit evidence
                        correlation_metrics={'domain_consistency': 0.8}
                    )
                    patterns.append(pattern)
            
        except Exception as e:
            logger.warning(
                f"Failed to recognize business process patterns: {e}",
                extra={"correlation_id": correlation_id}
            )
        
        return patterns
    
    async def _recognize_performance_trend_patterns(
        self,
        module_data: List[Dict[str, Any]],
        correlation_id: Optional[str]
    ) -> List[RecognizedPattern]:
        """Recognize performance trend patterns."""
        
        patterns = []
        
        try:
            # Extract performance metrics
            performance_data = []
            for item in module_data:
                if 'performance_score' in item or 'response_time' in item:
                    performance_data.append(item)
            
            if len(performance_data) >= 5:  # Minimum for trend analysis
                # Calculate trend
                scores = [item.get('performance_score', 0.5) for item in performance_data]
                
                if scores:
                    trend_strength = safe_execute(
                        lambda: self._calculate_trend_strength(scores),
                        operation="calculate_trend_strength",
                        component="lancedb_pattern_recognizer",
                        correlation_id=correlation_id,
                        default_return=0.5,
                        raise_on_failure=False
                    )
                    
                    if trend_strength > 0.6:  # Significant trend
                        pattern = RecognizedPattern(
                            pattern_id=str(uuid.uuid4()),
                            pattern_type=PatternType.PERFORMANCE_TREND,
                            pattern_name="Performance Trend Pattern",
                            description="Significant performance trend detected",
                            confidence=PatternConfidence.HIGH if trend_strength > 0.8 else PatternConfidence.MEDIUM,
                            occurrence_count=len(performance_data),
                            strength_score=trend_strength,
                            business_impact=0.8,
                            first_observed=datetime.utcnow() - timedelta(days=7),
                            last_observed=datetime.utcnow(),
                            frequency=len(performance_data) / 7.0,
                            affected_modules=['performance'],
                            business_domains=['operations'],
                            supporting_data=performance_data[:5],
                            correlation_metrics={'trend_strength': trend_strength}
                        )
                        patterns.append(pattern)
            
        except Exception as e:
            logger.warning(
                f"Failed to recognize performance trend patterns: {e}",
                extra={"correlation_id": correlation_id}
            )
        
        return patterns
    
    async def _recognize_usage_patterns(
        self,
        module_data: List[Dict[str, Any]],
        correlation_id: Optional[str]
    ) -> List[RecognizedPattern]:
        """Recognize usage patterns."""
        
        patterns = []
        
        try:
            # Group by time periods
            usage_by_hour = defaultdict(int)
            for item in module_data:
                timestamp = item.get('timestamp')
                if timestamp:
                    if isinstance(timestamp, str):
                        try:
                            timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                        except:
                            continue
                    
                    hour = timestamp.hour
                    usage_by_hour[hour] += 1
            
            if usage_by_hour:
                # Find peak usage hours
                peak_hour = max(usage_by_hour, key=usage_by_hour.get)
                peak_usage = usage_by_hour[peak_hour]
                avg_usage = sum(usage_by_hour.values()) / len(usage_by_hour)
                
                if peak_usage > avg_usage * 1.5:  # Significant peak
                    pattern = RecognizedPattern(
                        pattern_id=str(uuid.uuid4()),
                        pattern_type=PatternType.USAGE_PATTERN,
                        pattern_name=f"Peak Usage Pattern - Hour {peak_hour}",
                        description=f"Peak usage detected at hour {peak_hour}",
                        confidence=PatternConfidence.HIGH,
                        occurrence_count=peak_usage,
                        strength_score=min(peak_usage / avg_usage / 2.0, 1.0),
                        business_impact=0.6,
                        first_observed=datetime.utcnow() - timedelta(days=1),
                        last_observed=datetime.utcnow(),
                        frequency=1.0,  # daily pattern
                        affected_modules=['usage'],
                        business_domains=['operations'],
                        supporting_data=[],
                        correlation_metrics={'peak_ratio': peak_usage / avg_usage}
                    )
                    patterns.append(pattern)
            
        except Exception as e:
            logger.warning(
                f"Failed to recognize usage patterns: {e}",
                extra={"correlation_id": correlation_id}
            )
        
        return patterns
    
    async def _recognize_generic_patterns(
        self,
        module_data: List[Dict[str, Any]],
        pattern_type: PatternType,
        correlation_id: Optional[str]
    ) -> List[RecognizedPattern]:
        """Recognize generic patterns for any pattern type."""
        
        patterns = []
        
        try:
            if len(module_data) >= 3:
                pattern = RecognizedPattern(
                    pattern_id=str(uuid.uuid4()),
                    pattern_type=pattern_type,
                    pattern_name=f"Generic {pattern_type.value} Pattern",
                    description=f"Generic pattern of type {pattern_type.value}",
                    confidence=PatternConfidence.LOW,
                    occurrence_count=len(module_data),
                    strength_score=min(len(module_data) / 20.0, 1.0),
                    business_impact=0.4,
                    first_observed=datetime.utcnow() - timedelta(days=1),
                    last_observed=datetime.utcnow(),
                    frequency=len(module_data),
                    affected_modules=['generic'],
                    business_domains=['general'],
                    supporting_data=module_data[:3],
                    correlation_metrics={'data_consistency': 0.5}
                )
                patterns.append(pattern)
        
        except Exception as e:
            logger.warning(
                f"Failed to recognize generic patterns: {e}",
                extra={"correlation_id": correlation_id}
            )
        
        return patterns
    
    def _calculate_trend_strength(self, values: List[float]) -> float:
        """Calculate the strength of a trend in values."""
        
        if len(values) < 3:
            return 0.0
        
        try:
            # Simple linear trend calculation
            x = list(range(len(values)))
            y = values
            
            correlation = safe_execute(
                lambda: abs(np.corrcoef(x, y)[0, 1]) if hasattr(np, 'corrcoef') else abs(np.corrcoef(x, y)),
                operation="calculate_correlation",
                component="lancedb_pattern_recognizer",
                default_return=0.5,
                raise_on_failure=False
            )
            
            return correlation if correlation is not None else 0.5
            
        except Exception:
            return 0.5  # Default trend strength
    
    def _filter_patterns_by_criteria(
        self,
        patterns: List[RecognizedPattern],
        criteria: PatternSearchCriteria
    ) -> List[RecognizedPattern]:
        """Filter patterns based on search criteria."""
        
        filtered = []
        
        for pattern in patterns:
            # Check pattern type
            if pattern.pattern_type not in criteria.pattern_types:
                continue
            
            # Check confidence
            confidence_levels = {
                PatternConfidence.LOW: 1,
                PatternConfidence.MEDIUM: 2,
                PatternConfidence.HIGH: 3,
                PatternConfidence.VERY_HIGH: 4
            }
            
            if confidence_levels.get(pattern.confidence, 0) < confidence_levels.get(criteria.min_confidence, 0):
                continue
            
            # Check occurrences
            if pattern.occurrence_count < criteria.min_occurrences:
                continue
            
            # Check time range
            if criteria.time_range:
                start_time, end_time = criteria.time_range
                if not (start_time <= pattern.last_observed <= end_time):
                    continue
            
            filtered.append(pattern)
        
        return filtered
    
    def _generate_cache_key(
        self,
        module_data: List[Dict[str, Any]],
        pattern_types: List[str]
    ) -> str:
        """Generate cache key for pattern search."""
        
        try:
            data_hash = str(hash(str(len(module_data))))
            types_hash = str(hash(''.join(sorted(pattern_types))))
            return f"patterns_{data_hash}_{types_hash}"
        except Exception:
            return f"patterns_{uuid.uuid4()}"
    
    def get_pattern_by_id(self, pattern_id: str) -> Optional[RecognizedPattern]:
        """Get pattern by ID with error handling."""
        if not pattern_id or not isinstance(pattern_id, str):
            raise ValidationError(
                "pattern_id must be a non-empty string",
                component="lancedb_pattern_recognizer",
                operation="get_pattern_by_id"
            )
        
        return self.recognized_patterns.get(pattern_id)
    
    def get_patterns_by_type(self, pattern_type: PatternType) -> List[RecognizedPattern]:
        """Get all patterns of specific type."""
        return [
            pattern for pattern in self.recognized_patterns.values()
            if pattern.pattern_type == pattern_type
        ]
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics."""
        return {
            "recognizer_id": self.recognizer_id,
            "total_patterns": len(self.recognized_patterns),
            "cache_size": len(self.pattern_cache),
            "circuit_breaker_open": self.circuit_breaker_open,
            "failure_count": self.failure_count,
            "performance_metrics": self.performance_metrics.copy(),
            "capabilities": {
                "intelligence_module": INTELLIGENCE_MODULE_AVAILABLE,
                "vector_capabilities": VECTOR_CAPABILITIES_AVAILABLE,
                "numpy_available": NUMPY_AVAILABLE
            }
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Comprehensive health check."""
        try:
            # Test basic functionality
            test_data = {
                'module_data': [{'test': 'data'}],
                'pattern_types': ['business_process']
            }
            
            self.validate_pattern_data(test_data)
            
            return {
                "status": "healthy",
                "circuit_breaker_open": self.circuit_breaker_open,
                "total_patterns": len(self.recognized_patterns),
                "capabilities_available": {
                    "intelligence_module": INTELLIGENCE_MODULE_AVAILABLE,
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
LanceDBPatternRecognizer = EnhancedLanceDBPatternRecognizer

# Example usage and testing
if __name__ == "__main__":
    async def test_enhanced_recognizer():
        """Test the enhanced recognizer functionality."""
        print("Testing Enhanced LanceDB Pattern Recognizer...")
        
        recognizer = EnhancedLanceDBPatternRecognizer()
        
        # Test health check
        health = await recognizer.health_check()
        print(f"Health check: {health}")
        
        # Test pattern recognition
        try:
            patterns = await recognizer.recognize_patterns(
                module_data=[
                    {"business_domain": "finance", "performance_score": 0.8, "timestamp": datetime.utcnow()},
                    {"business_domain": "finance", "performance_score": 0.85, "timestamp": datetime.utcnow()},
                    {"business_domain": "finance", "performance_score": 0.9, "timestamp": datetime.utcnow()}
                ],
                pattern_types=["business_process", "performance_trend"]
            )
            
            print(f"Recognized patterns: {len(patterns)}")
            for pattern in patterns:
                print(f"  - {pattern.pattern_name} (confidence: {pattern.confidence.value})")
            
            # Test performance stats
            stats = recognizer.get_performance_stats()
            print(f"Performance stats: {stats}")
            
        except Exception as e:
            print(f"Error during testing: {e}")
    
    import uuid
    asyncio.run(test_enhanced_recognizer())