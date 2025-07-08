#!/usr/bin/env python3
"""
Vector-Enhanced Complexity Analyzer - Phase 1.2 Implementation
Integrates LanceDB vector capabilities with Intelligence module complexity analysis.
Provides historical pattern learning and enhanced time/resource estimation accuracy.
"""

import asyncio
import json
import time
import numpy as np
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field

# Import existing Intelligence module components
try:
    from .complexity_analyzer import ComplexityAnalyzer, ComplexityScore, ComplexityLevel, InvestigationMethodology, ComplexityDimensions
    from .domain_expert import DomainExpert, BusinessIntent, BusinessDomain, AnalysisType
    from .config import settings
    from .intelligence_logging import setup_logger, performance_monitor
    INTELLIGENCE_MODULE_AVAILABLE = True
except ImportError:
    try:
        from complexity_analyzer import ComplexityAnalyzer, ComplexityScore, ComplexityLevel, InvestigationMethodology, ComplexityDimensions
        from domain_expert import DomainExpert, BusinessIntent, BusinessDomain, AnalysisType
        from config import settings
        from intelligence_logging import setup_logger, performance_monitor
        INTELLIGENCE_MODULE_AVAILABLE = True
    except ImportError:
        print("⚠️ Warning: Intelligence module components not available")
        INTELLIGENCE_MODULE_AVAILABLE = False

# Import LanceDB vector infrastructure from Phase 0
try:
    import sys
    from pathlib import Path
    lance_db_path = Path(__file__).parent.parent / "lance_db" / "src"
    sys.path.insert(0, str(lance_db_path))
    
    from enterprise_vector_schema import (
        EnterpriseVectorSchema,
        VectorMetadata,
        ModuleSource,
        BusinessDomain as UnifiedBusinessDomain,
        PerformanceTier,
        AnalysisType as UnifiedAnalysisType,
        generate_vector_id,
        normalize_score_to_unified_scale
    )
    from vector_index_manager import VectorIndexManager
    from vector_performance_monitor import (
        VectorPerformanceMonitor, 
        PerformanceMetric, 
        PerformanceMetricType
    )
    VECTOR_INFRASTRUCTURE_AVAILABLE = True
except ImportError:
    print("⚠️ Warning: Vector infrastructure not available")
    VECTOR_INFRASTRUCTURE_AVAILABLE = False

# Import vector embedding capabilities
try:
    import lancedb
    from sentence_transformers import SentenceTransformer
    VECTOR_CAPABILITIES_AVAILABLE = True
except ImportError:
    print("⚠️ Warning: Vector capabilities not available")
    VECTOR_CAPABILITIES_AVAILABLE = False


@dataclass
class ComplexityPattern:
    """Historical complexity pattern from vector search."""
    query_id: str
    similarity_score: float
    original_query: str
    business_domain: str
    analysis_type: str
    
    # Complexity metrics
    complexity_level: str
    methodology: str
    actual_duration_minutes: Optional[int] = None
    actual_queries: Optional[int] = None
    actual_services: Optional[int] = None
    
    # Pattern analysis
    domain_match: float = 0.0
    methodology_match: float = 0.0
    dimension_correlation: Dict[str, float] = field(default_factory=dict)
    
    # Performance data
    accuracy_score: float = 0.0
    confidence_score: float = 0.0
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass 
class VectorEnhancedComplexityScore:
    """Enhanced complexity score with vector learning and historical patterns."""
    
    # Base complexity score
    base_complexity: ComplexityScore
    
    # Vector enhancement data
    vector_id: str
    historical_patterns: List[ComplexityPattern] = field(default_factory=list)
    pattern_based_adjustments: Dict[str, float] = field(default_factory=dict)
    
    # Enhanced estimates with confidence intervals
    duration_estimate_range: Tuple[int, int] = (0, 0)
    queries_estimate_range: Tuple[int, int] = (0, 0)
    services_estimate_range: Tuple[int, int] = (0, 0)
    estimate_confidence: float = 0.0
    
    # Learning metrics
    pattern_match_quality: float = 0.0
    estimation_accuracy_boost: float = 0.0
    methodology_confidence_boost: float = 0.0
    
    # Performance tracking
    vector_search_time_ms: float = 0.0
    total_enhancement_time_ms: float = 0.0


@dataclass
class ComplexityFeedback:
    """Feedback data for improving complexity estimation accuracy."""
    vector_id: str
    original_estimate: ComplexityScore
    actual_results: Dict[str, Union[int, float]]
    accuracy_metrics: Dict[str, float]
    lessons_learned: List[str]
    feedback_timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class VectorEnhancedComplexityAnalyzer:
    """
    Vector-enhanced complexity analyzer that learns from historical patterns.
    Integrates LanceDB capabilities to improve estimation accuracy over time.
    """
    
    def __init__(self, vector_index_manager: Optional[VectorIndexManager] = None):
        self.logger = setup_logger("vector_enhanced_complexity_analyzer")
        
        # Initialize base complexity analyzer
        if INTELLIGENCE_MODULE_AVAILABLE:
            self.complexity_analyzer = ComplexityAnalyzer()
        else:
            self.complexity_analyzer = None
            self.logger.warning("Base complexity analyzer not available")
        
        # Vector infrastructure
        self.vector_index_manager = vector_index_manager
        self.performance_monitor = None
        self.embedder = None
        self.vector_db = None
        self.vector_table = None
        
        # Pattern learning cache
        self.pattern_cache: Dict[str, List[ComplexityPattern]] = {}
        self.cache_ttl_seconds = 3600  # 1 hour
        
        # Estimation accuracy tracking
        self.accuracy_stats = {
            "total_estimates": 0,
            "pattern_enhanced": 0,
            "accuracy_improvements": 0,
            "duration_accuracy": [],
            "queries_accuracy": [],
            "services_accuracy": []
        }
        
        # Learning parameters
        self.learning_rate = 0.1
        self.min_pattern_similarity = 0.6
        self.max_historical_patterns = 10
        
        self.logger.info("VectorEnhancedComplexityAnalyzer initialized")
    
    async def initialize(self, db_path: str = None, table_name: str = "complexity_vectors"):
        """Initialize vector capabilities and connect to LanceDB."""
        try:
            if not VECTOR_CAPABILITIES_AVAILABLE:
                self.logger.warning("Vector capabilities not available - using base complexity analysis only")
                return
            
            # Initialize embedding model
            self.logger.info("Loading BGE-M3 embedding model...")
            start_time = time.time()
            self.embedder = SentenceTransformer("BAAI/bge-m3", device="cpu")
            load_time = time.time() - start_time
            self.logger.info(f"Embedding model loaded in {load_time:.2f}s")
            
            # Connect to LanceDB
            if db_path:
                self.logger.info(f"Connecting to LanceDB at: {db_path}")
                self.vector_db = await lancedb.connect_async(db_path)
                
                # Setup vector table
                await self._setup_vector_table(table_name)
                
                # Initialize performance monitor
                if VECTOR_INFRASTRUCTURE_AVAILABLE and self.performance_monitor is None:
                    self.performance_monitor = VectorPerformanceMonitor()
                    await self.performance_monitor.establish_baseline(
                        "complexity_analysis", 
                        ModuleSource.INTELLIGENCE
                    )
                
                self.logger.info("✅ Vector capabilities initialized")
            else:
                self.logger.info("No DB path provided - vector learning disabled")
                
        except Exception as e:
            self.logger.error(f"Failed to initialize vector capabilities: {e}")
            self.embedder = None
            self.vector_db = None
    
    async def _setup_vector_table(self, table_name: str):
        """Setup vector table for complexity analysis patterns."""
        try:
            existing_tables = await self.vector_db.table_names()
            
            if table_name in existing_tables:
                self.vector_table = await self.vector_db.open_table(table_name)
                self.logger.info(f"Opened existing vector table: {table_name}")
            else:
                # Create table with enterprise schema if available
                if VECTOR_INFRASTRUCTURE_AVAILABLE:
                    sample_data = EnterpriseVectorSchema.create_sample_data()
                    self.vector_table = await self.vector_db.create_table(
                        table_name,
                        data=sample_data,
                        schema=EnterpriseVectorSchema.get_arrow_schema()
                    )
                    # Remove sample data
                    await self.vector_table.delete("id = 'schema_sample_001'")
                    self.logger.info(f"Created enterprise vector table: {table_name}")
                else:
                    # Fallback to basic table
                    dummy_data = [{
                        "id": "dummy",
                        "vector": np.random.randn(1024).astype(np.float32).tolist(),
                        "content": "dummy content",
                        "complexity_level": "simple",
                        "methodology": "rapid_response"
                    }]
                    self.vector_table = await self.vector_db.create_table(table_name, data=dummy_data)
                    await self.vector_table.delete("id = 'dummy'")
                    self.logger.info(f"Created basic vector table: {table_name}")
                    
        except Exception as e:
            self.logger.error(f"Failed to setup vector table: {e}")
            self.vector_table = None
    
    @performance_monitor("vector_enhanced_complexity_analysis")
    async def analyze_complexity_with_vectors(
        self, 
        business_intent: BusinessIntent, 
        query: str,
        include_historical_patterns: bool = True,
        similarity_threshold: float = 0.6,
        max_patterns: int = 5
    ) -> VectorEnhancedComplexityScore:
        """
        Enhanced complexity analysis using vector similarity and historical patterns.
        
        Args:
            business_intent: Classified business intent from domain expert
            query: Original natural language query
            include_historical_patterns: Whether to search for similar historical patterns
            similarity_threshold: Minimum similarity score for pattern matching
            max_patterns: Maximum number of historical patterns to consider
            
        Returns:
            Vector-enhanced complexity score with pattern-based improvements
        """
        total_start_time = time.perf_counter()
        self.accuracy_stats["total_estimates"] += 1
        
        try:
            # Step 1: Base complexity analysis
            if self.complexity_analyzer:
                base_complexity = self.complexity_analyzer.analyze_complexity(business_intent, query)
            else:
                base_complexity = self._create_fallback_complexity(query)
            
            # Step 2: Vector enhancement
            vector_search_start = time.perf_counter()
            vector_id = generate_vector_id(query, ModuleSource.INTELLIGENCE) if VECTOR_INFRASTRUCTURE_AVAILABLE else f"complexity_{hash(query)}"
            
            historical_patterns = []
            pattern_adjustments = {}
            estimate_confidence = base_complexity.confidence
            
            if self.embedder and self.vector_table and include_historical_patterns:
                # Generate query embedding
                query_embedding = self.embedder.encode(
                    query,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                    show_progress_bar=False
                )
                
                # Search for similar complexity patterns
                historical_patterns = await self._find_similar_complexity_patterns(
                    query_embedding,
                    business_intent,
                    base_complexity,
                    similarity_threshold,
                    max_patterns
                )
                
                # Calculate pattern-based adjustments
                if historical_patterns:
                    pattern_adjustments = self._calculate_pattern_adjustments(
                        base_complexity, 
                        historical_patterns
                    )
                    
                    # Boost confidence based on pattern quality
                    pattern_quality = self._assess_pattern_quality(historical_patterns)
                    estimate_confidence = min(estimate_confidence * (1 + pattern_quality * 0.3), 0.95)
                    
                    self.accuracy_stats["pattern_enhanced"] += 1
            
            vector_search_time = (time.perf_counter() - vector_search_start) * 1000
            
            # Step 3: Enhanced estimates with confidence intervals
            duration_range, queries_range, services_range = self._calculate_enhanced_estimates(
                base_complexity, historical_patterns, pattern_adjustments
            )
            
            # Step 4: Calculate enhancement metrics
            pattern_match_quality = self._calculate_pattern_match_quality(historical_patterns)
            estimation_accuracy_boost = self._calculate_accuracy_boost(historical_patterns)
            methodology_confidence_boost = self._calculate_methodology_boost(
                base_complexity, historical_patterns
            )
            
            total_time = (time.perf_counter() - total_start_time) * 1000
            
            # Step 5: Create enhanced complexity score
            enhanced_score = VectorEnhancedComplexityScore(
                base_complexity=base_complexity,
                vector_id=vector_id,
                historical_patterns=historical_patterns,
                pattern_based_adjustments=pattern_adjustments,
                duration_estimate_range=duration_range,
                queries_estimate_range=queries_range,
                services_estimate_range=services_range,
                estimate_confidence=estimate_confidence,
                pattern_match_quality=pattern_match_quality,
                estimation_accuracy_boost=estimation_accuracy_boost,
                methodology_confidence_boost=methodology_confidence_boost,
                vector_search_time_ms=vector_search_time,
                total_enhancement_time_ms=total_time
            )
            
            # Step 6: Record performance metrics
            if self.performance_monitor:
                await self._record_performance_metrics(enhanced_score)
            
            self.logger.info(
                f"Enhanced complexity analysis: {base_complexity.level.value} "
                f"({base_complexity.methodology.value}) with {len(historical_patterns)} patterns "
                f"in {total_time:.1f}ms"
            )
            
            return enhanced_score
            
        except Exception as e:
            self.logger.error(f"Vector-enhanced complexity analysis failed: {e}")
            # Fallback to base analysis
            if self.complexity_analyzer:
                base_complexity = self.complexity_analyzer.analyze_complexity(business_intent, query)
            else:
                base_complexity = self._create_fallback_complexity(query)
            
            return VectorEnhancedComplexityScore(
                base_complexity=base_complexity,
                vector_id=f"fallback_{hash(query)}",
                total_enhancement_time_ms=(time.perf_counter() - total_start_time) * 1000
            )
    
    async def _find_similar_complexity_patterns(
        self,
        query_embedding: np.ndarray,
        business_intent: BusinessIntent,
        base_complexity: ComplexityScore,
        similarity_threshold: float,
        max_patterns: int
    ) -> List[ComplexityPattern]:
        """Find similar complexity patterns from historical data."""
        try:
            if not self.vector_table:
                return []
            
            # Search for similar vectors - handle async properly
            try:
                search_results = await self.vector_table.search(query_embedding.tolist()).limit(max_patterns * 2).to_list()
            except AttributeError:
                # Fallback for different LanceDB API versions
                search_query = self.vector_table.search(query_embedding.tolist()).limit(max_patterns * 2)
                search_results = search_query.to_pandas().to_dict('records') if hasattr(search_query, 'to_pandas') else []
            
            patterns = []
            for result in search_results:
                similarity = float(result.get("_distance", 0.0))
                similarity_score = 1.0 - similarity  # Convert distance to similarity
                
                if similarity_score >= similarity_threshold:
                    # Extract complexity data from result
                    content_json = json.loads(result.get("content_json", "{}"))
                    module_metadata = json.loads(result.get("module_metadata", "{}"))
                    complexity_data = module_metadata.get("complexity_analysis", {})
                    
                    pattern = ComplexityPattern(
                        query_id=result.get("id", "unknown"),
                        similarity_score=similarity_score,
                        original_query=content_json.get("query", ""),
                        business_domain=result.get("business_domain", "unknown"),
                        analysis_type=result.get("analysis_type", "descriptive"),
                        complexity_level=complexity_data.get("level", "simple"),
                        methodology=complexity_data.get("methodology", "rapid_response"),
                        actual_duration_minutes=complexity_data.get("actual_duration"),
                        actual_queries=complexity_data.get("actual_queries"),
                        actual_services=complexity_data.get("actual_services"),
                        accuracy_score=complexity_data.get("accuracy_score", 0.5),
                        confidence_score=float(result.get("confidence_score", 0.5)),
                        timestamp=datetime.fromisoformat(result.get("created_at", datetime.now().isoformat()))
                    )
                    
                    # Calculate pattern analysis metrics
                    pattern.domain_match = self._calculate_domain_match(business_intent, pattern)
                    pattern.methodology_match = self._calculate_methodology_match(base_complexity, pattern)
                    pattern.dimension_correlation = self._calculate_dimension_correlation(base_complexity, pattern)
                    
                    patterns.append(pattern)
            
            # Sort by combined similarity and relevance score
            patterns.sort(key=lambda p: p.similarity_score * p.domain_match * p.methodology_match, reverse=True)
            return patterns[:max_patterns]
            
        except Exception as e:
            self.logger.warning(f"Failed to find similar complexity patterns: {e}")
            return []
    
    def _calculate_pattern_adjustments(
        self, 
        base_complexity: ComplexityScore, 
        patterns: List[ComplexityPattern]
    ) -> Dict[str, float]:
        """Calculate adjustments to base estimates based on historical patterns."""
        if not patterns:
            return {}
        
        adjustments = {
            "duration_multiplier": 1.0,
            "queries_multiplier": 1.0,
            "services_multiplier": 1.0,
            "confidence_boost": 0.0
        }
        
        # Weight patterns by similarity and relevance
        total_weight = 0.0
        weighted_duration_ratios = []
        weighted_queries_ratios = []
        weighted_services_ratios = []
        
        for pattern in patterns:
            weight = pattern.similarity_score * pattern.domain_match * pattern.methodology_match
            total_weight += weight
            
            # Calculate actual vs estimated ratios if available
            if pattern.actual_duration_minutes:
                # Estimate what base complexity would have predicted for this pattern
                estimated_duration = self._estimate_base_duration(pattern)
                if estimated_duration > 0:
                    ratio = pattern.actual_duration_minutes / estimated_duration
                    weighted_duration_ratios.append((ratio, weight))
            
            if pattern.actual_queries:
                estimated_queries = self._estimate_base_queries(pattern)
                if estimated_queries > 0:
                    ratio = pattern.actual_queries / estimated_queries
                    weighted_queries_ratios.append((ratio, weight))
            
            if pattern.actual_services:
                estimated_services = self._estimate_base_services(pattern)
                if estimated_services > 0:
                    ratio = pattern.actual_services / estimated_services
                    weighted_services_ratios.append((ratio, weight))
        
        # Calculate weighted average adjustments
        if weighted_duration_ratios:
            duration_adjustment = sum(ratio * weight for ratio, weight in weighted_duration_ratios) / total_weight
            adjustments["duration_multiplier"] = min(max(duration_adjustment, 0.5), 2.0)  # Cap adjustments
        
        if weighted_queries_ratios:
            queries_adjustment = sum(ratio * weight for ratio, weight in weighted_queries_ratios) / total_weight
            adjustments["queries_multiplier"] = min(max(queries_adjustment, 0.5), 2.0)
        
        if weighted_services_ratios:
            services_adjustment = sum(ratio * weight for ratio, weight in weighted_services_ratios) / total_weight
            adjustments["services_multiplier"] = min(max(services_adjustment, 0.5), 2.0)
        
        # Confidence boost based on pattern quality
        avg_accuracy = sum(p.accuracy_score for p in patterns) / len(patterns)
        adjustments["confidence_boost"] = avg_accuracy * 0.2  # Max 20% boost
        
        return adjustments
    
    def _calculate_enhanced_estimates(
        self,
        base_complexity: ComplexityScore,
        patterns: List[ComplexityPattern],
        adjustments: Dict[str, float]
    ) -> Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]:
        """Calculate enhanced estimates with confidence intervals."""
        
        # Base estimates
        base_duration = base_complexity.estimated_duration_minutes
        base_queries = base_complexity.estimated_queries
        base_services = base_complexity.estimated_services
        
        # Apply pattern-based adjustments
        adjusted_duration = int(base_duration * adjustments.get("duration_multiplier", 1.0))
        adjusted_queries = int(base_queries * adjustments.get("queries_multiplier", 1.0))
        adjusted_services = int(base_services * adjustments.get("services_multiplier", 1.0))
        
        # Calculate confidence intervals based on pattern variance
        if patterns:
            duration_variance = self._calculate_estimate_variance([p.actual_duration_minutes for p in patterns if p.actual_duration_minutes])
            queries_variance = self._calculate_estimate_variance([p.actual_queries for p in patterns if p.actual_queries])
            services_variance = self._calculate_estimate_variance([p.actual_services for p in patterns if p.actual_services])
        else:
            # Default uncertainty ranges
            duration_variance = 0.3
            queries_variance = 0.25
            services_variance = 0.2
        
        # Create confidence intervals
        duration_range = (
            max(1, int(adjusted_duration * (1 - duration_variance))),
            int(adjusted_duration * (1 + duration_variance))
        )
        
        queries_range = (
            max(1, int(adjusted_queries * (1 - queries_variance))),
            int(adjusted_queries * (1 + queries_variance))
        )
        
        services_range = (
            max(1, int(adjusted_services * (1 - services_variance))),
            int(adjusted_services * (1 + services_variance))
        )
        
        return duration_range, queries_range, services_range
    
    def _calculate_estimate_variance(self, values: List[Optional[int]]) -> float:
        """Calculate variance in historical estimates for confidence intervals."""
        valid_values = [v for v in values if v is not None and v > 0]
        if len(valid_values) < 2:
            return 0.3  # Default 30% variance
        
        mean_val = sum(valid_values) / len(valid_values)
        variance = sum((v - mean_val) ** 2 for v in valid_values) / len(valid_values)
        std_dev = (variance ** 0.5)
        
        # Return coefficient of variation, capped at 50%
        return min(std_dev / mean_val if mean_val > 0 else 0.3, 0.5)
    
    def _calculate_domain_match(self, intent: BusinessIntent, pattern: ComplexityPattern) -> float:
        """Calculate domain alignment between intent and pattern."""
        if pattern.business_domain == intent.primary_domain.value:
            return 1.0
        elif pattern.business_domain in [d.value for d in intent.secondary_domains]:
            return 0.7
        else:
            return 0.3
    
    def _calculate_methodology_match(self, complexity: ComplexityScore, pattern: ComplexityPattern) -> float:
        """Calculate methodology similarity between current and pattern."""
        if pattern.methodology == complexity.methodology.value:
            return 1.0
        elif pattern.complexity_level == complexity.level.value:
            return 0.8
        else:
            return 0.5
    
    def _calculate_dimension_correlation(self, complexity: ComplexityScore, pattern: ComplexityPattern) -> Dict[str, float]:
        """Calculate correlation between complexity dimensions and pattern."""
        # Simplified correlation calculation
        return {
            "overall_correlation": 0.7,  # Placeholder
            "domain_correlation": self._calculate_domain_match(None, pattern) if hasattr(pattern, 'business_domain') else 0.5,
            "complexity_correlation": 1.0 if pattern.complexity_level == complexity.level.value else 0.5
        }
    
    def _assess_pattern_quality(self, patterns: List[ComplexityPattern]) -> float:
        """Assess overall quality of matched patterns."""
        if not patterns:
            return 0.0
        
        # Quality factors
        avg_similarity = sum(p.similarity_score for p in patterns) / len(patterns)
        avg_accuracy = sum(p.accuracy_score for p in patterns) / len(patterns)
        avg_confidence = sum(p.confidence_score for p in patterns) / len(patterns)
        
        # Recent patterns are more valuable
        recent_weight = sum(1.0 if (datetime.now(timezone.utc) - p.timestamp).days < 30 else 0.5 for p in patterns) / len(patterns)
        
        return (avg_similarity * 0.3 + avg_accuracy * 0.4 + avg_confidence * 0.2 + recent_weight * 0.1)
    
    def _calculate_pattern_match_quality(self, patterns: List[ComplexityPattern]) -> float:
        """Calculate overall pattern match quality score."""
        return self._assess_pattern_quality(patterns)
    
    def _calculate_accuracy_boost(self, patterns: List[ComplexityPattern]) -> float:
        """Calculate expected accuracy improvement from patterns."""
        if not patterns:
            return 0.0
        
        avg_accuracy = sum(p.accuracy_score for p in patterns) / len(patterns)
        return min(avg_accuracy * 0.25, 0.3)  # Max 30% boost
    
    def _calculate_methodology_boost(self, complexity: ComplexityScore, patterns: List[ComplexityPattern]) -> float:
        """Calculate methodology confidence boost from patterns."""
        if not patterns:
            return 0.0
        
        methodology_matches = sum(1 for p in patterns if p.methodology == complexity.methodology.value)
        match_rate = methodology_matches / len(patterns)
        return match_rate * 0.15  # Max 15% boost
    
    def _estimate_base_duration(self, pattern: ComplexityPattern) -> int:
        """Estimate what base complexity analyzer would predict for this pattern."""
        # Simplified estimation based on complexity level
        level_estimates = {
            "simple": 3,
            "analytical": 10,
            "computational": 30,
            "investigative": 75
        }
        return level_estimates.get(pattern.complexity_level, 10)
    
    def _estimate_base_queries(self, pattern: ComplexityPattern) -> int:
        """Estimate base queries for pattern."""
        level_estimates = {
            "simple": 2,
            "analytical": 5,
            "computational": 14,
            "investigative": 32
        }
        return level_estimates.get(pattern.complexity_level, 5)
    
    def _estimate_base_services(self, pattern: ComplexityPattern) -> int:
        """Estimate base services for pattern."""
        level_estimates = {
            "simple": 1,
            "analytical": 2,
            "computational": 3,
            "investigative": 4
        }
        return level_estimates.get(pattern.complexity_level, 2)
    
    async def store_complexity_analysis(self, enhanced_score: VectorEnhancedComplexityScore, query: str) -> bool:
        """Store complexity analysis in vector database for future learning."""
        try:
            if not self.vector_table or not self.embedder:
                return False
            
            # Generate embedding for the query
            query_embedding = self.embedder.encode(
                query,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False
            )
            
            # Create vector record
            if VECTOR_INFRASTRUCTURE_AVAILABLE:
                # Use enterprise schema
                now = datetime.now(timezone.utc)
                vector_metadata = VectorMetadata(
                    id=enhanced_score.vector_id,
                    module_source=ModuleSource.INTELLIGENCE,
                    created_at=now,
                    updated_at=now,
                    
                    # Map complexity scores to unified scale
                    complexity_score=enhanced_score.base_complexity.score,
                    confidence_score=enhanced_score.estimate_confidence,
                    actionability_score=0.8,  # Default for complexity analysis
                    business_value_score=0.7,  # Default for complexity analysis
                    
                    # Map to unified enums
                    business_domain=self._map_to_unified_domain(enhanced_score.base_complexity.level),
                    performance_tier=self._map_to_performance_tier(enhanced_score.base_complexity.level),
                    analysis_type=UnifiedAnalysisType.DIAGNOSTIC,  # Complexity analysis is diagnostic
                    
                    # Module-specific metadata
                    module_metadata={
                        "complexity_analysis": {
                            "level": enhanced_score.base_complexity.level.value,
                            "methodology": enhanced_score.base_complexity.methodology.value,
                            "estimated_duration": enhanced_score.base_complexity.estimated_duration_minutes,
                            "estimated_queries": enhanced_score.base_complexity.estimated_queries,
                            "estimated_services": enhanced_score.base_complexity.estimated_services,
                            "dimension_scores": enhanced_score.base_complexity.dimension_scores,
                            "risk_factors": enhanced_score.base_complexity.risk_factors,
                            "resource_requirements": enhanced_score.base_complexity.resource_requirements,
                            "pattern_adjustments": enhanced_score.pattern_based_adjustments,
                            "estimate_confidence": enhanced_score.estimate_confidence,
                            "pattern_match_quality": enhanced_score.pattern_match_quality,
                            "accuracy_boost": enhanced_score.estimation_accuracy_boost
                        }
                    }
                )
                
                record = vector_metadata.to_lancedb_record(
                    query_embedding.tolist(),
                    {"query": query}
                )
            else:
                # Fallback to basic record
                record = {
                    "id": enhanced_score.vector_id,
                    "vector": query_embedding.tolist(),
                    "content": query,
                    "complexity_level": enhanced_score.base_complexity.level.value,
                    "methodology": enhanced_score.base_complexity.methodology.value,
                    "estimated_duration": enhanced_score.base_complexity.estimated_duration_minutes,
                    "confidence": enhanced_score.estimate_confidence,
                    "created_at": datetime.now(timezone.utc)
                }
            
            # Store in vector table
            await self.vector_table.add([record])
            
            self.logger.info(f"Stored complexity analysis: {enhanced_score.vector_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to store complexity analysis: {e}")
            return False
    
    async def record_complexity_feedback(self, feedback: ComplexityFeedback) -> bool:
        """Record feedback on complexity estimation accuracy for learning."""
        try:
            # Update accuracy statistics
            actual_duration = feedback.actual_results.get("duration_minutes")
            actual_queries = feedback.actual_results.get("queries_count")
            actual_services = feedback.actual_results.get("services_count")
            
            estimated_duration = feedback.original_estimate.estimated_duration_minutes
            estimated_queries = feedback.original_estimate.estimated_queries
            estimated_services = feedback.original_estimate.estimated_services
            
            # Calculate accuracy metrics
            if actual_duration and estimated_duration:
                duration_accuracy = 1.0 - abs(actual_duration - estimated_duration) / max(actual_duration, estimated_duration)
                self.accuracy_stats["duration_accuracy"].append(duration_accuracy)
            
            if actual_queries and estimated_queries:
                queries_accuracy = 1.0 - abs(actual_queries - estimated_queries) / max(actual_queries, estimated_queries)
                self.accuracy_stats["queries_accuracy"].append(queries_accuracy)
            
            if actual_services and estimated_services:
                services_accuracy = 1.0 - abs(actual_services - estimated_services) / max(actual_services, estimated_services)
                self.accuracy_stats["services_accuracy"].append(services_accuracy)
            
            # Update vector record with actual results if available
            if self.vector_table:
                try:
                    # Note: LanceDB update operations would go here
                    # For now, we log the feedback for future implementation
                    self.logger.info(f"Recorded feedback for {feedback.vector_id}: {feedback.accuracy_metrics}")
                except Exception as e:
                    self.logger.warning(f"Failed to update vector record: {e}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to record complexity feedback: {e}")
            return False
    
    def _map_to_unified_domain(self, complexity_level: ComplexityLevel) -> UnifiedBusinessDomain:
        """Map complexity level to unified business domain (simplified mapping)."""
        if not VECTOR_INFRASTRUCTURE_AVAILABLE:
            return None
        
        # Simplified mapping - in practice this would be more sophisticated
        return UnifiedBusinessDomain.OPERATIONS
    
    def _map_to_performance_tier(self, complexity_level: ComplexityLevel) -> PerformanceTier:
        """Map complexity level to performance tier."""
        if not VECTOR_INFRASTRUCTURE_AVAILABLE:
            return None
        
        mapping = {
            ComplexityLevel.SIMPLE: PerformanceTier.SIMPLE,
            ComplexityLevel.ANALYTICAL: PerformanceTier.ANALYTICAL,
            ComplexityLevel.COMPUTATIONAL: PerformanceTier.COMPUTATIONAL,
            ComplexityLevel.INVESTIGATIVE: PerformanceTier.COMPUTATIONAL  # Map to closest available
        }
        return mapping.get(complexity_level, PerformanceTier.ANALYTICAL)
    
    def _create_fallback_complexity(self, query: str) -> ComplexityScore:
        """Create fallback complexity score when base analyzer unavailable."""
        # Very basic fallback implementation
        query_length = len(query.split())
        
        if query_length > 20:
            level = ComplexityLevel.COMPUTATIONAL
            methodology = InvestigationMethodology.SCENARIO_MODELING
            duration = 25
            queries = 12
            services = 3
        elif query_length > 10:
            level = ComplexityLevel.ANALYTICAL
            methodology = InvestigationMethodology.SYSTEMATIC_ANALYSIS
            duration = 8
            queries = 4
            services = 2
        else:
            level = ComplexityLevel.SIMPLE
            methodology = InvestigationMethodology.RAPID_RESPONSE
            duration = 3
            queries = 2
            services = 1
        
        return ComplexityScore(
            level=level,
            methodology=methodology,
            score=0.5,
            dimension_scores={"overall": 0.5},
            estimated_duration_minutes=duration,
            estimated_queries=queries,
            estimated_services=services,
            confidence=0.3,
            risk_factors=["Fallback analysis - limited accuracy"],
            resource_requirements={"fallback": True}
        )
    
    async def _record_performance_metrics(self, enhanced_score: VectorEnhancedComplexityScore):
        """Record performance metrics for monitoring."""
        try:
            if not self.performance_monitor:
                return
            
            # Complexity analysis latency metric
            latency_metric = PerformanceMetric(
                metric_type=PerformanceMetricType.QUERY_LATENCY,
                value=enhanced_score.total_enhancement_time_ms,
                timestamp=datetime.now(timezone.utc),
                module_source=ModuleSource.INTELLIGENCE,
                context={
                    "operation": "vector_enhanced_complexity_analysis",
                    "patterns_found": len(enhanced_score.historical_patterns),
                    "accuracy_boost": enhanced_score.estimation_accuracy_boost,
                    "complexity_level": enhanced_score.base_complexity.level.value
                }
            )
            self.performance_monitor.record_metric(latency_metric)
            
        except Exception as e:
            self.logger.warning(f"Failed to record performance metrics: {e}")
    
    async def get_complexity_statistics(self) -> Dict[str, Any]:
        """Get complexity analysis performance statistics."""
        stats = self.accuracy_stats.copy()
        
        # Calculate accuracy averages
        if stats["duration_accuracy"]:
            stats["avg_duration_accuracy"] = sum(stats["duration_accuracy"]) / len(stats["duration_accuracy"])
        if stats["queries_accuracy"]:
            stats["avg_queries_accuracy"] = sum(stats["queries_accuracy"]) / len(stats["queries_accuracy"])
        if stats["services_accuracy"]:
            stats["avg_services_accuracy"] = sum(stats["services_accuracy"]) / len(stats["services_accuracy"])
        
        # Calculate rates
        total = stats["total_estimates"]
        if total > 0:
            stats["pattern_enhancement_rate"] = stats["pattern_enhanced"] / total
            stats["accuracy_improvement_rate"] = stats["accuracy_improvements"] / total
        
        # Add capability status
        stats["capabilities"] = {
            "base_analyzer_available": self.complexity_analyzer is not None,
            "vector_infrastructure": VECTOR_INFRASTRUCTURE_AVAILABLE,
            "vector_capabilities": VECTOR_CAPABILITIES_AVAILABLE,
            "embedder_loaded": self.embedder is not None,
            "vector_table_available": self.vector_table is not None
        }
        
        return stats
    
    async def cleanup(self):
        """Cleanup vector-enhanced complexity analyzer resources."""
        try:
            # Clear pattern cache
            self.pattern_cache.clear()
            
            # Close vector connections if available
            if hasattr(self, 'vector_db') and self.vector_db:
                # LanceDB connections are typically auto-managed
                pass
            
            self.logger.info("VectorEnhancedComplexityAnalyzer cleanup completed")
            
        except Exception as e:
            self.logger.error(f"Cleanup error: {e}")


# Utility functions for external integration
async def create_vector_enhanced_complexity_analyzer(
    vector_index_manager: Optional[VectorIndexManager] = None,
    db_path: str = None
) -> VectorEnhancedComplexityAnalyzer:
    """Factory function to create and initialize vector-enhanced complexity analyzer."""
    analyzer = VectorEnhancedComplexityAnalyzer(vector_index_manager)
    await analyzer.initialize(db_path)
    return analyzer


# Export main classes and functions
__all__ = [
    "VectorEnhancedComplexityAnalyzer",
    "VectorEnhancedComplexityScore",
    "ComplexityPattern",
    "ComplexityFeedback",
    "create_vector_enhanced_complexity_analyzer"
]