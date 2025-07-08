#!/usr/bin/env python3
"""
Vector-Enhanced Investigator - Phase 2.1 Implementation
Integrates LanceDB vector capabilities with Investigation module for enhanced autonomous analysis.
Provides semantic query understanding, historical pattern learning, and cross-module intelligence.
"""

import asyncio
import json
import time
import numpy as np
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field, asdict
import uuid

# Import existing Investigation module components
try:
    from .runner import (
        AutonomousInvestigationEngine, 
        InvestigationResults, 
        InvestigationStep,
        conduct_autonomous_investigation
    )
    from .config import settings, runtime_config
    from .investigation_logging import InvestigationLogger
    from .prompts import InvestigationPrompts
    INVESTIGATION_MODULE_AVAILABLE = True
except ImportError:
    try:
        from runner import (
            AutonomousInvestigationEngine, 
            InvestigationResults, 
            InvestigationStep,
            conduct_autonomous_investigation
        )
        from config import settings, runtime_config
        from investigation_logging import InvestigationLogger
        from prompts import InvestigationPrompts
        INVESTIGATION_MODULE_AVAILABLE = True
    except ImportError:
        print("⚠️ Warning: Investigation module components not available")
        INVESTIGATION_MODULE_AVAILABLE = False

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
class InvestigationPattern:
    """Historical investigation pattern from vector search."""
    pattern_id: str
    investigation_id: str
    similarity_score: float
    
    # Investigation metadata
    investigation_request: str
    investigation_type: str
    business_domain: str
    complexity_level: str
    
    # Performance data
    total_duration_seconds: float
    overall_confidence: float
    completed_steps: int
    success_status: str
    
    # Step optimization data
    step_sequence: List[str]
    step_durations: Dict[str, float]
    step_confidences: Dict[str, float]
    
    # Business outcomes
    actionability_score: float
    business_value_delivered: float
    user_satisfaction_score: float
    
    # Pattern analysis
    request_similarity: float
    domain_match: float
    methodology_alignment: float
    
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class VectorEnhancedInvestigationResults:
    """Enhanced investigation results with vector learning and pattern insights."""
    
    # Base investigation results
    base_results: InvestigationResults
    
    # Vector enhancement data
    vector_id: str
    similar_investigations: List[InvestigationPattern] = field(default_factory=list)
    pattern_insights: Dict[str, Any] = field(default_factory=dict)
    
    # Enhanced confidence metrics
    confidence_boost: float = 0.0
    pattern_based_confidence: float = 0.0
    cross_module_validation_score: float = 0.0
    
    # Optimization insights
    suggested_step_optimizations: List[Dict[str, Any]] = field(default_factory=list)
    predicted_duration_minutes: float = 0.0
    estimated_success_probability: float = 0.0
    
    # Cross-module intelligence
    related_queries_from_other_modules: List[Dict[str, Any]] = field(default_factory=list)
    cross_module_patterns: List[str] = field(default_factory=list)
    
    # Performance tracking
    vector_search_time_ms: float = 0.0
    pattern_analysis_time_ms: float = 0.0
    total_enhancement_time_ms: float = 0.0


@dataclass
class StepOptimizationInsight:
    """Optimization insight for investigation steps based on patterns."""
    step_name: str
    recommended_approach: str
    expected_duration_seconds: float
    confidence_improvement: float
    based_on_patterns: List[str]
    optimization_rationale: str


class VectorEnhancedInvestigator(AutonomousInvestigationEngine):
    """
    Vector-enhanced investigation engine that learns from historical patterns.
    Integrates LanceDB capabilities to improve investigation quality and efficiency.
    """
    
    def __init__(self, vector_index_manager: Optional[VectorIndexManager] = None):
        super().__init__()
        self.vector_logger = InvestigationLogger("vector_enhanced_investigator")
        
        # Vector infrastructure
        self.vector_index_manager = vector_index_manager
        self.performance_monitor = None
        self.embedder = None
        self.vector_db = None
        self.vector_table = None
        
        # Pattern learning state
        self.investigation_patterns: Dict[str, InvestigationPattern] = {}
        self.pattern_cache_ttl_seconds = 3600  # 1 hour
        
        # Performance tracking
        self.vector_enhancement_stats = {
            "total_investigations": 0,
            "pattern_enhanced": 0,
            "step_optimizations_applied": 0,
            "cross_module_discoveries": 0,
            "avg_confidence_improvement": []
        }
        
        # Learning parameters
        self.min_similarity_threshold = 0.7
        self.max_historical_patterns = 10
        self.confidence_boost_factor = 0.15
        
        self.vector_logger.logger.info("VectorEnhancedInvestigator initialized")
    
    async def initialize(self, db_path: str = None, table_name: str = "investigation_vectors"):
        """Initialize vector capabilities and connect to LanceDB."""
        try:
            if not VECTOR_CAPABILITIES_AVAILABLE:
                self.vector_logger.logger.warning("Vector capabilities not available - using base investigation only")
                return
            
            # Initialize embedding model
            self.vector_logger.logger.info("Loading BGE-M3 embedding model...")
            start_time = time.time()
            self.embedder = SentenceTransformer("BAAI/bge-m3", device="cpu")
            load_time = time.time() - start_time
            self.vector_logger.logger.info(f"Embedding model loaded in {load_time:.2f}s")
            
            # Connect to LanceDB
            if db_path:
                self.vector_logger.logger.info(f"Connecting to LanceDB at: {db_path}")
                self.vector_db = await lancedb.connect_async(db_path)
                
                # Setup vector table
                await self._setup_vector_table(table_name)
                
                # Initialize performance monitor
                if VECTOR_INFRASTRUCTURE_AVAILABLE and self.performance_monitor is None:
                    self.performance_monitor = VectorPerformanceMonitor()
                    await self.performance_monitor.establish_baseline(
                        "investigation_execution", 
                        ModuleSource.INVESTIGATION
                    )
                
                self.vector_logger.logger.info("✅ Vector capabilities initialized")
            else:
                self.vector_logger.logger.info("No DB path provided - vector enhancement disabled")
                
        except Exception as e:
            self.vector_logger.logger.error(f"Failed to initialize vector capabilities: {e}")
            self.embedder = None
            self.vector_db = None
    
    async def _setup_vector_table(self, table_name: str):
        """Setup vector table for investigation patterns."""
        try:
            existing_tables = await self.vector_db.table_names()
            
            if table_name in existing_tables:
                self.vector_table = await self.vector_db.open_table(table_name)
                self.vector_logger.logger.info(f"Opened existing vector table: {table_name}")
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
                    self.vector_logger.logger.info(f"Created enterprise vector table: {table_name}")
                else:
                    # Fallback to basic table
                    dummy_data = [{
                        "id": "dummy",
                        "vector": np.random.randn(1024).astype(np.float32).tolist(),
                        "content": "dummy content",
                        "investigation_type": "diagnostic",
                        "confidence": 0.5
                    }]
                    self.vector_table = await self.vector_db.create_table(table_name, data=dummy_data)
                    await self.vector_table.delete("id = 'dummy'")
                    self.vector_logger.logger.info(f"Created basic vector table: {table_name}")
                    
        except Exception as e:
            self.vector_logger.logger.error(f"Failed to setup vector table: {e}")
            self.vector_table = None
    
    async def conduct_investigation(
        self,
        coordinated_services: Dict[str, Any],
        investigation_request: str,
        execution_context: Dict[str, Any],
        mcp_client_manager=None,
        use_vector_enhancement: bool = True
    ) -> VectorEnhancedInvestigationResults:
        """
        Enhanced investigation with vector pattern learning and optimization.
        
        Args:
            coordinated_services: Database services from orchestration
            investigation_request: Business question to investigate
            execution_context: Investigation parameters and context
            mcp_client_manager: MCP client manager for database access
            use_vector_enhancement: Whether to use vector capabilities
            
        Returns:
            VectorEnhancedInvestigationResults with pattern insights
        """
        enhancement_start_time = time.perf_counter()
        self.vector_enhancement_stats["total_investigations"] += 1
        
        try:
            # Step 1: Generate investigation embedding and find similar patterns
            similar_patterns = []
            pattern_insights = {}
            
            if use_vector_enhancement and self.embedder and self.vector_table:
                # Generate embedding for investigation request
                request_embedding = self.embedder.encode(
                    investigation_request,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                    show_progress_bar=False
                )
                
                # Search for similar historical investigations
                vector_search_start = time.perf_counter()
                similar_patterns = await self._find_similar_investigations(
                    request_embedding,
                    investigation_request,
                    execution_context
                )
                vector_search_time = (time.perf_counter() - vector_search_start) * 1000
                
                # Analyze patterns for insights
                pattern_analysis_start = time.perf_counter()
                pattern_insights = await self._analyze_investigation_patterns(
                    similar_patterns,
                    investigation_request,
                    execution_context
                )
                pattern_analysis_time = (time.perf_counter() - pattern_analysis_start) * 1000
                
                # Apply pattern-based optimizations to execution context
                if pattern_insights.get("step_optimizations"):
                    execution_context["vector_optimizations"] = pattern_insights["step_optimizations"]
                
                self.vector_enhancement_stats["pattern_enhanced"] += 1
            else:
                vector_search_time = 0.0
                pattern_analysis_time = 0.0
            
            # Step 2: Execute base investigation with potential optimizations
            base_results = await super().conduct_investigation(
                coordinated_services=coordinated_services,
                investigation_request=investigation_request,
                execution_context=execution_context,
                mcp_client_manager=mcp_client_manager
            )
            
            # Step 3: Calculate enhanced confidence metrics
            confidence_boost = self._calculate_confidence_boost(base_results, similar_patterns)
            pattern_based_confidence = self._calculate_pattern_based_confidence(
                base_results, similar_patterns
            )
            
            # Step 4: Generate cross-module insights
            cross_module_queries = await self._find_cross_module_queries(
                investigation_request, request_embedding if use_vector_enhancement else None
            )
            
            # Step 5: Create enhanced results
            vector_id = generate_vector_id(
                investigation_request, 
                ModuleSource.INVESTIGATION
            ) if VECTOR_INFRASTRUCTURE_AVAILABLE else f"investigation_{uuid.uuid4()}"
            
            total_enhancement_time = (time.perf_counter() - enhancement_start_time) * 1000
            
            enhanced_results = VectorEnhancedInvestigationResults(
                base_results=base_results,
                vector_id=vector_id,
                similar_investigations=similar_patterns,
                pattern_insights=pattern_insights,
                confidence_boost=confidence_boost,
                pattern_based_confidence=pattern_based_confidence,
                cross_module_validation_score=self._calculate_cross_module_validation(
                    cross_module_queries
                ),
                suggested_step_optimizations=pattern_insights.get("step_optimizations", []),
                predicted_duration_minutes=pattern_insights.get("predicted_duration", 0.0) / 60,
                estimated_success_probability=pattern_insights.get("success_probability", 0.0),
                related_queries_from_other_modules=cross_module_queries,
                cross_module_patterns=pattern_insights.get("cross_module_patterns", []),
                vector_search_time_ms=vector_search_time,
                pattern_analysis_time_ms=pattern_analysis_time,
                total_enhancement_time_ms=total_enhancement_time
            )
            
            # Step 6: Store enhanced investigation for future learning
            if use_vector_enhancement and self.vector_table:
                await self._store_enhanced_investigation(enhanced_results, request_embedding)
            
            # Step 7: Record performance metrics
            if self.performance_monitor:
                await self._record_performance_metrics(enhanced_results)
            
            self.vector_logger.logger.info(
                f"Enhanced investigation completed: {base_results.status} "
                f"confidence: {base_results.overall_confidence:.3f} "
                f"(+{confidence_boost:.3f} boost) in {total_enhancement_time:.1f}ms"
            )
            
            return enhanced_results
            
        except Exception as e:
            self.vector_logger.logger.error(f"Vector-enhanced investigation failed: {e}")
            # Fallback to base investigation
            base_results = await super().conduct_investigation(
                coordinated_services=coordinated_services,
                investigation_request=investigation_request,
                execution_context=execution_context,
                mcp_client_manager=mcp_client_manager
            )
            
            return VectorEnhancedInvestigationResults(
                base_results=base_results,
                vector_id=f"fallback_{uuid.uuid4()}",
                total_enhancement_time_ms=(time.perf_counter() - enhancement_start_time) * 1000
            )
    
    async def _find_similar_investigations(
        self,
        request_embedding: np.ndarray,
        investigation_request: str,
        execution_context: Dict[str, Any]
    ) -> List[InvestigationPattern]:
        """Find similar historical investigations using vector search."""
        try:
            if not self.vector_table:
                return []
            
            # Search for similar vectors - handle async properly
            try:
                search_results = await self.vector_table.search(
                    request_embedding.tolist()
                ).limit(self.max_historical_patterns * 2).to_list()
            except AttributeError:
                # Fallback for different LanceDB API versions
                search_query = self.vector_table.search(request_embedding.tolist()).limit(
                    self.max_historical_patterns * 2
                )
                search_results = search_query.to_pandas().to_dict('records') if hasattr(
                    search_query, 'to_pandas'
                ) else []
            
            patterns = []
            for result in search_results:
                similarity = float(result.get("_distance", 0.0))
                similarity_score = 1.0 - similarity  # Convert distance to similarity
                
                if similarity_score >= self.min_similarity_threshold:
                    # Extract investigation metadata
                    content_json = json.loads(result.get("content_json", "{}"))
                    module_metadata = json.loads(result.get("module_metadata", "{}"))
                    investigation_data = module_metadata.get("investigation", {})
                    
                    pattern = InvestigationPattern(
                        pattern_id=result.get("id", "unknown"),
                        investigation_id=investigation_data.get("investigation_id", "unknown"),
                        similarity_score=similarity_score,
                        investigation_request=content_json.get("investigation_request", ""),
                        investigation_type=investigation_data.get("investigation_type", "unknown"),
                        business_domain=result.get("business_domain", "unknown"),
                        complexity_level=investigation_data.get("complexity_level", "unknown"),
                        total_duration_seconds=investigation_data.get("total_duration_seconds", 0.0),
                        overall_confidence=float(result.get("confidence_score", 0.5)),
                        completed_steps=investigation_data.get("completed_steps", 0),
                        success_status=investigation_data.get("status", "unknown"),
                        step_sequence=investigation_data.get("step_sequence", []),
                        step_durations=investigation_data.get("step_durations", {}),
                        step_confidences=investigation_data.get("step_confidences", {}),
                        actionability_score=float(result.get("actionability_score", 0.5)),
                        business_value_delivered=float(result.get("business_value_score", 0.5)),
                        user_satisfaction_score=investigation_data.get("user_satisfaction", 0.7),
                        request_similarity=similarity_score,
                        domain_match=self._calculate_domain_match(
                            execution_context.get("business_domain", ""),
                            result.get("business_domain", "")
                        ),
                        methodology_alignment=self._calculate_methodology_alignment(
                            investigation_data.get("investigation_type", ""),
                            self._classify_investigation_type(investigation_request)
                        ),
                        timestamp=datetime.fromisoformat(
                            result.get("created_at", datetime.now().isoformat())
                        )
                    )
                    patterns.append(pattern)
            
            # Sort by combined score (similarity + domain match + success)
            patterns.sort(
                key=lambda p: (
                    p.similarity_score * 0.4 + 
                    p.domain_match * 0.3 + 
                    p.overall_confidence * 0.3
                ),
                reverse=True
            )
            
            return patterns[:self.max_historical_patterns]
            
        except Exception as e:
            self.vector_logger.logger.warning(f"Failed to find similar investigations: {e}")
            return []
    
    async def _analyze_investigation_patterns(
        self,
        patterns: List[InvestigationPattern],
        investigation_request: str,
        execution_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze patterns to generate optimization insights."""
        if not patterns:
            return {}
        
        insights = {
            "pattern_count": len(patterns),
            "avg_similarity": sum(p.similarity_score for p in patterns) / len(patterns),
            "step_optimizations": [],
            "predicted_duration": 0.0,
            "success_probability": 0.0,
            "cross_module_patterns": []
        }
        
        # Analyze successful patterns for step optimizations
        successful_patterns = [p for p in patterns if p.success_status == "completed" and p.overall_confidence > 0.7]
        
        if successful_patterns:
            # Calculate predicted duration based on similar investigations
            duration_weights = []
            weighted_durations = []
            
            for pattern in successful_patterns:
                weight = pattern.similarity_score * pattern.overall_confidence
                duration_weights.append(weight)
                weighted_durations.append(pattern.total_duration_seconds * weight)
            
            if duration_weights:
                insights["predicted_duration"] = sum(weighted_durations) / sum(duration_weights)
            
            # Calculate success probability
            insights["success_probability"] = sum(
                p.overall_confidence * p.similarity_score for p in successful_patterns
            ) / len(successful_patterns)
            
            # Analyze step sequences for optimization
            step_insights = self._analyze_step_sequences(successful_patterns)
            insights["step_optimizations"] = step_insights
            
            # Identify cross-module patterns
            unique_domains = set(p.business_domain for p in patterns)
            if len(unique_domains) > 1:
                insights["cross_module_patterns"] = list(unique_domains)
        
        return insights
    
    def _analyze_step_sequences(self, patterns: List[InvestigationPattern]) -> List[Dict[str, Any]]:
        """Analyze step sequences from successful patterns for optimization."""
        step_optimizations = []
        
        # Aggregate step performance across patterns
        step_stats = {}
        
        for pattern in patterns:
            for step_name, duration in pattern.step_durations.items():
                if step_name not in step_stats:
                    step_stats[step_name] = {
                        "durations": [],
                        "confidences": [],
                        "occurrences": 0
                    }
                
                step_stats[step_name]["durations"].append(duration)
                step_stats[step_name]["confidences"].append(
                    pattern.step_confidences.get(step_name, 0.5)
                )
                step_stats[step_name]["occurrences"] += 1
        
        # Generate optimization insights for frequently successful steps
        for step_name, stats in step_stats.items():
            if stats["occurrences"] >= len(patterns) * 0.5:  # Step appears in >50% of patterns
                avg_duration = sum(stats["durations"]) / len(stats["durations"])
                avg_confidence = sum(stats["confidences"]) / len(stats["confidences"])
                
                optimization = {
                    "step_name": step_name,
                    "recommended_approach": f"Pattern-optimized approach based on {stats['occurrences']} successful cases",
                    "expected_duration_seconds": avg_duration,
                    "confidence_improvement": avg_confidence - 0.5,  # Improvement over baseline
                    "based_on_patterns": [p.pattern_id for p in patterns[:3]],
                    "optimization_rationale": f"Historical success rate: {avg_confidence:.2f}"
                }
                
                step_optimizations.append(optimization)
        
        return step_optimizations
    
    async def _find_cross_module_queries(
        self,
        investigation_request: str,
        request_embedding: Optional[np.ndarray] = None
    ) -> List[Dict[str, Any]]:
        """Find related queries from other modules for cross-module intelligence."""
        cross_module_queries = []
        
        try:
            if not self.vector_db or request_embedding is None:
                return []
            
            # Search across different module tables
            module_tables = {
                "intelligence": "intelligence_vectors",
                "complexity": "complexity_vectors",
                "moq": "moq_vectors",
                "auto_generation": "auto_generation_vectors"
            }
            
            for module_name, table_name in module_tables.items():
                if module_name == "investigation":  # Skip own module
                    continue
                
                try:
                    # Check if table exists
                    existing_tables = await self.vector_db.table_names()
                    if table_name not in existing_tables:
                        continue
                    
                    # Open table and search
                    module_table = await self.vector_db.open_table(table_name)
                    
                    # Search for similar vectors
                    try:
                        search_results = await module_table.search(
                            request_embedding.tolist()
                        ).limit(3).to_list()
                    except AttributeError:
                        search_query = module_table.search(request_embedding.tolist()).limit(3)
                        search_results = search_query.to_pandas().to_dict('records') if hasattr(
                            search_query, 'to_pandas'
                        ) else []
                    
                    for result in search_results:
                        similarity = 1.0 - float(result.get("_distance", 0.0))
                        if similarity > 0.6:  # Cross-module threshold
                            content_json = json.loads(result.get("content_json", "{}"))
                            cross_module_queries.append({
                                "module": module_name,
                                "query_id": result.get("id", "unknown"),
                                "similarity": similarity,
                                "query": content_json.get("query", "") or content_json.get("sql_query", ""),
                                "business_domain": result.get("business_domain", "unknown"),
                                "confidence": float(result.get("confidence_score", 0.5))
                            })
                            
                            self.vector_enhancement_stats["cross_module_discoveries"] += 1
                
                except Exception as e:
                    self.vector_logger.logger.warning(f"Failed to search {module_name} table: {e}")
        
        except Exception as e:
            self.vector_logger.logger.warning(f"Failed to find cross-module queries: {e}")
        
        return cross_module_queries
    
    def _calculate_confidence_boost(
        self,
        base_results: InvestigationResults,
        patterns: List[InvestigationPattern]
    ) -> float:
        """Calculate confidence boost based on similar successful patterns."""
        if not patterns:
            return 0.0
        
        # Boost based on similar successful investigations
        successful_patterns = [
            p for p in patterns 
            if p.success_status == "completed" and p.overall_confidence > 0.7
        ]
        
        if not successful_patterns:
            return 0.0
        
        # Weight by similarity and confidence
        weighted_boost = 0.0
        total_weight = 0.0
        
        for pattern in successful_patterns:
            weight = pattern.similarity_score * pattern.overall_confidence
            weighted_boost += weight * self.confidence_boost_factor
            total_weight += weight
        
        if total_weight > 0:
            boost = weighted_boost / total_weight
            self.vector_enhancement_stats["avg_confidence_improvement"].append(boost)
            return min(boost, 0.2)  # Cap at 20% boost
        
        return 0.0
    
    def _calculate_pattern_based_confidence(
        self,
        base_results: InvestigationResults,
        patterns: List[InvestigationPattern]
    ) -> float:
        """Calculate pattern-based confidence score."""
        if not patterns:
            return base_results.overall_confidence
        
        # Combine base confidence with pattern confidence
        pattern_confidences = [p.overall_confidence for p in patterns[:5]]  # Top 5 patterns
        pattern_weights = [p.similarity_score for p in patterns[:5]]
        
        if pattern_confidences:
            weighted_pattern_confidence = sum(
                c * w for c, w in zip(pattern_confidences, pattern_weights)
            ) / sum(pattern_weights)
            
            # Blend with base confidence
            return 0.7 * base_results.overall_confidence + 0.3 * weighted_pattern_confidence
        
        return base_results.overall_confidence
    
    def _calculate_cross_module_validation(self, cross_module_queries: List[Dict[str, Any]]) -> float:
        """Calculate validation score based on cross-module query alignment."""
        if not cross_module_queries:
            return 0.5  # Neutral score
        
        # Higher score for queries with high similarity from multiple modules
        module_scores = {}
        
        for query in cross_module_queries:
            module = query["module"]
            similarity = query["similarity"]
            
            if module not in module_scores:
                module_scores[module] = []
            module_scores[module].append(similarity)
        
        # Calculate average similarity per module
        avg_scores = []
        for module, scores in module_scores.items():
            avg_scores.append(sum(scores) / len(scores))
        
        # Return overall cross-module validation score
        return sum(avg_scores) / len(avg_scores) if avg_scores else 0.5
    
    def _calculate_domain_match(self, domain1: str, domain2: str) -> float:
        """Calculate domain similarity score."""
        if not domain1 or not domain2:
            return 0.5
        
        if domain1.lower() == domain2.lower():
            return 1.0
        
        # Check for related domains
        domain_relationships = {
            "sales": ["marketing", "customer", "revenue"],
            "finance": ["cost", "budget", "accounting"],
            "production": ["manufacturing", "operations", "quality"],
            "hr": ["human_resources", "personnel", "employee"]
        }
        
        for primary, related in domain_relationships.items():
            if domain1.lower() == primary and domain2.lower() in related:
                return 0.7
            if domain2.lower() == primary and domain1.lower() in related:
                return 0.7
        
        return 0.3
    
    def _calculate_methodology_alignment(self, type1: str, type2: str) -> float:
        """Calculate methodology alignment between investigation types."""
        if type1 == type2:
            return 1.0
        
        # Define methodology relationships
        type_similarity = {
            ("diagnostic", "exploratory"): 0.7,
            ("predictive", "comparative"): 0.6,
            ("descriptive", "comparative"): 0.7,
            ("diagnostic", "predictive"): 0.5
        }
        
        pair = (type1, type2)
        return type_similarity.get(pair, type_similarity.get((type2, type1), 0.4))
    
    async def _store_enhanced_investigation(
        self,
        enhanced_results: VectorEnhancedInvestigationResults,
        request_embedding: np.ndarray
    ) -> bool:
        """Store enhanced investigation results for future pattern learning."""
        try:
            if not self.vector_table:
                return False
            
            base_results = enhanced_results.base_results
            
            # Create vector record
            if VECTOR_INFRASTRUCTURE_AVAILABLE:
                # Use enterprise schema
                now = datetime.now(timezone.utc)
                
                # Map investigation type to unified analysis type
                investigation_type = self._classify_investigation_type(base_results.investigation_request)
                analysis_type_map = {
                    "diagnostic": UnifiedAnalysisType.DIAGNOSTIC,
                    "predictive": UnifiedAnalysisType.PREDICTIVE,
                    "descriptive": UnifiedAnalysisType.DESCRIPTIVE,
                    "comparative": UnifiedAnalysisType.DESCRIPTIVE,
                    "exploratory": UnifiedAnalysisType.DIAGNOSTIC
                }
                
                vector_metadata = VectorMetadata(
                    id=enhanced_results.vector_id,
                    module_source=ModuleSource.INVESTIGATION,
                    created_at=now,
                    updated_at=now,
                    
                    # Map scores to unified scale
                    complexity_score=normalize_score_to_unified_scale(
                        len(base_results.completed_steps) / 7, "0.0-1.0"
                    ),
                    confidence_score=base_results.overall_confidence + enhanced_results.confidence_boost,
                    actionability_score=normalize_score_to_unified_scale(
                        base_results.business_context.get("actionability_score", 0.7), "0.0-1.0"
                    ),
                    business_value_score=enhanced_results.pattern_based_confidence,
                    
                    # Map to unified enums
                    business_domain=self._map_to_unified_domain(
                        base_results.business_context.get("business_domain", "general")
                    ),
                    performance_tier=self._map_to_performance_tier(
                        base_results.business_context.get("investigation_type", "analytical")
                    ),
                    analysis_type=analysis_type_map.get(
                        investigation_type, UnifiedAnalysisType.DESCRIPTIVE
                    ),
                    
                    # Module-specific metadata
                    module_metadata={
                        "investigation": {
                            "investigation_id": base_results.investigation_id,
                            "investigation_type": investigation_type,
                            "complexity_level": base_results.business_context.get(
                                "investigation_type", "analytical"
                            ),
                            "total_duration_seconds": base_results.total_duration_seconds,
                            "completed_steps": len(base_results.completed_steps),
                            "status": base_results.status,
                            "step_sequence": [s.step_name for s in base_results.completed_steps],
                            "step_durations": {
                                s.step_name: s.duration_seconds 
                                for s in base_results.completed_steps 
                                if s.duration_seconds
                            },
                            "step_confidences": {
                                s.step_name: s.confidence_score 
                                for s in base_results.completed_steps 
                                if s.confidence_score
                            },
                            "user_satisfaction": 0.8  # Default, could be from feedback
                        },
                        "vector_enhancement": {
                            "confidence_boost": enhanced_results.confidence_boost,
                            "pattern_matches": len(enhanced_results.similar_investigations),
                            "cross_module_queries": len(enhanced_results.related_queries_from_other_modules),
                            "predicted_duration_minutes": enhanced_results.predicted_duration_minutes,
                            "success_probability": enhanced_results.estimated_success_probability
                        }
                    }
                )
                
                record = vector_metadata.to_lancedb_record(
                    request_embedding.tolist(),
                    {"investigation_request": base_results.investigation_request}
                )
            else:
                # Fallback to basic record
                record = {
                    "id": enhanced_results.vector_id,
                    "vector": request_embedding.tolist(),
                    "content": base_results.investigation_request,
                    "investigation_type": self._classify_investigation_type(
                        base_results.investigation_request
                    ),
                    "confidence": base_results.overall_confidence + enhanced_results.confidence_boost,
                    "duration": base_results.total_duration_seconds,
                    "created_at": datetime.now(timezone.utc)
                }
            
            # Store in vector table
            await self.vector_table.add([record])
            
            self.vector_logger.logger.info(f"Stored enhanced investigation: {enhanced_results.vector_id}")
            return True
            
        except Exception as e:
            self.vector_logger.logger.error(f"Failed to store enhanced investigation: {e}")
            return False
    
    def _map_to_unified_domain(self, business_domain: str) -> UnifiedBusinessDomain:
        """Map investigation business domain to unified schema domain."""
        if not VECTOR_INFRASTRUCTURE_AVAILABLE:
            return None
        
        domain_map = {
            "sales": UnifiedBusinessDomain.SALES,
            "finance": UnifiedBusinessDomain.FINANCE,
            "hr": UnifiedBusinessDomain.HUMAN_RESOURCES,
            "human_resources": UnifiedBusinessDomain.HUMAN_RESOURCES,
            "production": UnifiedBusinessDomain.PRODUCTION,
            "manufacturing": UnifiedBusinessDomain.PRODUCTION,
            "marketing": UnifiedBusinessDomain.MARKETING,
            "quality": UnifiedBusinessDomain.QUALITY,
            "logistics": UnifiedBusinessDomain.SUPPLY_CHAIN,
            "supply_chain": UnifiedBusinessDomain.SUPPLY_CHAIN,
            "operations": UnifiedBusinessDomain.OPERATIONS,
            "strategic": UnifiedBusinessDomain.STRATEGIC,
            "analytics": UnifiedBusinessDomain.OPERATIONS,
            "general": UnifiedBusinessDomain.OPERATIONS
        }
        
        return domain_map.get(business_domain.lower(), UnifiedBusinessDomain.OPERATIONS)
    
    def _map_to_performance_tier(self, investigation_type: str) -> PerformanceTier:
        """Map investigation type to performance tier."""
        if not VECTOR_INFRASTRUCTURE_AVAILABLE:
            return None
        
        tier_map = {
            "simple": PerformanceTier.SIMPLE,
            "analytical": PerformanceTier.ANALYTICAL,
            "computational": PerformanceTier.COMPUTATIONAL,
            "comprehensive": PerformanceTier.COMPUTATIONAL,
            "diagnostic": PerformanceTier.ANALYTICAL,
            "predictive": PerformanceTier.COMPUTATIONAL,
            "descriptive": PerformanceTier.SIMPLE,
            "comparative": PerformanceTier.ANALYTICAL,
            "exploratory": PerformanceTier.ANALYTICAL
        }
        
        return tier_map.get(investigation_type.lower(), PerformanceTier.ANALYTICAL)
    
    async def _record_performance_metrics(self, enhanced_results: VectorEnhancedInvestigationResults):
        """Record performance metrics for monitoring."""
        try:
            if not self.performance_monitor:
                return
            
            # Investigation latency metric
            latency_metric = PerformanceMetric(
                metric_type=PerformanceMetricType.QUERY_LATENCY,
                value=enhanced_results.total_enhancement_time_ms,
                timestamp=datetime.now(timezone.utc),
                module_source=ModuleSource.INVESTIGATION,
                context={
                    "operation": "vector_enhanced_investigation",
                    "patterns_found": len(enhanced_results.similar_investigations),
                    "confidence_boost": enhanced_results.confidence_boost,
                    "cross_module_queries": len(enhanced_results.related_queries_from_other_modules),
                    "investigation_status": enhanced_results.base_results.status
                }
            )
            self.performance_monitor.record_metric(latency_metric)
            
        except Exception as e:
            self.vector_logger.logger.warning(f"Failed to record performance metrics: {e}")
    
    async def get_investigation_statistics(self) -> Dict[str, Any]:
        """Get vector-enhanced investigation statistics."""
        stats = self.vector_enhancement_stats.copy()
        
        # Calculate averages
        if stats["avg_confidence_improvement"]:
            stats["avg_confidence_boost"] = sum(
                stats["avg_confidence_improvement"]
            ) / len(stats["avg_confidence_improvement"])
        else:
            stats["avg_confidence_boost"] = 0.0
        
        # Calculate rates
        total = stats["total_investigations"]
        if total > 0:
            stats["pattern_enhancement_rate"] = stats["pattern_enhanced"] / total
            stats["cross_module_discovery_rate"] = stats["cross_module_discoveries"] / total
        
        # Add capability status
        stats["capabilities"] = {
            "investigation_module": INVESTIGATION_MODULE_AVAILABLE,
            "vector_infrastructure": VECTOR_INFRASTRUCTURE_AVAILABLE,
            "vector_capabilities": VECTOR_CAPABILITIES_AVAILABLE,
            "embedder_loaded": self.embedder is not None,
            "vector_table_available": self.vector_table is not None
        }
        
        return stats
    
    async def cleanup(self):
        """Cleanup vector-enhanced investigator resources."""
        try:
            # Clear pattern cache
            self.investigation_patterns.clear()
            
            # Close vector connections if available
            if hasattr(self, 'vector_db') and self.vector_db:
                # LanceDB connections are typically auto-managed
                pass
            
            self.vector_logger.logger.info("VectorEnhancedInvestigator cleanup completed")
            
        except Exception as e:
            self.vector_logger.logger.error(f"Cleanup error: {e}")


# Enhanced interface function
async def conduct_vector_enhanced_investigation(
    coordinated_services: Dict[str, Any],
    investigation_request: str,
    execution_context: Dict[str, Any],
    mcp_client_manager=None,
    db_path: str = None,
    use_vector_enhancement: bool = True
) -> VectorEnhancedInvestigationResults:
    """
    High-level interface for vector-enhanced investigation execution.
    
    Args:
        coordinated_services: Database services from orchestration
        investigation_request: Business question to investigate
        execution_context: Investigation parameters and context
        mcp_client_manager: MCP client manager for database access
        db_path: Path to LanceDB for vector storage
        use_vector_enhancement: Whether to use vector capabilities
        
    Returns:
        VectorEnhancedInvestigationResults with pattern insights
    """
    investigator = VectorEnhancedInvestigator()
    
    try:
        # Initialize vector capabilities if requested
        if use_vector_enhancement and db_path:
            await investigator.initialize(db_path)
        
        # Conduct enhanced investigation
        results = await investigator.conduct_investigation(
            coordinated_services=coordinated_services,
            investigation_request=investigation_request,
            execution_context=execution_context,
            mcp_client_manager=mcp_client_manager,
            use_vector_enhancement=use_vector_enhancement
        )
        
        return results
        
    except Exception as e:
        if investigator.vector_logger:
            investigator.vector_logger.logger.error(f"Vector-enhanced investigation failed: {e}")
        
        # Fallback to base investigation
        base_results = await conduct_autonomous_investigation(
            coordinated_services=coordinated_services,
            investigation_request=investigation_request,
            execution_context=execution_context,
            mcp_client_manager=mcp_client_manager
        )
        
        return VectorEnhancedInvestigationResults(
            base_results=base_results,
            vector_id=f"error_{uuid.uuid4()}",
            total_enhancement_time_ms=0.0
        )
    
    finally:
        await investigator.cleanup()


# Export main classes and functions
__all__ = [
    "VectorEnhancedInvestigator",
    "VectorEnhancedInvestigationResults",
    "InvestigationPattern",
    "StepOptimizationInsight",
    "conduct_vector_enhanced_investigation"
]