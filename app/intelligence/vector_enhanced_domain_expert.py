#!/usr/bin/env python3
"""
Vector-Enhanced Domain Expert - Phase 1.1 Implementation
Integrates LanceDB vector capabilities with Intelligence module domain classification.
Provides semantic similarity search and pattern recognition for business intelligence.
"""

import asyncio
import json
import time
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
import numpy as np

# Import existing Intelligence module components
try:
    from .domain_expert import DomainExpert, BusinessIntent, BusinessDomain, AnalysisType
    from .complexity_analyzer import ComplexityAnalyzer, ComplexityScore, ComplexityLevel
    from .config import settings
    from .intelligence_logging import setup_logger, performance_monitor
    INTELLIGENCE_MODULE_AVAILABLE = True
except ImportError:
    try:
        from domain_expert import DomainExpert, BusinessIntent, BusinessDomain, AnalysisType
        from complexity_analyzer import ComplexityAnalyzer, ComplexityScore, ComplexityLevel
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
class VectorEnhancedBusinessIntent:
    """Enhanced business intent with vector similarity and pattern matching."""
    
    # Original business intent
    business_intent: BusinessIntent
    
    # Vector enhancement data
    vector_id: str
    similarity_scores: Dict[str, float] = field(default_factory=dict)
    pattern_matches: List[Dict[str, Any]] = field(default_factory=list)
    confidence_boost: float = 0.0
    
    # Cross-module relationships
    related_investigations: List[str] = field(default_factory=list)
    related_syntheses: List[str] = field(default_factory=list)
    similar_queries: List[Dict[str, Any]] = field(default_factory=list)
    
    # Performance metrics
    classification_time_ms: float = 0.0
    vector_search_time_ms: float = 0.0
    total_processing_time_ms: float = 0.0


@dataclass
class SemanticPatternMatch:
    """Semantic pattern match from vector search."""
    query_id: str
    similarity_score: float
    business_domain: str
    analysis_type: str
    confidence_score: float
    business_value_score: float
    query_content: str
    timestamp: datetime
    
    # Pattern analysis
    domain_alignment: float
    methodology_similarity: float
    complexity_correlation: float


class VectorEnhancedDomainExpert:
    """
    Vector-enhanced domain expert integrating LanceDB capabilities with Intelligence module.
    Provides semantic similarity search, pattern recognition, and enhanced business intelligence.
    """
    
    def __init__(self, vector_index_manager: Optional[VectorIndexManager] = None):
        self.logger = setup_logger("vector_enhanced_domain_expert")
        
        # Initialize base Intelligence components
        if INTELLIGENCE_MODULE_AVAILABLE:
            self.domain_expert = DomainExpert()
            self.complexity_analyzer = ComplexityAnalyzer()
        else:
            self.domain_expert = None
            self.complexity_analyzer = None
            self.logger.warning("Intelligence module components not available")
        
        # Vector infrastructure
        self.vector_index_manager = vector_index_manager
        self.performance_monitor = None
        self.embedder = None
        self.vector_db = None
        self.vector_table = None
        
        # Semantic pattern cache
        self.pattern_cache: Dict[str, List[SemanticPatternMatch]] = {}
        self.cache_ttl_seconds = 3600  # 1 hour
        
        # Performance tracking
        self.classification_stats = {
            "total_queries": 0,
            "vector_enhanced": 0,
            "pattern_matches_found": 0,
            "confidence_improvements": 0
        }
        
        self.logger.info("VectorEnhancedDomainExpert initialized")
    
    async def initialize(self, db_path: str = None, table_name: str = "intelligence_vectors"):
        """Initialize vector capabilities and connect to LanceDB."""
        try:
            if not VECTOR_CAPABILITIES_AVAILABLE:
                self.logger.warning("Vector capabilities not available - using base Intelligence only")
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
                
                self.logger.info("✅ Vector capabilities initialized")
            else:
                self.logger.info("No DB path provided - vector search disabled")
                
        except Exception as e:
            self.logger.error(f"Failed to initialize vector capabilities: {e}")
            self.embedder = None
            self.vector_db = None
    
    async def _setup_vector_table(self, table_name: str):
        """Setup vector table for intelligence queries."""
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
                        "business_domain": "operations"
                    }]
                    self.vector_table = await self.vector_db.create_table(table_name, data=dummy_data)
                    await self.vector_table.delete("id = 'dummy'")
                    self.logger.info(f"Created basic vector table: {table_name}")
                    
        except Exception as e:
            self.logger.error(f"Failed to setup vector table: {e}")
            self.vector_table = None
    
    @performance_monitor("vector_enhanced_classification")
    async def classify_business_intent_with_vectors(
        self, 
        query: str,
        include_similar_patterns: bool = True,
        similarity_threshold: float = 0.75,
        max_similar_patterns: int = 5
    ) -> VectorEnhancedBusinessIntent:
        """
        Enhanced business intent classification using vector similarity and pattern matching.
        
        Args:
            query: Natural language business query
            include_similar_patterns: Whether to find similar historical patterns
            similarity_threshold: Minimum similarity score for pattern matches
            max_similar_patterns: Maximum number of similar patterns to return
            
        Returns:
            Vector-enhanced business intent with similarity data
        """
        total_start_time = time.perf_counter()
        self.classification_stats["total_queries"] += 1
        
        try:
            # Step 1: Base intelligence classification
            classification_start = time.perf_counter()
            
            if self.domain_expert:
                base_intent = self.domain_expert.classify_business_intent(query)
            else:
                # Fallback if Intelligence module not available
                base_intent = self._create_fallback_intent(query)
            
            classification_time = (time.perf_counter() - classification_start) * 1000
            
            # Step 2: Vector enhancement
            vector_search_start = time.perf_counter()
            vector_id = generate_vector_id(query, ModuleSource.INTELLIGENCE) if VECTOR_INFRASTRUCTURE_AVAILABLE else f"intelligence_{hash(query)}"
            
            similarity_scores = {}
            pattern_matches = []
            confidence_boost = 0.0
            similar_queries = []
            
            if self.embedder and self.vector_table and include_similar_patterns:
                # Generate query embedding
                query_embedding = self.embedder.encode(
                    query, 
                    convert_to_numpy=True, 
                    normalize_embeddings=True,
                    show_progress_bar=False
                )
                
                # Search for similar patterns
                pattern_matches = await self._find_similar_patterns(
                    query_embedding, 
                    base_intent,
                    similarity_threshold,
                    max_similar_patterns
                )
                
                # Calculate confidence boost based on pattern matches
                confidence_boost = self._calculate_confidence_boost(pattern_matches, base_intent)
                
                # Extract similar queries for cross-module learning
                similar_queries = [
                    {
                        "query_id": match.query_id,
                        "similarity": match.similarity_score,
                        "domain": match.business_domain,
                        "confidence": match.confidence_score
                    } for match in pattern_matches
                ]
                
                self.classification_stats["vector_enhanced"] += 1
                if pattern_matches:
                    self.classification_stats["pattern_matches_found"] += 1
                if confidence_boost > 0:
                    self.classification_stats["confidence_improvements"] += 1
            
            vector_search_time = (time.perf_counter() - vector_search_start) * 1000
            total_time = (time.perf_counter() - total_start_time) * 1000
            
            # Step 3: Create enhanced business intent
            enhanced_intent = VectorEnhancedBusinessIntent(
                business_intent=base_intent,
                vector_id=vector_id,
                similarity_scores=similarity_scores,
                pattern_matches=[self._pattern_to_dict(match) for match in pattern_matches],
                confidence_boost=confidence_boost,
                similar_queries=similar_queries,
                classification_time_ms=classification_time,
                vector_search_time_ms=vector_search_time,
                total_processing_time_ms=total_time
            )
            
            # Step 4: Record performance metrics
            if self.performance_monitor:
                await self._record_performance_metrics(enhanced_intent)
            
            self.logger.info(
                f"Enhanced classification completed: {base_intent.primary_domain.value} "
                f"({base_intent.analysis_type.value}) confidence: {base_intent.confidence:.3f} "
                f"(+{confidence_boost:.3f} boost) in {total_time:.1f}ms"
            )
            
            return enhanced_intent
            
        except Exception as e:
            self.logger.error(f"Vector-enhanced classification failed: {e}")
            # Fallback to base classification
            if self.domain_expert:
                base_intent = self.domain_expert.classify_business_intent(query)
            else:
                base_intent = self._create_fallback_intent(query)
            
            return VectorEnhancedBusinessIntent(
                business_intent=base_intent,
                vector_id=f"fallback_{hash(query)}",
                total_processing_time_ms=(time.perf_counter() - total_start_time) * 1000
            )
    
    async def _find_similar_patterns(
        self, 
        query_embedding: np.ndarray, 
        base_intent: BusinessIntent,
        similarity_threshold: float,
        max_patterns: int
    ) -> List[SemanticPatternMatch]:
        """Find semantically similar patterns from vector store."""
        try:
            if not self.vector_table:
                return []
            
            # Search for similar vectors
            search_query = self.vector_table.search(query_embedding.tolist()).limit(max_patterns * 2)
            search_results = await search_query.to_list()
            
            pattern_matches = []
            for result in search_results:
                similarity = float(result.get("_distance", 0.0))
                
                # Convert distance to similarity (assuming cosine distance)
                similarity_score = 1.0 - similarity
                
                if similarity_score >= similarity_threshold:
                    pattern_match = SemanticPatternMatch(
                        query_id=result.get("id", "unknown"),
                        similarity_score=similarity_score,
                        business_domain=result.get("business_domain", "unknown"),
                        analysis_type=result.get("analysis_type", "descriptive"),
                        confidence_score=float(result.get("confidence_score", 0.5)),
                        business_value_score=float(result.get("business_value_score", 0.5)),
                        query_content=json.loads(result.get("content_json", "{}")).get("sql_query", ""),
                        timestamp=datetime.fromisoformat(result.get("created_at", datetime.now().isoformat())),
                        domain_alignment=self._calculate_domain_alignment(base_intent, result),
                        methodology_similarity=self._calculate_methodology_similarity(base_intent, result),
                        complexity_correlation=self._calculate_complexity_correlation(base_intent, result)
                    )
                    pattern_matches.append(pattern_match)
            
            # Sort by similarity and return top matches
            pattern_matches.sort(key=lambda x: x.similarity_score, reverse=True)
            return pattern_matches[:max_patterns]
            
        except Exception as e:
            self.logger.warning(f"Failed to find similar patterns: {e}")
            return []
    
    def _calculate_confidence_boost(self, pattern_matches: List[SemanticPatternMatch], base_intent: BusinessIntent) -> float:
        """Calculate confidence boost based on pattern matches."""
        if not pattern_matches:
            return 0.0
        
        # Boost based on domain alignment and similarity scores
        domain_boost = 0.0
        similarity_boost = 0.0
        
        for match in pattern_matches:
            if match.business_domain == base_intent.primary_domain.value:
                domain_boost += match.similarity_score * 0.1  # Max 10% boost per exact domain match
            
            similarity_boost += match.similarity_score * 0.05  # Max 5% boost per high similarity
        
        # Cap total boost at 20%
        total_boost = min(domain_boost + similarity_boost, 0.2)
        return total_boost
    
    def _calculate_domain_alignment(self, base_intent: BusinessIntent, result: Dict) -> float:
        """Calculate domain alignment score between base intent and search result."""
        result_domain = result.get("business_domain", "unknown")
        
        if result_domain == base_intent.primary_domain.value:
            return 1.0
        elif result_domain in [d.value for d in base_intent.secondary_domains]:
            return 0.7
        else:
            return 0.3
    
    def _calculate_methodology_similarity(self, base_intent: BusinessIntent, result: Dict) -> float:
        """Calculate methodology similarity score."""
        result_analysis = result.get("analysis_type", "descriptive")
        
        if result_analysis == base_intent.analysis_type.value:
            return 1.0
        else:
            # Simple similarity based on analysis type relationships
            type_similarity = {
                ("descriptive", "diagnostic"): 0.6,
                ("diagnostic", "predictive"): 0.7,
                ("predictive", "prescriptive"): 0.8
            }
            
            pair = (base_intent.analysis_type.value, result_analysis)
            return type_similarity.get(pair, type_similarity.get((pair[1], pair[0]), 0.4))
    
    def _calculate_complexity_correlation(self, base_intent: BusinessIntent, result: Dict) -> float:
        """Calculate complexity correlation score."""
        result_complexity = float(result.get("complexity_score", 0.5))
        
        # Estimate base complexity (simplified)
        base_complexity = 0.5  # Default
        if len(base_intent.key_indicators) > 5:
            base_complexity += 0.2
        if len(base_intent.secondary_domains) > 1:
            base_complexity += 0.15
        if base_intent.analysis_type in [AnalysisType.PREDICTIVE, AnalysisType.PRESCRIPTIVE]:
            base_complexity += 0.15
        
        base_complexity = min(base_complexity, 1.0)
        
        # Correlation based on complexity difference
        diff = abs(base_complexity - result_complexity)
        return max(0.0, 1.0 - diff)
    
    async def store_enhanced_intent(self, enhanced_intent: VectorEnhancedBusinessIntent, query: str) -> bool:
        """Store enhanced business intent in vector database for future pattern matching."""
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
                    id=enhanced_intent.vector_id,
                    module_source=ModuleSource.INTELLIGENCE,
                    created_at=now,
                    updated_at=now,
                    
                    # Map Intelligence scores to unified scale
                    complexity_score=normalize_score_to_unified_scale(enhanced_intent.business_intent.confidence, "0.0-1.0"),
                    confidence_score=enhanced_intent.business_intent.confidence + enhanced_intent.confidence_boost,
                    actionability_score=0.7,  # Default for intelligence module
                    business_value_score=0.8,  # Default for intelligence module
                    
                    # Map to unified enums
                    business_domain=self._map_to_unified_domain(enhanced_intent.business_intent.primary_domain),
                    performance_tier=PerformanceTier.ANALYTICAL,  # Default for intelligence
                    analysis_type=self._map_to_unified_analysis_type(enhanced_intent.business_intent.analysis_type),
                    
                    # Module-specific metadata
                    module_metadata={
                        "business_intent": {
                            "primary_domain": enhanced_intent.business_intent.primary_domain.value,
                            "secondary_domains": [d.value for d in enhanced_intent.business_intent.secondary_domains],
                            "analysis_type": enhanced_intent.business_intent.analysis_type.value,
                            "confidence": enhanced_intent.business_intent.confidence,
                            "key_indicators": enhanced_intent.business_intent.key_indicators,
                            "business_metrics": enhanced_intent.business_intent.business_metrics,
                            "time_context": enhanced_intent.business_intent.time_context,
                            "urgency_level": enhanced_intent.business_intent.urgency_level
                        },
                        "vector_enhancement": {
                            "confidence_boost": enhanced_intent.confidence_boost,
                            "pattern_matches_count": len(enhanced_intent.pattern_matches),
                            "similar_queries_count": len(enhanced_intent.similar_queries),
                            "processing_time_ms": enhanced_intent.total_processing_time_ms
                        }
                    }
                )
                
                # Convert BusinessIntent to dictionary for JSON serialization
                business_intent_dict = {
                    "primary_domain": enhanced_intent.business_intent.primary_domain.value,
                    "secondary_domains": [d.value for d in enhanced_intent.business_intent.secondary_domains],
                    "analysis_type": enhanced_intent.business_intent.analysis_type.value,
                    "confidence": enhanced_intent.business_intent.confidence,
                    "key_indicators": enhanced_intent.business_intent.key_indicators,
                    "business_metrics": enhanced_intent.business_intent.business_metrics,
                    "time_context": enhanced_intent.business_intent.time_context,
                    "urgency_level": enhanced_intent.business_intent.urgency_level
                }
                
                record = vector_metadata.to_lancedb_record(
                    query_embedding.tolist(),
                    {"query": query, "business_intent": business_intent_dict}
                )
            else:
                # Fallback to basic record
                record = {
                    "id": enhanced_intent.vector_id,
                    "vector": query_embedding.tolist(),
                    "content": query,
                    "business_domain": enhanced_intent.business_intent.primary_domain.value,
                    "analysis_type": enhanced_intent.business_intent.analysis_type.value,
                    "confidence": enhanced_intent.business_intent.confidence + enhanced_intent.confidence_boost,
                    "created_at": datetime.now(timezone.utc)
                }
            
            # Store in vector table
            await self.vector_table.add([record])
            
            self.logger.info(f"Stored enhanced intent: {enhanced_intent.vector_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to store enhanced intent: {e}")
            return False
    
    def _map_to_unified_domain(self, intelligence_domain: BusinessDomain) -> UnifiedBusinessDomain:
        """Map Intelligence module BusinessDomain to unified schema BusinessDomain."""
        if not VECTOR_INFRASTRUCTURE_AVAILABLE:
            return None
        
        # Direct mapping since the enums are identical
        mapping = {
            BusinessDomain.PRODUCTION: UnifiedBusinessDomain.PRODUCTION,
            BusinessDomain.QUALITY: UnifiedBusinessDomain.QUALITY,
            BusinessDomain.SUPPLY_CHAIN: UnifiedBusinessDomain.SUPPLY_CHAIN,
            BusinessDomain.COST: UnifiedBusinessDomain.COST,
            BusinessDomain.ASSETS: UnifiedBusinessDomain.ASSETS,
            BusinessDomain.SAFETY: UnifiedBusinessDomain.SAFETY,
            BusinessDomain.CUSTOMER: UnifiedBusinessDomain.CUSTOMER,
            BusinessDomain.PLANNING: UnifiedBusinessDomain.PLANNING,
            BusinessDomain.HUMAN_RESOURCES: UnifiedBusinessDomain.HUMAN_RESOURCES,
            BusinessDomain.SALES: UnifiedBusinessDomain.SALES,
            BusinessDomain.FINANCE: UnifiedBusinessDomain.FINANCE,
            BusinessDomain.MARKETING: UnifiedBusinessDomain.MARKETING,
            BusinessDomain.OPERATIONS: UnifiedBusinessDomain.OPERATIONS,
            BusinessDomain.STRATEGIC: UnifiedBusinessDomain.STRATEGIC
        }
        return mapping.get(intelligence_domain, UnifiedBusinessDomain.OPERATIONS)
    
    def _map_to_unified_analysis_type(self, intelligence_analysis: AnalysisType) -> UnifiedAnalysisType:
        """Map Intelligence module AnalysisType to unified schema AnalysisType."""
        if not VECTOR_INFRASTRUCTURE_AVAILABLE:
            return None
        
        # Direct mapping since the enums are identical
        mapping = {
            AnalysisType.DESCRIPTIVE: UnifiedAnalysisType.DESCRIPTIVE,
            AnalysisType.DIAGNOSTIC: UnifiedAnalysisType.DIAGNOSTIC,
            AnalysisType.PREDICTIVE: UnifiedAnalysisType.PREDICTIVE,
            AnalysisType.PRESCRIPTIVE: UnifiedAnalysisType.PRESCRIPTIVE
        }
        return mapping.get(intelligence_analysis, UnifiedAnalysisType.DESCRIPTIVE)
    
    def _create_fallback_intent(self, query: str) -> BusinessIntent:
        """Create fallback business intent when Intelligence module unavailable."""
        return BusinessIntent(
            primary_domain=BusinessDomain.OPERATIONS if INTELLIGENCE_MODULE_AVAILABLE else "operations",
            secondary_domains=[],
            analysis_type=AnalysisType.DESCRIPTIVE if INTELLIGENCE_MODULE_AVAILABLE else "descriptive", 
            confidence=0.5,
            key_indicators=[],
            business_metrics=[],
            time_context=None,
            urgency_level="normal"
        )
    
    def _pattern_to_dict(self, pattern: SemanticPatternMatch) -> Dict[str, Any]:
        """Convert SemanticPatternMatch to dictionary for serialization."""
        return {
            "query_id": pattern.query_id,
            "similarity_score": pattern.similarity_score,
            "business_domain": pattern.business_domain,
            "analysis_type": pattern.analysis_type,
            "confidence_score": pattern.confidence_score,
            "business_value_score": pattern.business_value_score,
            "query_content": pattern.query_content[:100],  # Truncate for storage
            "timestamp": pattern.timestamp.isoformat(),
            "domain_alignment": pattern.domain_alignment,
            "methodology_similarity": pattern.methodology_similarity,
            "complexity_correlation": pattern.complexity_correlation
        }
    
    async def _record_performance_metrics(self, enhanced_intent: VectorEnhancedBusinessIntent):
        """Record performance metrics for monitoring."""
        try:
            if not self.performance_monitor:
                return
            
            # Classification latency metric
            classification_metric = PerformanceMetric(
                metric_type=PerformanceMetricType.QUERY_LATENCY,
                value=enhanced_intent.total_processing_time_ms,
                timestamp=datetime.now(timezone.utc),
                module_source=ModuleSource.INTELLIGENCE,
                context={
                    "operation": "vector_enhanced_classification",
                    "pattern_matches": len(enhanced_intent.pattern_matches),
                    "confidence_boost": enhanced_intent.confidence_boost
                }
            )
            self.performance_monitor.record_metric(classification_metric)
            
        except Exception as e:
            self.logger.warning(f"Failed to record performance metrics: {e}")
    
    async def get_classification_statistics(self) -> Dict[str, Any]:
        """Get classification performance statistics."""
        stats = self.classification_stats.copy()
        
        # Calculate rates
        total = stats["total_queries"]
        if total > 0:
            stats["vector_enhancement_rate"] = stats["vector_enhanced"] / total
            stats["pattern_match_rate"] = stats["pattern_matches_found"] / total
            stats["confidence_improvement_rate"] = stats["confidence_improvements"] / total
        else:
            stats["vector_enhancement_rate"] = 0.0
            stats["pattern_match_rate"] = 0.0
            stats["confidence_improvement_rate"] = 0.0
        
        # Add capability status
        stats["capabilities"] = {
            "intelligence_module": INTELLIGENCE_MODULE_AVAILABLE,
            "vector_infrastructure": VECTOR_INFRASTRUCTURE_AVAILABLE,
            "vector_capabilities": VECTOR_CAPABILITIES_AVAILABLE,
            "embedder_loaded": self.embedder is not None,
            "vector_table_available": self.vector_table is not None
        }
        
        return stats
    
    async def cleanup(self):
        """Cleanup vector-enhanced domain expert resources."""
        try:
            # Clear pattern cache
            self.pattern_cache.clear()
            
            # Close vector connections if available
            if hasattr(self, 'vector_db') and self.vector_db:
                # LanceDB connections are typically auto-managed
                pass
            
            self.logger.info("VectorEnhancedDomainExpert cleanup completed")
            
        except Exception as e:
            self.logger.error(f"Cleanup error: {e}")


# Utility functions for external integration
async def create_vector_enhanced_domain_expert(
    vector_index_manager: Optional[VectorIndexManager] = None,
    db_path: str = None
) -> VectorEnhancedDomainExpert:
    """Factory function to create and initialize vector-enhanced domain expert."""
    expert = VectorEnhancedDomainExpert(vector_index_manager)
    await expert.initialize(db_path)
    return expert


# Export main classes and functions
__all__ = [
    "VectorEnhancedDomainExpert",
    "VectorEnhancedBusinessIntent", 
    "SemanticPatternMatch",
    "create_vector_enhanced_domain_expert"
]