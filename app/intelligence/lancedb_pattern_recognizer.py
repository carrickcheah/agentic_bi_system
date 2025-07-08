#!/usr/bin/env python3
"""
LanceDB Pattern Recognizer - Phase 1.3 Implementation
Cross-module pattern recognition system for identifying and learning from patterns
across all modules in the LanceDB ecosystem. Provides unified intelligence and insights.
"""

import asyncio
import json
import time
import numpy as np
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union, Set
from dataclasses import dataclass, field
from enum import Enum
import math
from collections import defaultdict

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


class PatternType(Enum):
    """Types of patterns that can be recognized across modules."""
    BUSINESS_PROCESS = "business_process"
    DOMAIN_CORRELATION = "domain_correlation"
    COMPLEXITY_EVOLUTION = "complexity_evolution"
    PERFORMANCE_TREND = "performance_trend"
    RESOURCE_OPTIMIZATION = "resource_optimization"
    CROSS_MODULE_DEPENDENCY = "cross_module_dependency"
    SEASONAL_PATTERN = "seasonal_pattern"
    ANOMALY_DETECTION = "anomaly_detection"


class PatternStrength(Enum):
    """Strength levels for pattern confidence."""
    WEAK = "weak"          # 0.3-0.5
    MODERATE = "moderate"  # 0.5-0.7
    STRONG = "strong"      # 0.7-0.85
    VERY_STRONG = "very_strong"  # 0.85-1.0


@dataclass
class CrossModulePattern:
    """Comprehensive pattern data across multiple modules."""
    pattern_id: str
    pattern_type: PatternType
    strength: PatternStrength
    confidence: float
    
    # Pattern metadata
    modules_involved: Set[ModuleSource]
    business_domains: Set[str]
    time_span_days: int
    occurrence_count: int
    
    # Pattern characteristics
    semantic_cluster_center: List[float]
    pattern_keywords: List[str]
    business_metrics_involved: List[str]
    
    # Performance insights
    avg_complexity_score: float
    avg_processing_time_ms: float
    success_rate: float
    resource_efficiency_score: float
    
    # Trend analysis
    trend_direction: str  # "increasing", "decreasing", "stable", "cyclical"
    trend_strength: float
    last_occurrence: datetime
    prediction_accuracy: float
    
    # Cross-module relationships
    module_dependencies: Dict[str, List[str]]
    correlation_matrix: Dict[str, Dict[str, float]]
    causal_relationships: List[Dict[str, Any]]
    
    # Learning metadata
    discovery_timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_updated: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    validation_count: int = 0
    false_positive_rate: float = 0.0


@dataclass
class PatternInsight:
    """Actionable insight derived from pattern analysis."""
    insight_id: str
    pattern_id: str
    insight_type: str  # "optimization", "prediction", "risk", "opportunity"
    confidence: float
    
    # Insight content
    title: str
    description: str
    business_impact: str
    recommended_actions: List[str]
    
    # Quantitative measures
    potential_improvement: Dict[str, float]
    risk_factors: List[str]
    success_probability: float
    
    # Context
    applicable_domains: List[str]
    time_sensitivity: str  # "immediate", "short_term", "medium_term", "long_term"
    resource_requirements: Dict[str, Any]
    
    # Validation
    validation_status: str = "pending"  # "pending", "validated", "rejected"
    validation_timestamp: Optional[datetime] = None


@dataclass
class PatternCluster:
    """Cluster of related patterns for higher-level insights."""
    cluster_id: str
    cluster_name: str
    patterns: List[CrossModulePattern]
    cluster_center: List[float]
    
    # Cluster characteristics
    dominant_pattern_type: PatternType
    primary_business_domains: List[str]
    cluster_strength: float
    internal_coherence: float
    
    # Meta-insights
    cluster_insights: List[PatternInsight]
    strategic_implications: List[str]
    optimization_opportunities: List[str]


class LanceDBPatternRecognizer:
    """
    Advanced pattern recognition system for cross-module intelligence.
    Identifies, analyzes, and learns from patterns across all LanceDB modules.
    """
    
    def __init__(self, vector_index_manager: Optional[VectorIndexManager] = None):
        self.logger = setup_logger("lancedb_pattern_recognizer")
        
        # Core components
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
        self.vector_tables = {}  # Module-specific tables
        
        # Pattern recognition state
        self.discovered_patterns: Dict[str, CrossModulePattern] = {}
        self.pattern_clusters: Dict[str, PatternCluster] = {}
        self.pattern_insights: Dict[str, PatternInsight] = {}
        
        # Learning parameters
        self.min_pattern_support = 3  # Minimum occurrences to form pattern
        self.pattern_similarity_threshold = 0.75
        self.cluster_similarity_threshold = 0.6
        self.pattern_ttl_days = 90  # Pattern expiry
        
        # Performance tracking
        self.recognition_stats = {
            "total_vectors_analyzed": 0,
            "patterns_discovered": 0,
            "patterns_validated": 0,
            "insights_generated": 0,
            "cross_module_correlations": 0,
            "prediction_accuracy": []
        }
        
        # Clustering state
        self.last_clustering_time = datetime.now(timezone.utc)
        self.clustering_interval_hours = 24
        
        self.logger.info("LanceDBPatternRecognizer initialized")
    
    async def initialize(self, db_path: str = None, module_tables: Dict[str, str] = None):
        """Initialize pattern recognizer with access to multiple vector tables."""
        try:
            if not VECTOR_CAPABILITIES_AVAILABLE:
                self.logger.warning("Vector capabilities not available - using basic pattern recognition only")
                return
            
            # Initialize embedding model
            self.logger.info("Loading BGE-M3 embedding model for pattern recognition...")
            start_time = time.time()
            self.embedder = SentenceTransformer("BAAI/bge-m3", device="cpu")
            load_time = time.time() - start_time
            self.logger.info(f"Embedding model loaded in {load_time:.2f}s")
            
            # Connect to LanceDB
            if db_path:
                self.logger.info(f"Connecting to LanceDB at: {db_path}")
                self.vector_db = await lancedb.connect_async(db_path)
                
                # Setup access to module tables
                await self._setup_module_tables(module_tables or {})
                
                # Initialize performance monitor
                if VECTOR_INFRASTRUCTURE_AVAILABLE and self.performance_monitor is None:
                    self.performance_monitor = VectorPerformanceMonitor()
                    await self.performance_monitor.establish_baseline(
                        "pattern_recognition", 
                        ModuleSource.INTELLIGENCE
                    )
                
                self.logger.info("✅ Pattern recognizer capabilities initialized")
            else:
                self.logger.info("No DB path provided - pattern recognition disabled")
                
        except Exception as e:
            self.logger.error(f"Failed to initialize pattern recognizer: {e}")
            self.embedder = None
            self.vector_db = None
    
    async def _setup_module_tables(self, module_tables: Dict[str, str]):
        """Setup access to vector tables from different modules."""
        try:
            existing_tables = await self.vector_db.table_names()
            
            # Default table mappings if not provided
            default_tables = {
                "intelligence": "intelligence_vectors",
                "complexity": "complexity_vectors", 
                "auto_generation": "auto_generation_vectors",
                "moq": "moq_vectors",
                "investigation": "investigation_vectors",
                "insight_synthesis": "insight_synthesis_vectors"
            }
            
            table_mapping = {**default_tables, **module_tables}
            
            for module, table_name in table_mapping.items():
                if table_name in existing_tables:
                    try:
                        self.vector_tables[module] = await self.vector_db.open_table(table_name)
                        self.logger.info(f"Connected to {module} table: {table_name}")
                    except Exception as e:
                        self.logger.warning(f"Failed to open {module} table {table_name}: {e}")
                else:
                    self.logger.info(f"Table {table_name} not found for module {module}")
            
            self.logger.info(f"Connected to {len(self.vector_tables)} module tables")
            
        except Exception as e:
            self.logger.error(f"Failed to setup module tables: {e}")
    
    @performance_monitor("cross_module_pattern_analysis")
    async def analyze_cross_module_patterns(
        self,
        time_window_days: int = 30,
        min_pattern_strength: PatternStrength = PatternStrength.MODERATE,
        include_predictions: bool = True
    ) -> Dict[str, Any]:
        """
        Comprehensive cross-module pattern analysis.
        
        Args:
            time_window_days: Analysis window in days
            min_pattern_strength: Minimum pattern strength to report
            include_predictions: Whether to generate predictive insights
            
        Returns:
            Comprehensive pattern analysis results
        """
        analysis_start_time = time.perf_counter()
        self.logger.info(f"Starting cross-module pattern analysis for {time_window_days} days")
        
        try:
            # Step 1: Collect vectors from all modules
            all_vectors = await self._collect_cross_module_vectors(time_window_days)
            self.recognition_stats["total_vectors_analyzed"] += len(all_vectors)
            
            # Step 2: Identify semantic patterns
            semantic_patterns = await self._identify_semantic_patterns(all_vectors)
            
            # Step 3: Analyze business domain correlations
            domain_correlations = await self._analyze_domain_correlations(all_vectors)
            
            # Step 4: Detect performance trends
            performance_trends = await self._detect_performance_trends(all_vectors)
            
            # Step 5: Identify cross-module dependencies
            module_dependencies = await self._analyze_module_dependencies(all_vectors)
            
            # Step 6: Generate pattern clusters
            pattern_clusters = await self._generate_pattern_clusters()
            
            # Step 7: Create actionable insights
            insights = await self._generate_actionable_insights(pattern_clusters)
            
            # Step 8: Predictions (if requested)
            predictions = {}
            if include_predictions and self.embedder:
                predictions = await self._generate_pattern_predictions(pattern_clusters)
            
            analysis_time = (time.perf_counter() - analysis_start_time) * 1000
            
            # Compile comprehensive results
            results = {
                "analysis_summary": {
                    "time_window_days": time_window_days,
                    "vectors_analyzed": len(all_vectors),
                    "modules_included": list(self.vector_tables.keys()),
                    "patterns_discovered": len(semantic_patterns),
                    "domain_correlations": len(domain_correlations),
                    "performance_trends": len(performance_trends),
                    "cluster_count": len(pattern_clusters),
                    "insights_generated": len(insights),
                    "analysis_time_ms": analysis_time
                },
                
                "semantic_patterns": semantic_patterns,
                "domain_correlations": domain_correlations,
                "performance_trends": performance_trends,
                "module_dependencies": module_dependencies,
                "pattern_clusters": pattern_clusters,
                "actionable_insights": insights,
                "predictions": predictions,
                
                "recognition_statistics": self.recognition_stats.copy(),
                "quality_metrics": await self._calculate_analysis_quality_metrics()
            }
            
            # Update learning state
            await self._update_pattern_learning_state(results)
            
            self.logger.info(
                f"Cross-module pattern analysis completed: {len(semantic_patterns)} patterns, "
                f"{len(insights)} insights in {analysis_time:.1f}ms"
            )
            
            return results
            
        except Exception as e:
            self.logger.error(f"Cross-module pattern analysis failed: {e}")
            return {
                "error": str(e),
                "analysis_time_ms": (time.perf_counter() - analysis_start_time) * 1000
            }
    
    async def _collect_cross_module_vectors(self, time_window_days: int) -> List[Dict[str, Any]]:
        """Collect vectors from all connected module tables within time window."""
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=time_window_days)
        all_vectors = []
        
        for module_name, table in self.vector_tables.items():
            try:
                # Query vectors within time window - handle async properly
                try:
                    vectors = await table.search(np.random.randn(1024).tolist()).limit(1000).to_list()
                except AttributeError:
                    # Fallback for different LanceDB API versions
                    search_query = table.search(np.random.randn(1024).tolist()).limit(1000)
                    vectors = search_query.to_pandas().to_dict('records') if hasattr(search_query, 'to_pandas') else []
                
                module_vectors = []
                for vector in vectors:
                    created_at_str = vector.get("created_at", "")
                    try:
                        created_at = datetime.fromisoformat(created_at_str.replace('Z', '+00:00'))
                        if created_at >= cutoff_date:
                            vector["module_source"] = module_name
                            module_vectors.append(vector)
                    except (ValueError, AttributeError):
                        # Include vectors without valid timestamps
                        vector["module_source"] = module_name
                        module_vectors.append(vector)
                
                all_vectors.extend(module_vectors)
                self.logger.info(f"Collected {len(module_vectors)} vectors from {module_name}")
                
            except Exception as e:
                self.logger.warning(f"Failed to collect vectors from {module_name}: {e}")
        
        return all_vectors
    
    async def _identify_semantic_patterns(self, vectors: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Identify semantic patterns using clustering and similarity analysis."""
        if not self.embedder or len(vectors) < self.min_pattern_support:
            return []
        
        try:
            # Extract embeddings and metadata
            embeddings = []
            metadata = []
            
            for vector in vectors:
                if "vector" in vector:
                    embeddings.append(vector["vector"])
                    metadata.append({
                        "id": vector.get("id", "unknown"),
                        "module": vector.get("module_source", "unknown"),
                        "business_domain": vector.get("business_domain", "unknown"),
                        "content": json.loads(vector.get("content_json", "{}")).get("query", ""),
                        "complexity_score": vector.get("complexity_score", 0.5),
                        "confidence_score": vector.get("confidence_score", 0.5)
                    })
            
            if len(embeddings) < self.min_pattern_support:
                return []
            
            # Perform clustering to identify semantic patterns
            embeddings_array = np.array(embeddings)
            patterns = self._cluster_embeddings_for_patterns(embeddings_array, metadata)
            
            return patterns
            
        except Exception as e:
            self.logger.warning(f"Failed to identify semantic patterns: {e}")
            return []
    
    def _cluster_embeddings_for_patterns(self, embeddings: np.ndarray, metadata: List[Dict]) -> List[Dict[str, Any]]:
        """Cluster embeddings to identify semantic patterns."""
        from sklearn.cluster import DBSCAN
        from sklearn.metrics.pairwise import cosine_similarity
        
        # Use DBSCAN for density-based clustering
        similarity_matrix = cosine_similarity(embeddings)
        distance_matrix = 1 - similarity_matrix
        
        # DBSCAN clustering
        clustering = DBSCAN(
            metric='precomputed',
            eps=1 - self.pattern_similarity_threshold,
            min_samples=self.min_pattern_support
        ).fit(distance_matrix)
        
        patterns = []
        
        # Analyze each cluster
        for cluster_id in set(clustering.labels_):
            if cluster_id == -1:  # Noise points
                continue
            
            cluster_indices = np.where(clustering.labels_ == cluster_id)[0]
            cluster_metadata = [metadata[i] for i in cluster_indices]
            cluster_embeddings = embeddings[cluster_indices]
            
            # Calculate cluster characteristics
            cluster_center = np.mean(cluster_embeddings, axis=0)
            intra_cluster_similarity = np.mean(cosine_similarity(cluster_embeddings))
            
            # Extract pattern characteristics
            modules_involved = set(m["module"] for m in cluster_metadata)
            domains_involved = set(m["business_domain"] for m in cluster_metadata)
            avg_complexity = np.mean([m["complexity_score"] for m in cluster_metadata])
            avg_confidence = np.mean([m["confidence_score"] for m in cluster_metadata])
            
            # Determine pattern strength
            pattern_strength = self._calculate_pattern_strength(
                len(cluster_indices), intra_cluster_similarity, len(modules_involved)
            )
            
            # Extract keywords from cluster content
            all_content = " ".join(m["content"] for m in cluster_metadata if m["content"])
            keywords = self._extract_pattern_keywords(all_content)
            
            pattern = {
                "pattern_id": f"pattern_{cluster_id}_{int(time.time())}",
                "pattern_type": self._classify_pattern_type(cluster_metadata),
                "strength": pattern_strength.value,
                "confidence": intra_cluster_similarity,
                "modules_involved": list(modules_involved),
                "business_domains": list(domains_involved),
                "occurrence_count": len(cluster_indices),
                "avg_complexity_score": avg_complexity,
                "avg_confidence_score": avg_confidence,
                "keywords": keywords,
                "cluster_center": cluster_center.tolist(),
                "member_count": len(cluster_indices),
                "cross_module": len(modules_involved) > 1
            }
            
            patterns.append(pattern)
        
        return patterns
    
    def _calculate_pattern_strength(self, occurrence_count: int, similarity: float, module_count: int) -> PatternStrength:
        """Calculate pattern strength based on multiple factors."""
        # Base strength from similarity
        base_strength = similarity
        
        # Boost for occurrence count
        occurrence_boost = min(occurrence_count / 10, 0.2)
        
        # Boost for cross-module patterns
        cross_module_boost = (module_count - 1) * 0.1 if module_count > 1 else 0
        
        total_strength = base_strength + occurrence_boost + cross_module_boost
        
        if total_strength >= 0.85:
            return PatternStrength.VERY_STRONG
        elif total_strength >= 0.7:
            return PatternStrength.STRONG
        elif total_strength >= 0.5:
            return PatternStrength.MODERATE
        else:
            return PatternStrength.WEAK
    
    def _classify_pattern_type(self, metadata: List[Dict]) -> PatternType:
        """Classify the type of pattern based on metadata analysis."""
        # Simple heuristic classification
        modules = set(m["module"] for m in metadata)
        domains = set(m["business_domain"] for m in metadata)
        
        if len(modules) > 1:
            return PatternType.CROSS_MODULE_DEPENDENCY
        elif len(domains) > 1:
            return PatternType.DOMAIN_CORRELATION
        else:
            # Analyze content for pattern type
            content = " ".join(m["content"] for m in metadata if m["content"]).lower()
            
            if any(word in content for word in ["efficiency", "performance", "optimization"]):
                return PatternType.PERFORMANCE_TREND
            elif any(word in content for word in ["process", "workflow", "procedure"]):
                return PatternType.BUSINESS_PROCESS
            elif any(word in content for word in ["trend", "time", "historical"]):
                return PatternType.SEASONAL_PATTERN
            else:
                return PatternType.BUSINESS_PROCESS
    
    def _extract_pattern_keywords(self, content: str) -> List[str]:
        """Extract key terms from pattern content."""
        if not content:
            return []
        
        # Simple keyword extraction (could be enhanced with NLP)
        words = content.lower().split()
        
        # Filter for business-relevant terms
        business_terms = [
            word for word in words
            if len(word) > 3 and word.isalpha() and
            word not in {"this", "that", "with", "from", "they", "have", "been", "were", "will"}
        ]
        
        # Count frequency and return top terms
        from collections import Counter
        term_counts = Counter(business_terms)
        return [term for term, count in term_counts.most_common(10)]
    
    async def _analyze_domain_correlations(self, vectors: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Analyze correlations between business domains."""
        correlations = []
        
        try:
            # Group vectors by business domain
            domain_groups = defaultdict(list)
            for vector in vectors:
                domain = vector.get("business_domain", "unknown")
                domain_groups[domain].append(vector)
            
            # Calculate pairwise domain correlations
            domains = list(domain_groups.keys())
            for i, domain1 in enumerate(domains):
                for domain2 in domains[i+1:]:
                    correlation = self._calculate_domain_correlation(
                        domain_groups[domain1], 
                        domain_groups[domain2]
                    )
                    
                    if correlation["strength"] > 0.3:  # Threshold for meaningful correlation
                        correlations.append({
                            "domain_pair": [domain1, domain2],
                            "correlation_strength": correlation["strength"],
                            "shared_patterns": correlation["shared_patterns"],
                            "business_impact": correlation["business_impact"]
                        })
            
        except Exception as e:
            self.logger.warning(f"Failed to analyze domain correlations: {e}")
        
        return correlations
    
    def _calculate_domain_correlation(self, vectors1: List[Dict], vectors2: List[Dict]) -> Dict[str, Any]:
        """Calculate correlation between two domain vector sets."""
        # Simplified correlation calculation
        shared_keywords = self._find_shared_keywords(vectors1, vectors2)
        
        # Calculate similarity based on shared concepts
        correlation_strength = len(shared_keywords) / 10  # Normalized
        correlation_strength = min(correlation_strength, 1.0)
        
        return {
            "strength": correlation_strength,
            "shared_patterns": shared_keywords,
            "business_impact": "moderate" if correlation_strength > 0.5 else "low"
        }
    
    def _find_shared_keywords(self, vectors1: List[Dict], vectors2: List[Dict]) -> List[str]:
        """Find shared keywords between two vector sets."""
        def extract_keywords(vectors):
            all_content = ""
            for vector in vectors:
                content_json = json.loads(vector.get("content_json", "{}"))
                all_content += " " + content_json.get("query", "")
            return set(self._extract_pattern_keywords(all_content))
        
        keywords1 = extract_keywords(vectors1)
        keywords2 = extract_keywords(vectors2)
        
        return list(keywords1.intersection(keywords2))
    
    async def _detect_performance_trends(self, vectors: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Detect performance trends across modules and time."""
        trends = []
        
        try:
            # Group vectors by time periods (weekly)
            time_groups = defaultdict(list)
            
            for vector in vectors:
                created_at_str = vector.get("created_at", "")
                try:
                    created_at = datetime.fromisoformat(created_at_str.replace('Z', '+00:00'))
                    week_key = created_at.strftime("%Y-W%U")
                    time_groups[week_key].append(vector)
                except (ValueError, AttributeError):
                    continue
            
            # Analyze trends for each performance metric
            metrics = ["complexity_score", "confidence_score", "processing_time_ms"]
            
            for metric in metrics:
                trend = self._calculate_metric_trend(time_groups, metric)
                if trend["significance"] > 0.3:
                    trends.append(trend)
        
        except Exception as e:
            self.logger.warning(f"Failed to detect performance trends: {e}")
        
        return trends
    
    def _calculate_metric_trend(self, time_groups: Dict[str, List], metric: str) -> Dict[str, Any]:
        """Calculate trend for a specific metric over time."""
        time_series = []
        
        for week, vectors in sorted(time_groups.items()):
            values = []
            for vector in vectors:
                if metric == "processing_time_ms":
                    # Extract from module metadata
                    metadata = json.loads(vector.get("module_metadata", "{}"))
                    value = metadata.get("performance", {}).get("processing_time_ms", 0)
                else:
                    value = vector.get(metric, 0)
                
                if value > 0:
                    values.append(float(value))
            
            if values:
                avg_value = sum(values) / len(values)
                time_series.append((week, avg_value))
        
        if len(time_series) < 3:
            return {"metric": metric, "significance": 0.0}
        
        # Simple linear trend calculation
        x_values = list(range(len(time_series)))
        y_values = [y for _, y in time_series]
        
        # Calculate trend slope
        n = len(x_values)
        sum_x = sum(x_values)
        sum_y = sum(y_values)
        sum_xy = sum(x * y for x, y in zip(x_values, y_values))
        sum_x2 = sum(x * x for x in x_values)
        
        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
        
        # Determine trend direction and significance
        trend_direction = "increasing" if slope > 0 else "decreasing" if slope < 0 else "stable"
        significance = min(abs(slope) / max(y_values), 1.0)
        
        return {
            "metric": metric,
            "trend_direction": trend_direction,
            "slope": slope,
            "significance": significance,
            "time_periods": len(time_series)
        }
    
    async def _analyze_module_dependencies(self, vectors: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze dependencies and relationships between modules."""
        dependencies = {
            "direct_dependencies": {},
            "correlation_matrix": {},
            "dependency_strength": {}
        }
        
        try:
            # Group vectors by module
            module_groups = defaultdict(list)
            for vector in vectors:
                module = vector.get("module_source", "unknown")
                module_groups[module].append(vector)
            
            modules = list(module_groups.keys())
            
            # Build correlation matrix
            correlation_matrix = {}
            for module1 in modules:
                correlation_matrix[module1] = {}
                for module2 in modules:
                    if module1 == module2:
                        correlation_matrix[module1][module2] = 1.0
                    else:
                        correlation = self._calculate_module_correlation(
                            module_groups[module1], 
                            module_groups[module2]
                        )
                        correlation_matrix[module1][module2] = correlation
            
            dependencies["correlation_matrix"] = correlation_matrix
            
            # Identify strong dependencies (correlation > 0.6)
            for module1 in modules:
                deps = []
                for module2 in modules:
                    if module1 != module2 and correlation_matrix[module1][module2] > 0.6:
                        deps.append(module2)
                dependencies["direct_dependencies"][module1] = deps
            
        except Exception as e:
            self.logger.warning(f"Failed to analyze module dependencies: {e}")
        
        return dependencies
    
    def _calculate_module_correlation(self, vectors1: List[Dict], vectors2: List[Dict]) -> float:
        """Calculate correlation between two modules based on their vectors."""
        # Simplified correlation based on shared business domains and keywords
        domains1 = set(v.get("business_domain", "") for v in vectors1)
        domains2 = set(v.get("business_domain", "") for v in vectors2)
        
        domain_overlap = len(domains1.intersection(domains2)) / max(len(domains1.union(domains2)), 1)
        
        # Add keyword similarity
        keywords1 = set()
        keywords2 = set()
        
        for vector in vectors1:
            content = json.loads(vector.get("content_json", "{}")).get("query", "")
            keywords1.update(self._extract_pattern_keywords(content))
        
        for vector in vectors2:
            content = json.loads(vector.get("content_json", "{}")).get("query", "")
            keywords2.update(self._extract_pattern_keywords(content))
        
        keyword_overlap = len(keywords1.intersection(keywords2)) / max(len(keywords1.union(keywords2)), 1)
        
        return (domain_overlap + keyword_overlap) / 2
    
    async def _generate_pattern_clusters(self) -> List[Dict[str, Any]]:
        """Generate higher-level pattern clusters for strategic insights."""
        clusters = []
        
        try:
            if not self.discovered_patterns:
                return clusters
            
            # Convert patterns to feature vectors for clustering
            pattern_features = []
            pattern_ids = []
            
            for pattern_id, pattern in self.discovered_patterns.items():
                features = self._extract_pattern_features(pattern)
                pattern_features.append(features)
                pattern_ids.append(pattern_id)
            
            if len(pattern_features) < 2:
                return clusters
            
            # Cluster patterns
            from sklearn.cluster import KMeans
            n_clusters = min(len(pattern_features) // 2, 5)  # Max 5 clusters
            
            if n_clusters >= 2:
                clustering = KMeans(n_clusters=n_clusters, random_state=42)
                cluster_labels = clustering.fit_predict(pattern_features)
                
                # Create cluster summaries
                for cluster_id in range(n_clusters):
                    cluster_pattern_ids = [
                        pattern_ids[i] for i, label in enumerate(cluster_labels) 
                        if label == cluster_id
                    ]
                    
                    if cluster_pattern_ids:
                        cluster = self._create_cluster_summary(cluster_id, cluster_pattern_ids)
                        clusters.append(cluster)
        
        except Exception as e:
            self.logger.warning(f"Failed to generate pattern clusters: {e}")
        
        return clusters
    
    def _extract_pattern_features(self, pattern: CrossModulePattern) -> List[float]:
        """Extract numerical features from pattern for clustering."""
        return [
            pattern.confidence,
            pattern.avg_complexity_score,
            len(pattern.modules_involved),
            len(pattern.business_domains),
            pattern.occurrence_count / 10,  # Normalized
            pattern.success_rate,
            pattern.resource_efficiency_score
        ]
    
    def _create_cluster_summary(self, cluster_id: int, pattern_ids: List[str]) -> Dict[str, Any]:
        """Create summary for a pattern cluster."""
        cluster_patterns = [self.discovered_patterns[pid] for pid in pattern_ids]
        
        # Aggregate cluster characteristics
        avg_confidence = sum(p.confidence for p in cluster_patterns) / len(cluster_patterns)
        all_modules = set()
        all_domains = set()
        
        for pattern in cluster_patterns:
            all_modules.update(pattern.modules_involved)
            all_domains.update(pattern.business_domains)
        
        return {
            "cluster_id": f"cluster_{cluster_id}",
            "pattern_count": len(cluster_patterns),
            "avg_confidence": avg_confidence,
            "modules_involved": list(all_modules),
            "business_domains": list(all_domains),
            "dominant_pattern_type": self._find_dominant_pattern_type(cluster_patterns),
            "strategic_significance": "high" if len(all_modules) > 2 else "medium"
        }
    
    def _find_dominant_pattern_type(self, patterns: List[CrossModulePattern]) -> str:
        """Find the most common pattern type in a cluster."""
        type_counts = defaultdict(int)
        for pattern in patterns:
            type_counts[pattern.pattern_type.value] += 1
        
        return max(type_counts.items(), key=lambda x: x[1])[0] if type_counts else "unknown"
    
    async def _generate_actionable_insights(self, clusters: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate actionable business insights from pattern analysis."""
        insights = []
        
        for cluster in clusters:
            try:
                insight = self._create_insight_from_cluster(cluster)
                if insight:
                    insights.append(insight)
            except Exception as e:
                self.logger.warning(f"Failed to generate insight for cluster {cluster.get('cluster_id')}: {e}")
        
        # Add overall insights
        if len(clusters) > 1:
            overall_insight = self._create_overall_system_insight(clusters)
            if overall_insight:
                insights.append(overall_insight)
        
        return insights
    
    def _create_insight_from_cluster(self, cluster: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Create actionable insight from a pattern cluster."""
        cluster_id = cluster.get("cluster_id", "unknown")
        modules = cluster.get("modules_involved", [])
        domains = cluster.get("business_domains", [])
        
        if len(modules) > 1:
            # Cross-module optimization opportunity
            return {
                "insight_id": f"insight_{cluster_id}_{int(time.time())}",
                "type": "optimization",
                "title": f"Cross-Module Optimization Opportunity",
                "description": f"Strong patterns detected across {', '.join(modules)} modules in {', '.join(domains)} domains",
                "recommendations": [
                    f"Integrate {' and '.join(modules)} for improved efficiency",
                    "Standardize processes across identified domains",
                    "Consider unified data models for better consistency"
                ],
                "business_impact": "medium_to_high",
                "confidence": cluster.get("avg_confidence", 0.5),
                "modules_affected": modules,
                "domains_affected": domains
            }
        elif cluster.get("pattern_count", 0) > 5:
            # High-frequency pattern insight
            return {
                "insight_id": f"insight_{cluster_id}_{int(time.time())}",
                "type": "efficiency",
                "title": "High-Frequency Pattern Detected",
                "description": f"Recurring pattern in {domains[0] if domains else 'unknown'} domain with {cluster.get('pattern_count')} occurrences",
                "recommendations": [
                    "Consider automating this recurring process",
                    "Optimize resource allocation for this pattern",
                    "Create templates or shortcuts for efficiency"
                ],
                "business_impact": "medium",
                "confidence": cluster.get("avg_confidence", 0.5),
                "modules_affected": modules,
                "domains_affected": domains
            }
        
        return None
    
    def _create_overall_system_insight(self, clusters: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create system-wide insight from all clusters."""
        all_modules = set()
        all_domains = set()
        total_patterns = 0
        
        for cluster in clusters:
            all_modules.update(cluster.get("modules_involved", []))
            all_domains.update(cluster.get("business_domains", []))
            total_patterns += cluster.get("pattern_count", 0)
        
        return {
            "insight_id": f"system_insight_{int(time.time())}",
            "type": "strategic",
            "title": "System-Wide Pattern Analysis",
            "description": f"Analysis of {total_patterns} patterns across {len(all_modules)} modules and {len(all_domains)} business domains",
            "recommendations": [
                "Consider enterprise-wide data integration strategy",
                "Standardize cross-module interfaces",
                "Implement unified monitoring and analytics",
                "Develop cross-functional optimization initiatives"
            ],
            "business_impact": "high",
            "confidence": 0.8,
            "modules_affected": list(all_modules),
            "domains_affected": list(all_domains),
            "strategic_priority": "high"
        }
    
    async def _generate_pattern_predictions(self, clusters: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate predictive insights based on pattern analysis."""
        predictions = {
            "trend_predictions": [],
            "anomaly_forecasts": [],
            "optimization_forecasts": [],
            "confidence_level": 0.0
        }
        
        try:
            # Simple prediction logic - could be enhanced with ML
            for cluster in clusters:
                if cluster.get("pattern_count", 0) > 3:
                    prediction = {
                        "prediction_id": f"pred_{cluster['cluster_id']}",
                        "type": "trend_continuation",
                        "description": f"Pattern in {cluster.get('business_domains', ['unknown'])[0]} likely to continue",
                        "probability": min(cluster.get("avg_confidence", 0.5) * 1.2, 0.95),
                        "time_horizon": "30_days",
                        "expected_impact": "continued_efficiency" if cluster.get("avg_confidence", 0) > 0.7 else "potential_optimization"
                    }
                    predictions["trend_predictions"].append(prediction)
            
            if predictions["trend_predictions"]:
                predictions["confidence_level"] = sum(
                    p["probability"] for p in predictions["trend_predictions"]
                ) / len(predictions["trend_predictions"])
        
        except Exception as e:
            self.logger.warning(f"Failed to generate predictions: {e}")
        
        return predictions
    
    async def _calculate_analysis_quality_metrics(self) -> Dict[str, float]:
        """Calculate quality metrics for the pattern analysis."""
        return {
            "pattern_discovery_rate": len(self.discovered_patterns) / max(self.recognition_stats["total_vectors_analyzed"], 1),
            "cross_module_coverage": len(self.vector_tables) / 6,  # Assuming 6 total modules
            "insight_generation_rate": self.recognition_stats["insights_generated"] / max(len(self.discovered_patterns), 1),
            "validation_success_rate": self.recognition_stats["patterns_validated"] / max(self.recognition_stats["patterns_discovered"], 1),
            "avg_prediction_accuracy": sum(self.recognition_stats["prediction_accuracy"]) / max(len(self.recognition_stats["prediction_accuracy"]), 1) if self.recognition_stats["prediction_accuracy"] else 0.0
        }
    
    async def _update_pattern_learning_state(self, analysis_results: Dict[str, Any]):
        """Update internal learning state based on analysis results."""
        # Update discovered patterns
        semantic_patterns = analysis_results.get("semantic_patterns", [])
        for pattern_data in semantic_patterns:
            pattern_id = pattern_data["pattern_id"]
            
            # Convert to CrossModulePattern object
            pattern = CrossModulePattern(
                pattern_id=pattern_id,
                pattern_type=PatternType(pattern_data.get("pattern_type", "business_process")),
                strength=PatternStrength(pattern_data.get("strength", "moderate")),
                confidence=pattern_data.get("confidence", 0.5),
                modules_involved=set(pattern_data.get("modules_involved", [])),
                business_domains=set(pattern_data.get("business_domains", [])),
                time_span_days=30,  # Default
                occurrence_count=pattern_data.get("occurrence_count", 1),
                semantic_cluster_center=pattern_data.get("cluster_center", []),
                pattern_keywords=pattern_data.get("keywords", []),
                business_metrics_involved=[],
                avg_complexity_score=pattern_data.get("avg_complexity_score", 0.5),
                avg_processing_time_ms=100.0,  # Default
                success_rate=0.8,  # Default
                resource_efficiency_score=0.7,  # Default
                trend_direction="stable",
                trend_strength=0.0,
                last_occurrence=datetime.now(timezone.utc),
                prediction_accuracy=0.0,
                module_dependencies={},
                correlation_matrix={},
                causal_relationships=[]
            )
            
            self.discovered_patterns[pattern_id] = pattern
        
        # Update statistics
        self.recognition_stats["patterns_discovered"] = len(self.discovered_patterns)
        self.recognition_stats["insights_generated"] = len(analysis_results.get("actionable_insights", []))
    
    async def get_pattern_summary(self) -> Dict[str, Any]:
        """Get comprehensive summary of all discovered patterns."""
        return {
            "discovery_summary": {
                "total_patterns": len(self.discovered_patterns),
                "pattern_clusters": len(self.pattern_clusters),
                "actionable_insights": len(self.pattern_insights),
                "modules_connected": len(self.vector_tables),
                "last_analysis": self.last_clustering_time.isoformat()
            },
            
            "pattern_breakdown": {
                pattern_type.value: len([
                    p for p in self.discovered_patterns.values() 
                    if p.pattern_type == pattern_type
                ]) for pattern_type in PatternType
            },
            
            "strength_distribution": {
                strength.value: len([
                    p for p in self.discovered_patterns.values() 
                    if p.strength == strength
                ]) for strength in PatternStrength
            },
            
            "cross_module_patterns": len([
                p for p in self.discovered_patterns.values() 
                if len(p.modules_involved) > 1
            ]),
            
            "recognition_statistics": self.recognition_stats.copy(),
            
            "capability_status": {
                "vector_capabilities": VECTOR_CAPABILITIES_AVAILABLE,
                "vector_infrastructure": VECTOR_INFRASTRUCTURE_AVAILABLE,
                "intelligence_module": INTELLIGENCE_MODULE_AVAILABLE,
                "embedder_loaded": self.embedder is not None,
                "tables_connected": len(self.vector_tables)
            }
        }
    
    async def cleanup(self):
        """Cleanup pattern recognizer resources."""
        try:
            # Clear pattern caches
            self.discovered_patterns.clear()
            self.pattern_clusters.clear()
            self.pattern_insights.clear()
            
            # Close vector connections if available
            if hasattr(self, 'vector_db') and self.vector_db:
                # LanceDB connections are typically auto-managed
                pass
            
            self.logger.info("LanceDBPatternRecognizer cleanup completed")
            
        except Exception as e:
            self.logger.error(f"Cleanup error: {e}")


# Utility functions for external integration
async def create_lancedb_pattern_recognizer(
    vector_index_manager: Optional[VectorIndexManager] = None,
    db_path: str = None,
    module_tables: Dict[str, str] = None
) -> LanceDBPatternRecognizer:
    """Factory function to create and initialize LanceDB pattern recognizer."""
    recognizer = LanceDBPatternRecognizer(vector_index_manager)
    await recognizer.initialize(db_path, module_tables)
    return recognizer


# Export main classes and functions
__all__ = [
    "LanceDBPatternRecognizer",
    "CrossModulePattern",
    "PatternInsight", 
    "PatternCluster",
    "PatternType",
    "PatternStrength",
    "create_lancedb_pattern_recognizer"
]