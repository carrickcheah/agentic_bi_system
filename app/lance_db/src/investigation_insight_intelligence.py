#!/usr/bin/env python3
"""
Investigation-Insight Cross-Module Intelligence - Phase 2.3 Implementation
Connects Investigation and Insight Synthesis modules for enhanced business intelligence.
Provides bidirectional learning, pattern correlation, and predictive capabilities.
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

# Import enterprise vector infrastructure
try:
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

# Import vector capabilities
try:
    import lancedb
    from sentence_transformers import SentenceTransformer
    VECTOR_CAPABILITIES_AVAILABLE = True
except ImportError:
    print("⚠️ Warning: Vector capabilities not available")
    VECTOR_CAPABILITIES_AVAILABLE = False

# Try to import numpy with fallback
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    # Simple fallback implementations
    import random as _random
    
    class np:
        @staticmethod
        def mean(values):
            return sum(values) / len(values) if values else 0
        
        @staticmethod
        def std(values):
            if not values:
                return 0
            mean = sum(values) / len(values)
            variance = sum((x - mean) ** 2 for x in values) / len(values)
            return variance ** 0.5
        
        @staticmethod
        def median(values):
            if not values:
                return 0
            sorted_values = sorted(values)
            n = len(sorted_values)
            if n % 2 == 0:
                return (sorted_values[n//2-1] + sorted_values[n//2]) / 2
            else:
                return sorted_values[n//2]


class InvestigationInsightLinkType(Enum):
    """Types of links between investigations and insights."""
    DIRECT_GENERATION = "direct_generation"      # Investigation directly led to insight
    PATTERN_CORRELATION = "pattern_correlation"   # Similar patterns discovered
    DOMAIN_OVERLAP = "domain_overlap"            # Same business domain
    TEMPORAL_SEQUENCE = "temporal_sequence"      # Time-based relationship
    CAUSAL_CHAIN = "causal_chain"              # Causal relationship
    VALIDATION_SUPPORT = "validation_support"    # Insight validates investigation
    HYPOTHESIS_EVOLUTION = "hypothesis_evolution" # Investigation hypothesis became insight


class FeedbackLoopType(Enum):
    """Types of feedback loops between modules."""
    INSIGHT_QUALITY = "insight_quality"          # Insight quality improves investigations
    INVESTIGATION_DEPTH = "investigation_depth"   # Investigation depth improves insights
    PATTERN_DISCOVERY = "pattern_discovery"      # Cross-module pattern discovery
    CONFIDENCE_BOOST = "confidence_boost"        # Mutual confidence improvement
    DOMAIN_EXPANSION = "domain_expansion"        # Domain knowledge expansion


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


@dataclass
class FeedbackLoop:
    """Feedback loop between Investigation and Insight modules."""
    loop_id: str
    loop_type: FeedbackLoopType
    strength: float  # 0.0 to 1.0
    
    # Loop characteristics
    iterations_count: int
    improvement_rate: float
    convergence_speed: float
    
    # Impact metrics
    investigation_improvement: float
    insight_improvement: float
    overall_effectiveness: float
    
    # Active elements
    active_investigations: List[str]
    active_insights: List[str]
    
    # Status
    is_active: bool
    last_activation: datetime


@dataclass
class InvestigationInsightIntelligence:
    """Comprehensive cross-module intelligence between Investigation and Insight."""
    intelligence_id: str
    analysis_timestamp: datetime
    
    # Links and patterns
    active_links: List[InvestigationInsightLink]
    discovered_patterns: List[CrossModulePattern]
    feedback_loops: List[FeedbackLoop]
    
    # Performance metrics
    avg_investigation_to_insight_time: float
    insight_generation_success_rate: float
    pattern_discovery_rate: float
    
    # Quality metrics
    avg_insight_quality_score: float
    avg_investigation_confidence: float
    cross_validation_accuracy: float
    
    # Business impact
    total_business_value: float
    roi_multiplier: float
    strategic_alignment_score: float
    
    # Predictions
    predicted_insight_quality: Dict[str, float]
    recommended_investigation_areas: List[str]
    optimization_opportunities: List[Dict[str, Any]]


class InvestigationInsightIntelligenceEngine:
    """
    Engine for managing cross-module intelligence between Investigation and Insight Synthesis.
    Provides bidirectional learning, pattern discovery, and optimization.
    """
    
    def __init__(self):
        """Initialize the intelligence engine."""
        self.embedder = None
        self.vector_db = None
        self.vector_table = None
        self.schema_manager = None
        self.index_manager = None
        self.performance_monitor = None
        
        # Intelligence cache
        self.link_cache: Dict[str, InvestigationInsightLink] = {}
        self.pattern_cache: Dict[str, CrossModulePattern] = {}
        self.feedback_loops: Dict[str, FeedbackLoop] = {}
        
        # Logging
        self.logger = self._setup_logger()
        
    def _setup_logger(self):
        """Set up logging for the intelligence engine."""
        import logging
        logger = logging.getLogger("investigation_insight_intelligence")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    async def initialize(self, db_path: Optional[str] = None):
        """Initialize vector infrastructure for cross-module intelligence."""
        if not VECTOR_CAPABILITIES_AVAILABLE:
            self.logger.warning("Vector capabilities not available - running in limited mode")
            return
        
        try:
            # Initialize embedder
            self.embedder = SentenceTransformer('BAAI/bge-m3')
            
            # Initialize LanceDB
            from pathlib import Path
            db_path = db_path or str(Path(__file__).parent.parent / "data")
            self.vector_db = await lancedb.connect_async(db_path)
            
            # Initialize infrastructure
            if VECTOR_INFRASTRUCTURE_AVAILABLE:
                self.schema_manager = EnterpriseVectorSchema()
                self.index_manager = VectorIndexManager(self.vector_db)
                self.performance_monitor = VectorPerformanceMonitor()
            
            # Open vector table
            table_name = "enterprise_vectors"
            existing_tables = await self.vector_db.table_names()
            
            if table_name in existing_tables:
                self.vector_table = await self.vector_db.open_table(table_name)
                self.logger.info(f"Opened vector table: {table_name}")
            else:
                self.logger.warning(f"Vector table {table_name} not found")
            
            self.logger.info("Cross-module intelligence engine initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize: {e}")
    
    async def analyze_investigation_to_insight_flow(
        self,
        investigation_id: str,
        time_window_hours: int = 24
    ) -> List[InvestigationInsightLink]:
        """
        Analyze how an investigation led to insights.
        
        Args:
            investigation_id: ID of the investigation to analyze
            time_window_hours: Time window to look for related insights
            
        Returns:
            List of investigation-insight links
        """
        if not self.vector_table:
            return []
        
        links = []
        
        try:
            # Get investigation vector
            investigation_vector = await self._get_investigation_vector(investigation_id)
            if not investigation_vector:
                return []
            
            # Search for related insights within time window
            start_time = investigation_vector.get("timestamp", datetime.now(timezone.utc))
            end_time = start_time + timedelta(hours=time_window_hours)
            
            # Find insights in the time window
            insights = await self._find_insights_in_timeframe(start_time, end_time)
            
            # Analyze relationships
            for insight in insights:
                link = await self._analyze_investigation_insight_relationship(
                    investigation_vector, insight
                )
                if link and link.correlation_strength > 0.6:
                    links.append(link)
            
            # Sort by correlation strength
            links.sort(key=lambda x: x.correlation_strength, reverse=True)
            
            # Cache the links
            for link in links:
                self.link_cache[link.link_id] = link
            
            return links
            
        except Exception as e:
            self.logger.error(f"Failed to analyze investigation flow: {e}")
            return []
    
    async def discover_cross_module_patterns(
        self,
        min_occurrences: int = 3,
        time_window_days: int = 30
    ) -> List[CrossModulePattern]:
        """
        Discover patterns that span Investigation and Insight modules.
        
        Args:
            min_occurrences: Minimum occurrences to be considered a pattern
            time_window_days: Time window for pattern analysis
            
        Returns:
            List of discovered cross-module patterns
        """
        if not self.vector_table:
            return []
        
        patterns = []
        
        try:
            # Get vectors from both modules
            cutoff_time = datetime.now(timezone.utc) - timedelta(days=time_window_days)
            
            investigations = await self._get_module_vectors(
                ModuleSource.INVESTIGATION, cutoff_time
            )
            insights = await self._get_module_vectors(
                ModuleSource.INSIGHT_SYNTHESIS, cutoff_time
            )
            
            if not investigations or not insights:
                return []
            
            # Cluster investigations and insights
            investigation_clusters = self._cluster_vectors(investigations, n_clusters=10)
            insight_clusters = self._cluster_vectors(insights, n_clusters=10)
            
            # Find cross-cluster patterns
            pattern_candidates = self._find_cross_cluster_patterns(
                investigation_clusters, insight_clusters
            )
            
            # Validate patterns
            for candidate in pattern_candidates:
                if candidate["occurrence_count"] >= min_occurrences:
                    pattern = self._create_pattern_from_candidate(candidate)
                    patterns.append(pattern)
                    self.pattern_cache[pattern.pattern_id] = pattern
            
            # Sort by business value
            patterns.sort(key=lambda x: x.business_value_generated, reverse=True)
            
            return patterns
            
        except Exception as e:
            self.logger.error(f"Failed to discover patterns: {e}")
            return []
    
    async def establish_feedback_loops(
        self,
        min_iterations: int = 2
    ) -> List[FeedbackLoop]:
        """
        Establish and track feedback loops between modules.
        
        Args:
            min_iterations: Minimum iterations to establish a loop
            
        Returns:
            List of active feedback loops
        """
        loops = []
        
        try:
            # Analyze existing links for feedback patterns
            link_chains = self._find_link_chains(min_length=min_iterations)
            
            for chain in link_chains:
                loop_type = self._determine_feedback_type(chain)
                if loop_type:
                    loop = FeedbackLoop(
                        loop_id=str(uuid.uuid4()),
                        loop_type=loop_type,
                        strength=self._calculate_loop_strength(chain),
                        iterations_count=len(chain),
                        improvement_rate=self._calculate_improvement_rate(chain),
                        convergence_speed=self._calculate_convergence_speed(chain),
                        investigation_improvement=self._measure_investigation_improvement(chain),
                        insight_improvement=self._measure_insight_improvement(chain),
                        overall_effectiveness=self._calculate_loop_effectiveness(chain),
                        active_investigations=[link.investigation_id for link in chain],
                        active_insights=[id for link in chain for id in link.insight_ids],
                        is_active=True,
                        last_activation=datetime.now(timezone.utc)
                    )
                    loops.append(loop)
                    self.feedback_loops[loop.loop_id] = loop
            
            return loops
            
        except Exception as e:
            self.logger.error(f"Failed to establish feedback loops: {e}")
            return []
    
    async def predict_insight_quality(
        self,
        investigation_results: Dict[str, Any]
    ) -> Dict[str, float]:
        """
        Predict the quality of insights that will be generated from investigation results.
        
        Args:
            investigation_results: Results from an investigation
            
        Returns:
            Predicted quality scores for different insight types
        """
        predictions = {}
        
        try:
            # Extract investigation features
            features = self._extract_investigation_features(investigation_results)
            
            # Find similar past investigations
            similar_investigations = await self._find_similar_investigations(
                features, limit=10
            )
            
            if not similar_investigations:
                # Default predictions
                return {
                    "overall_quality": 0.7,
                    "actionability": 0.65,
                    "strategic_value": 0.6,
                    "confidence": 0.75
                }
            
            # Analyze insights from similar investigations
            insight_qualities = []
            for inv in similar_investigations:
                inv_id = inv.get("investigation_id")
                if inv_id in self.link_cache:
                    link = self.link_cache[inv_id]
                    insight_qualities.append({
                        "quality": link.insight_quality_improvement,
                        "impact": link.business_impact,
                        "confidence": link.confidence_score,
                        "similarity": inv.get("similarity", 0.5)
                    })
            
            # Weight by similarity
            if insight_qualities:
                weighted_quality = sum(
                    q["quality"] * q["similarity"] for q in insight_qualities
                ) / sum(q["similarity"] for q in insight_qualities)
                
                weighted_impact = sum(
                    q["impact"] * q["similarity"] for q in insight_qualities
                ) / sum(q["similarity"] for q in insight_qualities)
                
                weighted_confidence = sum(
                    q["confidence"] * q["similarity"] for q in insight_qualities
                ) / sum(q["similarity"] for q in insight_qualities)
                
                predictions = {
                    "overall_quality": weighted_quality,
                    "actionability": weighted_impact * 0.8,
                    "strategic_value": weighted_impact * 0.9,
                    "confidence": weighted_confidence
                }
            else:
                predictions = {
                    "overall_quality": 0.7,
                    "actionability": 0.65,
                    "strategic_value": 0.6,
                    "confidence": 0.75
                }
            
            return predictions
            
        except Exception as e:
            self.logger.error(f"Failed to predict insight quality: {e}")
            return {"overall_quality": 0.5, "error": str(e)}
    
    async def recommend_investigation_areas(
        self,
        current_insights: List[Dict[str, Any]],
        business_context: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Recommend investigation areas based on current insights and patterns.
        
        Args:
            current_insights: List of current insights
            business_context: Business context and goals
            
        Returns:
            Recommended investigation areas with rationale
        """
        recommendations = []
        
        try:
            # Analyze insight gaps
            insight_domains = self._extract_insight_domains(current_insights)
            domain_coverage = self._analyze_domain_coverage(insight_domains)
            
            # Find successful investigation patterns
            successful_patterns = [
                p for p in self.pattern_cache.values()
                if p.business_value_generated > 0.7
            ]
            
            # Generate recommendations
            for pattern in successful_patterns:
                # Check if pattern applies to current context
                if self._pattern_applies_to_context(pattern, business_context):
                    # Check for gaps
                    gap_score = self._calculate_gap_score(
                        pattern.business_domains, domain_coverage
                    )
                    
                    if gap_score > 0.5:
                        recommendation = {
                            "investigation_area": pattern.pattern_name,
                            "description": pattern.pattern_description,
                            "priority_score": gap_score * pattern.business_value_generated,
                            "expected_value": pattern.business_value_generated,
                            "confidence": pattern.prediction_accuracy,
                            "rationale": f"Based on pattern with {pattern.occurrence_count} successful instances",
                            "example_domains": pattern.business_domains[:3],
                            "estimated_duration": self._estimate_investigation_duration(pattern)
                        }
                        recommendations.append(recommendation)
            
            # Add exploratory recommendations for uncovered domains
            uncovered_domains = self._find_uncovered_domains(domain_coverage)
            for domain in uncovered_domains[:3]:
                recommendations.append({
                    "investigation_area": f"Explore {domain} opportunities",
                    "description": f"Investigate potential insights in {domain} domain",
                    "priority_score": 0.6,
                    "expected_value": 0.5,
                    "confidence": 0.4,
                    "rationale": "Domain not well covered by current insights",
                    "example_domains": [domain],
                    "estimated_duration": "2-4 hours"
                })
            
            # Sort by priority
            recommendations.sort(key=lambda x: x["priority_score"], reverse=True)
            
            return recommendations[:10]  # Top 10 recommendations
            
        except Exception as e:
            self.logger.error(f"Failed to recommend investigation areas: {e}")
            return []
    
    async def optimize_investigation_insight_pipeline(
        self,
        performance_threshold: float = 0.8
    ) -> List[Dict[str, Any]]:
        """
        Identify optimization opportunities in the investigation-to-insight pipeline.
        
        Args:
            performance_threshold: Threshold for identifying underperforming areas
            
        Returns:
            List of optimization opportunities
        """
        optimizations = []
        
        try:
            # Analyze pipeline performance
            pipeline_metrics = await self._analyze_pipeline_performance()
            
            # Identify bottlenecks
            if pipeline_metrics["avg_time_to_insight"] > 3600:  # More than 1 hour
                optimizations.append({
                    "type": "speed_optimization",
                    "area": "investigation_to_insight_latency",
                    "current_value": pipeline_metrics["avg_time_to_insight"],
                    "target_value": 1800,  # 30 minutes
                    "recommendation": "Implement real-time insight generation triggers",
                    "expected_improvement": 0.5,
                    "implementation_effort": "medium"
                })
            
            # Check success rates
            if pipeline_metrics["insight_generation_rate"] < performance_threshold:
                optimizations.append({
                    "type": "quality_optimization",
                    "area": "insight_generation_success",
                    "current_value": pipeline_metrics["insight_generation_rate"],
                    "target_value": 0.9,
                    "recommendation": "Enhance investigation depth and cross-validation",
                    "expected_improvement": 0.2,
                    "implementation_effort": "high"
                })
            
            # Analyze pattern utilization
            pattern_utilization = self._calculate_pattern_utilization()
            if pattern_utilization < 0.6:
                optimizations.append({
                    "type": "pattern_optimization",
                    "area": "pattern_learning_utilization",
                    "current_value": pattern_utilization,
                    "target_value": 0.85,
                    "recommendation": "Increase pattern matching sensitivity",
                    "expected_improvement": 0.25,
                    "implementation_effort": "low"
                })
            
            # Check feedback loop effectiveness
            avg_loop_effectiveness = np.mean([
                loop.overall_effectiveness 
                for loop in self.feedback_loops.values()
            ]) if self.feedback_loops else 0
            
            if avg_loop_effectiveness < 0.7:
                optimizations.append({
                    "type": "feedback_optimization",
                    "area": "feedback_loop_effectiveness",
                    "current_value": avg_loop_effectiveness,
                    "target_value": 0.85,
                    "recommendation": "Strengthen feedback mechanisms",
                    "expected_improvement": 0.15,
                    "implementation_effort": "medium"
                })
            
            # Sort by expected improvement
            optimizations.sort(
                key=lambda x: x["expected_improvement"], 
                reverse=True
            )
            
            return optimizations
            
        except Exception as e:
            self.logger.error(f"Failed to optimize pipeline: {e}")
            return []
    
    async def generate_intelligence_report(self) -> InvestigationInsightIntelligence:
        """
        Generate comprehensive intelligence report on Investigation-Insight relationship.
        
        Returns:
            Comprehensive intelligence report
        """
        try:
            # Gather all intelligence data
            active_links = list(self.link_cache.values())
            discovered_patterns = list(self.pattern_cache.values())
            feedback_loops = list(self.feedback_loops.values())
            
            # Calculate metrics
            pipeline_metrics = await self._analyze_pipeline_performance()
            
            # Generate predictions
            quality_predictions = {}
            if discovered_patterns:
                # Predict quality for each pattern
                for pattern in discovered_patterns[:5]:
                    quality_predictions[pattern.pattern_name] = pattern.avg_insight_quality
            
            # Get recommendations
            mock_context = {"domain": "general", "goal": "optimize"}
            recommended_areas = await self.recommend_investigation_areas([], mock_context)
            
            # Get optimizations
            optimizations = await self.optimize_investigation_insight_pipeline()
            
            # Create intelligence report
            report = InvestigationInsightIntelligence(
                intelligence_id=str(uuid.uuid4()),
                analysis_timestamp=datetime.now(timezone.utc),
                active_links=active_links,
                discovered_patterns=discovered_patterns,
                feedback_loops=feedback_loops,
                avg_investigation_to_insight_time=pipeline_metrics.get("avg_time_to_insight", 0),
                insight_generation_success_rate=pipeline_metrics.get("insight_generation_rate", 0),
                pattern_discovery_rate=len(discovered_patterns) / max(len(active_links), 1),
                avg_insight_quality_score=np.mean([p.avg_insight_quality for p in discovered_patterns]) if discovered_patterns else 0,
                avg_investigation_confidence=np.mean([l.confidence_score for l in active_links]) if active_links else 0,
                cross_validation_accuracy=np.mean([l.cross_validation_score for l in active_links]) if active_links else 0,
                total_business_value=sum(p.business_value_generated for p in discovered_patterns),
                roi_multiplier=self._calculate_roi_multiplier(active_links, discovered_patterns),
                strategic_alignment_score=self._calculate_strategic_alignment(discovered_patterns),
                predicted_insight_quality=quality_predictions,
                recommended_investigation_areas=[r["investigation_area"] for r in recommended_areas[:5]],
                optimization_opportunities=optimizations
            )
            
            return report
            
        except Exception as e:
            self.logger.error(f"Failed to generate intelligence report: {e}")
            # Return minimal report
            return InvestigationInsightIntelligence(
                intelligence_id=str(uuid.uuid4()),
                analysis_timestamp=datetime.now(timezone.utc),
                active_links=[],
                discovered_patterns=[],
                feedback_loops=[],
                avg_investigation_to_insight_time=0,
                insight_generation_success_rate=0,
                pattern_discovery_rate=0,
                avg_insight_quality_score=0,
                avg_investigation_confidence=0,
                cross_validation_accuracy=0,
                total_business_value=0,
                roi_multiplier=1.0,
                strategic_alignment_score=0,
                predicted_insight_quality={},
                recommended_investigation_areas=[],
                optimization_opportunities=[]
            )
    
    # Helper methods
    
    async def _get_investigation_vector(self, investigation_id: str) -> Optional[Dict[str, Any]]:
        """Get vector data for an investigation."""
        if not self.vector_table:
            return None
        
        try:
            # Search by investigation ID in metadata
            results = await self.vector_table.search([0] * 1024).limit(1000).to_list()
            
            for result in results:
                if (result.get("module_source") == ModuleSource.INVESTIGATION.value and
                    result.get("content_metadata", {}).get("investigation_id") == investigation_id):
                    return result
            
            return None
            
        except Exception:
            return None
    
    async def _find_insights_in_timeframe(
        self, 
        start_time: datetime, 
        end_time: datetime
    ) -> List[Dict[str, Any]]:
        """Find insights within a specific timeframe."""
        if not self.vector_table:
            return []
        
        try:
            # Get all insights
            results = await self.vector_table.search([0] * 1024).limit(1000).to_list()
            
            insights = []
            for result in results:
                if result.get("module_source") == ModuleSource.INSIGHT_SYNTHESIS.value:
                    timestamp_str = result.get("timestamp")
                    if timestamp_str:
                        timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                        if start_time <= timestamp <= end_time:
                            insights.append(result)
            
            return insights
            
        except Exception:
            return []
    
    async def _analyze_investigation_insight_relationship(
        self,
        investigation: Dict[str, Any],
        insight: Dict[str, Any]
    ) -> Optional[InvestigationInsightLink]:
        """Analyze relationship between investigation and insight."""
        try:
            # Calculate semantic similarity
            if self.embedder and "vector" in investigation and "vector" in insight:
                inv_vec = investigation["vector"]
                ins_vec = insight["vector"]
                
                # Cosine similarity
                dot_product = sum(a * b for a, b in zip(inv_vec, ins_vec))
                norm_inv = sum(a * a for a in inv_vec) ** 0.5
                norm_ins = sum(b * b for b in ins_vec) ** 0.5
                correlation = dot_product / (norm_inv * norm_ins) if norm_inv * norm_ins > 0 else 0
            else:
                correlation = 0.5
            
            # Extract domains
            inv_domain = investigation.get("business_domain", "general")
            ins_domain = insight.get("business_domain", "general")
            domain_match = 1.0 if inv_domain == ins_domain else 0.5
            
            # Calculate temporal distance
            inv_time = datetime.fromisoformat(investigation.get("timestamp", datetime.now(timezone.utc).isoformat()).replace('Z', '+00:00'))
            ins_time = datetime.fromisoformat(insight.get("timestamp", datetime.now(timezone.utc).isoformat()).replace('Z', '+00:00'))
            temporal_distance = abs((ins_time - inv_time).total_seconds() / 3600)
            
            # Determine link type
            if correlation > 0.85 and temporal_distance < 1:
                link_type = InvestigationInsightLinkType.DIRECT_GENERATION
            elif correlation > 0.7:
                link_type = InvestigationInsightLinkType.PATTERN_CORRELATION
            elif domain_match > 0.9:
                link_type = InvestigationInsightLinkType.DOMAIN_OVERLAP
            else:
                link_type = InvestigationInsightLinkType.TEMPORAL_SEQUENCE
            
            # Create link
            link = InvestigationInsightLink(
                link_id=str(uuid.uuid4()),
                link_type=link_type,
                investigation_id=investigation.get("content_metadata", {}).get("investigation_id", "unknown"),
                insight_ids=[insight.get("content_metadata", {}).get("synthesis_id", "unknown")],
                correlation_strength=correlation * domain_match,
                confidence_score=(investigation.get("confidence_score", 0.5) + insight.get("confidence_score", 0.5)) / 2,
                business_impact=insight.get("business_value_score", 0.5),
                shared_patterns=self._extract_shared_patterns(investigation, insight),
                domain_overlap=[inv_domain] if inv_domain == ins_domain else [],
                temporal_distance_hours=temporal_distance,
                insight_quality_improvement=insight.get("confidence_score", 0.5) - 0.5,
                investigation_efficiency_gain=0.1 if correlation > 0.7 else 0,
                cross_validation_score=correlation,
                link_created=datetime.now(timezone.utc),
                last_updated=datetime.now(timezone.utc)
            )
            
            return link
            
        except Exception:
            return None
    
    async def _get_module_vectors(
        self,
        module: ModuleSource,
        cutoff_time: datetime
    ) -> List[Dict[str, Any]]:
        """Get vectors from a specific module after cutoff time."""
        if not self.vector_table:
            return []
        
        try:
            results = await self.vector_table.search([0] * 1024).limit(1000).to_list()
            
            module_vectors = []
            for result in results:
                if result.get("module_source") == module.value:
                    timestamp_str = result.get("timestamp")
                    if timestamp_str:
                        timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                        if timestamp >= cutoff_time:
                            module_vectors.append(result)
            
            return module_vectors
            
        except Exception:
            return []
    
    def _cluster_vectors(self, vectors: List[Dict[str, Any]], n_clusters: int) -> Dict[int, List[Dict[str, Any]]]:
        """Cluster vectors using simple k-means-like approach."""
        if not vectors:
            return {}
        
        # Simple clustering by business domain for now
        clusters = defaultdict(list)
        
        for i, vector in enumerate(vectors):
            domain = vector.get("business_domain", "general")
            # Map domain to cluster ID (simple hash)
            cluster_id = hash(domain) % n_clusters
            clusters[cluster_id].append(vector)
        
        return dict(clusters)
    
    def _find_cross_cluster_patterns(
        self,
        investigation_clusters: Dict[int, List[Dict[str, Any]]],
        insight_clusters: Dict[int, List[Dict[str, Any]]]
    ) -> List[Dict[str, Any]]:
        """Find patterns across investigation and insight clusters."""
        patterns = []
        
        for inv_cluster_id, inv_vectors in investigation_clusters.items():
            for ins_cluster_id, ins_vectors in insight_clusters.items():
                # Check for connections
                connection_count = 0
                total_quality = 0
                domains = set()
                
                for inv in inv_vectors:
                    for ins in ins_vectors:
                        # Simple domain-based connection
                        if inv.get("business_domain") == ins.get("business_domain"):
                            connection_count += 1
                            total_quality += ins.get("confidence_score", 0.5)
                            domains.add(inv.get("business_domain", "general"))
                
                if connection_count >= 3:  # Minimum connections
                    patterns.append({
                        "pattern_name": f"Pattern_{inv_cluster_id}_{ins_cluster_id}",
                        "occurrence_count": connection_count,
                        "avg_quality": total_quality / connection_count if connection_count > 0 else 0,
                        "domains": list(domains),
                        "investigation_cluster": inv_cluster_id,
                        "insight_cluster": ins_cluster_id
                    })
        
        return patterns
    
    def _create_pattern_from_candidate(self, candidate: Dict[str, Any]) -> CrossModulePattern:
        """Create a CrossModulePattern from candidate data."""
        return CrossModulePattern(
            pattern_id=str(uuid.uuid4()),
            pattern_name=candidate["pattern_name"],
            pattern_description=f"Pattern linking investigations in cluster {candidate['investigation_cluster']} to insights in cluster {candidate['insight_cluster']}",
            occurrence_count=candidate["occurrence_count"],
            avg_correlation_strength=0.7,  # Default
            business_domains=candidate["domains"],
            avg_insight_quality=candidate["avg_quality"],
            avg_investigation_confidence=0.8,  # Default
            business_value_generated=candidate["avg_quality"] * 0.9,
            prediction_accuracy=min(0.9, candidate["occurrence_count"] / 10),
            recommendation_success_rate=0.7,  # Default
            example_investigations=[],
            example_insights=[],
            first_observed=datetime.now(timezone.utc) - timedelta(days=30),
            last_observed=datetime.now(timezone.utc),
            trend="stable"
        )
    
    def _find_link_chains(self, min_length: int) -> List[List[InvestigationInsightLink]]:
        """Find chains of linked investigations and insights."""
        chains = []
        
        # Build adjacency map
        insight_to_investigations = defaultdict(list)
        for link in self.link_cache.values():
            for insight_id in link.insight_ids:
                insight_to_investigations[insight_id].append(link)
        
        # Find chains
        visited = set()
        for link in self.link_cache.values():
            if link.link_id not in visited:
                chain = self._build_chain(link, insight_to_investigations, visited)
                if len(chain) >= min_length:
                    chains.append(chain)
        
        return chains
    
    def _build_chain(
        self,
        start_link: InvestigationInsightLink,
        insight_to_investigations: Dict[str, List[InvestigationInsightLink]],
        visited: Set[str]
    ) -> List[InvestigationInsightLink]:
        """Build a chain of linked investigations and insights."""
        chain = [start_link]
        visited.add(start_link.link_id)
        
        # Follow the chain
        current_insights = start_link.insight_ids
        
        while current_insights:
            next_insights = []
            for insight_id in current_insights:
                for link in insight_to_investigations.get(insight_id, []):
                    if link.link_id not in visited:
                        chain.append(link)
                        visited.add(link.link_id)
                        next_insights.extend(link.insight_ids)
            current_insights = next_insights
        
        return chain
    
    def _determine_feedback_type(self, chain: List[InvestigationInsightLink]) -> Optional[FeedbackLoopType]:
        """Determine the type of feedback loop from a chain."""
        if not chain:
            return None
        
        # Analyze improvement trends
        quality_improvements = [link.insight_quality_improvement for link in chain]
        efficiency_gains = [link.investigation_efficiency_gain for link in chain]
        
        avg_quality_improvement = np.mean(quality_improvements)
        avg_efficiency_gain = np.mean(efficiency_gains)
        
        if avg_quality_improvement > 0.1:
            return FeedbackLoopType.INSIGHT_QUALITY
        elif avg_efficiency_gain > 0.1:
            return FeedbackLoopType.INVESTIGATION_DEPTH
        elif len(set(link.link_type for link in chain)) > 2:
            return FeedbackLoopType.PATTERN_DISCOVERY
        else:
            return FeedbackLoopType.CONFIDENCE_BOOST
    
    def _calculate_loop_strength(self, chain: List[InvestigationInsightLink]) -> float:
        """Calculate the strength of a feedback loop."""
        if not chain:
            return 0.0
        
        # Average correlation strength
        avg_correlation = np.mean([link.correlation_strength for link in chain])
        
        # Consistency of improvements
        improvements = [link.insight_quality_improvement for link in chain]
        consistency = 1 - np.std(improvements) if len(improvements) > 1 else 1
        
        return avg_correlation * consistency
    
    def _calculate_improvement_rate(self, chain: List[InvestigationInsightLink]) -> float:
        """Calculate improvement rate in a chain."""
        if len(chain) < 2:
            return 0.0
        
        # Compare first and last
        initial_quality = chain[0].confidence_score
        final_quality = chain[-1].confidence_score
        
        return (final_quality - initial_quality) / len(chain)
    
    def _calculate_convergence_speed(self, chain: List[InvestigationInsightLink]) -> float:
        """Calculate how quickly the loop converges to stability."""
        if len(chain) < 3:
            return 0.5
        
        # Check variance reduction
        first_half = chain[:len(chain)//2]
        second_half = chain[len(chain)//2:]
        
        first_variance = np.std([link.confidence_score for link in first_half])
        second_variance = np.std([link.confidence_score for link in second_half])
        
        if first_variance > 0:
            return 1 - (second_variance / first_variance)
        return 0.5
    
    def _measure_investigation_improvement(self, chain: List[InvestigationInsightLink]) -> float:
        """Measure investigation improvement in a chain."""
        if not chain:
            return 0.0
        
        return np.mean([link.investigation_efficiency_gain for link in chain])
    
    def _measure_insight_improvement(self, chain: List[InvestigationInsightLink]) -> float:
        """Measure insight improvement in a chain."""
        if not chain:
            return 0.0
        
        return np.mean([link.insight_quality_improvement for link in chain])
    
    def _calculate_loop_effectiveness(self, chain: List[InvestigationInsightLink]) -> float:
        """Calculate overall effectiveness of a feedback loop."""
        if not chain:
            return 0.0
        
        # Combine multiple factors
        avg_impact = np.mean([link.business_impact for link in chain])
        avg_confidence = np.mean([link.confidence_score for link in chain])
        avg_validation = np.mean([link.cross_validation_score for link in chain])
        
        return (avg_impact + avg_confidence + avg_validation) / 3
    
    def _extract_investigation_features(self, investigation_results: Dict[str, Any]) -> Dict[str, Any]:
        """Extract features from investigation results."""
        return {
            "confidence": investigation_results.get("overall_confidence", 0.5),
            "domain": investigation_results.get("business_context", {}).get("domain", "general"),
            "complexity": investigation_results.get("business_context", {}).get("complexity_level", "medium"),
            "findings_count": len(investigation_results.get("investigation_findings", {}).get("key_findings", [])),
            "duration": investigation_results.get("total_duration_seconds", 0)
        }
    
    async def _find_similar_investigations(
        self,
        features: Dict[str, Any],
        limit: int
    ) -> List[Dict[str, Any]]:
        """Find similar past investigations."""
        if not self.vector_table or not self.embedder:
            return []
        
        try:
            # Create feature text
            feature_text = f"Domain: {features['domain']}, Complexity: {features['complexity']}, Confidence: {features['confidence']}"
            
            # Generate embedding
            embedding = self.embedder.encode(feature_text).tolist()
            
            # Search
            results = await self.vector_table.search(embedding).limit(limit).to_list()
            
            # Filter for investigations
            investigations = []
            for result in results:
                if result.get("module_source") == ModuleSource.INVESTIGATION.value:
                    result["similarity"] = 1 - result.get("_distance", 0.5)
                    investigations.append(result)
            
            return investigations
            
        except Exception:
            return []
    
    def _extract_insight_domains(self, insights: List[Dict[str, Any]]) -> List[str]:
        """Extract domains from insights."""
        domains = []
        for insight in insights:
            domain = insight.get("business_domain", insight.get("related_domains", ["general"])[0])
            domains.append(domain)
        return domains
    
    def _analyze_domain_coverage(self, domains: List[str]) -> Dict[str, float]:
        """Analyze coverage of different domains."""
        if not domains:
            return {}
        
        domain_counts = defaultdict(int)
        for domain in domains:
            domain_counts[domain] += 1
        
        total = len(domains)
        coverage = {
            domain: count / total 
            for domain, count in domain_counts.items()
        }
        
        return coverage
    
    def _pattern_applies_to_context(
        self,
        pattern: CrossModulePattern,
        context: Dict[str, Any]
    ) -> bool:
        """Check if a pattern applies to the given context."""
        # Check domain match
        context_domain = context.get("domain", "general")
        if context_domain in pattern.business_domains:
            return True
        
        # Check for related domains
        if context_domain == "general":
            return True
        
        return False
    
    def _calculate_gap_score(
        self,
        pattern_domains: List[str],
        coverage: Dict[str, float]
    ) -> float:
        """Calculate gap score for pattern domains."""
        gap_scores = []
        
        for domain in pattern_domains:
            if domain not in coverage:
                gap_scores.append(1.0)
            else:
                gap_scores.append(1.0 - coverage[domain])
        
        return np.mean(gap_scores) if gap_scores else 0.0
    
    def _estimate_investigation_duration(self, pattern: CrossModulePattern) -> str:
        """Estimate investigation duration based on pattern."""
        # Simple estimation based on pattern complexity
        if pattern.occurrence_count > 10:
            return "1-2 hours"
        elif pattern.occurrence_count > 5:
            return "2-4 hours"
        else:
            return "4-8 hours"
    
    def _find_uncovered_domains(self, coverage: Dict[str, float]) -> List[str]:
        """Find domains not well covered."""
        # All possible domains
        all_domains = [
            "sales", "customer", "operations", "finance", 
            "supply_chain", "inventory", "quality", "hr",
            "marketing", "production", "logistics", "compliance"
        ]
        
        uncovered = []
        for domain in all_domains:
            if domain not in coverage or coverage[domain] < 0.2:
                uncovered.append(domain)
        
        return uncovered
    
    async def _analyze_pipeline_performance(self) -> Dict[str, float]:
        """Analyze overall pipeline performance metrics."""
        if not self.link_cache:
            return {
                "avg_time_to_insight": 0,
                "insight_generation_rate": 0,
                "pattern_utilization": 0
            }
        
        # Calculate average time to insight
        time_distances = [link.temporal_distance_hours for link in self.link_cache.values()]
        avg_time = np.mean(time_distances) * 3600 if time_distances else 0  # Convert to seconds
        
        # Calculate success rate
        total_investigations = len(set(link.investigation_id for link in self.link_cache.values()))
        successful_insights = len([link for link in self.link_cache.values() if link.confidence_score > 0.7])
        success_rate = successful_insights / total_investigations if total_investigations > 0 else 0
        
        return {
            "avg_time_to_insight": avg_time,
            "insight_generation_rate": success_rate,
            "pattern_utilization": self._calculate_pattern_utilization()
        }
    
    def _calculate_pattern_utilization(self) -> float:
        """Calculate how well patterns are being utilized."""
        if not self.pattern_cache or not self.link_cache:
            return 0.0
        
        # Check how many links use discovered patterns
        pattern_domains = set()
        for pattern in self.pattern_cache.values():
            pattern_domains.update(pattern.business_domains)
        
        link_domains = set()
        for link in self.link_cache.values():
            link_domains.update(link.domain_overlap)
        
        if not pattern_domains:
            return 0.0
        
        overlap = len(link_domains & pattern_domains)
        return overlap / len(pattern_domains)
    
    def _extract_shared_patterns(
        self,
        investigation: Dict[str, Any],
        insight: Dict[str, Any]
    ) -> List[str]:
        """Extract patterns shared between investigation and insight."""
        patterns = []
        
        # Check query patterns
        inv_patterns = json.loads(investigation.get("query_patterns", "[]"))
        ins_patterns = json.loads(insight.get("query_patterns", "[]"))
        
        shared = set(inv_patterns) & set(ins_patterns)
        patterns.extend(list(shared))
        
        return patterns[:5]  # Limit to 5
    
    def _calculate_roi_multiplier(
        self,
        links: List[InvestigationInsightLink],
        patterns: List[CrossModulePattern]
    ) -> float:
        """Calculate ROI multiplier from intelligence."""
        if not links:
            return 1.0
        
        # Base ROI from business impact
        avg_impact = np.mean([link.business_impact for link in links])
        
        # Boost from patterns
        pattern_boost = 1 + (len(patterns) * 0.1)
        
        return avg_impact * pattern_boost
    
    def _calculate_strategic_alignment(self, patterns: List[CrossModulePattern]) -> float:
        """Calculate strategic alignment score."""
        if not patterns:
            return 0.5
        
        # Check business value and prediction accuracy
        avg_value = np.mean([p.business_value_generated for p in patterns])
        avg_accuracy = np.mean([p.prediction_accuracy for p in patterns])
        
        return (avg_value + avg_accuracy) / 2
    
    async def cleanup(self):
        """Cleanup resources."""
        # Clear caches
        self.link_cache.clear()
        self.pattern_cache.clear()
        self.feedback_loops.clear()


# High-level interface functions

async def analyze_investigation_insight_intelligence(
    investigation_id: Optional[str] = None,
    time_window_days: int = 30,
    db_path: Optional[str] = None
) -> InvestigationInsightIntelligence:
    """
    Analyze cross-module intelligence between Investigation and Insight Synthesis.
    
    Args:
        investigation_id: Optional specific investigation to analyze
        time_window_days: Time window for analysis
        db_path: Optional path to vector database
        
    Returns:
        Comprehensive intelligence report
    """
    engine = InvestigationInsightIntelligenceEngine()
    await engine.initialize(db_path=db_path)
    
    try:
        # Analyze specific investigation if provided
        if investigation_id:
            await engine.analyze_investigation_to_insight_flow(
                investigation_id, time_window_days * 24
            )
        
        # Discover patterns
        await engine.discover_cross_module_patterns(
            min_occurrences=2, time_window_days=time_window_days
        )
        
        # Establish feedback loops
        await engine.establish_feedback_loops(min_iterations=2)
        
        # Generate comprehensive report
        report = await engine.generate_intelligence_report()
        
        return report
        
    finally:
        await engine.cleanup()


if __name__ == "__main__":
    # Test the cross-module intelligence engine
    import asyncio
    
    async def test_intelligence_engine():
        print("Testing Investigation-Insight Intelligence Engine...")
        
        # Run analysis
        report = await analyze_investigation_insight_intelligence(
            time_window_days=30
        )
        
        print(f"\nIntelligence Report Generated!")
        print(f"Report ID: {report.intelligence_id}")
        print(f"Active links: {len(report.active_links)}")
        print(f"Discovered patterns: {len(report.discovered_patterns)}")
        print(f"Feedback loops: {len(report.feedback_loops)}")
        print(f"Pattern discovery rate: {report.pattern_discovery_rate:.2%}")
        print(f"Average insight quality: {report.avg_insight_quality_score:.3f}")
        print(f"ROI multiplier: {report.roi_multiplier:.2f}x")
        
        if report.recommended_investigation_areas:
            print(f"\nRecommended investigation areas:")
            for area in report.recommended_investigation_areas[:3]:
                print(f"  - {area}")
        
        if report.optimization_opportunities:
            print(f"\nOptimization opportunities:")
            for opt in report.optimization_opportunities[:3]:
                print(f"  - {opt['area']}: {opt['recommendation']}")
    
    asyncio.run(test_intelligence_engine())