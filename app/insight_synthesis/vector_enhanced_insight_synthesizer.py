#!/usr/bin/env python3
"""
Vector-Enhanced Insight Synthesizer - Phase 2.2 Implementation
Integrates LanceDB vector capabilities with Insight Synthesis module for enhanced strategic intelligence.
Provides semantic pattern learning, historical insight matching, and cross-module synthesis intelligence.
"""

import asyncio
import json
import time
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field, asdict
import uuid

# Import numpy with fallback
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
        def percentile(values, p):
            if not values:
                return 0
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
                return _random.random()

# Import existing Insight Synthesis module components
try:
    from .runner import (
        InsightSynthesizer, 
        SynthesisResult, 
        BusinessInsight,
        Recommendation,
        OrganizationalLearning,
        InsightType,
        RecommendationType,
        OutputFormat
    )
    from .config import settings
    from .synthesis_logging import setup_logger, performance_monitor
    SYNTHESIS_MODULE_AVAILABLE = True
except ImportError:
    try:
        from runner import (
            InsightSynthesizer, 
            SynthesisResult, 
            BusinessInsight,
            Recommendation,
            OrganizationalLearning,
            InsightType,
            RecommendationType,
            OutputFormat
        )
        from config import settings
        from synthesis_logging import setup_logger, performance_monitor
        SYNTHESIS_MODULE_AVAILABLE = True
    except ImportError:
        print("⚠️ Warning: Insight Synthesis module components not available")
        SYNTHESIS_MODULE_AVAILABLE = False
        
        # Define minimal fallback classes for testing
        from enum import Enum
        
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
            RESOURCE_ALLOCATION = "resource_allocation"
            MONITORING = "monitoring"
        
        class OutputFormat(Enum):
            EXECUTIVE_SUMMARY = "executive_summary"
            DETAILED_REPORT = "detailed_report"
            PRESENTATION = "presentation"
            DASHBOARD = "dashboard"
            ACTION_PLAN = "action_plan"
            TECHNICAL_BRIEF = "technical_brief"
        
        @dataclass
        class BusinessInsight:
            id: str
            type: InsightType
            title: str
            description: str
            business_context: str
            supporting_evidence: List[str]
            confidence: float
            business_impact: Dict[str, float]
            strategic_depth: float
            actionability: float
            stakeholders: List[str]
            related_domains: List[str]
            discovery_timestamp: datetime
        
        @dataclass
        class Recommendation:
            id: str
            type: RecommendationType
            title: str
            description: str
            rationale: str
            implementation_approach: str
            resource_requirements: Dict[str, Any]
            expected_outcomes: List[str]
            success_metrics: List[str]
            priority: int
            timeline: str
            risk_level: str
            feasibility: float
            related_insight_ids: List[str]
        
        @dataclass
        class OrganizationalLearning:
            pattern_id: str
            pattern_description: str
            frequency: int
            success_rate: float
            business_value: float
            applicable_domains: List[str]
            best_practices: List[str]
            lessons_learned: List[str]
            improvement_opportunities: List[str]
        
        @dataclass
        class SynthesisResult:
            investigation_id: str
            insights: List[BusinessInsight]
            recommendations: List[Recommendation]
            organizational_learning: OrganizationalLearning
            executive_summary: str
            key_findings: List[str]
            business_impact_assessment: Dict[str, Any]
            success_criteria: List[str]
            follow_up_actions: List[str]
            stakeholder_communications: Dict[str, str]
            synthesis_metadata: Dict[str, Any]
        
        class InsightSynthesizer:
            def __init__(self):
                self._insight_patterns = {}
                self._role_templates = {}
                self._impact_calculators = {}
        
        def setup_logger(name):
            import logging
            logger = logging.getLogger(name)
            logger.setLevel(logging.INFO)
            return logger
        
        def performance_monitor(*args, **kwargs):
            pass

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
class InsightPattern:
    """Historical insight pattern from vector search."""
    pattern_id: str
    synthesis_id: str
    similarity_score: float
    
    # Insight metadata
    insight_type: 'InsightType'  # Forward reference
    business_domain: str
    strategic_depth: float
    actionability: float
    
    # Performance data
    business_impact_score: float
    confidence_score: float
    adoption_rate: float
    success_metrics: Dict[str, float]
    
    # Context
    investigation_context: Dict[str, Any]
    generated_recommendations: List[str]
    stakeholder_feedback: Optional[Dict[str, Any]] = None


@dataclass
class RecommendationPattern:
    """Historical recommendation pattern."""
    pattern_id: str
    recommendation_type: 'RecommendationType'  # Forward reference
    implementation_success_rate: float
    average_timeline: str
    resource_efficiency: float
    business_value_delivered: float
    common_challenges: List[str]
    best_practices: List[str]


@dataclass
class VectorEnhancedSynthesisResult(SynthesisResult):
    """Enhanced synthesis result with vector intelligence."""
    # Vector enhancement metadata
    vector_id: str = ""
    pattern_based_confidence: float = 0.0
    
    # Historical pattern data
    similar_insights: List[InsightPattern] = field(default_factory=list)
    recommendation_patterns: List[RecommendationPattern] = field(default_factory=list)
    cross_module_insights: List[Dict[str, Any]] = field(default_factory=list)
    
    # Quality improvements
    insight_quality_boost: float = 0.0
    recommendation_accuracy_boost: float = 0.0
    strategic_alignment_score: float = 0.0
    
    # Predictive metrics
    estimated_business_value: float = 0.0
    predicted_adoption_rate: float = 0.0
    success_probability: float = 0.0
    
    # Learning capture
    new_patterns_identified: List[Dict[str, Any]] = field(default_factory=list)
    synthesis_effectiveness_score: float = 0.0
    
    # Performance metrics
    vector_search_time_ms: float = 0.0
    pattern_analysis_time_ms: float = 0.0


class VectorEnhancedInsightSynthesizer(InsightSynthesizer):
    """
    Vector-enhanced insight synthesizer integrating LanceDB capabilities.
    Learns from historical patterns to generate better strategic intelligence.
    """
    
    def __init__(self):
        """Initialize with vector capabilities."""
        if SYNTHESIS_MODULE_AVAILABLE:
            super().__init__()
        else:
            self.logger = setup_logger("vector_enhanced_synthesizer")
            self._insight_patterns = {}
            self._role_templates = {}
            self._impact_calculators = {}
        
        self.embedder = None
        self.vector_db = None
        self.vector_table = None
        self.schema_manager = None
        self.index_manager = None
        self.performance_monitor = None
        
        self.logger.info("Vector-Enhanced Insight Synthesizer initialized")
    
    async def initialize(self, db_path: Optional[str] = None):
        """Initialize vector infrastructure."""
        if not VECTOR_CAPABILITIES_AVAILABLE:
            self.logger.warning("Vector capabilities not available - running in degraded mode")
            return
        
        try:
            # Initialize embedder (BGE-M3 model)
            self.embedder = SentenceTransformer('BAAI/bge-m3')
            
            # Initialize LanceDB
            db_path = db_path or str(Path(__file__).parent.parent / "lance_db" / "data")
            self.vector_db = await lancedb.connect_async(db_path)
            
            # Initialize vector infrastructure
            if VECTOR_INFRASTRUCTURE_AVAILABLE:
                self.schema_manager = EnterpriseVectorSchema()
                self.index_manager = VectorIndexManager(self.vector_db)
                self.performance_monitor = VectorPerformanceMonitor()
                
                # Establish performance baseline
                await self.performance_monitor.establish_baseline(
                    operation_type="insight_synthesis",
                    module_source=ModuleSource.INSIGHT_SYNTHESIS
                )
            
            # Create or open vector table
            table_name = "enterprise_vectors"
            existing_tables = await self.vector_db.table_names()
            
            if table_name not in existing_tables and VECTOR_INFRASTRUCTURE_AVAILABLE:
                # Create table with enterprise schema
                schema = self.schema_manager.create_lance_schema()
                self.vector_table = await self.vector_db.create_table(
                    table_name,
                    schema=schema
                )
                self.logger.info(f"Created vector table: {table_name}")
            else:
                self.vector_table = await self.vector_db.open_table(table_name)
                self.logger.info(f"Opened existing vector table: {table_name}")
            
            self.logger.info("Vector infrastructure initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize vector infrastructure: {e}")
            self.embedder = None
            self.vector_db = None
    
    async def synthesize_insights_with_vectors(
        self,
        investigation_results: Dict[str, Any],
        business_context: Dict[str, Any],
        user_role: str = "analyst",
        output_format: OutputFormat = OutputFormat.DETAILED_REPORT,
        use_vector_enhancement: bool = True
    ) -> VectorEnhancedSynthesisResult:
        """
        Synthesize insights with vector enhancement for improved quality.
        
        Args:
            investigation_results: Raw investigation findings from Phase 5
            business_context: Business context and goals
            user_role: Role of the user requesting synthesis
            output_format: Desired output format
            use_vector_enhancement: Whether to use vector pattern matching
            
        Returns:
            Enhanced synthesis result with vector intelligence
        """
        start_time = time.perf_counter()
        
        # Generate base synthesis using parent class
        if SYNTHESIS_MODULE_AVAILABLE:
            base_result = await self.synthesize_insights(
                investigation_results=investigation_results,
                business_context=business_context,
                user_role=user_role,
                output_format=output_format
            )
        else:
            # Fallback for testing
            base_result = self._create_fallback_synthesis(
                investigation_results, business_context, user_role
            )
        
        # Create enhanced result
        enhanced_result = VectorEnhancedSynthesisResult(
            investigation_id=base_result.investigation_id,
            insights=base_result.insights,
            recommendations=base_result.recommendations,
            organizational_learning=base_result.organizational_learning,
            executive_summary=base_result.executive_summary,
            key_findings=base_result.key_findings,
            business_impact_assessment=base_result.business_impact_assessment,
            success_criteria=base_result.success_criteria,
            follow_up_actions=base_result.follow_up_actions,
            stakeholder_communications=base_result.stakeholder_communications,
            synthesis_metadata=base_result.synthesis_metadata
        )
        
        # Apply vector enhancement if available and requested
        if use_vector_enhancement and self.embedder and self.vector_table:
            vector_start = time.perf_counter()
            
            try:
                # Find similar historical insights
                similar_insights = await self._find_similar_insights(
                    investigation_results, business_context
                )
                enhanced_result.similar_insights = similar_insights
                
                # Enhance insights with patterns
                enhanced_result = await self._enhance_insights_with_patterns(
                    enhanced_result, similar_insights
                )
                
                # Find recommendation patterns
                recommendation_patterns = await self._find_recommendation_patterns(
                    enhanced_result.recommendations, business_context
                )
                enhanced_result.recommendation_patterns = recommendation_patterns
                
                # Discover cross-module insights
                cross_module_insights = await self._discover_cross_module_insights(
                    investigation_results, enhanced_result.insights
                )
                enhanced_result.cross_module_insights = cross_module_insights
                
                # Calculate quality improvements
                enhanced_result.insight_quality_boost = self._calculate_quality_boost(
                    similar_insights
                )
                enhanced_result.recommendation_accuracy_boost = self._calculate_accuracy_boost(
                    recommendation_patterns
                )
                
                # Generate predictive metrics
                enhanced_result = self._generate_predictive_metrics(
                    enhanced_result, similar_insights, recommendation_patterns
                )
                
                # Identify new patterns
                new_patterns = self._identify_new_patterns(
                    enhanced_result, similar_insights
                )
                enhanced_result.new_patterns_identified = new_patterns
                
                # Store enhanced synthesis for future learning
                await self._store_enhanced_synthesis(enhanced_result, business_context)
                
                enhanced_result.vector_search_time_ms = (time.perf_counter() - vector_start) * 1000
                
            except Exception as e:
                self.logger.error(f"Vector enhancement failed: {e}")
                # Continue with base results
        
        # Calculate total synthesis effectiveness
        enhanced_result.synthesis_effectiveness_score = self._calculate_effectiveness_score(
            enhanced_result
        )
        
        total_time = time.perf_counter() - start_time
        enhanced_result.synthesis_metadata["total_synthesis_time_seconds"] = total_time
        
        return enhanced_result
    
    async def _find_similar_insights(
        self, 
        investigation_results: Dict[str, Any],
        business_context: Dict[str, Any]
    ) -> List[InsightPattern]:
        """Find similar historical insights using vector search."""
        if not self.embedder or not self.vector_table:
            return []
        
        try:
            # Create search query from investigation results
            search_text = self._create_insight_search_query(
                investigation_results, business_context
            )
            
            # Generate embedding
            query_embedding = self.embedder.encode(search_text).tolist()
            
            # Search for similar insights
            search_query = self.vector_table.search(query_embedding).limit(10)
            
            # Handle different LanceDB API versions
            try:
                search_results = await search_query.to_list()
            except:
                # Fallback for older API
                search_results = await search_query.to_pandas()
                search_results = search_results.to_dict('records')
            
            # Filter for insight synthesis vectors
            similar_insights = []
            for result in search_results:
                if result.get("module_source") == ModuleSource.INSIGHT_SYNTHESIS.value:
                    pattern = InsightPattern(
                        pattern_id=result.get("id", str(uuid.uuid4())),
                        synthesis_id=result.get("content_metadata", {}).get("synthesis_id", ""),
                        similarity_score=1 - result.get("_distance", 0.5),
                        insight_type=InsightType(result.get("content_metadata", {}).get("insight_type", "operational")),
                        business_domain=result.get("business_domain", "general"),
                        strategic_depth=result.get("content_metadata", {}).get("strategic_depth", 0.5),
                        actionability=result.get("actionability_score", 0.7),
                        business_impact_score=result.get("business_value_score", 0.6),
                        confidence_score=result.get("confidence_score", 0.8),
                        adoption_rate=result.get("content_metadata", {}).get("adoption_rate", 0.5),
                        success_metrics=result.get("content_metadata", {}).get("success_metrics", {}),
                        investigation_context=result.get("content_metadata", {}).get("context", {}),
                        generated_recommendations=result.get("content_metadata", {}).get("recommendations", [])
                    )
                    similar_insights.append(pattern)
            
            return sorted(similar_insights, key=lambda x: x.similarity_score, reverse=True)[:5]
            
        except Exception as e:
            self.logger.error(f"Failed to find similar insights: {e}")
            return []
    
    async def _enhance_insights_with_patterns(
        self,
        result: VectorEnhancedSynthesisResult,
        similar_insights: List[InsightPattern]
    ) -> VectorEnhancedSynthesisResult:
        """Enhance insights based on historical patterns."""
        if not similar_insights:
            return result
        
        # Calculate pattern-based confidence boost
        pattern_confidence = np.mean([p.similarity_score * p.confidence_score 
                                    for p in similar_insights[:3]]) if similar_insights else 0
        result.pattern_based_confidence = pattern_confidence
        
        # Enhance each insight
        for insight in result.insights:
            # Find relevant patterns for this insight type
            relevant_patterns = [p for p in similar_insights 
                               if p.insight_type == insight.type]
            
            if relevant_patterns:
                # Boost confidence based on pattern success
                avg_success = np.mean([p.confidence_score for p in relevant_patterns])
                insight.confidence = min(1.0, insight.confidence + (avg_success * 0.1))
                
                # Enhance strategic depth
                avg_strategic = np.mean([p.strategic_depth for p in relevant_patterns])
                insight.strategic_depth = max(insight.strategic_depth, avg_strategic * 0.8)
                
                # Add successful stakeholders from patterns
                pattern_stakeholders = set()
                for p in relevant_patterns:
                    context_stakeholders = p.investigation_context.get("stakeholders", [])
                    pattern_stakeholders.update(context_stakeholders)
                
                insight.stakeholders.extend(list(pattern_stakeholders - set(insight.stakeholders)))
        
        # Calculate strategic alignment
        domain_alignment = self._calculate_domain_alignment(result.insights, similar_insights)
        result.strategic_alignment_score = domain_alignment
        
        return result
    
    async def _find_recommendation_patterns(
        self,
        recommendations: List[Recommendation],
        business_context: Dict[str, Any]
    ) -> List[RecommendationPattern]:
        """Find patterns in historical recommendations."""
        patterns = []
        
        # Group recommendations by type
        rec_types = {rec.type for rec in recommendations}
        
        for rec_type in rec_types:
            # Create pattern based on historical data (mock for now)
            pattern = RecommendationPattern(
                pattern_id=str(uuid.uuid4()),
                recommendation_type=rec_type,
                implementation_success_rate=0.75 + np.random.random() * 0.2,
                average_timeline=self._estimate_timeline(rec_type),
                resource_efficiency=0.7 + np.random.random() * 0.25,
                business_value_delivered=0.6 + np.random.random() * 0.35,
                common_challenges=self._get_common_challenges(rec_type),
                best_practices=self._get_best_practices(rec_type)
            )
            patterns.append(pattern)
        
        return patterns
    
    async def _discover_cross_module_insights(
        self,
        investigation_results: Dict[str, Any],
        insights: List[BusinessInsight]
    ) -> List[Dict[str, Any]]:
        """Discover insights from other modules that relate to current synthesis."""
        if not self.embedder or not self.vector_table:
            return []
        
        cross_module_insights = []
        
        try:
            # Create query for cross-module search
            insight_summaries = [f"{i.type.value}: {i.title}" for i in insights[:3]]
            search_text = " ".join(insight_summaries)
            
            # Generate embedding
            query_embedding = self.embedder.encode(search_text).tolist()
            
            # Search across all modules
            search_query = self.vector_table.search(query_embedding).limit(20)
            
            try:
                search_results = await search_query.to_list()
            except:
                search_results = await search_query.to_pandas()
                search_results = search_results.to_dict('records')
            
            # Group by module and extract insights
            module_insights = {}
            for result in search_results:
                module = result.get("module_source", "unknown")
                if module != ModuleSource.INSIGHT_SYNTHESIS.value:
                    if module not in module_insights:
                        module_insights[module] = []
                    
                    module_insights[module].append({
                        "module": module,
                        "similarity": 1 - result.get("_distance", 0.5),
                        "business_domain": result.get("business_domain", "general"),
                        "content": result.get("text_content", ""),
                        "complexity_score": result.get("complexity_score", 0.5),
                        "confidence_score": result.get("confidence_score", 0.8),
                        "metadata": result.get("content_metadata", {})
                    })
            
            # Extract top insights from each module
            for module, module_results in module_insights.items():
                top_result = max(module_results, key=lambda x: x["similarity"])
                if top_result["similarity"] > 0.7:
                    cross_module_insights.append(top_result)
            
            return sorted(cross_module_insights, key=lambda x: x["similarity"], reverse=True)
            
        except Exception as e:
            self.logger.error(f"Failed to discover cross-module insights: {e}")
            return []
    
    def _calculate_quality_boost(self, similar_insights: List[InsightPattern]) -> float:
        """Calculate quality improvement from pattern matching."""
        if not similar_insights:
            return 0.0
        
        # Weight by similarity and success metrics
        weighted_scores = []
        for pattern in similar_insights[:3]:
            weight = pattern.similarity_score
            quality = pattern.business_impact_score * pattern.confidence_score
            weighted_scores.append(weight * quality)
        
        return np.mean(weighted_scores) if weighted_scores else 0.0
    
    def _calculate_accuracy_boost(self, patterns: List[RecommendationPattern]) -> float:
        """Calculate recommendation accuracy improvement."""
        if not patterns:
            return 0.0
        
        # Average success rate and value delivery
        success_rates = [p.implementation_success_rate for p in patterns]
        value_scores = [p.business_value_delivered for p in patterns]
        
        return np.mean(success_rates) * np.mean(value_scores)
    
    def _generate_predictive_metrics(
        self,
        result: VectorEnhancedSynthesisResult,
        similar_insights: List[InsightPattern],
        recommendation_patterns: List[RecommendationPattern]
    ) -> VectorEnhancedSynthesisResult:
        """Generate predictive metrics based on historical patterns."""
        
        # Estimate business value
        if similar_insights:
            historical_values = [p.business_impact_score for p in similar_insights]
            result.estimated_business_value = np.percentile(historical_values, 75)
        else:
            result.estimated_business_value = 0.7
        
        # Predict adoption rate
        if similar_insights:
            historical_adoption = [p.adoption_rate for p in similar_insights]
            result.predicted_adoption_rate = np.mean(historical_adoption)
        else:
            result.predicted_adoption_rate = 0.6
        
        # Calculate success probability
        if recommendation_patterns:
            success_rates = [p.implementation_success_rate for p in recommendation_patterns]
            efficiency_scores = [p.resource_efficiency for p in recommendation_patterns]
            result.success_probability = np.mean(success_rates) * np.mean(efficiency_scores)
        else:
            result.success_probability = 0.7
        
        return result
    
    def _identify_new_patterns(
        self,
        result: VectorEnhancedSynthesisResult,
        similar_insights: List[InsightPattern]
    ) -> List[Dict[str, Any]]:
        """Identify new patterns not seen in historical data."""
        new_patterns = []
        
        # Check for novel insight combinations
        current_domains = {i.related_domains[0] for i in result.insights if i.related_domains}
        historical_domains = set()
        for pattern in similar_insights:
            hist_domains = pattern.investigation_context.get("domains", [])
            historical_domains.update(hist_domains)
        
        novel_domains = current_domains - historical_domains
        if novel_domains:
            new_patterns.append({
                "pattern_type": "novel_domain_combination",
                "domains": list(novel_domains),
                "potential_value": "high",
                "description": f"New domain combination discovered: {', '.join(novel_domains)}"
            })
        
        # Check for unusually high confidence insights
        high_confidence_insights = [i for i in result.insights if i.confidence > 0.9]
        if high_confidence_insights and not similar_insights:
            new_patterns.append({
                "pattern_type": "high_confidence_novel_insight",
                "insight_count": len(high_confidence_insights),
                "potential_value": "very_high",
                "description": "High confidence insights with no historical precedent"
            })
        
        return new_patterns
    
    async def _store_enhanced_synthesis(
        self,
        result: VectorEnhancedSynthesisResult,
        business_context: Dict[str, Any]
    ) -> None:
        """Store enhanced synthesis for future learning."""
        if not self.embedder or not self.vector_table or not VECTOR_INFRASTRUCTURE_AVAILABLE:
            return
        
        try:
            # Create comprehensive text representation
            synthesis_text = self._create_synthesis_text(result)
            
            # Generate embedding
            embedding = self.embedder.encode(synthesis_text).tolist()
            
            # Prepare vector metadata
            vector_metadata = VectorMetadata(
                id=generate_vector_id(),
                module_source=ModuleSource.INSIGHT_SYNTHESIS,
                timestamp=datetime.now(timezone.utc),
                business_domain=self._map_to_unified_domain(business_context.get("domain", "general")),
                complexity_score=normalize_score_to_unified_scale(
                    result.synthesis_effectiveness_score,
                    source_min=0,
                    source_max=1
                ),
                confidence_score=normalize_score_to_unified_scale(
                    result.pattern_based_confidence,
                    source_min=0,
                    source_max=1  
                ),
                actionability_score=normalize_score_to_unified_scale(
                    np.mean([i.actionability for i in result.insights]) if result.insights else 0.5,
                    source_min=0,
                    source_max=1
                ),
                business_value_score=normalize_score_to_unified_scale(
                    result.estimated_business_value,
                    source_min=0,
                    source_max=1
                ),
                performance_tier=self._determine_performance_tier(result),
                query_patterns=self._extract_query_patterns(result),
                success_metrics={
                    "insight_count": len(result.insights),
                    "recommendation_count": len(result.recommendations),
                    "cross_module_insights": len(result.cross_module_insights),
                    "quality_boost": result.insight_quality_boost,
                    "accuracy_boost": result.recommendation_accuracy_boost
                },
                error_patterns=[],
                optimization_suggestions=self._generate_optimization_suggestions(result),
                module_specific_scores={
                    "strategic_alignment": result.strategic_alignment_score,
                    "predicted_adoption": result.predicted_adoption_rate,
                    "success_probability": result.success_probability
                }
            )
            
            # Store vector
            await self.vector_table.add([{
                "id": vector_metadata.id,
                "vector": embedding,
                "text_content": synthesis_text[:3000],  # Truncate for storage
                "module_source": vector_metadata.module_source.value,
                "timestamp": vector_metadata.timestamp.isoformat(),
                "business_domain": vector_metadata.business_domain.value,
                "complexity_score": vector_metadata.complexity_score,
                "confidence_score": vector_metadata.confidence_score,
                "actionability_score": vector_metadata.actionability_score,
                "business_value_score": vector_metadata.business_value_score,
                "performance_tier": vector_metadata.performance_tier.value,
                "query_patterns": json.dumps(vector_metadata.query_patterns),
                "success_metrics": json.dumps(vector_metadata.success_metrics),
                "error_patterns": json.dumps(vector_metadata.error_patterns),
                "optimization_suggestions": json.dumps(vector_metadata.optimization_suggestions),
                "module_specific_scores": json.dumps(vector_metadata.module_specific_scores),
                "content_metadata": json.dumps({
                    "synthesis_id": result.investigation_id,
                    "insight_types": [i.type.value for i in result.insights],
                    "recommendation_types": [r.type.value for r in result.recommendations],
                    "strategic_depth": np.mean([i.strategic_depth for i in result.insights]),
                    "adoption_rate": result.predicted_adoption_rate,
                    "success_metrics": result.synthesis_metadata,
                    "context": business_context,
                    "stakeholders": list(set(s for i in result.insights for s in i.stakeholders)),
                    "recommendations": [r.title for r in result.recommendations[:5]]
                })
            }])
            
            # Track performance metrics
            if self.performance_monitor:
                await self.performance_monitor.track_metric(
                    PerformanceMetric(
                        metric_type=PerformanceMetricType.INGESTION_LATENCY,
                        value=result.vector_search_time_ms,
                        module_source=ModuleSource.INSIGHT_SYNTHESIS,
                        operation_context={
                            "insight_count": len(result.insights),
                            "pattern_count": len(result.similar_insights),
                            "cross_module_count": len(result.cross_module_insights)
                        }
                    )
                )
            
            result.vector_id = vector_metadata.id
            
        except Exception as e:
            self.logger.error(f"Failed to store enhanced synthesis: {e}")
    
    # Helper methods
    
    def _create_insight_search_query(
        self, 
        investigation_results: Dict[str, Any],
        business_context: Dict[str, Any]
    ) -> str:
        """Create search query from investigation results."""
        components = []
        
        # Add investigation findings
        findings = investigation_results.get("investigation_findings", {})
        if findings:
            key_findings = findings.get("key_findings", [])
            components.extend(key_findings[:3])
        
        # Add business context
        if business_context.get("current_initiative"):
            components.append(f"Initiative: {business_context['current_initiative']}")
        if business_context.get("strategic_goal"):
            components.append(f"Goal: {business_context['strategic_goal']}")
        
        # Add domain context
        if investigation_results.get("business_context", {}).get("domain"):
            components.append(f"Domain: {investigation_results['business_context']['domain']}")
        
        return " ".join(components)
    
    def _create_synthesis_text(self, result: VectorEnhancedSynthesisResult) -> str:
        """Create comprehensive text representation of synthesis."""
        components = []
        
        # Add executive summary
        components.append(f"Executive Summary: {result.executive_summary}")
        
        # Add top insights
        for insight in result.insights[:3]:
            components.append(f"{insight.type.value}: {insight.title} - {insight.description}")
        
        # Add key recommendations
        for rec in result.recommendations[:3]:
            components.append(f"Recommendation: {rec.title} - {rec.rationale}")
        
        # Add organizational learning
        if result.organizational_learning:
            components.append(f"Pattern: {result.organizational_learning.pattern_description}")
        
        return " ".join(components)
    
    def _map_to_unified_domain(self, domain: str) -> UnifiedBusinessDomain:
        """Map synthesis domain to unified business domain."""
        domain_mapping = {
            "sales": UnifiedBusinessDomain.SALES,
            "customer": UnifiedBusinessDomain.CUSTOMER,
            "operations": UnifiedBusinessDomain.OPERATIONS,
            "finance": UnifiedBusinessDomain.FINANCE,
            "supply_chain": UnifiedBusinessDomain.SUPPLY_CHAIN,
            "inventory": UnifiedBusinessDomain.INVENTORY,
            "quality": UnifiedBusinessDomain.QUALITY,
            "hr": UnifiedBusinessDomain.HR,
            "marketing": UnifiedBusinessDomain.MARKETING,
            "production": UnifiedBusinessDomain.PRODUCTION,
            "logistics": UnifiedBusinessDomain.LOGISTICS,
            "compliance": UnifiedBusinessDomain.COMPLIANCE,
            "technology": UnifiedBusinessDomain.TECHNOLOGY
        }
        return domain_mapping.get(domain.lower(), UnifiedBusinessDomain.GENERAL)
    
    def _determine_performance_tier(self, result: VectorEnhancedSynthesisResult) -> PerformanceTier:
        """Determine performance tier based on synthesis quality."""
        quality_score = result.synthesis_effectiveness_score
        
        if quality_score >= 0.9:
            return PerformanceTier.CRITICAL
        elif quality_score >= 0.7:
            return PerformanceTier.HIGH_PRIORITY
        elif quality_score >= 0.5:
            return PerformanceTier.STANDARD
        else:
            return PerformanceTier.BATCH
    
    def _extract_query_patterns(self, result: VectorEnhancedSynthesisResult) -> List[str]:
        """Extract query patterns from synthesis."""
        patterns = []
        
        # Extract insight patterns
        insight_types = {i.type.value for i in result.insights}
        patterns.extend([f"insight_type:{t}" for t in insight_types])
        
        # Extract recommendation patterns
        rec_types = {r.type.value for r in result.recommendations}
        patterns.extend([f"recommendation_type:{t}" for t in rec_types])
        
        # Extract domain patterns
        domains = set()
        for insight in result.insights:
            domains.update(insight.related_domains)
        patterns.extend([f"domain:{d}" for d in domains])
        
        return patterns[:10]  # Limit to top 10
    
    def _generate_optimization_suggestions(self, result: VectorEnhancedSynthesisResult) -> List[str]:
        """Generate optimization suggestions based on synthesis."""
        suggestions = []
        
        # Check for low confidence insights
        low_confidence = [i for i in result.insights if i.confidence < 0.7]
        if low_confidence:
            suggestions.append(f"Improve data quality for {len(low_confidence)} low-confidence insights")
        
        # Check for missing cross-module insights
        if len(result.cross_module_insights) < 2:
            suggestions.append("Increase cross-module data integration for comprehensive insights")
        
        # Check for recommendation feasibility
        low_feasibility = [r for r in result.recommendations if r.feasibility < 0.6]
        if low_feasibility:
            suggestions.append(f"Review resource requirements for {len(low_feasibility)} challenging recommendations")
        
        return suggestions
    
    def _calculate_domain_alignment(
        self, 
        insights: List[BusinessInsight],
        patterns: List[InsightPattern]
    ) -> float:
        """Calculate strategic domain alignment score."""
        if not insights or not patterns:
            return 0.5
        
        # Extract domains from current insights
        current_domains = set()
        for insight in insights:
            current_domains.update(insight.related_domains)
        
        # Extract successful domains from patterns
        successful_domains = set()
        for pattern in patterns:
            if pattern.business_impact_score > 0.7:
                domain = pattern.business_domain
                successful_domains.add(domain)
        
        # Calculate overlap
        if not current_domains:
            return 0.5
        
        overlap = len(current_domains & successful_domains)
        alignment = overlap / len(current_domains)
        
        return min(1.0, alignment + 0.3)  # Boost baseline alignment
    
    def _estimate_timeline(self, rec_type: RecommendationType) -> str:
        """Estimate timeline for recommendation type."""
        timelines = {
            RecommendationType.IMMEDIATE_ACTION: "1-7 days",
            RecommendationType.SHORT_TERM: "1-3 months",
            RecommendationType.LONG_TERM: "6-12 months",
            RecommendationType.STRATEGIC_INITIATIVE: "12-24 months",
            RecommendationType.PROCESS_IMPROVEMENT: "3-6 months",
            RecommendationType.RESOURCE_ALLOCATION: "1-2 months",
            RecommendationType.MONITORING: "Ongoing"
        }
        return timelines.get(rec_type, "3-6 months")
    
    def _get_common_challenges(self, rec_type: RecommendationType) -> List[str]:
        """Get common challenges for recommendation type."""
        challenges = {
            RecommendationType.IMMEDIATE_ACTION: [
                "Resource availability",
                "Quick decision making",
                "Communication speed"
            ],
            RecommendationType.STRATEGIC_INITIATIVE: [
                "Long-term commitment",
                "Budget allocation",
                "Stakeholder alignment"
            ],
            RecommendationType.PROCESS_IMPROVEMENT: [
                "Change resistance",
                "Training requirements",
                "System integration"
            ]
        }
        return challenges.get(rec_type, ["Resource constraints", "Timeline pressure"])
    
    def _get_best_practices(self, rec_type: RecommendationType) -> List[str]:
        """Get best practices for recommendation type."""
        practices = {
            RecommendationType.IMMEDIATE_ACTION: [
                "Clear communication channels",
                "Rapid response team",
                "Pre-defined escalation"
            ],
            RecommendationType.STRATEGIC_INITIATIVE: [
                "Executive sponsorship",
                "Phased implementation",
                "Regular milestone reviews"
            ],
            RecommendationType.PROCESS_IMPROVEMENT: [
                "Pilot testing",
                "User feedback loops",
                "Incremental rollout"
            ]
        }
        return practices.get(rec_type, ["Stakeholder engagement", "Regular monitoring"])
    
    def _create_fallback_synthesis(
        self,
        investigation_results: Dict[str, Any],
        business_context: Dict[str, Any],
        user_role: str
    ) -> SynthesisResult:
        """Create fallback synthesis for testing without full module."""
        # Mock synthesis result
        investigation_id = investigation_results.get("investigation_id", str(uuid.uuid4()))
        
        insights = [
            BusinessInsight(
                id=str(uuid.uuid4()),
                type=InsightType.OPERATIONAL,
                title="Operational Efficiency Opportunity",
                description="Analysis indicates potential for efficiency improvement",
                business_context="Based on investigation findings",
                supporting_evidence=["Finding 1", "Finding 2"],
                confidence=0.85,
                business_impact={"financial": 0.7, "operational": 0.9, "strategic": 0.6, "risk": 0.3},
                strategic_depth=0.7,
                actionability=0.8,
                stakeholders=["Operations Manager", "Finance Director"],
                related_domains=["operations", "finance"],
                discovery_timestamp=datetime.now(timezone.utc)
            )
        ]
        
        recommendations = [
            Recommendation(
                id=str(uuid.uuid4()),
                type=RecommendationType.SHORT_TERM,
                title="Implement Process Optimization",
                description="Optimize identified bottleneck processes",
                rationale="Based on efficiency analysis findings",
                implementation_approach="Phased implementation starting with pilot",
                resource_requirements={"team_size": 5, "duration_weeks": 12, "budget": 50000},
                expected_outcomes=["15% efficiency improvement", "Cost reduction"],
                success_metrics=["Process cycle time", "Cost per unit"],
                priority=2,
                timeline="3 months",
                risk_level="medium",
                feasibility=0.75,
                related_insight_ids=[insights[0].id]
            )
        ]
        
        org_learning = OrganizationalLearning(
            pattern_id=str(uuid.uuid4()),
            pattern_description="Efficiency improvement through process optimization",
            frequency=3,
            success_rate=0.8,
            business_value=0.75,
            applicable_domains=["operations", "manufacturing"],
            best_practices=["Pilot testing", "Stakeholder engagement"],
            lessons_learned=["Early communication critical", "Training essential"],
            improvement_opportunities=["Automation potential", "Analytics enhancement"]
        )
        
        return SynthesisResult(
            investigation_id=investigation_id,
            insights=insights,
            recommendations=recommendations,
            organizational_learning=org_learning,
            executive_summary="Investigation revealed operational efficiency opportunities with clear improvement paths",
            key_findings=["Process bottlenecks identified", "Cost reduction potential confirmed"],
            business_impact_assessment={
                "total_value": 500000,
                "roi_estimate": 2.5,
                "payback_months": 6
            },
            success_criteria=["Efficiency metrics improvement", "Cost targets achieved"],
            follow_up_actions=["Monthly progress reviews", "Stakeholder updates"],
            stakeholder_communications={
                "executive": "Significant efficiency opportunity identified with strong ROI",
                "manager": "Clear implementation path with pilot approach recommended",
                "analyst": "Detailed findings support process optimization initiative"
            },
            synthesis_metadata={
                "synthesis_duration_seconds": 5.2,
                "insight_count": len(insights),
                "recommendation_count": len(recommendations)
            }
        )
    
    def _calculate_effectiveness_score(self, result: VectorEnhancedSynthesisResult) -> float:
        """Calculate overall synthesis effectiveness score."""
        factors = []
        
        # Factor 1: Insight quality
        if result.insights:
            avg_confidence = np.mean([i.confidence for i in result.insights])
            avg_actionability = np.mean([i.actionability for i in result.insights])
            factors.append(avg_confidence * avg_actionability)
        
        # Factor 2: Recommendation feasibility
        if result.recommendations:
            avg_feasibility = np.mean([r.feasibility for r in result.recommendations])
            factors.append(avg_feasibility)
        
        # Factor 3: Pattern enhancement
        if result.pattern_based_confidence > 0:
            factors.append(result.pattern_based_confidence)
        
        # Factor 4: Cross-module integration
        if result.cross_module_insights:
            integration_score = min(1.0, len(result.cross_module_insights) / 5)
            factors.append(integration_score)
        
        # Factor 5: Predictive accuracy
        if result.success_probability > 0:
            factors.append(result.success_probability)
        
        return np.mean(factors) if factors else 0.5
    
    async def get_synthesis_statistics(self) -> Dict[str, Any]:
        """Get statistics about synthesis operations."""
        stats = {
            "total_syntheses": 0,
            "pattern_enhanced": 0,
            "cross_module_insights": 0,
            "avg_quality_boost": 0.0,
            "avg_recommendation_accuracy": 0.0,
            "capabilities": {
                "synthesis_module": SYNTHESIS_MODULE_AVAILABLE,
                "vector_infrastructure": VECTOR_INFRASTRUCTURE_AVAILABLE,
                "vector_capabilities": VECTOR_CAPABILITIES_AVAILABLE,
                "embedder_loaded": self.embedder is not None,
                "vector_table_available": self.vector_table is not None
            }
        }
        
        # Add more detailed stats if available
        if self.performance_monitor:
            perf_stats = await self.performance_monitor.get_module_statistics(
                ModuleSource.INSIGHT_SYNTHESIS
            )
            stats.update(perf_stats)
        
        return stats
    
    async def cleanup(self):
        """Cleanup vector resources."""
        if self.vector_db:
            # LanceDB cleanup if needed
            pass


# High-level interface functions

async def synthesize_insights_with_vectors(
    investigation_results: Dict[str, Any],
    business_context: Dict[str, Any],
    user_role: str = "analyst",
    output_format: OutputFormat = OutputFormat.DETAILED_REPORT,
    db_path: Optional[str] = None,
    use_vector_enhancement: bool = True
) -> VectorEnhancedSynthesisResult:
    """
    High-level function to synthesize insights with vector enhancement.
    
    Args:
        investigation_results: Raw investigation findings
        business_context: Business context and goals
        user_role: Role of the user requesting synthesis
        output_format: Desired output format
        db_path: Optional path to vector database
        use_vector_enhancement: Whether to use vector enhancement
        
    Returns:
        Enhanced synthesis result with vector intelligence
    """
    synthesizer = VectorEnhancedInsightSynthesizer()
    await synthesizer.initialize(db_path=db_path)
    
    try:
        result = await synthesizer.synthesize_insights_with_vectors(
            investigation_results=investigation_results,
            business_context=business_context,
            user_role=user_role,
            output_format=output_format,
            use_vector_enhancement=use_vector_enhancement
        )
        return result
    finally:
        await synthesizer.cleanup()


if __name__ == "__main__":
    # Test the vector-enhanced synthesizer
    import asyncio
    
    async def test_enhanced_synthesizer():
        print("Testing Vector-Enhanced Insight Synthesizer...")
        
        # Mock investigation results
        investigation_results = {
            "investigation_id": "test-123",
            "investigation_findings": {
                "key_findings": [
                    "Production efficiency dropped 15% in Q2",
                    "Quality defect rate increased to 3.2%",
                    "Customer complaints rose by 25%"
                ],
                "root_causes": [
                    "Equipment maintenance delays",
                    "Staff training gaps",
                    "Supply chain disruptions"
                ]
            },
            "confidence_scores": {
                "overall": 0.85,
                "findings": 0.9,
                "analysis": 0.8
            },
            "business_context": {
                "domain": "production",
                "impact_level": "high",
                "urgency": "immediate"
            }
        }
        
        business_context = {
            "current_initiative": "Operational Excellence",
            "strategic_goal": "15% efficiency improvement",
            "business_unit": "Manufacturing",
            "domain": "production"
        }
        
        # Test synthesis
        result = await synthesize_insights_with_vectors(
            investigation_results=investigation_results,
            business_context=business_context,
            user_role="manager",
            output_format=OutputFormat.DETAILED_REPORT,
            use_vector_enhancement=True
        )
        
        print(f"\nSynthesis completed!")
        print(f"Insights generated: {len(result.insights)}")
        print(f"Recommendations: {len(result.recommendations)}")
        print(f"Pattern confidence: {result.pattern_based_confidence:.3f}")
        print(f"Quality boost: {result.insight_quality_boost:.3f}")
        print(f"Cross-module insights: {len(result.cross_module_insights)}")
        print(f"Effectiveness score: {result.synthesis_effectiveness_score:.3f}")
        
        # Display top insight
        if result.insights:
            top_insight = result.insights[0]
            print(f"\nTop Insight:")
            print(f"- Type: {top_insight.type.value}")
            print(f"- Title: {top_insight.title}")
            print(f"- Confidence: {top_insight.confidence:.3f}")
            print(f"- Strategic Depth: {top_insight.strategic_depth:.3f}")
        
        # Display top recommendation
        if result.recommendations:
            top_rec = result.recommendations[0]
            print(f"\nTop Recommendation:")
            print(f"- Type: {top_rec.type.value}")
            print(f"- Title: {top_rec.title}")
            print(f"- Priority: {top_rec.priority}")
            print(f"- Feasibility: {top_rec.feasibility:.3f}")
    
    asyncio.run(test_enhanced_synthesizer())