"""
Pattern Recognizer - Dynamic Pattern Discovery Component
Self-contained pattern recognition and discovery for expanding investigation methodologies.
Zero external dependencies beyond module boundary.
"""

from enum import Enum
from typing import Dict, List, Tuple, Optional, Union, Set
from dataclasses import dataclass, field
import re
import json
from collections import defaultdict, Counter
from datetime import datetime, timezone

try:
    from .config import settings
    from .intelligence_logging import setup_logger, performance_monitor
    from .domain_expert import BusinessDomain, AnalysisType, BusinessIntent
    from .complexity_analyzer import ComplexityLevel, InvestigationMethodology
    from .business_context import ContextualStrategy
    from .hypothesis_generator import HypothesisSet
except ImportError:
    from config import settings
    from intelligence_logging import setup_logger, performance_monitor
    from domain_expert import BusinessDomain, AnalysisType, BusinessIntent
    from complexity_analyzer import ComplexityLevel, InvestigationMethodology
    from business_context import ContextualStrategy
    from hypothesis_generator import HypothesisSet


class PatternType(Enum):
    """Types of discoverable patterns."""
    QUERY_PATTERN = "query_pattern"              # User query patterns
    METHODOLOGY_PATTERN = "methodology_pattern"  # Investigation methodology patterns
    SUCCESS_PATTERN = "success_pattern"          # Success/failure patterns
    DOMAIN_PATTERN = "domain_pattern"           # Business domain patterns
    TEMPORAL_PATTERN = "temporal_pattern"       # Time-based patterns
    CONTEXTUAL_PATTERN = "contextual_pattern"   # Context-dependent patterns


class PatternSource(Enum):
    """Sources of pattern data."""
    USER_QUERIES = "user_queries"
    INVESTIGATION_RESULTS = "investigation_results"
    SUCCESS_METRICS = "success_metrics"
    FAILURE_ANALYSIS = "failure_analysis"
    DOMAIN_FEEDBACK = "domain_feedback"
    CONTEXTUAL_ADAPTATION = "contextual_adaptation"


@dataclass
class PatternEvidence:
    """Evidence supporting a discovered pattern."""
    observation_count: int
    success_rate: float
    confidence_interval: Tuple[float, float]
    statistical_significance: float
    examples: List[str]
    counter_examples: List[str]
    last_observed: datetime
    trend_direction: str  # increasing, decreasing, stable


@dataclass
class DiscoveredPattern:
    """A discovered pattern with supporting evidence."""
    id: str
    type: PatternType
    source: PatternSource
    description: str
    conditions: Dict[str, Union[str, float, List[str]]]
    evidence: PatternEvidence
    business_value: float  # 0.0 to 1.0
    implementation_complexity: str  # low, medium, high
    recommended_action: str
    related_domains: List[BusinessDomain]
    metadata: Dict[str, Union[str, float, int]] = field(default_factory=dict)


@dataclass
class PatternLibraryUpdate:
    """Update to pattern library based on discoveries."""
    new_patterns: List[DiscoveredPattern]
    updated_patterns: List[str]  # Pattern IDs that were modified
    deprecated_patterns: List[str]  # Pattern IDs that should be removed
    confidence_updates: Dict[str, float]  # Pattern ID -> new confidence
    usage_statistics: Dict[str, int]  # Pattern usage counts
    effectiveness_metrics: Dict[str, float]  # Pattern effectiveness scores


class PatternRecognizer:
    """
    Dynamic pattern recognition and discovery system.
    Identifies new investigation patterns and improves existing methodologies.
    """
    
    def __init__(self):
        self.logger = setup_logger("pattern_recognizer")
        self._pattern_templates = self._load_pattern_templates()
        self._recognition_algorithms = self._load_recognition_algorithms()
        self._statistical_thresholds = self._load_statistical_thresholds()
        self._pattern_cache = {}
        self._observation_history = []
        
        self.logger.info("Pattern Recognizer initialized with discovery algorithms")
    
    def _load_pattern_templates(self) -> Dict[PatternType, Dict[str, Union[str, List[str]]]]:
        """Load pattern recognition templates."""
        return {
            PatternType.QUERY_PATTERN: {
                "features": [
                    "keyword_frequency", "domain_distribution", "complexity_levels",
                    "time_context_usage", "urgency_indicators", "metric_references"
                ],
                "clustering_method": "semantic_similarity",
                "min_observations": 10,
                "confidence_threshold": 0.7
            },
            
            PatternType.METHODOLOGY_PATTERN: {
                "features": [
                    "methodology_success_rate", "context_effectiveness", "user_satisfaction",
                    "duration_efficiency", "resource_utilization", "complexity_handling"
                ],
                "clustering_method": "performance_based",
                "min_observations": 5,
                "confidence_threshold": 0.8
            },
            
            PatternType.SUCCESS_PATTERN: {
                "features": [
                    "investigation_outcome", "user_role", "domain_expertise",
                    "methodology_choice", "complexity_level", "time_investment"
                ],
                "clustering_method": "outcome_correlation",
                "min_observations": 15,
                "confidence_threshold": 0.75
            },
            
            PatternType.DOMAIN_PATTERN: {
                "features": [
                    "domain_specific_queries", "indicator_patterns", "methodology_preferences",
                    "success_factors", "common_issues", "solution_approaches"
                ],
                "clustering_method": "domain_similarity",
                "min_observations": 8,
                "confidence_threshold": 0.7
            },
            
            PatternType.TEMPORAL_PATTERN: {
                "features": [
                    "time_of_day", "day_of_week", "seasonal_trends",
                    "business_cycle_correlation", "urgency_timing", "completion_patterns"
                ],
                "clustering_method": "temporal_analysis",
                "min_observations": 20,
                "confidence_threshold": 0.6
            },
            
            PatternType.CONTEXTUAL_PATTERN: {
                "features": [
                    "organizational_context", "user_experience_level", "resource_constraints",
                    "business_priorities", "cultural_factors", "industry_specifics"
                ],
                "clustering_method": "contextual_similarity",
                "min_observations": 12,
                "confidence_threshold": 0.7
            }
        }
    
    def _load_recognition_algorithms(self) -> Dict[str, Dict[str, Union[str, float]]]:
        """Load pattern recognition algorithms."""
        return {
            "semantic_similarity": {
                "method": "cosine_similarity",
                "threshold": 0.8,
                "clustering_algorithm": "dbscan",
                "min_cluster_size": 3
            },
            
            "performance_based": {
                "method": "performance_correlation",
                "threshold": 0.7,
                "clustering_algorithm": "kmeans",
                "optimization_metric": "effectiveness_score"
            },
            
            "outcome_correlation": {
                "method": "statistical_correlation",
                "threshold": 0.6,
                "significance_level": 0.05,
                "correlation_method": "pearson"
            },
            
            "domain_similarity": {
                "method": "domain_embedding",
                "threshold": 0.75,
                "clustering_algorithm": "hierarchical",
                "linkage_method": "ward"
            },
            
            "temporal_analysis": {
                "method": "time_series_clustering",
                "threshold": 0.65,
                "seasonality_detection": True,
                "trend_analysis": True
            },
            
            "contextual_similarity": {
                "method": "multi_dimensional_scaling",
                "threshold": 0.7,
                "feature_weighting": "adaptive",
                "dimensionality_reduction": "pca"
            }
        }
    
    def _load_statistical_thresholds(self) -> Dict[str, float]:
        """Load statistical significance thresholds."""
        return {
            "min_sample_size": 10,
            "confidence_level": 0.95,
            "significance_level": 0.05,
            "effect_size_threshold": 0.3,
            "correlation_threshold": 0.5,
            "pattern_stability_threshold": 0.8,
            "trend_significance_threshold": 0.7
        }
    
    @performance_monitor("pattern_recognition")
    def analyze_investigation_patterns(
        self,
        investigation_history: List[Dict[str, Union[str, float, bool]]],
        current_investigation: Optional[Dict[str, Union[str, float]]] = None
    ) -> PatternLibraryUpdate:
        """
        Analyze investigation history to discover new patterns.
        
        Args:
            investigation_history: Historical investigation data
            current_investigation: Current investigation context (optional)
            
        Returns:
            PatternLibraryUpdate with discovered patterns and recommendations
        """
        if len(investigation_history) < self._statistical_thresholds["min_sample_size"]:
            self.logger.warning(
                f"Insufficient data for pattern analysis: {len(investigation_history)} investigations"
            )
            return PatternLibraryUpdate(
                new_patterns=[], updated_patterns=[], deprecated_patterns=[],
                confidence_updates={}, usage_statistics={}, effectiveness_metrics={}
            )
        
        # Store observation for future analysis
        if current_investigation:
            self._observation_history.append({
                **current_investigation,
                "timestamp": datetime.now(timezone.utc)
            })
        
        # Discover patterns by type
        discovered_patterns = []
        
        for pattern_type in PatternType:
            type_patterns = self._discover_patterns_by_type(
                investigation_history, pattern_type
            )
            discovered_patterns.extend(type_patterns)
        
        # Validate and filter patterns
        validated_patterns = self._validate_patterns(discovered_patterns, investigation_history)
        
        # Calculate pattern library updates
        library_update = self._calculate_library_updates(
            validated_patterns, investigation_history
        )
        
        self.logger.info(
            f"Discovered {len(validated_patterns)} patterns from "
            f"{len(investigation_history)} investigations"
        )
        
        return library_update
    
    def _discover_patterns_by_type(
        self, investigation_history: List[Dict], pattern_type: PatternType
    ) -> List[DiscoveredPattern]:
        """Discover patterns of a specific type."""
        
        template = self._pattern_templates[pattern_type]
        min_observations = template["min_observations"]
        
        if len(investigation_history) < min_observations:
            return []
        
        # Extract features based on pattern type
        features = self._extract_features(investigation_history, pattern_type)
        
        # Apply clustering/analysis algorithm
        clustering_method = template["clustering_method"]
        clusters = self._apply_clustering_algorithm(features, clustering_method)
        
        # Generate patterns from clusters
        patterns = []
        for cluster_id, cluster_data in clusters.items():
            if len(cluster_data) >= min_observations:
                pattern = self._generate_pattern_from_cluster(
                    cluster_data, pattern_type, cluster_id
                )
                if pattern:
                    patterns.append(pattern)
        
        return patterns
    
    def _extract_features(
        self, investigation_history: List[Dict], pattern_type: PatternType
    ) -> Dict[str, List[Union[str, float]]]:
        """Extract relevant features for pattern type."""
        
        features = defaultdict(list)
        
        for investigation in investigation_history:
            if pattern_type == PatternType.QUERY_PATTERN:
                features["keywords"].append(self._extract_keywords(investigation.get("query", "")))
                features["domain"].append(investigation.get("domain", "unknown"))
                features["complexity"].append(investigation.get("complexity_level", "medium"))
                features["urgency"].append(investigation.get("urgency_level", "normal"))
                
            elif pattern_type == PatternType.METHODOLOGY_PATTERN:
                features["methodology"].append(investigation.get("methodology", "unknown"))
                features["success_rate"].append(investigation.get("success", 0.5))
                features["duration"].append(investigation.get("duration_minutes", 30))
                features["user_satisfaction"].append(investigation.get("user_satisfaction", 0.5))
                
            elif pattern_type == PatternType.SUCCESS_PATTERN:
                features["outcome"].append(investigation.get("success", False))
                features["user_role"].append(investigation.get("user_role", "unknown"))
                features["domain_expertise"].append(investigation.get("domain_expertise", 0.5))
                features["methodology"].append(investigation.get("methodology", "unknown"))
                
            elif pattern_type == PatternType.DOMAIN_PATTERN:
                features["domain"].append(investigation.get("domain", "unknown"))
                features["query_indicators"].append(
                    self._extract_domain_indicators(investigation.get("query", ""))
                )
                features["methodology_used"].append(investigation.get("methodology", "unknown"))
                
            elif pattern_type == PatternType.TEMPORAL_PATTERN:
                timestamp = investigation.get("timestamp", datetime.now(timezone.utc))
                if isinstance(timestamp, str):
                    timestamp = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
                
                features["hour_of_day"].append(timestamp.hour)
                features["day_of_week"].append(timestamp.weekday())
                features["completion_time"].append(investigation.get("duration_minutes", 30))
                
            elif pattern_type == PatternType.CONTEXTUAL_PATTERN:
                features["org_context"].append(investigation.get("organizational_context", "unknown"))
                features["user_experience"].append(investigation.get("user_experience_level", "intermediate"))
                features["resource_constraints"].append(investigation.get("resource_constraints", "none"))
        
        return dict(features)
    
    def _extract_keywords(self, query: str) -> List[str]:
        """Extract meaningful keywords from query."""
        # Simple keyword extraction - could be enhanced with NLP
        words = re.findall(r'\b\w+\b', query.lower())
        # Filter out common words
        stop_words = {
            "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
            "of", "with", "by", "is", "are", "was", "were", "be", "been", "have",
            "has", "had", "do", "does", "did", "will", "would", "could", "should"
        }
        return [word for word in words if word not in stop_words and len(word) > 2]
    
    def _extract_domain_indicators(self, query: str) -> List[str]:
        """Extract business domain indicators from query."""
        domain_keywords = {
            "production": ["production", "manufacturing", "output", "efficiency", "line"],
            "quality": ["quality", "defect", "inspection", "testing", "compliance"],
            "supply_chain": ["supplier", "inventory", "delivery", "logistics", "procurement"],
            "cost": ["cost", "expense", "budget", "price", "margin", "savings"],
            "customer": ["customer", "satisfaction", "complaint", "service", "feedback"],
            "assets": ["equipment", "maintenance", "downtime", "reliability", "asset"]
        }
        
        query_lower = query.lower()
        indicators = []
        
        for domain, keywords in domain_keywords.items():
            if any(keyword in query_lower for keyword in keywords):
                indicators.append(domain)
        
        return indicators
    
    def _apply_clustering_algorithm(
        self, features: Dict[str, List], clustering_method: str
    ) -> Dict[str, List[Dict]]:
        """Apply clustering algorithm to identify patterns."""
        
        # Simplified clustering - in production, would use actual ML algorithms
        clusters = defaultdict(list)
        
        if clustering_method == "semantic_similarity":
            clusters = self._semantic_clustering(features)
        elif clustering_method == "performance_based":
            clusters = self._performance_clustering(features)
        elif clustering_method == "outcome_correlation":
            clusters = self._outcome_clustering(features)
        elif clustering_method == "domain_similarity":
            clusters = self._domain_clustering(features)
        elif clustering_method == "temporal_analysis":
            clusters = self._temporal_clustering(features)
        elif clustering_method == "contextual_similarity":
            clusters = self._contextual_clustering(features)
        
        return clusters
    
    def _semantic_clustering(self, features: Dict[str, List]) -> Dict[str, List[Dict]]:
        """Cluster based on semantic similarity of keywords."""
        clusters = defaultdict(list)
        
        if "keywords" in features:
            # Group by common keyword patterns
            keyword_combinations = {}
            for i, keywords in enumerate(features["keywords"]):
                key = tuple(sorted(keywords))
                if key not in keyword_combinations:
                    keyword_combinations[key] = []
                keyword_combinations[key].append(i)
            
            # Create clusters from groups with sufficient size
            cluster_id = 0
            for keyword_combo, indices in keyword_combinations.items():
                if len(indices) >= 3:  # Minimum cluster size
                    for idx in indices:
                        clusters[f"semantic_{cluster_id}"].append({
                            "index": idx,
                            "keywords": features["keywords"][idx],
                            "domain": features.get("domain", ["unknown"] * len(features["keywords"]))[idx]
                        })
                    cluster_id += 1
        
        return dict(clusters)
    
    def _performance_clustering(self, features: Dict[str, List]) -> Dict[str, List[Dict]]:
        """Cluster based on performance metrics."""
        clusters = defaultdict(list)
        
        if "methodology" in features and "success_rate" in features:
            # Group by methodology and performance level
            method_performance = defaultdict(list)
            
            for i, (method, success) in enumerate(zip(features["methodology"], features["success_rate"])):
                performance_tier = "high" if success > 0.8 else "medium" if success > 0.5 else "low"
                key = f"{method}_{performance_tier}"
                method_performance[key].append({
                    "index": i,
                    "methodology": method,
                    "success_rate": success,
                    "performance_tier": performance_tier
                })
            
            # Create clusters from significant groups
            for key, group in method_performance.items():
                if len(group) >= 3:
                    clusters[key] = group
        
        return dict(clusters)
    
    def _outcome_clustering(self, features: Dict[str, List]) -> Dict[str, List[Dict]]:
        """Cluster based on investigation outcomes."""
        clusters = defaultdict(list)
        
        if "outcome" in features:
            success_group = []
            failure_group = []
            
            for i, outcome in enumerate(features["outcome"]):
                data = {"index": i, "outcome": outcome}
                
                # Add additional context
                for feature_name in ["user_role", "methodology", "domain_expertise"]:
                    if feature_name in features:
                        data[feature_name] = features[feature_name][i]
                
                if outcome:
                    success_group.append(data)
                else:
                    failure_group.append(data)
            
            if len(success_group) >= 5:
                clusters["success_patterns"] = success_group
            if len(failure_group) >= 5:
                clusters["failure_patterns"] = failure_group
        
        return dict(clusters)
    
    def _domain_clustering(self, features: Dict[str, List]) -> Dict[str, List[Dict]]:
        """Cluster based on business domain patterns."""
        clusters = defaultdict(list)
        
        if "domain" in features:
            domain_groups = defaultdict(list)
            
            for i, domain in enumerate(features["domain"]):
                data = {"index": i, "domain": domain}
                
                # Add domain-specific features
                if "query_indicators" in features:
                    data["indicators"] = features["query_indicators"][i]
                if "methodology_used" in features:
                    data["methodology"] = features["methodology_used"][i]
                
                domain_groups[domain].append(data)
            
            # Create clusters for domains with sufficient data
            for domain, group in domain_groups.items():
                if len(group) >= 5:
                    clusters[f"domain_{domain}"] = group
        
        return dict(clusters)
    
    def _temporal_clustering(self, features: Dict[str, List]) -> Dict[str, List[Dict]]:
        """Cluster based on temporal patterns."""
        clusters = defaultdict(list)
        
        if "hour_of_day" in features:
            # Group by time periods
            time_periods = {
                "morning": range(6, 12),
                "afternoon": range(12, 18),
                "evening": range(18, 24),
                "night": list(range(0, 6))
            }
            
            period_groups = defaultdict(list)
            
            for i, hour in enumerate(features["hour_of_day"]):
                for period, hours in time_periods.items():
                    if hour in hours:
                        period_groups[period].append({
                            "index": i,
                            "hour": hour,
                            "period": period,
                            "completion_time": features.get("completion_time", [30] * len(features["hour_of_day"]))[i]
                        })
                        break
            
            # Create clusters for significant time periods
            for period, group in period_groups.items():
                if len(group) >= 8:
                    clusters[f"temporal_{period}"] = group
        
        return dict(clusters)
    
    def _contextual_clustering(self, features: Dict[str, List]) -> Dict[str, List[Dict]]:
        """Cluster based on contextual similarities."""
        clusters = defaultdict(list)
        
        if "org_context" in features:
            context_groups = defaultdict(list)
            
            for i, context in enumerate(features["org_context"]):
                data = {
                    "index": i,
                    "org_context": context,
                    "user_experience": features.get("user_experience", ["intermediate"] * len(features["org_context"]))[i],
                    "resource_constraints": features.get("resource_constraints", ["none"] * len(features["org_context"]))[i]
                }
                
                context_groups[context].append(data)
            
            # Create clusters for contexts with sufficient data
            for context, group in context_groups.items():
                if len(group) >= 6:
                    clusters[f"context_{context}"] = group
        
        return dict(clusters)
    
    def _generate_pattern_from_cluster(
        self, cluster_data: List[Dict], pattern_type: PatternType, cluster_id: str
    ) -> Optional[DiscoveredPattern]:
        """Generate a discovered pattern from cluster analysis."""
        
        if len(cluster_data) < 3:
            return None
        
        # Calculate pattern evidence
        evidence = self._calculate_pattern_evidence(cluster_data, pattern_type)
        
        # Generate pattern description and conditions
        description, conditions = self._generate_pattern_description(
            cluster_data, pattern_type, cluster_id
        )
        
        # Calculate business value
        business_value = self._calculate_business_value(cluster_data, pattern_type)
        
        # Determine implementation complexity
        implementation_complexity = self._assess_implementation_complexity(
            pattern_type, cluster_data
        )
        
        # Generate recommended action
        recommended_action = self._generate_recommended_action(
            pattern_type, cluster_data, evidence
        )
        
        # Determine related domains
        related_domains = self._identify_related_domains(cluster_data)
        
        pattern = DiscoveredPattern(
            id=f"pattern_{pattern_type.value}_{cluster_id}_{datetime.now().strftime('%Y%m%d')}",
            type=pattern_type,
            source=self._determine_pattern_source(pattern_type),
            description=description,
            conditions=conditions,
            evidence=evidence,
            business_value=business_value,
            implementation_complexity=implementation_complexity,
            recommended_action=recommended_action,
            related_domains=related_domains,
            metadata={
                "cluster_size": len(cluster_data),
                "discovery_date": datetime.now(timezone.utc).isoformat(),
                "pattern_version": "1.0"
            }
        )
        
        return pattern
    
    def _calculate_pattern_evidence(
        self, cluster_data: List[Dict], pattern_type: PatternType
    ) -> PatternEvidence:
        """Calculate statistical evidence for pattern."""
        
        observation_count = len(cluster_data)
        
        # Calculate success rate based on pattern type
        if pattern_type == PatternType.SUCCESS_PATTERN:
            successes = sum(1 for item in cluster_data if item.get("outcome", False))
            success_rate = successes / observation_count if observation_count > 0 else 0.0
        elif pattern_type == PatternType.METHODOLOGY_PATTERN:
            success_rates = [item.get("success_rate", 0.5) for item in cluster_data]
            success_rate = sum(success_rates) / len(success_rates) if success_rates else 0.5
        else:
            success_rate = 0.7  # Default for non-outcome patterns
        
        # Calculate confidence interval
        if observation_count > 1:
            # Simplified confidence interval calculation
            std_error = (success_rate * (1 - success_rate) / observation_count) ** 0.5
            margin_error = 1.96 * std_error  # 95% confidence
            confidence_interval = (
                max(0.0, success_rate - margin_error),
                min(1.0, success_rate + margin_error)
            )
        else:
            confidence_interval = (0.0, 1.0)
        
        # Calculate statistical significance
        if observation_count >= 10:
            statistical_significance = min(0.95, 0.5 + (observation_count / 100))
        else:
            statistical_significance = observation_count / 10
        
        # Generate examples
        examples = [
            str(item.get("description", f"Example {i+1}"))
            for i, item in enumerate(cluster_data[:3])
        ]
        
        return PatternEvidence(
            observation_count=observation_count,
            success_rate=success_rate,
            confidence_interval=confidence_interval,
            statistical_significance=statistical_significance,
            examples=examples,
            counter_examples=[],  # Would be populated with contrasting examples
            last_observed=datetime.now(timezone.utc),
            trend_direction="stable"  # Would be calculated from time series
        )
    
    def _generate_pattern_description(
        self, cluster_data: List[Dict], pattern_type: PatternType, cluster_id: str
    ) -> Tuple[str, Dict[str, Union[str, float, List[str]]]]:
        """Generate human-readable pattern description and conditions."""
        
        conditions = {}
        
        if pattern_type == PatternType.QUERY_PATTERN:
            # Analyze common keywords and domains
            all_keywords = []
            domains = []
            
            for item in cluster_data:
                if "keywords" in item:
                    all_keywords.extend(item["keywords"])
                if "domain" in item:
                    domains.append(item["domain"])
            
            common_keywords = [word for word, count in Counter(all_keywords).most_common(5)]
            common_domain = Counter(domains).most_common(1)[0][0] if domains else "unknown"
            
            description = (
                f"Query pattern involving {', '.join(common_keywords[:3])} "
                f"primarily in {common_domain} domain with "
                f"{len(cluster_data)} similar investigations"
            )
            
            conditions = {
                "common_keywords": common_keywords,
                "primary_domain": common_domain,
                "min_keyword_overlap": 0.6
            }
            
        elif pattern_type == PatternType.METHODOLOGY_PATTERN:
            # Analyze methodology effectiveness
            methodologies = [item.get("methodology", "unknown") for item in cluster_data]
            primary_methodology = Counter(methodologies).most_common(1)[0][0]
            
            avg_success = sum(item.get("success_rate", 0.5) for item in cluster_data) / len(cluster_data)
            
            description = (
                f"Methodology pattern showing {primary_methodology} "
                f"achieving {avg_success:.1%} success rate across "
                f"{len(cluster_data)} investigations"
            )
            
            conditions = {
                "methodology": primary_methodology,
                "success_rate_threshold": avg_success,
                "min_observations": len(cluster_data)
            }
            
        elif pattern_type == PatternType.SUCCESS_PATTERN:
            # Analyze success factors
            success_factors = []
            
            if any("user_role" in item for item in cluster_data):
                roles = [item.get("user_role", "unknown") for item in cluster_data]
                common_role = Counter(roles).most_common(1)[0][0]
                success_factors.append(f"user role: {common_role}")
                conditions["user_role"] = common_role
            
            if any("methodology" in item for item in cluster_data):
                methods = [item.get("methodology", "unknown") for item in cluster_data]
                common_method = Counter(methods).most_common(1)[0][0]
                success_factors.append(f"methodology: {common_method}")
                conditions["methodology"] = common_method
            
            description = (
                f"Success pattern identified with {', '.join(success_factors)} "
                f"showing consistent positive outcomes in {len(cluster_data)} cases"
            )
            
        elif pattern_type == PatternType.DOMAIN_PATTERN:
            domains = [item.get("domain", "unknown") for item in cluster_data]
            primary_domain = Counter(domains).most_common(1)[0][0]
            
            description = (
                f"Domain-specific pattern for {primary_domain} investigations "
                f"with recurring characteristics across {len(cluster_data)} cases"
            )
            
            conditions = {
                "domain": primary_domain,
                "pattern_strength": len(cluster_data) / 10  # Normalized strength
            }
            
        elif pattern_type == PatternType.TEMPORAL_PATTERN:
            if cluster_data and "period" in cluster_data[0]:
                period = cluster_data[0]["period"]
                avg_completion = sum(item.get("completion_time", 30) for item in cluster_data) / len(cluster_data)
                
                description = (
                    f"Temporal pattern showing investigations during {period} "
                    f"typically complete in {avg_completion:.1f} minutes "
                    f"based on {len(cluster_data)} observations"
                )
                
                conditions = {
                    "time_period": period,
                    "avg_completion_time": avg_completion,
                    "pattern_frequency": len(cluster_data)
                }
            else:
                description = f"Temporal pattern with {len(cluster_data)} observations"
                conditions = {"observation_count": len(cluster_data)}
                
        elif pattern_type == PatternType.CONTEXTUAL_PATTERN:
            contexts = [item.get("org_context", "unknown") for item in cluster_data]
            primary_context = Counter(contexts).most_common(1)[0][0]
            
            description = (
                f"Contextual pattern for {primary_context} organizations "
                f"showing consistent behavior across {len(cluster_data)} investigations"
            )
            
            conditions = {
                "organizational_context": primary_context,
                "pattern_consistency": 0.8  # High consistency threshold
            }
        
        else:
            description = f"Pattern of type {pattern_type.value} with {len(cluster_data)} observations"
            conditions = {"observation_count": len(cluster_data)}
        
        return description, conditions
    
    def _calculate_business_value(
        self, cluster_data: List[Dict], pattern_type: PatternType
    ) -> float:
        """Calculate business value score for pattern (0.0 to 1.0)."""
        
        base_value = 0.5
        
        # Value based on pattern type
        type_values = {
            PatternType.SUCCESS_PATTERN: 0.9,     # High value for success insights
            PatternType.METHODOLOGY_PATTERN: 0.8, # High value for methodology optimization
            PatternType.QUERY_PATTERN: 0.6,       # Medium value for query insights
            PatternType.DOMAIN_PATTERN: 0.7,      # Good value for domain specificity
            PatternType.TEMPORAL_PATTERN: 0.5,    # Moderate value for timing insights
            PatternType.CONTEXTUAL_PATTERN: 0.6   # Medium value for context adaptation
        }
        
        base_value = type_values.get(pattern_type, 0.5)
        
        # Adjust based on cluster size (more observations = higher confidence)
        size_multiplier = min(1.2, 1.0 + (len(cluster_data) - 5) / 20)
        
        # Adjust based on success rate if available
        if pattern_type in [PatternType.SUCCESS_PATTERN, PatternType.METHODOLOGY_PATTERN]:
            if pattern_type == PatternType.SUCCESS_PATTERN:
                successes = sum(1 for item in cluster_data if item.get("outcome", False))
                success_rate = successes / len(cluster_data) if cluster_data else 0.5
            else:
                success_rates = [item.get("success_rate", 0.5) for item in cluster_data]
                success_rate = sum(success_rates) / len(success_rates) if success_rates else 0.5
            
            success_multiplier = 0.5 + success_rate
            base_value *= success_multiplier
        
        return min(1.0, base_value * size_multiplier)
    
    def _assess_implementation_complexity(
        self, pattern_type: PatternType, cluster_data: List[Dict]
    ) -> str:
        """Assess implementation complexity for pattern."""
        
        # Base complexity by pattern type
        type_complexity = {
            PatternType.QUERY_PATTERN: "low",       # Easy to implement query routing
            PatternType.METHODOLOGY_PATTERN: "medium", # Requires methodology updates
            PatternType.SUCCESS_PATTERN: "medium",  # Requires success factor integration
            PatternType.DOMAIN_PATTERN: "low",      # Domain-specific adaptations
            PatternType.TEMPORAL_PATTERN: "low",    # Time-based rules
            PatternType.CONTEXTUAL_PATTERN: "high"  # Complex context integration
        }
        
        base_complexity = type_complexity.get(pattern_type, "medium")
        
        # Adjust based on cluster characteristics
        if len(cluster_data) > 20:
            # More data makes implementation more reliable but potentially complex
            if base_complexity == "low":
                return "medium"
        
        return base_complexity
    
    def _generate_recommended_action(
        self, pattern_type: PatternType, cluster_data: List[Dict], evidence: PatternEvidence
    ) -> str:
        """Generate recommended action for implementing pattern."""
        
        if pattern_type == PatternType.QUERY_PATTERN:
            return (
                "Create query template or routing rule to automatically classify "
                "similar queries and suggest appropriate investigation approaches"
            )
            
        elif pattern_type == PatternType.METHODOLOGY_PATTERN:
            if evidence.success_rate > 0.8:
                return (
                    "Promote this methodology as preferred approach for similar contexts "
                    "and update methodology selection algorithms"
                )
            else:
                return (
                    "Investigate factors contributing to lower success rate and "
                    "develop methodology improvements"
                )
                
        elif pattern_type == PatternType.SUCCESS_PATTERN:
            return (
                "Incorporate identified success factors into investigation planning "
                "and user guidance systems to improve overall outcomes"
            )
            
        elif pattern_type == PatternType.DOMAIN_PATTERN:
            return (
                "Develop domain-specific investigation templates and guidance "
                "based on recurring patterns in this business area"
            )
            
        elif pattern_type == PatternType.TEMPORAL_PATTERN:
            return (
                "Adjust resource allocation and expectations based on timing patterns "
                "to optimize investigation scheduling and duration estimates"
            )
            
        elif pattern_type == PatternType.CONTEXTUAL_PATTERN:
            return (
                "Create context-specific adaptation rules to automatically adjust "
                "investigation approaches based on organizational context"
            )
        
        return "Monitor pattern stability and gather additional evidence before implementation"
    
    def _identify_related_domains(self, cluster_data: List[Dict]) -> List[BusinessDomain]:
        """Identify business domains related to the pattern."""
        
        domains = []
        domain_mentions = []
        
        for item in cluster_data:
            if "domain" in item:
                domain_mentions.append(item["domain"])
        
        if domain_mentions:
            # Get most common domains
            domain_counts = Counter(domain_mentions)
            for domain_name, count in domain_counts.most_common(3):
                try:
                    domain = BusinessDomain(domain_name.lower())
                    domains.append(domain)
                except ValueError:
                    # Skip invalid domain names
                    continue
        
        return domains if domains else [BusinessDomain.OPERATIONS]
    
    def _determine_pattern_source(self, pattern_type: PatternType) -> PatternSource:
        """Determine the source of pattern data."""
        
        source_mapping = {
            PatternType.QUERY_PATTERN: PatternSource.USER_QUERIES,
            PatternType.METHODOLOGY_PATTERN: PatternSource.INVESTIGATION_RESULTS,
            PatternType.SUCCESS_PATTERN: PatternSource.SUCCESS_METRICS,
            PatternType.DOMAIN_PATTERN: PatternSource.DOMAIN_FEEDBACK,
            PatternType.TEMPORAL_PATTERN: PatternSource.INVESTIGATION_RESULTS,
            PatternType.CONTEXTUAL_PATTERN: PatternSource.CONTEXTUAL_ADAPTATION
        }
        
        return source_mapping.get(pattern_type, PatternSource.INVESTIGATION_RESULTS)
    
    def _validate_patterns(
        self, discovered_patterns: List[DiscoveredPattern], investigation_history: List[Dict]
    ) -> List[DiscoveredPattern]:
        """Validate discovered patterns against statistical thresholds."""
        
        validated_patterns = []
        
        for pattern in discovered_patterns:
            # Check minimum sample size
            if pattern.evidence.observation_count < self._statistical_thresholds["min_sample_size"]:
                continue
            
            # Check statistical significance
            if pattern.evidence.statistical_significance < self._statistical_thresholds["significance_level"]:
                continue
            
            # Check pattern stability (confidence interval width)
            ci_width = pattern.evidence.confidence_interval[1] - pattern.evidence.confidence_interval[0]
            if ci_width > 0.5:  # Too wide confidence interval
                continue
            
            # Check business value threshold
            if pattern.business_value < 0.4:  # Minimum business value
                continue
            
            validated_patterns.append(pattern)
        
        return validated_patterns
    
    def _calculate_library_updates(
        self, validated_patterns: List[DiscoveredPattern], investigation_history: List[Dict]
    ) -> PatternLibraryUpdate:
        """Calculate updates to pattern library."""
        
        # For now, all validated patterns are new
        new_patterns = validated_patterns
        
        # Calculate usage statistics
        usage_statistics = {}
        for pattern in validated_patterns:
            usage_statistics[pattern.id] = pattern.evidence.observation_count
        
        # Calculate effectiveness metrics
        effectiveness_metrics = {}
        for pattern in validated_patterns:
            effectiveness_metrics[pattern.id] = pattern.evidence.success_rate
        
        return PatternLibraryUpdate(
            new_patterns=new_patterns,
            updated_patterns=[],  # No existing patterns updated in this implementation
            deprecated_patterns=[],  # No patterns deprecated
            confidence_updates={},  # No confidence updates
            usage_statistics=usage_statistics,
            effectiveness_metrics=effectiveness_metrics
        )


# Standalone execution for testing
if __name__ == "__main__":
    recognizer = PatternRecognizer()
    
    # Test pattern recognition with mock data
    investigation_history = [
        {
            "query": "Why did production efficiency drop on Line 2?",
            "domain": "production",
            "methodology": "multi_phase_root_cause",
            "success": True,
            "duration_minutes": 25,
            "user_role": "manager",
            "complexity_level": "analytical",
            "timestamp": "2024-01-15T10:30:00Z"
        },
        {
            "query": "Analyze production line efficiency issues",
            "domain": "production", 
            "methodology": "systematic_analysis",
            "success": True,
            "duration_minutes": 18,
            "user_role": "engineer",
            "complexity_level": "analytical",
            "timestamp": "2024-01-16T14:20:00Z"
        },
        {
            "query": "Production efficiency analysis for manufacturing line",
            "domain": "production",
            "methodology": "multi_phase_root_cause", 
            "success": False,
            "duration_minutes": 35,
            "user_role": "analyst",
            "complexity_level": "investigative",
            "timestamp": "2024-01-17T09:15:00Z"
        }
        # Would include many more investigations in real usage
    ] * 5  # Simulate more data
    
    library_update = recognizer.analyze_investigation_patterns(investigation_history)
    
    print("Pattern Recognition Test")
    print("=" * 50)
    print(f"Discovered {len(library_update.new_patterns)} new patterns")
    
    for i, pattern in enumerate(library_update.new_patterns[:3]):
        print(f"\nPattern {i+1}:")
        print(f"  Type: {pattern.type.value}")
        print(f"  Description: {pattern.description}")
        print(f"  Business Value: {pattern.business_value:.3f}")
        print(f"  Evidence: {pattern.evidence.observation_count} observations, "
              f"{pattern.evidence.success_rate:.1%} success rate")
        print(f"  Recommended Action: {pattern.recommended_action[:100]}...")