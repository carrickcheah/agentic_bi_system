"""
Business Intelligence Pattern Library

Central pattern management system for Phase 1: Query Processing.
Stores, matches, and evolves 150+ manufacturing business intelligence patterns
with automatic success rate tracking and organizational learning.

Key Features:
- Pattern loading from JSON files organized by business domain
- Semantic similarity matching via Qdrant vector database
- Success rate tracking with Bayesian updates
- Business context awareness for role-specific recommendations
- Organizational learning and pattern evolution
"""

import json
import hashlib
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass
from enum import Enum

from ..utils.logging import logger
from ..fastmcp.qdrant_client import QdrantClient
from ..fastmcp.postgres_client import PostgresClient


class DataSource(Enum):
    """Pattern success rate data source classification."""
    BOOTSTRAP_ESTIMATE = "bootstrap_estimate"
    REAL_USAGE_DATA = "real_usage_data"
    STATISTICAL_CONFIDENCE = "statistical_confidence"


@dataclass
class PatternMatch:
    """Represents a pattern match with confidence scoring."""
    pattern_id: str
    pattern_data: Dict[str, Any]
    semantic_similarity: float
    business_context_score: float
    role_relevance_score: float
    success_rate_weight: float
    total_score: float
    confidence_interval: Tuple[float, float]
    match_explanation: str


@dataclass
class PatternStatistics:
    """Pattern usage and success statistics."""
    pattern_id: str
    success_rate: float
    confidence_interval: Tuple[float, float]
    data_source: DataSource
    sample_size: int
    successes: int
    failures: int
    last_updated: datetime
    usage_count: int
    average_investigation_time: Optional[float] = None


@dataclass
class InvestigationOutcome:
    """Investigation outcome for pattern success tracking."""
    pattern_id: str
    investigation_id: str
    user_id: str
    completion_success: bool
    user_satisfaction_score: float  # 0.0 - 1.0
    accuracy_validation: bool
    implementation_success: Optional[bool]
    investigation_time_minutes: float
    timestamp: datetime


class PatternLibrary:
    """
    Core pattern library for business intelligence patterns.
    
    Manages 150+ manufacturing patterns with semantic matching,
    success rate tracking, and organizational learning capabilities.
    """
    
    def __init__(self, qdrant_client: QdrantClient, postgres_client: PostgresClient):
        self.qdrant_client = qdrant_client
        self.postgres_client = postgres_client
        self.patterns: Dict[str, Dict[str, Any]] = {}
        self.pattern_statistics: Dict[str, PatternStatistics] = {}
        self.patterns_loaded = False
        
        # Pattern file organization by business domain
        self.pattern_files = {
            # Manufacturing domains
            "production": "production_operations.json",
            "quality": "quality_management.json",
            "supply_chain": "supply_chain_inventory.json",
            "cost_management": "cost_management.json",
            "asset_management": "asset_equipment.json",
            "safety": "safety_compliance.json",
            "customer": "customer_demand.json",
            "planning": "planning_scheduling.json",
            "hr": "hr_workforce.json",
            # Business domains
            "sales": "sales_revenue.json",
            "product": "product_management.json",
            "marketing": "marketing_campaigns.json",
            "operations": "operations_efficiency.json",
            "finance": "finance_budgeting.json"
        }
        
        # Business context scoring weights
        self.context_weights = {
            "role_match": 0.15,
            "domain_match": 0.25,
            "complexity_match": 0.10,
            "urgency_match": 0.05,
            "success_rate": 0.20,
            "semantic_similarity": 0.25
        }
        
        # Success metrics weighting for multi-dimensional scoring
        self.success_dimensions = {
            "investigation_completion": 0.3,
            "user_satisfaction": 0.25,
            "accuracy_validation": 0.25,
            "implementation_success": 0.2
        }
    
    async def initialize(self) -> None:
        """Initialize pattern library with data loading and indexing."""
        try:
            logger.info("🔧 Initializing Pattern Library...")
            
            # Load patterns from JSON files
            await self._load_all_patterns()
            
            # Load pattern statistics from database
            await self._load_pattern_statistics()
            
            # Initialize Qdrant collection if needed
            await self._initialize_qdrant_collection()
            
            # Index patterns in Qdrant for semantic search
            await self._index_patterns_in_qdrant()
            
            self.patterns_loaded = True
            logger.info(f"✅ Pattern Library initialized with {len(self.patterns)} patterns")
            
        except Exception as e:
            logger.error(f"❌ Pattern Library initialization failed: {e}")
            raise
    
    async def find_matching_patterns(
        self,
        business_question: str,
        user_context: Dict[str, Any],
        top_k: int = 5
    ) -> List[PatternMatch]:
        """
        Find matching patterns using multi-dimensional scoring.
        
        Args:
            business_question: Natural language business question
            user_context: User role, department, permissions, etc.
            top_k: Number of top patterns to return
            
        Returns:
            List of PatternMatch objects sorted by total score
        """
        if not self.patterns_loaded:
            await self.initialize()
        
        try:
            logger.info(f"🔍 Finding patterns for: {business_question[:100]}...")
            
            # 1. Semantic similarity search via Qdrant
            semantic_matches = await self._get_semantic_matches(business_question, top_k * 2)
            
            # 2. Business context scoring
            context_scored = self._apply_business_context_scoring(
                semantic_matches, user_context
            )
            
            # 3. Role relevance scoring
            role_scored = self._apply_role_relevance_scoring(
                context_scored, user_context
            )
            
            # 4. Success rate weighting
            success_weighted = self._apply_success_rate_weighting(role_scored)
            
            # 5. Calculate total scores and confidence
            final_matches = self._calculate_final_scores(success_weighted)
            
            # Sort by total score and return top_k
            sorted_matches = sorted(final_matches, key=lambda x: x.total_score, reverse=True)
            top_matches = sorted_matches[:top_k]
            
            logger.info(f"📊 Found {len(top_matches)} pattern matches")
            return top_matches
            
        except Exception as e:
            logger.error(f"❌ Pattern matching failed: {e}")
            return []
    
    async def track_investigation_outcome(
        self,
        outcome: InvestigationOutcome
    ) -> PatternStatistics:
        """
        Track investigation outcome and update pattern success rates.
        
        Args:
            outcome: Investigation outcome data
            
        Returns:
            Updated pattern statistics
        """
        try:
            logger.info(f"📈 Tracking outcome for pattern {outcome.pattern_id}")
            
            # Calculate multi-dimensional success score
            success_score = self._calculate_success_score(outcome)
            
            # Get current pattern statistics
            current_stats = self.pattern_statistics.get(outcome.pattern_id)
            if not current_stats:
                # Initialize new statistics for this pattern
                current_stats = PatternStatistics(
                    pattern_id=outcome.pattern_id,
                    success_rate=0.55,  # Default moderate bootstrap
                    confidence_interval=(0.45, 0.65),
                    data_source=DataSource.BOOTSTRAP_ESTIMATE,
                    sample_size=0,
                    successes=0,
                    failures=0,
                    last_updated=datetime.utcnow(),
                    usage_count=0
                )
            
            # Bayesian update of success rate
            updated_stats = self._bayesian_update_success_rate(current_stats, success_score)
            
            # Update data source classification based on sample size
            updated_stats.data_source = self._determine_data_source(updated_stats.sample_size)
            
            # Update usage tracking
            updated_stats.usage_count += 1
            if updated_stats.average_investigation_time:
                # Running average of investigation time
                n = updated_stats.usage_count
                updated_stats.average_investigation_time = (
                    (updated_stats.average_investigation_time * (n - 1) + outcome.investigation_time_minutes) / n
                )
            else:
                updated_stats.average_investigation_time = outcome.investigation_time_minutes
            
            # Store updated statistics
            self.pattern_statistics[outcome.pattern_id] = updated_stats
            
            # Persist to database
            await self._persist_pattern_statistics(updated_stats)
            
            # Store detailed outcome for learning
            await self._store_investigation_outcome(outcome)
            
            logger.info(f"✅ Updated pattern {outcome.pattern_id} success rate: {updated_stats.success_rate:.3f}")
            return updated_stats
            
        except Exception as e:
            logger.error(f"❌ Failed to track investigation outcome: {e}")
            raise
    
    async def get_pattern_statistics(self, pattern_id: str) -> Optional[PatternStatistics]:
        """Get statistics for a specific pattern."""
        if not self.patterns_loaded:
            await self.initialize()
        
        return self.pattern_statistics.get(pattern_id)
    
    async def get_domain_patterns(self, business_domain: str) -> List[Dict[str, Any]]:
        """Get all patterns for a specific business domain."""
        if not self.patterns_loaded:
            await self.initialize()
        
        return [
            pattern for pattern in self.patterns.values()
            if pattern["metadata"]["business_domain"] == business_domain
        ]
    
    async def _load_all_patterns(self) -> None:
        """Load all patterns from JSON files."""
        logger.info("📂 Loading patterns from JSON files...")
        
        pattern_dir = Path(__file__).parent.parent / "data" / "patterns"
        
        for domain, filename in self.pattern_files.items():
            file_path = pattern_dir / filename
            
            try:
                if file_path.exists():
                    with open(file_path, 'r', encoding='utf-8') as f:
                        domain_patterns = json.load(f)
                    
                    # Add patterns with generated IDs
                    for i, pattern in enumerate(domain_patterns, 1):
                        pattern_id = f"{domain}_{i:03d}"
                        pattern["pattern_id"] = pattern_id
                        self.patterns[pattern_id] = pattern
                    
                    logger.info(f"✅ Loaded {len(domain_patterns)} patterns from {filename}")
                else:
                    logger.warning(f"⚠️  Pattern file not found: {file_path}")
                    
            except Exception as e:
                logger.error(f"❌ Failed to load patterns from {filename}: {e}")
                raise
    
    async def _load_pattern_statistics(self) -> None:
        """Load pattern statistics from PostgreSQL database."""
        logger.info("📊 Loading pattern statistics from database...")
        
        try:
            # Query pattern statistics from database
            query = """
            SELECT pattern_id, success_rate, confidence_lower, confidence_upper,
                   data_source, sample_size, successes, failures, last_updated, usage_count,
                   average_investigation_time
            FROM pattern_statistics
            WHERE active = true
            """
            
            results = await self.postgres_client.execute_query(query)
            
            for row in results:
                stats = PatternStatistics(
                    pattern_id=row["pattern_id"],
                    success_rate=row["success_rate"],
                    confidence_interval=(row["confidence_lower"], row["confidence_upper"]),
                    data_source=DataSource(row["data_source"]),
                    sample_size=row["sample_size"],
                    successes=row["successes"],
                    failures=row["failures"],
                    last_updated=row["last_updated"],
                    usage_count=row["usage_count"],
                    average_investigation_time=row["average_investigation_time"]
                )
                self.pattern_statistics[row["pattern_id"]] = stats
            
            logger.info(f"✅ Loaded statistics for {len(self.pattern_statistics)} patterns")
            
        except Exception as e:
            logger.warning(f"⚠️  Could not load pattern statistics: {e}")
            # Continue with bootstrap estimates
    
    async def _initialize_qdrant_collection(self) -> None:
        """Initialize Qdrant collection for pattern embeddings."""
        collection_name = "business_investigation_patterns"
        
        try:
            # Check if collection exists
            collections = await self.qdrant_client.list_collections()
            
            if collection_name not in [c["name"] for c in collections]:
                logger.info(f"🔧 Creating Qdrant collection: {collection_name}")
                
                await self.qdrant_client.create_collection(
                    collection_name=collection_name,
                    vector_size=384,  # sentence-transformers/all-MiniLM-L6-v2 dimension
                    distance_metric="cosine"
                )
                
                logger.info(f"✅ Created Qdrant collection: {collection_name}")
            else:
                logger.info(f"✅ Qdrant collection already exists: {collection_name}")
                
        except Exception as e:
            logger.error(f"❌ Failed to initialize Qdrant collection: {e}")
            raise
    
    async def _index_patterns_in_qdrant(self) -> None:
        """Index all patterns in Qdrant for semantic search."""
        collection_name = "business_investigation_patterns"
        
        try:
            logger.info("🔍 Indexing patterns in Qdrant...")
            
            # Check if patterns are already indexed
            collection_info = await self.qdrant_client.get_collection_info(collection_name)
            if collection_info["points_count"] >= len(self.patterns):
                logger.info("✅ Patterns already indexed in Qdrant")
                return
            
            # Index patterns
            for pattern_id, pattern in self.patterns.items():
                try:
                    # Create text for embedding
                    embedding_text = self._create_embedding_text(pattern)
                    
                    # Store in Qdrant
                    await self.qdrant_client.store_embedding(
                        collection_name=collection_name,
                        information=embedding_text,
                        metadata={
                            "pattern_id": pattern_id,
                            "business_domain": pattern["metadata"]["business_domain"],
                            "complexity": pattern["metadata"]["complexity"],
                            "user_roles": pattern["metadata"]["user_roles"],
                            "timeframe": pattern["metadata"]["timeframe"]
                        }
                    )
                    
                except Exception as e:
                    logger.warning(f"⚠️  Failed to index pattern {pattern_id}: {e}")
                    continue
            
            logger.info(f"✅ Indexed {len(self.patterns)} patterns in Qdrant")
            
        except Exception as e:
            logger.error(f"❌ Failed to index patterns in Qdrant: {e}")
            raise
    
    def _create_embedding_text(self, pattern: Dict[str, Any]) -> str:
        """Create comprehensive text for pattern embedding."""
        metadata = pattern["metadata"]
        
        # Combine multiple text fields for rich embedding
        embedding_parts = [
            pattern["information"],
            " ".join(metadata["confidence_indicators"]),
            metadata["pattern"],
            " ".join(metadata["expected_deliverables"]),
            metadata["business_domain"],
            metadata["timeframe"],
            metadata["complexity"]
        ]
        
        return " | ".join(embedding_parts)
    
    async def _get_semantic_matches(
        self, 
        business_question: str, 
        top_k: int
    ) -> List[Tuple[Dict[str, Any], float]]:
        """Get semantic matches from Qdrant."""
        collection_name = "business_investigation_patterns"
        
        try:
            # Search for similar patterns
            results = await self.qdrant_client.search_similar(
                collection_name=collection_name,
                query_text=business_question,
                top_k=top_k,
                score_threshold=0.1  # Very permissive for initial filtering
            )
            
            # Convert to pattern data and similarity scores
            matches = []
            for result in results:
                pattern_id = result["metadata"]["pattern_id"]
                if pattern_id in self.patterns:
                    pattern_data = self.patterns[pattern_id]
                    similarity_score = result["score"]
                    matches.append((pattern_data, similarity_score))
            
            return matches
            
        except Exception as e:
            logger.error(f"❌ Semantic search failed: {e}")
            return []
    
    def _apply_business_context_scoring(
        self, 
        semantic_matches: List[Tuple[Dict[str, Any], float]],
        user_context: Dict[str, Any]
    ) -> List[Tuple[Dict[str, Any], float, float]]:
        """Apply business context scoring to semantic matches."""
        
        context_scored = []
        
        for pattern_data, semantic_score in semantic_matches:
            metadata = pattern_data["metadata"]
            
            # Business domain match
            user_domain = user_context.get("department", "").lower()
            pattern_domain = metadata["business_domain"]
            domain_score = 1.0 if user_domain in pattern_domain or pattern_domain in user_domain else 0.5
            
            # Complexity appropriateness
            user_experience = user_context.get("experience_level", "intermediate")
            pattern_complexity = metadata["complexity"]
            
            complexity_score = 1.0
            if user_experience == "beginner" and pattern_complexity == "complex":
                complexity_score = 0.3
            elif user_experience == "expert" and pattern_complexity == "simple":
                complexity_score = 0.7
            
            # Calculate weighted business context score
            business_score = (
                domain_score * 0.6 +
                complexity_score * 0.4
            )
            
            context_scored.append((pattern_data, semantic_score, business_score))
        
        return context_scored
    
    def _apply_role_relevance_scoring(
        self,
        context_scored: List[Tuple[Dict[str, Any], float, float]],
        user_context: Dict[str, Any]
    ) -> List[Tuple[Dict[str, Any], float, float, float]]:
        """Apply role-based relevance scoring."""
        
        user_role = user_context.get("role", "").lower()
        role_scored = []
        
        for pattern_data, semantic_score, business_score in context_scored:
            metadata = pattern_data["metadata"]
            pattern_roles = [role.lower() for role in metadata["user_roles"]]
            
            # Direct role match
            role_score = 1.0 if any(user_role in role or role in user_role for role in pattern_roles) else 0.6
            
            # Boost for manager roles on strategic patterns
            if "manager" in user_role and metadata["complexity"] in ["moderate", "complex"]:
                role_score *= 1.1
            
            role_scored.append((pattern_data, semantic_score, business_score, role_score))
        
        return role_scored
    
    def _apply_success_rate_weighting(
        self,
        role_scored: List[Tuple[Dict[str, Any], float, float, float]]
    ) -> List[Tuple[Dict[str, Any], float, float, float, float]]:
        """Apply success rate weighting based on pattern performance."""
        
        success_weighted = []
        
        for pattern_data, semantic_score, business_score, role_score in role_scored:
            pattern_id = pattern_data["pattern_id"]
            
            # Get pattern statistics
            stats = self.pattern_statistics.get(pattern_id)
            if stats:
                success_weight = stats.success_rate
                # Boost confidence for patterns with more data
                if stats.data_source == DataSource.STATISTICAL_CONFIDENCE:
                    success_weight *= 1.1
                elif stats.data_source == DataSource.BOOTSTRAP_ESTIMATE:
                    success_weight *= 0.9
            else:
                # Default bootstrap success rate based on complexity
                complexity = pattern_data["metadata"]["complexity"]
                success_weight = {"simple": 0.65, "moderate": 0.55, "complex": 0.45}.get(complexity, 0.55)
            
            success_weighted.append((pattern_data, semantic_score, business_score, role_score, success_weight))
        
        return success_weighted
    
    def _calculate_final_scores(
        self,
        success_weighted: List[Tuple[Dict[str, Any], float, float, float, float]]
    ) -> List[PatternMatch]:
        """Calculate final pattern match scores and create PatternMatch objects."""
        
        matches = []
        
        for pattern_data, semantic_score, business_score, role_score, success_weight in success_weighted:
            # Calculate weighted total score
            total_score = (
                semantic_score * self.context_weights["semantic_similarity"] +
                business_score * self.context_weights["domain_match"] +
                role_score * self.context_weights["role_match"] +
                success_weight * self.context_weights["success_rate"]
            )
            
            # Get confidence interval
            pattern_id = pattern_data["pattern_id"]
            stats = self.pattern_statistics.get(pattern_id)
            confidence_interval = stats.confidence_interval if stats else (0.45, 0.65)
            
            # Generate match explanation
            explanation = self._generate_match_explanation(
                pattern_data, semantic_score, business_score, role_score, success_weight
            )
            
            match = PatternMatch(
                pattern_id=pattern_id,
                pattern_data=pattern_data,
                semantic_similarity=semantic_score,
                business_context_score=business_score,
                role_relevance_score=role_score,
                success_rate_weight=success_weight,
                total_score=total_score,
                confidence_interval=confidence_interval,
                match_explanation=explanation
            )
            
            matches.append(match)
        
        return matches
    
    def _generate_match_explanation(
        self,
        pattern_data: Dict[str, Any],
        semantic_score: float,
        business_score: float,
        role_score: float,
        success_weight: float
    ) -> str:
        """Generate human-readable explanation for pattern match."""
        
        metadata = pattern_data["metadata"]
        
        explanations = []
        
        # Semantic similarity
        if semantic_score > 0.8:
            explanations.append("high semantic similarity")
        elif semantic_score > 0.6:
            explanations.append("good semantic match")
        else:
            explanations.append("moderate semantic relevance")
        
        # Business context
        if business_score > 0.8:
            explanations.append(f"strong {metadata['business_domain']} domain alignment")
        
        # Role relevance
        if role_score > 0.9:
            explanations.append("excellent role match")
        elif role_score > 0.7:
            explanations.append("good role alignment")
        
        # Success rate
        if success_weight > 0.7:
            explanations.append("high success rate")
        elif success_weight > 0.5:
            explanations.append("proven effectiveness")
        
        return f"Match based on {', '.join(explanations)}."
    
    def _calculate_success_score(self, outcome: InvestigationOutcome) -> float:
        """Calculate multi-dimensional success score from investigation outcome."""
        
        # Normalize user satisfaction score (0-1 range)
        satisfaction_score = max(0.0, min(1.0, outcome.user_satisfaction_score))
        
        # Convert booleans to scores
        completion_score = 1.0 if outcome.completion_success else 0.0
        accuracy_score = 1.0 if outcome.accuracy_validation else 0.0
        implementation_score = 1.0 if outcome.implementation_success else 0.0 if outcome.implementation_success is False else 0.5
        
        # Calculate weighted success score
        weighted_score = (
            completion_score * self.success_dimensions["investigation_completion"] +
            satisfaction_score * self.success_dimensions["user_satisfaction"] +
            accuracy_score * self.success_dimensions["accuracy_validation"] +
            implementation_score * self.success_dimensions["implementation_success"]
        )
        
        return weighted_score
    
    def _bayesian_update_success_rate(
        self,
        current_stats: PatternStatistics,
        success_score: float
    ) -> PatternStatistics:
        """Update success rate using Bayesian approach."""
        
        # Treat success_score as probability of success
        is_success = success_score > 0.5
        
        # Update counts
        new_successes = current_stats.successes + (1 if is_success else 0)
        new_failures = current_stats.failures + (0 if is_success else 1)
        new_sample_size = current_stats.sample_size + 1
        
        # Simple Bayesian update (Beta-Binomial)
        # Use Beta(1,1) as uninformative prior
        alpha = new_successes + 1
        beta = new_failures + 1
        
        new_success_rate = alpha / (alpha + beta)
        
        # Calculate confidence interval using normal approximation
        variance = (alpha * beta) / ((alpha + beta) ** 2 * (alpha + beta + 1))
        std_error = variance ** 0.5
        
        # 95% confidence interval
        confidence_lower = max(0.0, new_success_rate - 1.96 * std_error)
        confidence_upper = min(1.0, new_success_rate + 1.96 * std_error)
        
        return PatternStatistics(
            pattern_id=current_stats.pattern_id,
            success_rate=new_success_rate,
            confidence_interval=(confidence_lower, confidence_upper),
            data_source=current_stats.data_source,
            sample_size=new_sample_size,
            successes=new_successes,
            failures=new_failures,
            last_updated=datetime.utcnow(),
            usage_count=current_stats.usage_count,
            average_investigation_time=current_stats.average_investigation_time
        )
    
    def _determine_data_source(self, sample_size: int) -> DataSource:
        """Determine data source classification based on sample size."""
        if sample_size == 0:
            return DataSource.BOOTSTRAP_ESTIMATE
        elif sample_size < 15:
            return DataSource.REAL_USAGE_DATA
        else:
            return DataSource.STATISTICAL_CONFIDENCE
    
    async def _persist_pattern_statistics(self, stats: PatternStatistics) -> None:
        """Persist pattern statistics to PostgreSQL database."""
        try:
            query = """
            INSERT INTO pattern_statistics (
                pattern_id, success_rate, confidence_lower, confidence_upper,
                data_source, sample_size, successes, failures, last_updated,
                usage_count, average_investigation_time, active
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, true)
            ON CONFLICT (pattern_id) 
            DO UPDATE SET
                success_rate = EXCLUDED.success_rate,
                confidence_lower = EXCLUDED.confidence_lower,
                confidence_upper = EXCLUDED.confidence_upper,
                data_source = EXCLUDED.data_source,
                sample_size = EXCLUDED.sample_size,
                successes = EXCLUDED.successes,
                failures = EXCLUDED.failures,
                last_updated = EXCLUDED.last_updated,
                usage_count = EXCLUDED.usage_count,
                average_investigation_time = EXCLUDED.average_investigation_time
            """
            
            await self.postgres_client.execute_query(query, [
                stats.pattern_id,
                stats.success_rate,
                stats.confidence_interval[0],
                stats.confidence_interval[1],
                stats.data_source.value,
                stats.sample_size,
                stats.successes,
                stats.failures,
                stats.last_updated,
                stats.usage_count,
                stats.average_investigation_time
            ])
            
        except Exception as e:
            logger.warning(f"⚠️  Failed to persist pattern statistics: {e}")
    
    async def _store_investigation_outcome(self, outcome: InvestigationOutcome) -> None:
        """Store detailed investigation outcome for learning."""
        try:
            query = """
            INSERT INTO investigation_outcomes (
                pattern_id, investigation_id, user_id, completion_success,
                user_satisfaction_score, accuracy_validation, implementation_success,
                investigation_time_minutes, timestamp
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
            """
            
            await self.postgres_client.execute_query(query, [
                outcome.pattern_id,
                outcome.investigation_id,
                outcome.user_id,
                outcome.completion_success,
                outcome.user_satisfaction_score,
                outcome.accuracy_validation,
                outcome.implementation_success,
                outcome.investigation_time_minutes,
                outcome.timestamp
            ])
            
        except Exception as e:
            logger.warning(f"⚠️  Failed to store investigation outcome: {e}")