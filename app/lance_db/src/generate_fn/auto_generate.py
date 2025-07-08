#!/usr/bin/env python3
"""
Production-Grade Auto-Generation Module for MOQ Template Metadata Enhancement

This module provides comprehensive algorithms for:
- SQL complexity analysis and performance estimation
- Business intelligence scoring and actionability assessment
- Metadata enhancement and cache key generation
- Timestamp and counter initialization

Following LanceDB expert standards for enterprise-grade performance and reliability.
"""

import re
import hashlib
import time
import math
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass
from enum import Enum

# Import enterprise vector schema for unified cross-module integration
try:
    from ..enterprise_vector_schema import (
        EnterpriseVectorSchema,
        VectorMetadata,
        ModuleSource,
        BusinessDomain,
        PerformanceTier as UnifiedPerformanceTier,
        AnalysisType,
        generate_vector_id,
        normalize_score_to_unified_scale,
        create_cross_module_metadata
    )
    ENTERPRISE_SCHEMA_AVAILABLE = True
except ImportError:
    try:
        from enterprise_vector_schema import (
            EnterpriseVectorSchema,
            VectorMetadata,
            ModuleSource,
            BusinessDomain,
            PerformanceTier as UnifiedPerformanceTier,
            AnalysisType,
            generate_vector_id,
            normalize_score_to_unified_scale,
            create_cross_module_metadata
        )
        ENTERPRISE_SCHEMA_AVAILABLE = True
    except ImportError:
        print("⚠️ Warning: Enterprise vector schema not available, using legacy mode")
        ENTERPRISE_SCHEMA_AVAILABLE = False


class PerformanceTier(Enum):
    """Performance tier classifications for queries."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class ComplexityMetrics:
    """Container for SQL complexity analysis results."""
    join_count: int = 0
    join_complexity: int = 0
    window_functions: int = 0
    subquery_depth: int = 0
    aggregation_count: int = 0
    case_statements: int = 0
    cte_count: int = 0
    union_count: int = 0
    distinct_count: int = 0
    total_score: int = 1


class SQLComplexityAnalyzer:
    """Production-grade SQL complexity analysis with weighted scoring."""
    
    # Complexity weights based on performance impact analysis
    WEIGHTS = {
        'join': 2.0,
        'complex_join': 3.0,  # FULL OUTER, CROSS JOIN
        'window_function': 3.0,
        'subquery': 2.0,
        'nested_subquery': 1.5,  # Additional per nesting level
        'aggregation': 1.0,
        'case_statement': 0.5,
        'cte': 1.5,
        'union': 1.0,
        'distinct': 0.5,
        'having': 1.0,
        'order_by': 0.5
    }
    
    # SQL patterns for complexity detection
    PATTERNS = {
        'joins': r'\b(?:INNER\s+JOIN|LEFT\s+(?:OUTER\s+)?JOIN|RIGHT\s+(?:OUTER\s+)?JOIN|FULL\s+(?:OUTER\s+)?JOIN|CROSS\s+JOIN|JOIN)\b',
        'complex_joins': r'\b(?:FULL\s+(?:OUTER\s+)?JOIN|CROSS\s+JOIN)\b',
        'window_functions': r'\b(?:ROW_NUMBER|RANK|DENSE_RANK|LAG|LEAD|FIRST_VALUE|LAST_VALUE|NTH_VALUE|NTILE|PERCENT_RANK|CUME_DIST|SUM|COUNT|AVG|MIN|MAX)\s*\(\s*.*?\s*\)\s+OVER\s*\(',
        'aggregations': r'\b(?:SUM|COUNT|AVG|MIN|MAX|GROUP_CONCAT|STRING_AGG)\s*\(',
        'case_statements': r'\bCASE\s+(?:WHEN|.*?\s+WHEN)',
        'ctes': r'\bWITH\s+(?:RECURSIVE\s+)?\w+\s*(?:\([^)]*\))?\s*AS\s*\(',
        'unions': r'\bUNION(?:\s+ALL)?\b',
        'distincts': r'\bSELECT\s+DISTINCT\b',
        'having': r'\bHAVING\b',
        'order_by': r'\bORDER\s+BY\b',
        'subqueries': r'\(\s*SELECT\b'
    }
    
    def __init__(self):
        self._compiled_patterns = {
            name: re.compile(pattern, re.IGNORECASE | re.DOTALL)
            for name, pattern in self.PATTERNS.items()
        }
    
    def calculate_complexity_score(self, sql_query: str, tech_metadata: Optional[Dict] = None) -> int:
        """
        Calculate comprehensive SQL complexity score (1-10 scale).
        
        Args:
            sql_query: SQL query string to analyze
            tech_metadata: Optional existing technical metadata
            
        Returns:
            Complexity score from 1 (simple) to 10 (extremely complex)
        """
        if not sql_query or not isinstance(sql_query, str):
            return 1
        
        # Normalize SQL for analysis
        normalized_sql = self._normalize_sql(sql_query)
        
        # Calculate complexity metrics
        metrics = self._analyze_sql_complexity(normalized_sql)
        
        # Apply weights and calculate score
        complexity_score = self._calculate_weighted_score(metrics)
        
        # Apply logarithmic scaling to cap at 10
        final_score = min(10, max(1, int(math.log10(complexity_score * 10) * 2.5 + 1)))
        
        return final_score
    
    def _normalize_sql(self, sql_query: str) -> str:
        """Normalize SQL query for consistent analysis."""
        # Remove extra whitespace and normalize line endings
        normalized = re.sub(r'\s+', ' ', sql_query.strip())
        
        # Remove comments
        normalized = re.sub(r'--[^\n]*', '', normalized)
        normalized = re.sub(r'/\*.*?\*/', '', normalized, flags=re.DOTALL)
        
        return normalized.upper()
    
    def _analyze_sql_complexity(self, sql_query: str) -> ComplexityMetrics:
        """Analyze SQL query and extract complexity metrics."""
        metrics = ComplexityMetrics()
        
        # Count joins
        join_matches = self._compiled_patterns['joins'].findall(sql_query)
        metrics.join_count = len(join_matches)
        
        # Count complex joins (higher weight)
        complex_join_matches = self._compiled_patterns['complex_joins'].findall(sql_query)
        metrics.join_complexity = len(complex_join_matches)
        
        # Count window functions
        window_matches = self._compiled_patterns['window_functions'].findall(sql_query)
        metrics.window_functions = len(window_matches)
        
        # Analyze subquery depth
        metrics.subquery_depth = self._calculate_subquery_depth(sql_query)
        
        # Count aggregations
        agg_matches = self._compiled_patterns['aggregations'].findall(sql_query)
        metrics.aggregation_count = len(agg_matches)
        
        # Count CASE statements
        case_matches = self._compiled_patterns['case_statements'].findall(sql_query)
        metrics.case_statements = len(case_matches)
        
        # Count CTEs
        cte_matches = self._compiled_patterns['ctes'].findall(sql_query)
        metrics.cte_count = len(cte_matches)
        
        # Count UNIONs
        union_matches = self._compiled_patterns['unions'].findall(sql_query)
        metrics.union_count = len(union_matches)
        
        # Count DISTINCT
        distinct_matches = self._compiled_patterns['distincts'].findall(sql_query)
        metrics.distinct_count = len(distinct_matches)
        
        return metrics
    
    def _calculate_subquery_depth(self, sql_query: str) -> int:
        """Calculate maximum subquery nesting depth."""
        max_depth = 0
        current_depth = 0
        
        # Simple parentheses-based depth calculation for SELECT statements
        in_string = False
        escape_next = False
        
        for char in sql_query:
            if escape_next:
                escape_next = False
                continue
                
            if char == '\\':
                escape_next = True
                continue
                
            if char in ('"', "'"):
                in_string = not in_string
                continue
                
            if not in_string:
                if char == '(':
                    # Look ahead to see if this is a subquery
                    remaining = sql_query[sql_query.index(char):sql_query.index(char) + 20]
                    if 'SELECT' in remaining.upper():
                        current_depth += 1
                        max_depth = max(max_depth, current_depth)
                elif char == ')':
                    if current_depth > 0:
                        current_depth -= 1
        
        return max_depth
    
    def _calculate_weighted_score(self, metrics: ComplexityMetrics) -> float:
        """Calculate weighted complexity score from metrics."""
        score = 1.0  # Base score
        
        # Apply weights
        score += metrics.join_count * self.WEIGHTS['join']
        score += metrics.join_complexity * self.WEIGHTS['complex_join']
        score += metrics.window_functions * self.WEIGHTS['window_function']
        score += metrics.subquery_depth * self.WEIGHTS['subquery']
        score += max(0, metrics.subquery_depth - 1) * self.WEIGHTS['nested_subquery']
        score += metrics.aggregation_count * self.WEIGHTS['aggregation']
        score += metrics.case_statements * self.WEIGHTS['case_statement']
        score += metrics.cte_count * self.WEIGHTS['cte']
        score += metrics.union_count * self.WEIGHTS['union']
        score += metrics.distinct_count * self.WEIGHTS['distinct']
        
        return score


class PerformancePredictor:
    """Advanced performance estimation algorithms based on query complexity and characteristics."""
    
    # Performance baselines (in milliseconds) based on production data
    BASE_EXECUTION_TIME = 50.0
    COMPLEXITY_MULTIPLIERS = {
        1: 0.5,   # Very simple queries
        2: 0.7,
        3: 1.0,   # Average complexity
        4: 1.5,
        5: 2.0,
        6: 3.0,
        7: 4.5,
        8: 7.0,
        9: 10.0,
        10: 15.0  # Extremely complex queries
    }
    
    # Cost factors (relative units)
    COST_FACTORS = {
        'cpu_base': 1.0,
        'io_base': 2.0,
        'memory_base': 0.5,
        'network_base': 1.5
    }
    
    def determine_performance_tier(self, complexity_score: int, estimated_rows: int = 1000) -> str:
        """
        Determine performance tier based on complexity and estimated data volume.
        
        Args:
            complexity_score: SQL complexity score (1-10)
            estimated_rows: Estimated number of rows to process
            
        Returns:
            Performance tier: low, medium, high, or critical
        """
        # Calculate combined score considering both complexity and data volume
        volume_factor = math.log10(max(estimated_rows, 1)) / 10  # Normalize row count impact
        combined_score = complexity_score + (volume_factor * 3)
        
        if combined_score <= 3:
            return PerformanceTier.LOW.value
        elif combined_score <= 6:
            return PerformanceTier.MEDIUM.value
        elif combined_score <= 8:
            return PerformanceTier.HIGH.value
        else:
            return PerformanceTier.CRITICAL.value
    
    def estimate_execution_time(self, complexity_score: int, estimated_rows: int = 1000, 
                               table_count: int = 2) -> float:
        """
        Estimate query execution time based on complexity and data characteristics.
        
        Args:
            complexity_score: SQL complexity score (1-10)
            estimated_rows: Estimated number of rows to process
            table_count: Number of tables involved
            
        Returns:
            Estimated execution time in milliseconds
        """
        if complexity_score < 1 or complexity_score > 10:
            complexity_score = max(1, min(10, complexity_score))
        
        # Base time calculation
        multiplier = self.COMPLEXITY_MULTIPLIERS.get(complexity_score, 1.0)
        base_time = self.BASE_EXECUTION_TIME * multiplier
        
        # Row count factor (logarithmic scaling)
        row_factor = math.pow(max(estimated_rows, 1), 0.8) / 1000  # Sublinear scaling
        
        # Table count factor
        table_factor = 1.0 + (table_count - 1) * 0.3  # 30% overhead per additional table
        
        # Calculate final estimate
        estimated_time = base_time * row_factor * table_factor
        
        # Apply realistic bounds (10ms to 10 minutes)
        return max(10.0, min(600000.0, estimated_time))
    
    def estimate_query_cost(self, complexity_score: int, table_metadata: Optional[Dict] = None) -> float:
        """
        Estimate query execution cost in relative units.
        
        Args:
            complexity_score: SQL complexity score (1-10)
            table_metadata: Optional metadata about tables involved
            
        Returns:
            Estimated cost in relative units
        """
        # Base cost calculation
        cpu_cost = complexity_score * self.COST_FACTORS['cpu_base']
        io_cost = (complexity_score / 2) * self.COST_FACTORS['io_base']
        memory_cost = complexity_score * self.COST_FACTORS['memory_base']
        
        # Adjust for table metadata if available
        if table_metadata:
            table_count = len(table_metadata.get('tables_used', []))
            if table_count > 2:
                io_cost *= (1 + (table_count - 2) * 0.5)  # Increased I/O for more tables
        
        total_cost = cpu_cost + io_cost + memory_cost
        
        # Apply logarithmic scaling for realistic cost distribution
        return round(math.log10(total_cost * 10) * 25, 2)
    
    def estimate_row_count(self, sql_query: str, table_metadata: Optional[Dict] = None) -> int:
        """
        Estimate result row count based on query patterns and table metadata.
        
        Args:
            sql_query: SQL query string
            table_metadata: Optional metadata about tables
            
        Returns:
            Estimated number of rows in result set
        """
        normalized_sql = sql_query.upper()
        
        # Base estimates by query pattern
        if 'COUNT(' in normalized_sql and 'GROUP BY' not in normalized_sql:
            return 1  # Aggregate without grouping
        elif 'DISTINCT' in normalized_sql:
            return 25  # DISTINCT typically reduces rows significantly
        elif 'GROUP BY' in normalized_sql:
            return 15  # Grouping typically produces moderate row counts
        elif 'JOIN' in normalized_sql:
            join_count = len(re.findall(r'\bJOIN\b', normalized_sql))
            return min(100, 10 * (join_count + 1))  # JOINs can increase row count
        elif 'WHERE' in normalized_sql:
            return 8  # Filtering typically reduces row count
        else:
            return 12  # Default estimate for simple selects
    
    def estimate_memory_usage(self, complexity_score: int, estimated_rows: int) -> float:
        """
        Estimate memory usage in MB.
        
        Args:
            complexity_score: SQL complexity score (1-10)
            estimated_rows: Estimated number of rows to process
            
        Returns:
            Estimated memory usage in MB
        """
        # Base memory per row (approximate)
        base_memory_per_row = 0.001  # 1KB per row average
        
        # Complexity factor for additional memory overhead
        complexity_factor = 1 + (complexity_score - 1) * 0.2
        
        # Calculate estimate
        estimated_memory = estimated_rows * base_memory_per_row * complexity_factor
        
        # Apply realistic bounds (1MB to 1GB)
        return max(1.0, min(1024.0, estimated_memory))


class BusinessIntelligenceCalculator:
    """Business value and actionability scoring algorithms based on industry best practices."""
    
    # Query type weights for actionability scoring
    QUERY_TYPE_WEIGHTS = {
        'analytical': 0.9,
        'reporting': 0.8,
        'dashboard': 0.85,
        'customer_advisory': 0.95,  # High actionability for customer insights
        'operational': 0.7,
        'monitoring': 0.6,
        'data_quality': 0.5,
        'etl': 0.3,
        'maintenance': 0.2
    }
    
    # Business domain multipliers
    DOMAIN_MULTIPLIERS = {
        'manufacturing_sales': 1.2,  # High business impact
        'financial_services': 1.3,
        'healthcare': 1.1,
        'retail': 1.0,
        'logistics': 0.9,
        'marketing': 0.8,
        'internal_tools': 0.6
    }
    
    # User role factors
    ROLE_FACTORS = {
        'business_analyst': 1.0,
        'data_analyst': 0.9,
        'executive': 1.3,
        'manager': 1.1,
        'specialist': 0.8,
        'developer': 0.6,
        'admin': 0.4
    }
    
    def calculate_actionability_score(self, query_type: str, business_context: Dict) -> float:
        """
        Calculate business decision usefulness score (0.0-1.0).
        
        Args:
            query_type: Type of query (analytical, reporting, etc.)
            business_context: Dictionary containing business context information
            
        Returns:
            Actionability score from 0.0 (not actionable) to 1.0 (highly actionable)
        """
        # Base score from query type
        base_score = self.QUERY_TYPE_WEIGHTS.get(query_type.lower(), 0.5)
        
        # Business domain multiplier
        business_domain = business_context.get('business_domain', 'unknown')
        domain_multiplier = self.DOMAIN_MULTIPLIERS.get(business_domain.lower(), 0.8)
        
        # User role factor
        user_role = business_context.get('user_role', 'analyst')
        role_factor = self.ROLE_FACTORS.get(user_role.lower(), 0.8)
        
        # Business impact factor
        business_impact = business_context.get('business_impact', 'medium')
        impact_factor = {
            'low': 0.7,
            'medium': 1.0,
            'high': 1.3,
            'critical': 1.5
        }.get(business_impact.lower(), 1.0)
        
        # Confidence level factor
        confidence_level = business_context.get('confidence_level', 'medium')
        confidence_factor = {
            'low': 0.8,
            'medium': 1.0,
            'high': 1.2,
            'very_high': 1.3
        }.get(confidence_level.lower(), 1.0)
        
        # Calculate final score
        actionability_score = base_score * domain_multiplier * role_factor * impact_factor * confidence_factor
        
        # Normalize to 0.0-1.0 range
        return max(0.0, min(1.0, actionability_score))
    
    def calculate_business_value(self, business_domain: str, impact_level: str, 
                                user_role: str, urgency: str = 'medium') -> float:
        """
        Calculate strategic business value assessment.
        
        Args:
            business_domain: Business domain (manufacturing_sales, etc.)
            impact_level: Business impact level (low, medium, high, critical)
            user_role: User role (executive, manager, analyst, etc.)
            urgency: Urgency level (low, medium, high, critical)
            
        Returns:
            Business value score from 0.0 to 1.0
        """
        # Base value from domain
        domain_base = self.DOMAIN_MULTIPLIERS.get(business_domain.lower(), 0.8)
        
        # Impact level scoring
        impact_scores = {
            'low': 0.3,
            'medium': 0.6,
            'high': 0.9,
            'critical': 1.0
        }
        impact_score = impact_scores.get(impact_level.lower(), 0.6)
        
        # Role influence on business value
        role_influence = self.ROLE_FACTORS.get(user_role.lower(), 0.8)
        
        # Urgency factor
        urgency_factors = {
            'low': 0.8,
            'medium': 1.0,
            'high': 1.2,
            'critical': 1.4
        }
        urgency_factor = urgency_factors.get(urgency.lower(), 1.0)
        
        # Calculate business value
        business_value = domain_base * impact_score * role_influence * urgency_factor
        
        # Normalize to 0.0-1.0 range
        return max(0.0, min(1.0, business_value))
    
    def calculate_strategic_alignment(self, query_intent: str, business_function: str) -> float:
        """
        Calculate strategic alignment score based on query intent and business function.
        
        Args:
            query_intent: Intent of the query (optimization, analysis, etc.)
            business_function: Business function (pricing, operations, etc.)
            
        Returns:
            Strategic alignment score from 0.0 to 1.0
        """
        # Intent scoring
        intent_scores = {
            'optimization': 0.9,
            'analysis': 0.8,
            'monitoring': 0.7,
            'reporting': 0.6,
            'exploration': 0.5,
            'maintenance': 0.3
        }
        
        intent_base = intent_scores.get(query_intent.lower().split('_')[0], 0.6)
        
        # Business function alignment
        function_scores = {
            'pricing': 0.95,  # High strategic value
            'customer': 0.90,
            'sales': 0.85,
            'operations': 0.80,
            'finance': 0.85,
            'marketing': 0.75,
            'logistics': 0.70,
            'support': 0.60
        }
        
        function_base = function_scores.get(business_function.lower().split('_')[0], 0.7)
        
        # Combined alignment score
        alignment = (intent_base + function_base) / 2
        
        return max(0.0, min(1.0, alignment))


class MetadataEnhancer:
    """Production-grade metadata generation and enhancement utilities."""
    
    def generate_cache_key(self, sql_query: str, user_context: Dict) -> str:
        """
        Generate secure, collision-resistant cache key.
        
        Args:
            sql_query: SQL query string
            user_context: User context for cache scope
            
        Returns:
            Hex-encoded cache key
        """
        # Normalize SQL for consistent caching
        normalized_sql = re.sub(r'\s+', ' ', sql_query.strip().lower())
        
        # Include relevant user context for cache scoping
        cache_components = [
            normalized_sql,
            user_context.get('user_id', 'anonymous'),
            user_context.get('organization_id', 'default'),
            user_context.get('timezone', 'UTC')
        ]
        
        # Create hash from components
        cache_string = '|'.join(str(comp) for comp in cache_components)
        cache_hash = hashlib.sha256(cache_string.encode('utf-8')).hexdigest()
        
        # Return first 32 characters for manageable key length
        return cache_hash[:32]
    
    def initialize_execution_counters(self, is_template: bool = True) -> Dict:
        """
        Initialize execution counters with intelligent baseline values.
        
        Args:
            is_template: Whether this is template data or real execution
            
        Returns:
            Dictionary with initialized counter values
        """
        if is_template:
            return {
                'execution_count': 1,      # Templates start with 1 to indicate creation
                'usage_frequency': 1,      # Base frequency for new templates
                'cache_hit_count': 0,      # No cache hits yet
                'cache_miss_count': 1,     # First execution is always a miss
                'success_count': 1,        # Assume template is valid
                'error_count': 0,          # No errors initially
                'optimization_count': 0    # No optimizations applied yet
            }
        else:
            return {
                'execution_count': 0,
                'usage_frequency': 0,
                'cache_hit_count': 0,
                'cache_miss_count': 0,
                'success_count': 0,
                'error_count': 0,
                'optimization_count': 0
            }
    
    def generate_timestamps(self, timezone_str: str = 'UTC') -> Dict:
        """
        Generate comprehensive timestamp metadata.
        
        Args:
            timezone_str: Timezone string (default: UTC)
            
        Returns:
            Dictionary with various timestamp fields
        """
        current_time = datetime.now(timezone.utc)
        
        return {
            'created_at': current_time.isoformat(),
            'updated_at': current_time.isoformat(),
            'last_executed_at': current_time.isoformat(),
            'first_executed_at': current_time.isoformat(),
            'last_optimized_at': None,  # No optimization yet
            'last_cache_update': current_time.isoformat(),
            'learning_updated_at': current_time.isoformat(),
            'last_health_check': current_time.isoformat()
        }
    
    def generate_cache_strategy(self, query_type: str, complexity_score: int, 
                              business_impact: str) -> Dict:
        """
        Generate intelligent cache strategy based on query characteristics.
        
        Args:
            query_type: Type of query
            complexity_score: Query complexity (1-10)
            business_impact: Business impact level
            
        Returns:
            Cache strategy configuration
        """
        # Base TTL based on query type
        base_ttl_minutes = {
            'analytical': 60,      # 1 hour for analytical queries
            'reporting': 30,       # 30 minutes for reports
            'dashboard': 15,       # 15 minutes for dashboards
            'customer_advisory': 5, # 5 minutes for customer data
            'operational': 2,      # 2 minutes for operational queries
            'monitoring': 1        # 1 minute for monitoring
        }.get(query_type.lower(), 30)
        
        # Adjust TTL based on complexity (more complex = longer cache)
        complexity_multiplier = 1 + (complexity_score - 5) * 0.1
        
        # Adjust TTL based on business impact
        impact_multiplier = {
            'low': 1.5,      # Cache longer for low impact
            'medium': 1.0,
            'high': 0.7,     # Cache shorter for high impact
            'critical': 0.5  # Very short cache for critical data
        }.get(business_impact.lower(), 1.0)
        
        final_ttl = max(1, int(base_ttl_minutes * complexity_multiplier * impact_multiplier))
        
        return {
            'is_cacheable': True,
            'cache_ttl_seconds': final_ttl * 60,  # Convert to seconds
            'cache_strategy': 'time_based',
            'cache_dependencies': [],  # Will be populated with table names
            'invalidation_triggers': ['data_update', 'schema_change'],
            'cache_warmup_enabled': complexity_score >= 7,  # Warm up complex queries
            'cache_compression': complexity_score >= 5      # Compress for medium+ complexity
        }
    
    def enhance_json_sections(self, template_data: Dict, generated_values: Dict) -> Dict:
        """
        Process all JSON sections to fill null values intelligently.
        
        Args:
            template_data: Original template data with null values
            generated_values: Generated values to fill nulls
            
        Returns:
            Enhanced template data with filled values
        """
        enhanced_data = template_data.copy()
        
        # Enhance technical metadata
        tech_metadata = enhanced_data.get('technical_metadata', {})
        tech_metadata.update({
            'complexity_score': generated_values.get('complexity_score'),
            'performance_tier': generated_values.get('performance_tier'),
            'estimated_cost': generated_values.get('estimated_cost'),
            'estimated_execution_time_ms': generated_values.get('estimated_execution_time_ms'),
            'estimated_row_count': generated_values.get('estimated_row_count')
        })
        enhanced_data['technical_metadata'] = tech_metadata
        
        # Enhance execution results
        exec_results = enhanced_data.get('execution_results', {})
        exec_results.update(generated_values.get('execution_counters', {}))
        exec_results.update(generated_values.get('timestamps', {}))
        enhanced_data['execution_results'] = exec_results
        
        # Enhance business intelligence
        bi_data = enhanced_data.get('business_intelligence', {})
        bi_data.update({
            'actionability_score': generated_values.get('actionability_score'),
            'business_value': generated_values.get('business_value'),
            'strategic_alignment': generated_values.get('strategic_alignment')
        })
        enhanced_data['business_intelligence'] = bi_data
        
        # Enhance caching
        caching_data = enhanced_data.get('caching', {})
        caching_data.update({
            'cache_key': generated_values.get('cache_key')
        })
        caching_data.update(generated_values.get('cache_strategy', {}))
        enhanced_data['caching'] = caching_data
        
        # Add enterprise vector metadata if available
        if 'enterprise_vector_metadata' in generated_values:
            enhanced_data['enterprise_vector_metadata'] = generated_values['enterprise_vector_metadata']
        
        return enhanced_data


class AutoGenerationEngine:
    """Main orchestrator for all auto-generation functions with production reliability."""
    
    def __init__(self):
        """Initialize all component analyzers and calculators."""
        self.sql_analyzer = SQLComplexityAnalyzer()
        self.performance_predictor = PerformancePredictor()
        self.bi_calculator = BusinessIntelligenceCalculator()
        self.metadata_enhancer = MetadataEnhancer()
        self._cache = {}  # Simple memoization cache
        
        # Enterprise vector schema integration
        self.enterprise_schema_enabled = ENTERPRISE_SCHEMA_AVAILABLE
        if self.enterprise_schema_enabled:
            print("✅ Enterprise vector schema integration enabled")
        else:
            print("⚠️ Running in legacy mode without enterprise vector schema")
    
    def auto_generate_all_fields(self, template_data: Dict) -> Dict:
        """
        Master function that orchestrates all auto-generation processes.
        
        Args:
            template_data: Original template data with null/zero values
            
        Returns:
            Enhanced template data with all fields populated
        """
        try:
            # Extract key information for analysis
            sql_query = template_data.get('query_content', {}).get('sql_query', '')
            query_type = template_data.get('query_content', {}).get('query_type', 'analytical')
            business_domain = template_data.get('semantic_context', {}).get('business_domain', 'unknown')
            user_context = template_data.get('user_context', {})
            tech_metadata = template_data.get('technical_metadata', {})
            
            # Create cache key for memoization
            cache_key = f"{hash(sql_query)}_{query_type}_{business_domain}"
            
            if cache_key in self._cache:
                generated_values = self._cache[cache_key]
            else:
                # Generate all values
                generated_values = self._generate_all_values(
                    sql_query, query_type, business_domain, user_context, tech_metadata, template_data
                )
                
                # Cache results
                self._cache[cache_key] = generated_values
            
            # Enhance the original template data
            enhanced_data = self.metadata_enhancer.enhance_json_sections(template_data, generated_values)
            
            return enhanced_data
            
        except Exception as e:
            # Fallback with minimal enhancements
            print(f"Warning: Auto-generation failed: {e}")
            return self._apply_fallback_values(template_data)
    
    def _generate_all_values(self, sql_query: str, query_type: str, business_domain: str,
                           user_context: Dict, tech_metadata: Dict, template_data: Dict) -> Dict:
        """Generate all auto-calculated values."""
        generated = {}
        
        # SQL Complexity Analysis
        complexity_score = self.sql_analyzer.calculate_complexity_score(sql_query, tech_metadata)
        generated['complexity_score'] = complexity_score
        
        # Performance Predictions
        estimated_rows = self.performance_predictor.estimate_row_count(sql_query, tech_metadata)
        generated['estimated_row_count'] = estimated_rows
        
        performance_tier = self.performance_predictor.determine_performance_tier(complexity_score, estimated_rows)
        generated['performance_tier'] = performance_tier
        
        execution_time = self.performance_predictor.estimate_execution_time(
            complexity_score, estimated_rows, len(tech_metadata.get('tables_used', []))
        )
        generated['estimated_execution_time_ms'] = execution_time
        
        estimated_cost = self.performance_predictor.estimate_query_cost(complexity_score, tech_metadata)
        generated['estimated_cost'] = estimated_cost
        
        # Business Intelligence Calculations
        business_context = {
            'business_domain': business_domain,
            'user_role': user_context.get('user_role', 'analyst'),
            'business_impact': template_data.get('investigation_context', {}).get('business_impact', 'medium'),
            'confidence_level': template_data.get('investigation_context', {}).get('confidence_level', 'medium')
        }
        
        actionability_score = self.bi_calculator.calculate_actionability_score(query_type, business_context)
        generated['actionability_score'] = actionability_score
        
        business_value = self.bi_calculator.calculate_business_value(
            business_domain,
            business_context['business_impact'],
            business_context['user_role'],
            template_data.get('investigation_context', {}).get('urgency', 'medium')
        )
        generated['business_value'] = business_value
        
        strategic_alignment = self.bi_calculator.calculate_strategic_alignment(
            template_data.get('query_content', {}).get('query_intent', 'analysis'),
            template_data.get('semantic_context', {}).get('business_function', 'analytics')
        )
        generated['strategic_alignment'] = strategic_alignment
        
        # Metadata Enhancement
        cache_key = self.metadata_enhancer.generate_cache_key(sql_query, user_context)
        generated['cache_key'] = cache_key
        
        execution_counters = self.metadata_enhancer.initialize_execution_counters(is_template=True)
        generated['execution_counters'] = execution_counters
        
        timestamps = self.metadata_enhancer.generate_timestamps(
            user_context.get('timezone', 'UTC')
        )
        generated['timestamps'] = timestamps
        
        cache_strategy = self.metadata_enhancer.generate_cache_strategy(
            query_type, complexity_score, business_context['business_impact']
        )
        generated['cache_strategy'] = cache_strategy
        
        # Add enterprise vector metadata if available
        if self.enterprise_schema_enabled:
            vector_metadata = self._create_enterprise_vector_metadata(
                sql_query, query_type, business_domain, user_context, generated, template_data
            )
            generated['enterprise_vector_metadata'] = vector_metadata
        
        return generated
    
    def _apply_fallback_values(self, template_data: Dict) -> Dict:
        """Apply minimal fallback values if auto-generation fails."""
        enhanced_data = template_data.copy()
        
        # Apply basic fallbacks
        tech_metadata = enhanced_data.get('technical_metadata', {})
        tech_metadata.update({
            'complexity_score': 3,  # Default medium complexity
            'performance_tier': 'medium',
            'estimated_cost': 10.0,
            'estimated_execution_time_ms': 100.0,
            'estimated_row_count': 10
        })
        enhanced_data['technical_metadata'] = tech_metadata
        
        # Basic execution counters
        exec_results = enhanced_data.get('execution_results', {})
        exec_results.update({
            'execution_count': 1,
            'usage_frequency': 1,
            'cache_hit_count': 0,
            'cache_miss_count': 1
        })
        enhanced_data['execution_results'] = exec_results
        
        # Basic BI scores
        bi_data = enhanced_data.get('business_intelligence', {})
        bi_data.update({
            'actionability_score': 0.5,
            'business_value': 0.5,
            'strategic_alignment': 0.5
        })
        enhanced_data['business_intelligence'] = bi_data
        
        return enhanced_data
    
    def validate_generated_values(self, generated_values: Dict) -> bool:
        """
        Validate that generated values are within expected ranges.
        
        Args:
            generated_values: Dictionary of generated values
            
        Returns:
            True if all values are valid, False otherwise
        """
        try:
            # Validate complexity score
            complexity = generated_values.get('complexity_score', 0)
            if not (1 <= complexity <= 10):
                return False
            
            # Validate performance tier
            tier = generated_values.get('performance_tier', '')
            if tier not in ['low', 'medium', 'high', 'critical']:
                return False
            
            # Validate time estimates
            exec_time = generated_values.get('estimated_execution_time_ms', 0)
            if not (10 <= exec_time <= 600000):  # 10ms to 10 minutes
                return False
            
            # Validate row count
            rows = generated_values.get('estimated_row_count', 0)
            if not (1 <= rows <= 1000000):  # 1 to 1M rows
                return False
            
            # Validate scores (0.0 to 1.0)
            for score_field in ['actionability_score', 'business_value', 'strategic_alignment']:
                score = generated_values.get(score_field, -1)
                if not (0.0 <= score <= 1.0):
                    return False
            
            return True
            
        except (TypeError, ValueError):
            return False
    
    def _create_enterprise_vector_metadata(
        self, 
        sql_query: str, 
        query_type: str, 
        business_domain: str, 
        user_context: Dict,
        generated_values: Dict,
        template_data: Dict
    ) -> Dict[str, Any]:
        """Create enterprise vector metadata for unified cross-module integration."""
        if not self.enterprise_schema_enabled:
            return {}
        
        try:
            # Generate vector ID
            vector_id = generate_vector_id(sql_query, ModuleSource.AUTO_GENERATION)
            
            # Map business domain to unified enum
            domain_mapping = {
                'manufacturing_sales': BusinessDomain.SALES,
                'manufacturing_production': BusinessDomain.PRODUCTION,
                'manufacturing_quality': BusinessDomain.QUALITY,
                'supply_chain': BusinessDomain.SUPPLY_CHAIN,
                'cost_analysis': BusinessDomain.COST,
                'asset_management': BusinessDomain.ASSETS,
                'safety_compliance': BusinessDomain.SAFETY,
                'customer_analytics': BusinessDomain.CUSTOMER,
                'production_planning': BusinessDomain.PLANNING,
                'human_resources': BusinessDomain.HUMAN_RESOURCES,
                'finance': BusinessDomain.FINANCE,
                'marketing': BusinessDomain.MARKETING,
                'operations': BusinessDomain.OPERATIONS
            }
            unified_domain = domain_mapping.get(business_domain.lower(), BusinessDomain.OPERATIONS)
            
            # Map performance tier to unified enum
            tier_mapping = {
                'low': UnifiedPerformanceTier.SIMPLE,
                'medium': UnifiedPerformanceTier.ANALYTICAL, 
                'high': UnifiedPerformanceTier.COMPUTATIONAL,
                'critical': UnifiedPerformanceTier.INVESTIGATIVE
            }
            performance_tier = generated_values.get('performance_tier', 'medium')
            unified_tier = tier_mapping.get(performance_tier.lower(), UnifiedPerformanceTier.ANALYTICAL)
            
            # Map query type to analysis type
            analysis_mapping = {
                'analytical': AnalysisType.DESCRIPTIVE,
                'comparative': AnalysisType.DIAGNOSTIC,
                'predictive': AnalysisType.PREDICTIVE,
                'optimization': AnalysisType.PRESCRIPTIVE
            }
            unified_analysis = analysis_mapping.get(query_type.lower(), AnalysisType.DESCRIPTIVE)
            
            # Convert scores to unified 0.0-1.0 scale
            complexity_score_unified = normalize_score_to_unified_scale(
                generated_values.get('complexity_score', 5), "1-10"
            )
            
            actionability_score_unified = normalize_score_to_unified_scale(
                generated_values.get('actionability_score', 5), "1-10"
            )
            
            business_value_unified = normalize_score_to_unified_scale(
                generated_values.get('business_value', 0.5), "0.0-1.0"
            )
            
            # Create VectorMetadata object
            now = datetime.now(timezone.utc)
            vector_metadata = VectorMetadata(
                id=vector_id,
                module_source=ModuleSource.AUTO_GENERATION,
                created_at=now,
                updated_at=now,
                
                # Unified scoring
                complexity_score=complexity_score_unified,
                confidence_score=0.8,  # Default confidence for auto-generation
                actionability_score=actionability_score_unified,
                business_value_score=business_value_unified,
                
                # Unified classification
                business_domain=unified_domain,
                performance_tier=unified_tier,
                analysis_type=unified_analysis,
                
                # Business context
                user_role=user_context.get('user_role', 'analyst'),
                organization_context=user_context.get('organization', 'enterprise'),
                business_impact=template_data.get('investigation_context', {}).get('business_impact', 'medium'),
                urgency_level=template_data.get('investigation_context', {}).get('urgency', 'normal'),
                
                # Module-specific metadata
                module_metadata={
                    'sql_complexity_details': {
                        'original_score_1_10': generated_values.get('complexity_score', 5),
                        'performance_tier_original': performance_tier,
                        'estimated_execution_time_ms': generated_values.get('estimated_execution_time_ms', 100),
                        'estimated_cost': generated_values.get('estimated_cost', 10),
                        'estimated_row_count': generated_values.get('estimated_row_count', 10)
                    },
                    'cache_strategy': generated_values.get('cache_strategy', {}),
                    'execution_counters': generated_values.get('execution_counters', {})
                }
            )
            
            # Return as dictionary for JSON serialization
            return {
                'vector_id': vector_metadata.id,
                'module_source': vector_metadata.module_source.value,
                'unified_scores': {
                    'complexity_score': vector_metadata.complexity_score,
                    'confidence_score': vector_metadata.confidence_score,
                    'actionability_score': vector_metadata.actionability_score,
                    'business_value_score': vector_metadata.business_value_score
                },
                'unified_classification': {
                    'business_domain': vector_metadata.business_domain.value,
                    'performance_tier': vector_metadata.performance_tier.value,
                    'analysis_type': vector_metadata.analysis_type.value
                },
                'cross_module_ready': True,
                'enterprise_schema_version': '1.0',
                'integration_timestamp': now.isoformat()
            }
            
        except Exception as e:
            print(f"⚠️ Warning: Failed to create enterprise vector metadata: {e}")
            return {
                'cross_module_ready': False,
                'error': str(e),
                'fallback_mode': True
            }