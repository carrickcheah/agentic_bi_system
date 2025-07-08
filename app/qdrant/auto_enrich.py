"""
Auto-enrich pattern data by generating values for null/zero fields.
Analyzes SQL queries and generates intelligent metadata.
"""

import re
import hashlib
from datetime import datetime, timezone
from typing import Dict, Any, List
import uuid


class PatternEnricher:
    """Auto-generate values for pattern metadata fields."""
    
    def __init__(self):
        self.sql_complexity_keywords = {
            'simple': ['SELECT', 'FROM', 'WHERE'],
            'moderate': ['JOIN', 'GROUP BY', 'HAVING', 'DISTINCT'],
            'complex': ['UNION', 'INTERSECT', 'EXCEPT', 'EXISTS', 'RECURSIVE'],
            'advanced': ['WINDOW', 'PARTITION BY', 'LAG', 'LEAD', 'RANK', 'ROW_NUMBER', 'CTE', 'WITH']
        }
    
    def enrich_pattern(self, pattern: Dict[str, Any], source_file: str = "") -> Dict[str, Any]:
        """Enrich a pattern by auto-generating missing values."""
        # Generate ID if null
        if pattern.get("_id") is None:
            pattern["_id"] = self._generate_id(pattern, source_file)
        
        # Enrich query content
        query_content = pattern.get("query_content", {})
        sql_query = query_content.get("sql_query", "")
        
        # Enrich technical metadata
        technical = pattern.get("technical_metadata", {})
        if technical:
            pattern["technical_metadata"] = self._enrich_technical_metadata(technical, sql_query)
        
        # Enrich execution results
        execution = pattern.get("execution_results", {})
        if execution:
            pattern["execution_results"] = self._enrich_execution_results(execution)
        
        # Enrich learning metadata
        learning = pattern.get("learning_metadata", {})
        if learning:
            pattern["learning_metadata"] = self._enrich_learning_metadata(learning)
        
        # Enrich business intelligence
        bi = pattern.get("business_intelligence", {})
        if bi:
            pattern["business_intelligence"] = self._enrich_business_intelligence(bi)
        
        # Enrich version control
        version = pattern.get("version_control", {})
        if version:
            pattern["version_control"] = self._enrich_version_control(version)
        
        # Enrich caching
        caching = pattern.get("caching", {})
        if caching:
            pattern["caching"] = self._enrich_caching(caching, pattern)
        
        # Enrich monitoring
        monitoring = pattern.get("monitoring", {})
        if monitoring:
            pattern["monitoring"] = self._enrich_monitoring(monitoring, technical.get("complexity_score", 0.5))
        
        # Enrich security
        security = pattern.get("security", {})
        if security:
            pattern["security"] = self._enrich_security(security, sql_query)
        
        # Enrich embeddings metadata (not actual embeddings)
        embeddings = pattern.get("embeddings", {})
        if embeddings:
            pattern["embeddings"] = self._enrich_embeddings_metadata(embeddings)
        
        return pattern
    
    def _generate_id(self, pattern: Dict[str, Any], source_file: str = "") -> str:
        """Generate unique ID based on pattern content and source."""
        sql_query = pattern.get("query_content", {}).get("sql_query", "")
        # Include source file in hash to ensure uniqueness
        unique_content = f"{source_file}:{sql_query}"
        hash_content = hashlib.md5(unique_content.encode()).hexdigest()
        return f"pat_{hash_content[:12]}"
    
    def _calculate_complexity_score(self, sql_query: str) -> float:
        """Calculate complexity score from 0.0 to 1.0."""
        score = 0.1  # Base score
        sql_upper = sql_query.upper()
        
        # Check for complexity indicators
        if 'JOIN' in sql_upper:
            score += 0.1 * sql_upper.count('JOIN')
        if 'CASE WHEN' in sql_upper:
            score += 0.15
        if 'PARTITION BY' in sql_upper or 'WINDOW' in sql_upper:
            score += 0.25
        if 'WITH' in sql_upper and sql_upper.index('WITH') < 10:  # CTE
            score += 0.2
        if 'UNION' in sql_upper:
            score += 0.15
        if 'GROUP BY' in sql_upper:
            score += 0.1
        if 'HAVING' in sql_upper:
            score += 0.1
        
        # Subqueries
        subquery_count = sql_upper.count('(SELECT')
        score += 0.15 * subquery_count
        
        # Length factor
        if len(sql_query) > 500:
            score += 0.1
        if len(sql_query) > 1000:
            score += 0.1
        
        return min(score, 1.0)
    
    def _estimate_execution_time(self, complexity_score: float, table_count: int) -> int:
        """Estimate execution time in milliseconds."""
        base_time = 50
        complexity_factor = complexity_score * 500
        table_factor = table_count * 20
        return int(base_time + complexity_factor + table_factor)
    
    def _enrich_technical_metadata(self, technical: Dict[str, Any], sql_query: str) -> Dict[str, Any]:
        """Enrich technical metadata with calculated values."""
        if technical.get("complexity_score") is None:
            technical["complexity_score"] = self._calculate_complexity_score(sql_query)
        
        complexity = technical["complexity_score"]
        
        if technical.get("performance_tier") is None:
            if complexity < 0.3:
                technical["performance_tier"] = "fast"
            elif complexity < 0.6:
                technical["performance_tier"] = "moderate"
            else:
                technical["performance_tier"] = "intensive"
        
        if technical.get("estimated_cost") is None:
            technical["estimated_cost"] = round(complexity * 10, 2)
        
        if technical.get("estimated_execution_time_ms") is None:
            table_count = len(technical.get("tables_used", []))
            technical["estimated_execution_time_ms"] = self._estimate_execution_time(complexity, table_count)
        
        if technical.get("estimated_row_count") is None:
            # Estimate based on query type
            if "COUNT" in sql_query.upper():
                technical["estimated_row_count"] = 1
            elif "LIMIT" in sql_query.upper():
                match = re.search(r'LIMIT\s+(\d+)', sql_query, re.IGNORECASE)
                if match:
                    technical["estimated_row_count"] = int(match.group(1))
                else:
                    technical["estimated_row_count"] = 100
            else:
                technical["estimated_row_count"] = 1000  # Default estimate
        
        if not technical.get("optimization_hints"):
            hints = []
            if technical.get("has_window_functions"):
                hints.append("Consider indexing partition columns")
            if len(technical.get("tables_used", [])) > 3:
                hints.append("Review join order for optimization")
            if complexity > 0.7:
                hints.append("Consider query result caching")
            technical["optimization_hints"] = hints
        
        return technical
    
    def _enrich_execution_results(self, execution: Dict[str, Any]) -> Dict[str, Any]:
        """Enrich execution results with initial values."""
        now = datetime.now(timezone.utc).isoformat()
        
        if execution.get("success") is None:
            execution["success"] = False
        
        if execution.get("execution_time_ms") is None:
            execution["execution_time_ms"] = 0
        
        if execution.get("rows_returned") is None:
            execution["rows_returned"] = 0
        
        if execution.get("data_quality_score") is None:
            execution["data_quality_score"] = 0.95  # Assume high quality by default
        
        if execution.get("business_relevance_score") is None:
            execution["business_relevance_score"] = 0.8  # Assume good relevance
        
        if execution.get("user_satisfaction") is None:
            execution["user_satisfaction"] = 0.0
        
        if execution.get("led_to_followup") is None:
            execution["led_to_followup"] = False
        
        if execution.get("average_execution_time_ms") is None:
            execution["average_execution_time_ms"] = 0
        
        if execution.get("fastest_execution_ms") is None:
            execution["fastest_execution_ms"] = 0
        
        if execution.get("slowest_execution_ms") is None:
            execution["slowest_execution_ms"] = 0
        
        if execution.get("cache_hit_rate") is None:
            execution["cache_hit_rate"] = 0.0
        
        return execution
    
    def _enrich_learning_metadata(self, learning: Dict[str, Any]) -> Dict[str, Any]:
        """Enrich learning metadata with initial values."""
        now = datetime.now(timezone.utc).isoformat()
        
        if learning.get("success_rate") is None:
            learning["success_rate"] = 0.0
        
        if learning.get("average_user_rating") is None:
            learning["average_user_rating"] = 0.0
        
        if learning.get("first_used_at") is None:
            learning["first_used_at"] = now
        
        if learning.get("last_used_at") is None:
            learning["last_used_at"] = now
        
        if learning.get("learning_updated_at") is None:
            learning["learning_updated_at"] = now
        
        return learning
    
    def _enrich_business_intelligence(self, bi: Dict[str, Any]) -> Dict[str, Any]:
        """Enrich business intelligence metadata."""
        if bi.get("actionability_score") is None:
            # Calculate based on decision support level
            level = bi.get("decision_support_level", "operational")
            scores = {
                "strategic": 0.9,
                "tactical": 0.7,
                "operational": 0.5
            }
            bi["actionability_score"] = scores.get(level, 0.5)
        
        if bi.get("business_value") is None:
            bi["business_value"] = "medium"  # Default value
        
        return bi
    
    def _enrich_version_control(self, version: Dict[str, Any]) -> Dict[str, Any]:
        """Enrich version control metadata."""
        now = datetime.now(timezone.utc).isoformat()
        
        if version.get("created_at") is None:
            version["created_at"] = now
        
        if version.get("updated_at") is None:
            version["updated_at"] = now
        
        if version.get("peer_review_score") is None:
            version["peer_review_score"] = 0.0
        
        return version
    
    def _enrich_caching(self, caching: Dict[str, Any], pattern: Dict[str, Any]) -> Dict[str, Any]:
        """Enrich caching metadata."""
        if caching.get("cache_key") is None:
            # Generate cache key from SQL query
            sql_query = pattern.get("query_content", {}).get("sql_query", "")
            caching["cache_key"] = hashlib.md5(sql_query.encode()).hexdigest()
        
        if caching.get("last_cache_update") is None:
            caching["last_cache_update"] = datetime.now(timezone.utc).isoformat()
        
        return caching
    
    def _enrich_monitoring(self, monitoring: Dict[str, Any], complexity_score: float) -> Dict[str, Any]:
        """Enrich monitoring metadata based on complexity."""
        thresholds = monitoring.get("alert_thresholds", {})
        
        if thresholds.get("execution_time_ms") is None:
            # Set threshold based on complexity
            if complexity_score < 0.3:
                thresholds["execution_time_ms"] = 100
            elif complexity_score < 0.6:
                thresholds["execution_time_ms"] = 500
            else:
                thresholds["execution_time_ms"] = 2000
        
        if thresholds.get("row_count_variance") is None:
            thresholds["row_count_variance"] = 0.2  # 20% variance
        
        if thresholds.get("error_rate") is None:
            thresholds["error_rate"] = 0.05  # 5% error rate
        
        monitoring["alert_thresholds"] = thresholds
        
        baseline = monitoring.get("performance_baseline", {})
        
        if baseline.get("expected_execution_time_ms") is None:
            baseline["expected_execution_time_ms"] = thresholds["execution_time_ms"] // 2
        
        if baseline.get("expected_row_count_range") is None:
            baseline["expected_row_count_range"] = [10, 10000]
        
        if baseline.get("resource_usage") is None:
            baseline["resource_usage"] = "low" if complexity_score < 0.5 else "medium"
        
        monitoring["performance_baseline"] = baseline
        
        if monitoring.get("health_check_frequency") is None:
            monitoring["health_check_frequency"] = "daily"
        
        if monitoring.get("last_health_check") is None:
            monitoring["last_health_check"] = datetime.now(timezone.utc).isoformat()
        
        return monitoring
    
    def _enrich_security(self, security: Dict[str, Any], sql_query: str) -> Dict[str, Any]:
        """Enrich security metadata."""
        sql_upper = sql_query.upper()
        
        if security.get("contains_sensitive_data") is None:
            # Check for sensitive table/column names
            sensitive_keywords = ['PASSWORD', 'SSN', 'CREDIT_CARD', 'EMAIL', 'PHONE', 'ADDRESS']
            security["contains_sensitive_data"] = any(kw in sql_upper for kw in sensitive_keywords)
        
        if security.get("data_classification") is None:
            if security["contains_sensitive_data"]:
                security["data_classification"] = "confidential"
            else:
                security["data_classification"] = "internal"
        
        if security.get("requires_audit_log") is None:
            security["requires_audit_log"] = security["contains_sensitive_data"]
        
        if not security.get("allowed_roles"):
            security["allowed_roles"] = ["analyst", "manager", "admin"]
        
        if security.get("data_retention_days") is None:
            security["data_retention_days"] = 90
        
        return security
    
    def _enrich_embeddings_metadata(self, embeddings: Dict[str, Any]) -> Dict[str, Any]:
        """Enrich embeddings metadata (not the actual embeddings)."""
        if embeddings.get("embedding_created_at") is None:
            embeddings["embedding_created_at"] = datetime.now(timezone.utc).isoformat()
        
        # Update to OpenAI embeddings
        embeddings["embedding_model"] = "text-embedding-3-small"
        embeddings["embedding_dimension"] = 1536
        
        return embeddings


def enrich_patterns_file(file_path: str) -> Dict[str, Any]:
    """Enrich all patterns in a JSON file."""
    import json
    
    enricher = PatternEnricher()
    
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    # Handle single pattern or array of patterns
    if isinstance(data, dict):
        return enricher.enrich_pattern(data)
    elif isinstance(data, list):
        return [enricher.enrich_pattern(pattern) for pattern in data]
    else:
        raise ValueError(f"Unexpected data type in {file_path}")


if __name__ == "__main__":
    # Test enrichment
    import json
    from pathlib import Path
    
    test_file = Path(__file__).parent / "patterns" / "template_moq.json"
    
    if test_file.exists():
        enriched = enrich_patterns_file(str(test_file))
        print(json.dumps(enriched, indent=2))