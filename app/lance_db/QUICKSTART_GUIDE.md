# LanceDB Vector Integration - Quick Start Guide

## Prerequisites

```bash
# Install required dependencies
uv add lancedb
uv add sentence-transformers
uv add pyarrow
uv add numpy
```

## 1. Basic Setup

### Initialize Vector Infrastructure

```python
import asyncio
from pathlib import Path

# Import vector infrastructure
from app.lance_db.src import (
    EnterpriseVectorSchema,
    VectorIndexManager,
    VectorPerformanceMonitor
)

async def setup_vector_infrastructure():
    """Initialize LanceDB vector infrastructure."""
    import lancedb
    
    # Connect to LanceDB
    db_path = Path("./data/lancedb")
    db_path.mkdir(parents=True, exist_ok=True)
    db = await lancedb.connect_async(str(db_path))
    
    # Create schema
    schema_manager = EnterpriseVectorSchema()
    schema = schema_manager.create_lance_schema()
    
    # Create table
    table = await db.create_table(
        "enterprise_vectors",
        schema=schema
    )
    
    # Initialize index manager
    index_manager = VectorIndexManager(db)
    await index_manager.create_index(
        table_name="enterprise_vectors",
        index_type="IVF_PQ",
        metric_type="L2"
    )
    
    print("✅ Vector infrastructure initialized!")
    return db, table

# Run setup
asyncio.run(setup_vector_infrastructure())
```

## 2. Intelligence Module Usage

### Enhance Business Intent Classification

```python
from app.intelligence import VectorEnhancedDomainExpert

async def classify_with_vectors():
    # Initialize expert
    expert = VectorEnhancedDomainExpert()
    await expert.initialize()
    
    # Classify query
    query = "Show me why customer churn increased last month"
    result = await expert.classify_business_intent_with_vectors(query)
    
    print(f"Domain: {result.primary_domain}")
    print(f"Analysis Type: {result.analysis_type}")
    print(f"Confidence: {result.confidence:.2%}")
    print(f"Pattern Boost: {result.confidence_boost:.2%}")
    
    # View similar queries
    for q in result.similar_queries[:3]:
        print(f"Similar: {q.query_text} (score: {q.similarity_score:.2f})")

asyncio.run(classify_with_vectors())
```

### Analyze Complexity with Patterns

```python
from app.intelligence import VectorEnhancedComplexityAnalyzer

async def analyze_complexity():
    analyzer = VectorEnhancedComplexityAnalyzer()
    await analyzer.initialize()
    
    # Analyze with historical patterns
    result = await analyzer.analyze_complexity_with_vectors(
        business_intent=business_intent,
        query="Complex multi-table analysis with ML predictions"
    )
    
    print(f"Complexity Score: {result.overall_score:.2f}")
    print(f"Estimated Time: {result.estimated_execution_time_seconds}s")
    print(f"Pattern-based adjustment: {result.pattern_based_adjustment:.2%}")

asyncio.run(analyze_complexity())
```

## 3. Investigation Module Usage

### Run Vector-Enhanced Investigation

```python
from app.investigation import conduct_vector_enhanced_investigation

async def investigate_with_patterns():
    result = await conduct_vector_enhanced_investigation(
        coordinated_services={
            "mariadb": {"enabled": True, "priority": 1},
            "postgresql": {"enabled": True, "priority": 2}
        },
        investigation_request="Why did production costs increase in Q2?",
        execution_context={
            "business_domain": "production",
            "complexity_level": "high",
            "urgency": "medium"
        },
        use_vector_enhancement=True
    )
    
    print(f"Investigation Status: {result.base_results.status}")
    print(f"Confidence Boost: {result.confidence_boost:.2%}")
    print(f"Similar Investigations: {len(result.similar_investigations)}")
    
    # View optimization suggestions
    for opt in result.suggested_step_optimizations[:3]:
        print(f"Optimize {opt['step_name']}: {opt['optimization_rationale']}")

asyncio.run(investigate_with_patterns())
```

## 4. Insight Synthesis Usage

### Generate Enhanced Insights

```python
from app.insight_synthesis import synthesize_insights_with_vectors

async def synthesize_insights():
    # Mock investigation results
    investigation_results = {
        "investigation_id": "inv-001",
        "investigation_findings": {
            "key_findings": [
                "Production efficiency dropped 15%",
                "Quality defects increased to 3.2%"
            ],
            "root_causes": [
                "Equipment maintenance delays",
                "Staff training gaps"
            ]
        },
        "overall_confidence": 0.85
    }
    
    result = await synthesize_insights_with_vectors(
        investigation_results=investigation_results,
        business_context={
            "strategic_goal": "Improve efficiency by 20%",
            "domain": "production"
        },
        user_role="executive",
        use_vector_enhancement=True
    )
    
    print(f"Insights Generated: {len(result.insights)}")
    print(f"Quality Boost: {result.insight_quality_boost:.2%}")
    print(f"Predicted Adoption: {result.predicted_adoption_rate:.2%}")
    
    # Top insight
    if result.insights:
        top = result.insights[0]
        print(f"\nTop Insight: {top.title}")
        print(f"Confidence: {top.confidence:.2%}")
        print(f"Impact: {top.business_impact}")

asyncio.run(synthesize_insights())
```

## 5. Cross-Module Intelligence

### Analyze Module Relationships

```python
from app.lance_db.src import analyze_investigation_insight_intelligence

async def analyze_cross_module():
    # Generate intelligence report
    report = await analyze_investigation_insight_intelligence(
        time_window_days=30
    )
    
    print(f"Active Links: {len(report.active_links)}")
    print(f"Patterns: {len(report.discovered_patterns)}")
    print(f"Feedback Loops: {len(report.feedback_loops)}")
    print(f"ROI Multiplier: {report.roi_multiplier:.1f}x")
    
    # Top patterns
    for pattern in report.discovered_patterns[:3]:
        print(f"\nPattern: {pattern.pattern_name}")
        print(f"Occurrences: {pattern.occurrence_count}")
        print(f"Business Value: {pattern.business_value_generated:.2f}")
    
    # Recommendations
    print("\nRecommended Investigation Areas:")
    for area in report.recommended_investigation_areas[:3]:
        print(f"- {area}")

asyncio.run(analyze_cross_module())
```

## 6. Common Patterns

### Pattern 1: Full Pipeline Execution

```python
async def full_pipeline():
    """Execute complete intelligence -> investigation -> synthesis pipeline."""
    
    # Step 1: Classify intent
    from app.intelligence import VectorEnhancedDomainExpert
    expert = VectorEnhancedDomainExpert()
    await expert.initialize()
    
    query = "Analyze customer satisfaction decline and recommend improvements"
    intent = await expert.classify_business_intent_with_vectors(query)
    
    # Step 2: Investigate
    from app.investigation import conduct_vector_enhanced_investigation
    investigation = await conduct_vector_enhanced_investigation(
        coordinated_services={"mariadb": {"enabled": True}},
        investigation_request=query,
        execution_context={
            "business_domain": intent.primary_domain,
            "complexity_level": "high"
        },
        use_vector_enhancement=True
    )
    
    # Step 3: Synthesize insights
    from app.insight_synthesis import synthesize_insights_with_vectors
    insights = await synthesize_insights_with_vectors(
        investigation_results=investigation.base_results.__dict__,
        business_context={
            "domain": intent.primary_domain,
            "strategic_goal": "Improve customer satisfaction"
        },
        user_role="manager",
        use_vector_enhancement=True
    )
    
    return intent, investigation, insights
```

### Pattern 2: Batch Pattern Learning

```python
async def batch_learn_patterns():
    """Process multiple queries to build pattern knowledge."""
    
    queries = [
        "Why are sales declining in region X?",
        "What caused the production bottleneck?",
        "How can we reduce customer churn?",
        "Where are the cost overruns coming from?"
    ]
    
    from app.intelligence import LanceDBPatternRecognizer
    recognizer = LanceDBPatternRecognizer()
    await recognizer.initialize()
    
    # Process queries
    for query in queries:
        # Process and store patterns
        await recognizer.process_and_learn(query)
    
    # Analyze patterns
    patterns = await recognizer.analyze_cross_module_patterns()
    return patterns
```

## 7. Performance Optimization

### Monitor and Optimize

```python
from app.lance_db.src import VectorPerformanceMonitor

async def monitor_performance():
    monitor = VectorPerformanceMonitor()
    
    # Get performance metrics
    metrics = await monitor.get_performance_summary()
    
    print("Performance Summary:")
    print(f"Avg Search Latency: {metrics['avg_search_latency_ms']}ms")
    print(f"P95 Latency: {metrics['p95_latency_ms']}ms")
    print(f"Queries per Second: {metrics['queries_per_second']}")
    
    # Check for anomalies
    anomalies = await monitor.detect_anomalies()
    if anomalies:
        print(f"\n⚠️ Detected {len(anomalies)} anomalies")
```

## 8. Troubleshooting

### Common Issues and Solutions

```python
# Issue: Vector search returning no results
async def debug_vector_search():
    from app.lance_db.src import vector_debug_utils
    
    # Check table contents
    count = await vector_debug_utils.get_vector_count()
    print(f"Total vectors: {count}")
    
    # Verify embeddings
    test_text = "test query"
    embedding = await vector_debug_utils.generate_embedding(test_text)
    print(f"Embedding dimension: {len(embedding)}")
    
    # Test search
    results = await vector_debug_utils.test_search(test_text)
    print(f"Search results: {len(results)}")

# Issue: Slow performance
async def optimize_performance():
    from app.lance_db.src import VectorIndexManager
    
    manager = VectorIndexManager()
    
    # Rebuild index
    await manager.rebuild_index("enterprise_vectors")
    
    # Optimize query patterns
    await manager.optimize_for_query_pattern("frequent_similarity_search")
```

## 9. Best Practices

1. **Always Initialize**: Call `initialize()` on vector-enhanced modules
2. **Use Batch Operations**: Process multiple items together when possible
3. **Monitor Performance**: Regular checks on latency and accuracy
4. **Clean Up Resources**: Call `cleanup()` when done
5. **Handle Fallbacks**: Always have non-vector fallback logic

## 10. Next Steps

1. **Explore Advanced Features**:
   - Custom embedding models
   - Multi-modal search
   - Real-time pattern updates

2. **Integrate with Your Workflow**:
   - Add to existing pipelines
   - Create custom patterns
   - Build domain-specific intelligence

3. **Monitor and Improve**:
   - Track business metrics
   - Analyze pattern effectiveness
   - Optimize for your use cases

## Support

- **Documentation**: `/app/lance_db/INTEGRATION_COMPLETE.md`
- **Examples**: `/app/lance_db/examples/`
- **Tests**: Run test files for working examples
- **Issues**: Check module-specific READMEs

Happy Vector Intelligence Building! 🚀