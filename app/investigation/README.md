# Investigation Module - Phase 4: Autonomous Investigation Execution

## Overview

The Investigation module implements Phase 4 of the Agentic SQL system - the core autonomous investigation engine that conducts intelligent, hypothesis-driven analysis across coordinated database services. This module transforms business questions into comprehensive investigations using a 7-step framework with adaptive reasoning.

## Architecture

### Core Components

1. **AutonomousInvestigationEngine** (`runner.py`)
   - Main orchestrator for the 7-step investigation framework
   - Manages adaptive step execution based on complexity
   - Integrates AI reasoning with database operations
   - Implements timeout protection and error recovery

2. **InvestigationPrompts** (`prompts.py`)
   - AI prompt templates for each investigation step
   - Adaptive reasoning prompts for methodology evolution
   - Business context integration prompts

3. **InvestigationLogger** (`investigation_logging.py`)
   - Step-by-step investigation tracking
   - Performance metrics collection
   - Error and recovery logging

## The 7-Step Investigation Framework

### Step 1: Schema Analysis
**Purpose**: Discover and understand database structures to identify relevant data sources

**What it does**:
- Analyzes available tables based on business question keywords
- Maps table relationships and foreign keys
- Identifies data types and constraints
- Evaluates indexing for performance optimization
- Detects business logic embedded in schema design

**Example**:
- Question: "Why did Q4 sales drop?"
- Discovers: `sales`, `orders`, `customers`, `products` tables
- Maps: Foreign key relationships between tables
- Notes: Date columns for temporal analysis

**Complexity-based execution**:
- Simple queries: Analyze 1 most relevant table
- Complex queries: Analyze up to 15 related tables

### Step 2: Data Exploration
**Purpose**: Assess data quality and identify initial patterns

**What it does**:
- Evaluates data completeness and null value patterns
- Identifies data distributions and outliers
- Checks temporal coverage and data currency
- Validates data consistency across sources
- Estimates row counts and data volumes

**Example**:
- Finds: 3 months of missing data in Q3
- Identifies: 15% null values in customer region field
- Discovers: Data skew in December (holiday season)

**Complexity-based execution**:
- Skipped for simple queries (<0.3 complexity)
- Full exploration for moderate+ queries

### Step 3: Hypothesis Generation
**Purpose**: Generate testable business theories based on initial findings

**What it does**:
- Creates 3-5 specific hypotheses addressing the business question
- Defines clear success/failure criteria for each hypothesis
- Identifies required data sources for validation
- Considers confounding variables and alternative explanations
- Assigns expected confidence levels

**Example hypotheses for "Why did Q4 sales drop?":
1. "Holiday promotions were less effective than previous years"
2. "Key product categories experienced supply chain issues"
3. "Competitor pricing changes impacted market share"
4. "Customer satisfaction issues led to reduced repeat purchases"

**Complexity-based execution**:
- Skipped for simple and moderate queries (<0.5 complexity)
- Essential for analytical and complex investigations

### Step 4: Core Analysis
**Purpose**: Execute primary investigation through systematic data analysis

**What it does**:
- Tests each hypothesis using appropriate analytical methods
- Executes optimized SQL queries across data sources
- Performs statistical analysis where applicable
- Calculates confidence scores for findings
- Documents evidence quality and limitations

**Example execution**:
- Hypothesis: "Holiday promotions were less effective"
- Analysis: Compare promotion response rates YoY
- Finding: 23% lower conversion rate on promotions
- Confidence: 0.85 (high statistical significance)

**Complexity-based execution**:
- Always executed (core step for all complexity levels)
- Simple queries: Direct SQL execution
- Complex queries: Multi-step analysis with statistical validation

### Step 5: Pattern Discovery
**Purpose**: Identify unexpected patterns, anomalies, and hidden correlations

**What it does**:
- Detects temporal patterns and seasonality
- Identifies anomalies requiring investigation
- Discovers cross-domain correlations
- Recognizes emerging trends
- Evaluates pattern business significance

**Example discoveries**:
- Unusual spike in returns 2 weeks after Black Friday
- Strong correlation between weather events and sales
- Emerging trend: Mobile orders growing 40% QoQ
- Anomaly: Southwest region outperforming by 200%

**Complexity-based execution**:
- Skipped for simple and moderate queries (<0.5 complexity)
- Full pattern analysis for analytical and complex queries

### Step 6: Cross Validation
**Purpose**: Validate findings across data sources to ensure reliability

**What it does**:
- Cross-references findings between databases
- Validates temporal relationships and causality
- Checks metric consistency across systems
- Assesses statistical significance
- Identifies potential bias or data quality issues

**Example validation**:
- Sales drop confirmed in both transaction and inventory systems
- Customer satisfaction metrics align with sales decline
- No data quality issues found that would invalidate findings

**Complexity-based execution**:
- Only executed for complex queries (>0.8 complexity)
- Critical for high-stakes business decisions

### Step 7: Results Synthesis
**Purpose**: Compile all findings into coherent, actionable results

**What it does**:
- Integrates findings from all completed steps
- Prioritizes insights by business impact
- Calculates overall confidence scores
- Prepares structured results for Phase 5
- Documents limitations and uncertainties

**Example synthesis**:
- Primary finding: Q4 sales dropped 18% due to promotion effectiveness
- Secondary factors: Supply chain (5% impact), competition (3% impact)
- Confidence: 0.78 overall (high confidence in primary finding)
- Recommendation: Revise promotion strategy for next quarter

**Complexity-based execution**:
- Always executed (required for all investigations)
- Simple queries: Basic result formatting
- Complex queries: Comprehensive synthesis with recommendations

## Adaptive Step Execution

The investigation engine intelligently selects which steps to execute based on query complexity:

| Complexity Score | Classification | Steps Executed | Typical Duration |
|-----------------|----------------|----------------|------------------|
| < 0.3 | Simple | 1, 4, 7 (3 steps) | 2-3 seconds |
| 0.3-0.5 | Moderate | 1, 2, 4, 7 (4 steps) | 4-5 seconds |
| 0.5-0.8 | Analytical | 1, 2, 3, 4, 5, 7 (6 steps) | 8-10 seconds |
| > 0.8 | Complex | All 7 steps | 12-15 seconds |

### Why Steps Are Skipped

- **Data Exploration**: Skipped for simple queries that need direct answers
- **Hypothesis Generation**: Not needed for straightforward metric retrieval
- **Pattern Discovery**: Unnecessary for basic operational queries
- **Cross Validation**: Only required for complex investigations needing high confidence

## Integration Points

### Input from Phase 3
```python
{
    "coordinated_services": {
        "mariadb": MariaDBClient,
        "qdrant": QdrantClient  # Only for complex queries
    },
    "investigation_request": "Why did Q4 sales drop?",
    "execution_context": {
        "complexity_score": 0.85,
        "business_intent": "root_cause_analysis",
        "user_context": {...}
    }
}
```

### Output to Phase 5
```python
InvestigationResults(
    investigation_id="uuid",
    status="completed",
    investigation_findings={
        "schema_analysis": {...},
        "hypothesis_tests": {...},
        "pattern_discoveries": {...},
        "cross_validation": {...}
    },
    confidence_scores={
        "overall": 0.78,
        "by_hypothesis": {...}
    },
    business_context={
        "domain": "sales",
        "impact": "high",
        "urgency": "immediate"
    }
)
```

## Configuration

Configure via `settings.env`:
```bash
# Timeout settings
STEP_TIMEOUT_SIMPLE=10
STEP_TIMEOUT_COMPLEX=30

# Adaptive execution
ENABLE_ADAPTIVE_REASONING=true
ENABLE_RECOVERY_MODE=true

# Table limits by complexity
MAX_TABLES_SIMPLE=1
MAX_TABLES_MODERATE=5
MAX_TABLES_COMPLEX=15
```

## Performance Optimizations

1. **Timeout Protection**: Each step has complexity-based timeouts
2. **Table Limiting**: Restricts tables analyzed based on complexity
3. **Parallel Execution**: Where possible, independent analyses run concurrently
4. **Early Termination**: Failed critical steps can terminate investigation
5. **Caching Integration**: Reuses patterns from Qdrant when available

## Error Handling

The module implements sophisticated error recovery:

1. **Step-Level Recovery**: Individual step failures don't crash investigation
2. **Graceful Degradation**: Missing optional steps reduce confidence, not failure
3. **Timeout Handling**: Steps that exceed time limits return partial results
4. **Service Failures**: Continues with available services when some fail

## Logging and Monitoring

Comprehensive logging includes:
- Step start/completion with duration
- Key findings with confidence scores
- Database query execution times
- Error recovery attempts
- Overall investigation summary

## Usage Example

```python
# Initialize engine
engine = AutonomousInvestigationEngine()

# Conduct investigation
results = await engine.conduct_investigation(
    coordinated_services={"mariadb": mariadb_client},
    investigation_request="Show me top selling products this month",
    execution_context={
        "complexity_score": 0.4,
        "business_intent": "metric_retrieval"
    }
)

# Results contain structured findings ready for Phase 5
print(f"Investigation {results.investigation_id} completed")
print(f"Confidence: {results.overall_confidence}")
print(f"Steps executed: {len(results.completed_steps)}")
```

## Best Practices

1. **Let Complexity Drive Execution**: Trust the adaptive framework
2. **Monitor Timeouts**: Adjust based on your database performance
3. **Review Confidence Scores**: Low confidence may indicate data quality issues
4. **Check Error Logs**: Failed steps may hide important insights
5. **Validate Critical Findings**: Use cross-validation for high-stakes decisions

## Future Enhancements

- **Streaming Progress**: Real-time updates via WebSocket
- **Custom Step Framework**: User-defined investigation steps
- **ML Pattern Learning**: Improve hypothesis generation over time
- **Collaborative Investigation**: Multi-analyst coordination
- **Investigation Templates**: Pre-built frameworks for common analyses