# 🔍 Complete Investigation Flow - Five-Phase Workflow

## 🏗️ **Architecture Overview**

```
User Question → API → AutonomousBusinessAnalyst → Five-Phase Investigation → Results
```

## 🎯 **Revolutionary Principles**

1. **Business Intelligence First, Technology Second** - Thinks about business analysis methodology
2. **Autonomous Investigation, Not Query Translation** - Conducts multi-phase investigations  
3. **Organizational Learning, Not Individual Tools** - Every investigation improves system for everyone

---

## 📋 **Five-Phase Investigation Workflow**

### **Phase 1: Query Processing** 
*Transform natural language to business intent*

**Input**: `"What were last month's sales by region?"`

**Process**:
- **Business Intent Extraction**: Parse business meaning and context
- **Domain Classification**: Classify as "sales", "finance", "customer", etc.
- **Semantic Hashing**: Generate unique hash for cache lookup
- **Complexity Analysis**: Determine investigation complexity (simple → comprehensive)
- **User Context Integration**: Consider user role, permissions, department
- **Organization Context**: Apply business rules, fiscal calendar, data classification

**Output**:
```json
{
  "original_question": "What were last month's sales by region?",
  "business_intent": "revenue_analysis_by_geographic_dimension",
  "business_domain": "sales",
  "semantic_hash": "abc123...",
  "complexity_indicators": {
    "complexity_level": "simple",
    "data_sources_required": ["sales_data"],
    "analysis_type": "descriptive"
  },
  "user_context": {
    "role": "sales_manager",
    "permissions": ["read_sales_data"],
    "department": "sales"
  }
}
```

---

### **Phase 2: Strategy Planning**
*Create investigation methodology and execution plan*

**Input**: Semantic intent from Phase 1

**Process**:
- **Complexity Determination**: Simple → Moderate → Complex → Comprehensive
- **Methodology Selection**: Descriptive, Diagnostic, Predictive, or Prescriptive
- **Multi-Phase Planning**: Break complex investigations into manageable phases
- **Resource Estimation**: Calculate time, queries, and services needed
- **Risk Assessment**: Identify potential data quality issues
- **Service Selection**: Determine which databases/services to use

**Output**:
```json
{
  "investigation_strategy": {
    "complexity": "simple",
    "methodology": "descriptive",
    "estimated_duration_minutes": 3,
    "phases": [
      {
        "phase_name": "data_discovery",
        "phase_type": "discovery",
        "estimated_duration_minutes": 1,
        "required_services": ["business_data"],
        "estimated_queries": 2
      },
      {
        "phase_name": "sales_analysis",
        "phase_type": "analysis", 
        "estimated_duration_minutes": 2,
        "required_services": ["business_data", "analytics"],
        "estimated_queries": 3
      }
    ]
  }
}
```

---

### **Phase 3: Service Orchestration**
*Coordinate database services and MCP tools*

**Input**: Investigation strategy from Phase 2

**Process**:
- **Service Selection**: Choose optimal database services (MariaDB, PostgreSQL, etc.)
- **MCP Tool Coordination**: Coordinate Multiple Computer Protocols for tool access
- **Connection Management**: Establish and manage database connections
- **Schema Analysis**: Understand available tables and relationships
- **Business Logic Integration**: Apply business rules and domain knowledge
- **Cross-Service Data Correlation**: Coordinate data from multiple sources

**Services Available**:
- **Business Data Service**: MariaDB with sales, customers, products
- **Memory Service**: PostgreSQL for organizational memory
- **Vector Service**: Qdrant for semantic search
- **Analytics Service**: Advanced analytics and computation
- **External Service**: Supabase for additional operations

**Output**: Configured service connections and execution context

---

### **Phase 4: Investigation Execution**
*Autonomous multi-step analysis with adaptive methodology*

**Input**: Service connections and investigation plan

**Process**: **Autonomous Investigation Engine** executes investigation steps:

#### **Step Types**:
1. **Schema Analysis**: Understand database structure
2. **Data Exploration**: Initial data discovery and quality assessment
3. **Hypothesis Testing**: Test business hypotheses with data
4. **Pattern Discovery**: Identify trends, anomalies, and correlations
5. **Validation**: Cross-validate findings across data sources
6. **Optimization**: Refine queries and analysis for performance
7. **Synthesis**: Combine findings into coherent insights

#### **Adaptive Flow**:
```
Start Investigation
    ↓
Schema Analysis → Find sales and region tables
    ↓
Data Exploration → Check data quality and date ranges
    ↓
Core Analysis → Calculate sales by region for last month
    ↓
Validation → Cross-check totals and verify completeness
    ↓
Synthesis → Generate business insights and recommendations
```

#### **Intelligent Adaptation**:
- **Dynamic Planning**: Adapt investigation based on findings
- **Error Recovery**: Handle missing data or query failures
- **Hypothesis Evolution**: Form new hypotheses based on discoveries
- **Deep Diving**: Investigate unexpected patterns or anomalies

**Example Execution**:
```sql
-- Step 1: Schema Analysis
SHOW TABLES LIKE '%sales%';
DESCRIBE sales_data;

-- Step 2: Data Exploration  
SELECT COUNT(*), MIN(date), MAX(date) FROM sales_data;

-- Step 3: Core Analysis
SELECT 
    region,
    SUM(sales_amount) as total_sales,
    COUNT(*) as transaction_count
FROM sales_data 
WHERE date >= '2025-05-01' AND date < '2025-06-01'
GROUP BY region
ORDER BY total_sales DESC;

-- Step 4: Validation
SELECT SUM(sales_amount) as grand_total FROM sales_data 
WHERE date >= '2025-05-01' AND date < '2025-06-01';
```

---

### **Phase 5: Insight Synthesis**
*Generate strategic recommendations and business insights*

**Input**: Investigation results and analysis findings

**Process**:
- **Business Context Integration**: Apply business knowledge to raw findings
- **Strategic Recommendation Generation**: Create actionable business recommendations
- **Confidence Assessment**: Determine confidence levels for each insight
- **Organizational Learning**: Store insights for future organizational knowledge
- **User-Specific Formatting**: Format results for user's role and preferences
- **Uncertainty Management**: Communicate limitations and data quality issues

**Output**: Final business intelligence report

```json
{
  "investigation_summary": "Analyzed last month's sales performance across regions",
  "key_insights": [
    {
      "insight": "West region generated $2.3M (45% of total sales)",
      "confidence": "high",
      "supporting_data": "1,247 transactions"
    },
    {
      "insight": "East region shows 15% decline vs previous month", 
      "confidence": "medium",
      "recommendation": "Investigate East region performance drivers"
    }
  ],
  "recommendations": [
    "Focus resources on replicating West region success",
    "Conduct deep-dive analysis on East region decline",
    "Consider seasonal factors in Q2 planning"
  ],
  "metadata": {
    "investigation_id": "inv_20250625_143022",
    "phases_completed": 5,
    "execution_time_minutes": 2.8,
    "queries_executed": 5,
    "confidence_score": 0.85
  }
}
```

---

## 🔄 **Cache Integration Throughout Flow**

### **Multi-Tier Cache Cascade**:

1. **Before Phase 1**: Check **Anthropic Cache** (50ms) - Organization-wide conversation sharing
2. **Before Phase 2**: Check **PostgreSQL Cache** (100ms) - Personal + Organizational insights  
3. **Before Phase 3**: Check **Semantic Cache** (200ms) - Pattern matching and learning
4. **After Phase 5**: **Store Results** in all applicable cache tiers with dynamic TTL

### **Cache Flow Example**:
```
"What were last month's sales?" 
    ↓
Check Anthropic Cache → MISS
    ↓  
Check PostgreSQL Cache → MISS
    ↓
Check Semantic Cache → MISS
    ↓
Execute Full 5-Phase Investigation (2.8 minutes)
    ↓
Store in ALL caches with TTL:
- Anthropic: 24 hours (managed by Anthropic)
- PostgreSQL Personal: 8 hours (daily sales reports)
- PostgreSQL Organizational: 12 hours (team sharing)
- Semantic: Indefinite (pattern learning)
```

---

## ⚡ **Performance Characteristics**

| **Scenario** | **Response Time** | **Cost** | **Source** |
|--------------|------------------|----------|------------|
| **Cache Hit (Anthropic)** | 50ms | $0.0015 | Cached conversation |
| **Cache Hit (PostgreSQL)** | 100ms | $0.003 | Cached insights |
| **Cache Hit (Semantic)** | 200ms | $0.005 | Pattern match |
| **Cache Miss (Full Investigation)** | 2-120 minutes | $0.015 | Complete analysis |

---

## 🎯 **Investigation Complexity Levels**

### **Simple** (2-5 minutes)
- Single data source
- Descriptive analysis
- Basic SQL queries
- Example: "Last month's total sales"

### **Moderate** (5-15 minutes) 
- Multiple data sources
- Comparative analysis
- Cross-validation required
- Example: "Sales performance vs last year by product category"

### **Complex** (15-45 minutes)
- Advanced analytics
- Multiple hypotheses
- Pattern discovery
- Example: "Customer churn analysis with predictive factors"

### **Comprehensive** (30-120 minutes)
- Full business intelligence
- Predictive modeling
- Cross-domain validation
- Example: "Complete market analysis with competitive positioning"

---

## 🚀 **Key Advantages**

### **Autonomous Intelligence**
- **Self-Planning**: Creates its own investigation strategy
- **Adaptive**: Changes approach based on findings  
- **Self-Healing**: Recovers from errors and data issues
- **Learning**: Improves from each investigation

### **Business-First Approach**
- **Business Context**: Understands business meaning, not just data
- **Strategic Insights**: Generates actionable recommendations
- **Domain Aware**: Applies industry and organizational knowledge
- **Role-Sensitive**: Adapts output to user's role and needs

### **Organizational Learning**
- **Knowledge Sharing**: Insights benefit entire organization
- **Pattern Recognition**: Learns from investigation patterns
- **Efficiency Gains**: Each investigation makes future ones faster
- **Collective Intelligence**: Builds organizational business intelligence

---

This five-phase flow transforms simple business questions into comprehensive strategic insights while learning and improving with each investigation!