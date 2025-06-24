# Agentic SQL Business Intelligence - Database-Boundary Architecture

> **Status**: Finalized Architecture - Database-Boundary Multi-Agent System
> **Purpose**: Autonomous business intelligence agent using database boundaries to determine single vs multi-agent execution

## Executive Summary

This document defines the finalized architecture for a Claude Sonnet 4-powered autonomous business intelligence agent. The system uses **database boundaries as the decision criteria** for single vs multi-agent execution, solving context fragmentation concerns while optimizing for both speed and intelligence.

### Key Requirements
- **Claude Code-like Behavior**: Autonomous tool selection and multi-step reasoning
- **100+ Business Tools**: Complete access to pre-built business functions
- **5000+ Word System Prompt**: Comprehensive instructions from `prompt_engineering/system_prompts.py`
- **4-Database MCP Architecture**: MariaDB, PostgreSQL, Supabase, Qdrant integration
- **Database-Boundary Intelligence**: Single agent for single database, multi-agent for cross-database queries
- **Two-Layer Caching**: Anthropic cache + PostgreSQL application cache

---

## Database-Boundary Multi-Agent Architecture

### Core Principle: Database Boundaries as Agent Boundaries

**Single Database = Single Agent** (context naturally coherent)
**Multiple Databases = Multi-Agent** (natural specialization boundaries)

### Why This Approach Works

#### Single Database Context
```
MariaDB Query: "Analyze sales, customers, products"
├── All tables in same schema
├── Relationships are defined (foreign keys)
├── Business logic is consistent
└── Single agent maintains full context naturally
```

**Cognition.ai's concerns don't apply** - context is naturally coherent within one database.

#### Multiple Database Context
```
Cross-Database Query: "Compare sales performance with support tickets and marketing campaigns"
├── MariaDB: Sales data
├── PostgreSQL: Support tickets  
├── Supabase: Marketing campaigns
└── Each database = natural agent boundary
```

**Multi-agent makes sense** - each agent is expert in its database domain.

### Database Scope Decision Logic

```python
if query_spans_multiple_databases(query):
    # Multi-agent orchestration by database boundaries
    results = await asyncio.gather(
        mariadb_specialist_agent.handle(query_part),
        postgres_specialist_agent.handle(query_part),
        supabase_specialist_agent.handle(query_part),
        qdrant_specialist_agent.handle(query_part)
    )
    return orchestrator_agent.synthesize(results)

else:
    # Single specialist agent for single database
    return await single_database_agent.handle(query)
```

---

## Enhanced 4-Tier Architecture with Database-Boundary Logic

### Tier 1: Two-Layer Cache System (50-100ms)

#### Layer 1a: Anthropic Built-in Cache (50ms)
```python
async def tier1a_anthropic_cache(user_query: str) -> Optional[dict]:
    """Anthropic's automatic cache - fastest possible response"""
    
    # Automatic via API cache headers
    cache_result = await check_anthropic_cache(user_query)
    
    if cache_result.hit and cache_result.confidence > 0.95:
        return {
            "response": cache_result.content,
            "source": "anthropic_cache",
            "response_time": "50ms",
            "cost": "90% savings"
        }
    
    return None  # Proceed to Layer 1b
```

#### Layer 1b: PostgreSQL Application Cache - Hybrid Approach (100ms)
```python
async def tier1b_hybrid_postgres_cache(user_query: str, user_id: str) -> Optional[dict]:
    """Hybrid personal + organizational cache with smart TTL"""
    
    query_hash = hash_query(user_query)
    
    # Check personal cache first (most relevant)
    personal_result = await postgres_agent.check_personal_cache(
        user_id=user_id,
        query_hash=query_hash,
        current_time=datetime.now()
    )
    
    if personal_result.hit and not personal_result.expired:
        await postgres_agent.increment_hit_count(personal_result.id, cache_type="personal")
        return personal_result
    
    # Check organizational cache (shared knowledge)
    organizational_result = await postgres_agent.check_organizational_cache(
        query_hash=query_hash,
        organization_id=get_user_organization(user_id),
        current_time=datetime.now()
    )
    
    if organizational_result.hit and not organizational_result.expired:
        await postgres_agent.increment_hit_count(organizational_result.id, cache_type="organizational")
        return organizational_result
    
    return None  # Proceed to Tier 2
```

**PostgreSQL Cache Schema - Hybrid Approach:**
```sql
-- Personal Cache Table
CREATE TABLE personal_cache (
    id SERIAL PRIMARY KEY,
    user_id VARCHAR(64),
    query_hash VARCHAR(64),
    original_query TEXT,
    response_data JSONB,
    complexity_type VARCHAR(20),        -- simple/analytical/computational/investigative
    database_scope VARCHAR(20),         -- single/multi
    databases_used TEXT[],              -- [mariadb, postgres, supabase, qdrant]
    created_at TIMESTAMP DEFAULT NOW(),
    expires_at TIMESTAMP,
    hit_count INTEGER DEFAULT 0,
    user_satisfaction BOOLEAN,
    UNIQUE(user_id, query_hash)
);

-- Organizational Shared Cache Table
CREATE TABLE organizational_cache (
    id SERIAL PRIMARY KEY,
    query_hash VARCHAR(64) UNIQUE,
    original_query TEXT,
    response_data JSONB,
    complexity_type VARCHAR(20),
    database_scope VARCHAR(20),
    databases_used TEXT[],
    created_at TIMESTAMP DEFAULT NOW(),
    expires_at TIMESTAMP,
    hit_count INTEGER DEFAULT 0,
    user_satisfaction BOOLEAN,
    created_by VARCHAR(64),             -- Who first created this cache entry
    organization_id VARCHAR(64)         -- Organization scope
);

-- Cache Strategy: Check personal first, then organizational
-- Time-based TTL by data type:
-- Sales data (24h), Inventory (4h), Real-time metrics (1h)
-- User context: Personal vs shared based on query sensitivity
```

### Tier 2: Complexity + Database Scope Detection (200-500ms)

```python
async def tier2_enhanced_analysis(user_query: str) -> QueryAnalysis:
    """Analyze both complexity AND database scope"""
    
    analysis = await anthropic_client.messages.create(
        model="claude-sonnet-4-20250514",
        system="""
        Analyze this business query across multiple dimensions:
        
        COMPLEXITY TYPES:
        - simple: Direct data retrieval ("What are today's sales?")
        - analytical: Trend analysis, comparisons ("Why did sales drop?")
        - computational: Scenarios, modeling ("What if costs increase 10%?")
        - investigative: Root cause, recommendations ("How to fix customer churn?")
        
        DATABASE SCOPE:
        - MariaDB: Business operations (sales, customers, products, orders)
        - PostgreSQL: Agent memory and sessions
        - Supabase: Additional operations and data processing
        - Qdrant: Vector search and semantic analysis
        
        Return which databases are needed and whether single or multi-agent approach.
        """,
        messages=[{"role": "user", "content": f"Analyze: {user_query}"}],
        max_tokens=500
    )
    
    return QueryAnalysis(
        complexity=analysis.complexity_type,
        database_scope=analysis.database_scope,
        databases_needed=analysis.database_mapping,
        agent_strategy="single" if analysis.database_scope == "single" else "multi"
    )
```

### Tier 3: Context Building by Database Scope (100-300ms)

```python
async def tier3_scope_aware_context(query: str, analysis: QueryAnalysis) -> dict:
    """Build context based on database scope"""
    
    if analysis.agent_strategy == "single":
        # Single database - comprehensive context within boundary
        return await build_single_database_context(
            query=query,
            database=analysis.databases_needed[0],
            complexity=analysis.complexity
        )
    
    else:
        # Multi-database - context for each database boundary
        return await build_multi_database_context(
            query=query,
            databases=analysis.databases_needed,
            complexity=analysis.complexity
        )

async def build_single_database_context(query: str, database: str, complexity: str) -> dict:
    """Complete context within single database boundary"""
    
    if database == "mariadb":
        return {
            "optimization_mode": f"{complexity}_mariadb",
            "business_schema": await get_mariadb_schema(),
            "business_rules": await get_business_logic_rules(),
            "successful_patterns": await get_mariadb_query_patterns(query),
            "performance_hints": await get_mariadb_optimization_hints()
        }
    
    elif database == "postgres":
        return {
            "optimization_mode": f"{complexity}_postgres",
            "memory_schema": await get_postgres_schema(),
            "session_context": await get_current_session_context(),
            "memory_patterns": await get_postgres_query_patterns(query)
        }
    
    # Similar for supabase and qdrant...

async def build_multi_database_context(query: str, databases: list, complexity: str) -> dict:
    """Context for cross-database orchestration"""
    
    context = {"optimization_mode": f"{complexity}_multi_database"}
    
    for database in databases:
        context[f"{database}_context"] = await build_single_database_context(
            query, database, complexity
        )
    
    context["integration_strategy"] = await plan_cross_database_integration(
        query, databases, complexity
    )
    
    return context
```

### Tier 4: Database-Boundary Agent Selection (2-15s)

```python
async def tier4_database_boundary_execution(
    query: str, 
    analysis: QueryAnalysis, 
    context: dict
) -> dict:
    """Execute based on database boundaries"""
    
    if analysis.agent_strategy == "single":
        # Single database agent with full context
        specialist_agent = get_database_specialist(analysis.databases_needed[0])
        return await specialist_agent.handle_with_complete_context(query, context)
    
    else:
        # Multi-agent orchestration by database boundaries
        return await orchestrate_cross_database_query(query, analysis, context)
```

---

## Database Specialist Agents

### MariaDB Specialist Agent
```python
class MariaDBSpecialistAgent:
    """Expert in business operations data (sales, customers, products)"""
    
    async def handle_with_complete_context(self, query: str, context: dict):
        """Single agent with full MariaDB context"""
        
        # Complete business context within MariaDB boundary
        mariadb_context = {
            "business_schema": context.business_schema,
            "business_rules": context.business_rules,
            "optimization_hints": context.performance_hints,
            "query_patterns": context.successful_patterns
        }
        
        # Enhanced system prompt for MariaDB specialization
        system_prompt = f"""
        {load_base_system_prompt()}
        
        MARIADB BUSINESS EXPERT:
        You are a specialist in business operations data analysis.
        
        Available Business Schema: {mariadb_context['business_schema']}
        Business Rules: {mariadb_context['business_rules']}
        Optimization Hints: {mariadb_context['optimization_hints']}
        
        Execute this query with full context awareness of business operations.
        """
        
        return await claude_mariadb_execution(query, system_prompt, mariadb_context)
```

### PostgreSQL Specialist Agent
```python
class PostgreSQLSpecialistAgent:
    """Expert in agent memory and session data"""
    
    async def handle_with_complete_context(self, query: str, context: dict):
        """Single agent with full PostgreSQL context"""
        
        postgres_context = {
            "memory_schema": context.memory_schema,
            "session_context": context.session_context,
            "memory_patterns": context.memory_patterns
        }
        
        system_prompt = f"""
        {load_base_system_prompt()}
        
        POSTGRESQL MEMORY EXPERT:
        You are a specialist in agent memory and session management.
        
        Available Memory Schema: {postgres_context['memory_schema']}
        Current Session Context: {postgres_context['session_context']}
        
        Execute this query with full awareness of agent memory patterns.
        """
        
        return await claude_postgres_execution(query, system_prompt, postgres_context)
```

### Supabase Specialist Agent
```python
class SupabaseSpecialistAgent:
    """Expert in additional operations data"""
    
    async def handle_with_complete_context(self, query: str, context: dict):
        """Single agent with full Supabase context"""
        
        return await claude_supabase_execution(query, context)
```

### Qdrant Specialist Agent
```python
class QdrantSpecialistAgent:
    """Expert in vector search and embeddings"""
    
    async def handle_with_complete_context(self, query: str, context: dict):
        """Single agent with full Qdrant context"""
        
        return await claude_qdrant_execution(query, context)
```

---

## Cross-Database Orchestration

```python
async def orchestrate_cross_database_query(
    query: str, 
    analysis: QueryAnalysis, 
    context: dict
) -> dict:
    """Multi-agent orchestration respecting database boundaries"""
    
    # Parallel execution by database boundary
    database_tasks = []
    
    if "mariadb" in analysis.databases_needed:
        database_tasks.append(
            mariadb_agent.analyze_business_aspects(query, context.mariadb_context)
        )
    
    if "postgres" in analysis.databases_needed:
        database_tasks.append(
            postgres_agent.analyze_memory_aspects(query, context.postgres_context)
        )
    
    if "supabase" in analysis.databases_needed:
        database_tasks.append(
            supabase_agent.analyze_operations_aspects(query, context.supabase_context)
        )
    
    if "qdrant" in analysis.databases_needed:
        database_tasks.append(
            qdrant_agent.search_semantic_patterns(query, context.qdrant_context)
        )
    
    # Execute all database agents in parallel
    database_results = await asyncio.gather(*database_tasks)
    
    # Orchestrator synthesizes cross-database insights
    return await cross_database_orchestrator.synthesize(
        query=query,
        database_findings=database_results,
        integration_strategy=context.integration_strategy
    )

class CrossDatabaseOrchestrator:
    """Synthesizes results from multiple database specialists"""
    
    async def synthesize(self, query: str, database_findings: list, integration_strategy: dict):
        """Combine insights from different database domains"""
        
        synthesis_prompt = f"""
        {load_base_system_prompt()}
        
        CROSS-DATABASE SYNTHESIS EXPERT:
        You coordinate insights from multiple database specialists.
        
        Original Query: {query}
        Integration Strategy: {integration_strategy}
        
        Database Findings:
        {format_database_findings(database_findings)}
        
        Synthesize these findings into coherent business insights.
        Identify correlations, patterns, and actionable recommendations.
        """
        
        return await claude_synthesis_execution(query, synthesis_prompt, database_findings)
```

---

## Implementation Examples

### Single Database Query (Single Agent)

**Query**: "What are our top-selling products by category last month?"

```
Flow:
├── Tier 1a: Anthropic Cache → MISS
├── Tier 1b: PostgreSQL Cache → MISS  
├── Tier 2: Analysis → {database: "mariadb", strategy: "single", complexity: "simple"}
├── Tier 3: MariaDB Context → {business_schema, optimization_hints}
└── Tier 4: MariaDB Specialist Agent → Complete analysis within business data boundary

Execution: MariaDB Agent with full business context
Result: "Top products: Electronics (45%), Clothing (32%), Books (23%)"
Performance: ~2-3 seconds
```

### Multi-Database Query (Multi-Agent)

**Query**: "Correlate sales performance with customer support satisfaction and semantic search trends"

```
Flow:
├── Tier 1a: Anthropic Cache → MISS
├── Tier 1b: PostgreSQL Cache → MISS
├── Tier 2: Analysis → {databases: ["mariadb", "postgres", "qdrant"], strategy: "multi"}
├── Tier 3: Multi-Database Context → Contexts for each database + integration strategy
└── Tier 4: Multi-Agent Orchestration

Parallel Execution:
├── MariaDB Agent: Sales performance analysis
├── PostgreSQL Agent: Support ticket patterns  
└── Qdrant Agent: Customer sentiment search

Orchestrator Synthesis: "Sales decline correlates with 40% increase in support tickets and negative sentiment spike"
Performance: ~5-10 seconds
```

---

## Cache Strategy Implementation

### Cache Storage After Successful Execution

```python
async def cache_successful_result(
    query: str, 
    result: dict, 
    analysis: QueryAnalysis,
    execution_time: float
):
    """Store successful results in PostgreSQL cache"""
    
    # Calculate TTL based on complexity and data volatility
    ttl_hours = {
        "simple": 24,        # Simple queries cache longer
        "analytical": 12,    # Medium TTL for analytical  
        "computational": 6,  # Shorter for computational (data may change)
        "investigative": 2   # Shortest for complex investigations
    }
    
    cache_entry = {
        "query_hash": hash_query(query),
        "original_query": query,
        "response_data": result,
        "complexity_type": analysis.complexity,
        "database_scope": analysis.agent_strategy,
        "databases_used": analysis.databases_needed,
        "expires_at": datetime.now() + timedelta(hours=ttl_hours[analysis.complexity]),
        "execution_time_saved": execution_time
    }
    
    await postgres_agent.store_cache_entry(cache_entry)
```

### Complete Hybrid Cache Flow

```
User Query (with user_id)
    ↓
Tier 1a: Anthropic Cache (50ms) → Hit? Return (organization-wide cache benefit)
    ↓ (miss)
Tier 1b: Hybrid PostgreSQL Cache (100ms)
    ├── Personal Cache Check → Hit? Return (user-specific results)
    ├── Organizational Cache Check → Hit? Return (team knowledge sharing)
    └── Both MISS ↓
Tier 2: Complexity + Database Scope Detection (200-500ms) 
Tier 3: Context Building (100-300ms)
Tier 4: Database-Boundary Execution (2-15s)
    ↓
Store Result → Personal + Organizational Cache (based on query sensitivity)
    ├── Sensitive queries → Personal cache only
    └── Business insights → Both personal + organizational cache
```

---

## Master Implementation

```python
@mcp.tool()
async def database_boundary_business_agent(user_query: str, user_id: str) -> dict:
    """Complete database-boundary intelligent business agent with hybrid caching"""
    
    # Tier 1a: Anthropic cache (organization-wide)
    anthropic_result = await tier1a_anthropic_cache(user_query)
    if anthropic_result:
        return anthropic_result
    
    # Tier 1b: Hybrid PostgreSQL cache (personal + organizational)
    postgres_cache_result = await tier1b_hybrid_postgres_cache(user_query, user_id)
    if postgres_cache_result:
        return postgres_cache_result
    
    # Tier 2: Enhanced analysis (complexity + database scope) - Sonnet 4.0
    analysis = await tier2_enhanced_analysis(user_query)
    
    # Tier 3: Scope-aware context building
    context = await tier3_scope_aware_context(user_query, analysis)
    
    # Tier 4: Database-boundary execution
    result = await tier4_database_boundary_execution(user_query, analysis, context)
    
    # Cache successful results (hybrid storage)
    await cache_successful_result_hybrid(user_query, result, analysis, result.execution_time, user_id)
    
    return {
        "query": user_query,
        "user_id": user_id,
        "analysis": analysis,
        "execution_strategy": analysis.agent_strategy,
        "databases_used": analysis.databases_needed,
        "response": result,
        "performance_metrics": {
            "tier1a_anthropic": "miss",
            "tier1b_personal": "miss", 
            "tier1b_organizational": "miss",
            "tier2_analysis": f"{analysis.confidence:.1%} confidence",
            "tier3_context": context["optimization_mode"],
            "tier4_execution": analysis.agent_strategy,
            "total_time": result.execution_time,
            "cache_strategy": "hybrid_personal_organizational"
        }
    }
```

---

## Performance Analysis

### Response Time Characteristics

| Query Type | Cache Hit | Single DB | Multi-DB | 
|------------|-----------|-----------|----------|
| Simple | 50-100ms | 2-3s | N/A |
| Analytical | 50-100ms | 3-5s | 5-8s |
| Computational | 50-100ms | 5-8s | 8-12s |
| Investigative | 50-100ms | 8-12s | 10-15s |

### Database Scope Distribution (Estimated)

- **Single Database**: 70-80% of queries (MariaDB business queries most common)
- **Multi-Database**: 20-30% of queries (complex investigations, correlations)

### Cache Effectiveness & Team Collaboration

**Anthropic's Built-in Cache (Organization-Level):**
- **Scope**: Organization-wide sharing across all team members using same API key
- **Hit Rate**: 15-25% individual, 40-60% organization-wide
- **Benefits**: 90% cost savings, team prompt engineering ROI
- **Storage**: Anthropic's servers (automatic, invisible to application)
- **Duration**: 5 minutes standard, 1 hour enterprise

**PostgreSQL Application Cache (Hybrid Approach):**
- **Personal Cache**: Individual user queries and sensitive data
- **Organizational Cache**: Shared business insights and common patterns
- **Hit Rate**: 35-45% personal + 25-35% organizational = 60-80% combined
- **Team Benefits**: Knowledge building, cache warming, learning acceleration

**Team Collaboration Benefits:**
- **Prompt Engineering ROI**: One person optimizes prompt, whole team benefits
- **Knowledge Building**: Each query improves cache hit rate for everyone  
- **Cost Efficiency**: Team's combined usage creates better cache performance
- **Learning Acceleration**: New team members immediately benefit from existing cache

**Cache Strategy for Team Efficiency:**
- **Morning Queries**: First person "warms up" cache for entire team
- **Common Questions**: FAQ-style queries become instant for everyone after first ask
- **Schema Queries**: Database exploration cached for all analysts
- **Business Rules**: Complex business logic cached across all team investigations

**PostgreSQL Cache Optimization:**
- **Time-based TTL**: Sales data (24h), Inventory (4h), Real-time metrics (1h)
- **User Context**: Personal vs shared cache based on query sensitivity
- **Data Freshness**: Balance between speed and accuracy requirements  
- **Storage Costs**: PostgreSQL storage vs API costs optimization

---

## Benefits of Database-Boundary Architecture

### 1. Solves Context Fragmentation (Cognition.ai Concerns)
- ✅ Context preserved within natural database boundaries
- ✅ No fragmentation within coherent data domains  
- ✅ Specialization aligns with data architecture

### 2. Leverages Multi-Agent Benefits (Anthropic Patterns)
- ✅ Multi-agent for genuinely independent domains
- ✅ Parallel processing across database boundaries
- ✅ Orchestrated synthesis of cross-domain insights

### 3. Natural Scalability
- ✅ Easy to add new database specialists
- ✅ Clear responsibility boundaries
- ✅ Database expertise naturally concentrated

### 4. Practical Implementation
- ✅ Database boundaries are well-defined
- ✅ Agent responsibilities are clear
- ✅ Context scope is naturally limited

---

## Implementation Timeline

### Week 1-2: Foundation
- **Tier 1-2**: Cache + Enhanced Analysis with Database Scope Detection
- **Database Scope Logic**: Implement query-to-database mapping
- **PostgreSQL Cache**: Set up application-level caching infrastructure

### Week 3-4: Single Database Specialists  
- **Tier 3**: Single database context building
- **MariaDB Agent**: Business operations specialist
- **PostgreSQL Agent**: Memory and sessions specialist
- **Basic Supabase/Qdrant Agents**: Foundational implementations

### Week 5-6: Cross-Database Orchestration
- **Multi-Agent Coordination**: Cross-database orchestration
- **Synthesis Agent**: Results combination and correlation analysis
- **Integration Testing**: Single vs multi-agent decision accuracy

### Week 7-8: Optimization and Production Readiness
- **Performance Tuning**: Cache optimization, response time improvements
- **Quality Assurance**: Accuracy testing, error handling
- **Monitoring**: Comprehensive metrics and alerting

---

## Conclusion

The **Database-Boundary Multi-Agent Architecture** provides the optimal solution for autonomous business intelligence by:

- **Natural Boundaries**: Database boundaries determine agent boundaries, solving context fragmentation
- **Intelligent Routing**: Single agent for single database, multi-agent for cross-database queries
- **Performance Optimization**: Two-layer caching + complexity-aware execution
- **Scalable Design**: Easy to add databases and specialists while maintaining coherent context

This architecture leverages the best insights from both Anthropic's multi-agent research and Cognition.ai's context preservation principles, creating a system that is both sophisticated and reliable for production business intelligence workloads.