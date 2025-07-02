# Agentic SQL Backend

**Autonomous Business Intelligence System** - AI-powered backend that transforms natural language business questions into strategic insights through a sophisticated 6-phase investigation workflow.

## Architecture Overview

The system implements a **6-Phase Investigation Workflow** that progressively transforms user queries into comprehensive business intelligence:

```
Natural Language Query -> Strategic Business Intelligence

Phase 1: Multi-Tier Caching  -> Cache Hit or Investigation Needed
Phase 2: Query Processing     -> Business Intent + Context
Phase 3: Strategy Planning    -> Investigation Methodology  
Phase 4: Service Orchestration -> Coordinated Database Services
Phase 5: Investigation Execution -> Raw Investigation Findings
Phase 6: Insight Synthesis   -> Strategic Recommendations
```

## 6-Phase Architecture

### Phase 1: Multi-Tier Caching
**Intelligent cache cascade for instant responses**

**3-Tier Cache Architecture**:
- **Tier 1a - Anthropic Cache** (50ms): Organization-wide conversation caching, 90% cost savings
- **Tier 1b - PostgreSQL Cache** (100ms): Personal + organizational insights with permission control
- **Semantic Cache**: Intent-based caching for business intelligence patterns

**Cache Management Features**:
- **Dynamic TTL Management**: Business context-aware cache expiration
- **Cache Warming Engine**: Predictive cache population for common queries
- **Permission-Aware Sharing**: Role-based cache access and organizational learning
- **Intelligent Tier Selection**: Optimal cache tier based on data volatility
- **Cross-User Learning**: Accelerated knowledge sharing across organization

**Cache Hit Flow**:
```
Query -> Semantic Hash -> Cache Lookup -> Hit? -> Instant Response (50-100ms)
                                      -> Miss? -> Continue to Phase 2
```

**Output**: Cached response (on hit) or investigation trigger (on miss)

### Phase 2: Query Processing
**Transform natural language to business intent**

- **Business Intent Extraction**: Parse meaning and context from user queries
- **Domain Classification**: Classify queries (sales, finance, hr, production)
- **Semantic Hashing**: Generate unique cache lookup optimization
- **Complexity Analysis**: Determine investigation complexity level
- **User Context Integration**: Consider role, permissions, department
- **Organization Context**: Apply business rules and data classification

**Output**: Structured business intent with complexity indicators

### Phase 3: Strategy Planning
**Create investigation methodology and execution plan**

- **Complexity Determination**: Simple -> Moderate -> Complex -> Comprehensive
- **Methodology Selection**: Descriptive, Diagnostic, Predictive, Prescriptive
- **Multi-Phase Planning**: Break complex investigations into phases
- **Resource Estimation**: Calculate time, queries, services needed
- **Risk Assessment**: Identify potential data quality issues
- **Service Selection**: Determine optimal database services

**Output**: Complete investigation strategy with execution plan

### Phase 4: Service Orchestration
**Coordinate database services and MCP tools**

- **Intelligent Service Selection**:
  - Simple (< 0.3): MariaDB only
  - Analytical (< 0.6): MariaDB + PostgreSQL
  - Computational (< 0.8): + LanceDB
  - Investigative (≥ 0.8): + GraphRAG (all services)
- **MCP Tool Coordination**: Multiple Computer Protocol integration
- **Connection Management**: Database connections and health monitoring
- **Schema Analysis**: Understand tables and relationships
- **Business Logic Integration**: Apply domain knowledge

**Output**: Configured services ready for investigation

### Phase 5: Investigation Execution
**Autonomous multi-step analysis with adaptive methodology**

**7-Step Investigation Flow**:
1. **Schema Analysis** - AI discovers database structure
2. **Data Exploration** - AI assesses data quality and patterns
3. **Hypothesis Generation** - AI generates business hypotheses
4. **Core Analysis** - AI executes analysis using services
5. **Pattern Discovery** - AI identifies anomalies and trends
6. **Cross-Validation** - AI validates across data sources
7. **Results Synthesis** - AI combines findings

**Adaptive AI Reasoning**:
- Hypothesis evolution and methodology adaptation
- Intelligent error recovery and fallback strategies
- Business context application throughout process

**Output**: Raw investigation findings with confidence scores

### Phase 6: Insight Synthesis
**Generate strategic recommendations and business insights**

- **Strategic Insight Generation**: Transform findings into actionable intelligence
- **Multi-dimensional Analysis**: Operational, tactical, strategic integration
- **Business Impact Assessment**: Financial, operational, strategic, risk impact
- **Actionable Recommendations**: Specific recommendations with priorities
- **Role-Specific Formatting**: Adapt for executives, managers, analysts
- **Organizational Learning**: Capture insights for future knowledge
- **Confidence Assessment**: Reliability levels for each insight

**Output**: Strategic business intelligence with recommendations

## Module Structure

```
app/
├── cache/                   # Phase 1: Multi-tier caching system
│   ├── cache_manager.py        # Cache orchestrator
│   ├── anthropic_cache.py      # Tier 1a: Organization-wide caching
│   ├── postgresql_cache.py     # Tier 1b: Permission-aware caching
│   ├── semantic_cache.py       # Intent-based business intelligence cache
│   ├── cache_warming.py        # Predictive cache population
│   └── ttl_manager.py          # Dynamic TTL management
├── core/                    # Core business logic and orchestration
│   ├── service_orchestration/   # Phase 4: Service coordination
│   └── test_integration.py     # Integration testing
├── investigation/           # Phase 5: Investigation execution
│   ├── config.py
│   ├── runner.py
│   └── README.md
├── insight_synthesis/       # Phase 6: Strategic insight generation
│   ├── config.py
│   ├── runner.py
│   └── README.md
├── lance_db/               # Vector embeddings and similarity search
│   ├── embedded_sql/           # SQL query embeddings
│   ├── patterns/               # Business intelligence patterns
│   ├── runner.py
│   └── README.md
├── model/                  # AI model management and configuration
│   ├── config.py
│   ├── runner.py
│   └── settings.env
└── config/                 # Global configuration (databases, infrastructure)
    ├── cfg_databases.py
    ├── cfg_infrastructure.py
    └── __init__.py
```

## Self-Contained Module Pattern

Each module follows the **self-contained architecture**:

```
module_name/
├── config.py              # Pydantic settings (no hardcoded defaults)
├── settings.env           # Environment variables
├── module_logging.py      # Local logging
├── runner.py              # Main business logic orchestrator
├── __init__.py           # Clean package exports
├── test_standalone.py    # Independent testing
└── README.md             # Complete documentation
```

**Benefits**:
- **Independent execution**: Each module runs standalone
- **Zero external dependencies**: Beyond module boundary
- **Configuration-driven**: Type-safe environment management
- **Production-grade reliability**: Health monitoring and fallbacks

## Technology Stack

### Core Technologies
- **Python 3.11+**: Modern language features and performance
- **FastAPI**: High-performance async web framework
- **Pydantic Settings**: Type-safe configuration management
- **uv**: Fast Python package management

### Database Services
- **MariaDB**: Primary operational database (MCP)
- **PostgreSQL**: Analytics and caching (MCP + Supabase)
- **LanceDB**: Vector embeddings and similarity search
- **GraphRAG**: Knowledge graph and relationship analysis

### AI and Intelligence
- **Claude Models**: Advanced reasoning and analysis
- **BGE-M3**: Multilingual embedding model
- **Sentence Transformers**: Text embedding generation
- **MCP (Model Context Protocol)**: AI tool integration

### Infrastructure
- **React Frontend**: Modern UI with TypeScript
- **WebSocket**: Real-time investigation streaming
- **Zeabur Deployment**: Cloud platform with volume support
- **Git Hooks**: Pre-commit quality checks

## Development Standards

### Dependency Management
- **ONLY use uv**: Never pip for package management
- **Installation**: `uv add package`
- **Running**: `uv run tool`
- **MCP-based**: No SQLAlchemy/asyncpg needed

### Code Quality
- **Formatting**: `uv run --frozen ruff format .`
- **Linting**: `uv run --frozen ruff check . --fix`
- **Type Checking**: `uv run --frozen pyright`
- **Pre-commit**: Automated quality checks

### Documentation Standards
- **NO EMOJIS**: Use only standard ASCII characters
- **Professional formatting**: Markdown headers, bullets, tables
- **Clear sections**: Use `##`, `###`, `**bold**`, `*italic*`
- **Descriptive text**: Replace symbols with words

## Use Cases by Complexity

### Simple Queries (30 seconds)
**Example**: "What are today's sales numbers?"
- **Phases**: 1 (cache miss), 2, 3, 4 (MariaDB only), 5 (steps 1,2,4,6,7), 6
- **Output**: Direct numerical results with confidence

### Analytical Queries (2-5 minutes)  
**Example**: "How do Q3 sales compare to last year by region?"
- **Phases**: All 6 phases, MariaDB + PostgreSQL
- **Output**: Comparative analysis with business insights

### Complex Investigations (15-30 minutes)
**Example**: "Why did customer satisfaction drop in Q2?"
- **Phases**: All 6 phases, all 7 investigation steps, multiple iterations
- **Output**: Causal analysis with recommended actions

### Strategic Reviews (30-60 minutes)
**Example**: "Analyze market position and competitive threats"
- **Phases**: All 6 phases with extensive cross-validation
- **Output**: Strategic intelligence with actionable recommendations

## Performance Characteristics

| Query Type | Services Used | Duration | Confidence | Cache Hit Rate |
|------------|---------------|----------|------------|----------------|
| Simple     | MariaDB       | 30s      | 95%+       | 80%+           |
| Analytical | MariaDB+PG    | 2-5m     | 90%+       | 60%+           |
| Complex    | +LanceDB      | 15-30m   | 80%+       | 40%+           |
| Strategic  | +GraphRAG     | 30-60m   | 75%+       | 20%+           |

## Getting Started

### Environment Setup
```bash
# Install uv package manager
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and setup
git clone <repository>
cd agentic_sql/app

# Install dependencies
uv sync

# Configure environment
cp model/settings.env.example model/settings.env
# Edit with your API keys and database connections
```

### Module Testing
```bash
# Test individual modules
cd app/model && python test_standalone.py
cd app/lance_db && python test_standalone.py
cd app/investigation && python test_standalone.py

# Run integration tests
cd app && python core/test_integration.py
```

### Development Workflow
```bash
# Format and check code
uv run --frozen ruff format .
uv run --frozen ruff check . --fix
uv run --frozen pyright

# Run specific module
cd app/model && python runner.py
cd app/lance_db && python ingest_patterns.py
```

## Key Innovations

1. **Autonomous Investigation**: AI conducts genuine business analysis, not just query execution

2. **Adaptive Methodology**: Investigation approach evolves based on real-time findings

3. **Complexity-Driven Resource Allocation**: Simple queries use minimal resources, complex investigations get full power

4. **Organizational Learning**: Every investigation improves system knowledge

5. **Self-Contained Architecture**: Modules are completely independent and production-ready

6. **Multi-tier Caching**: Intelligent cache cascade for optimal performance

7. **Business Intelligence Focus**: Domain knowledge integrated throughout the investigation process

## Contributing

1. **Follow Standards**: Use uv, ruff, pyright, and no-emoji documentation
2. **Self-Contained Modules**: Maintain independence and standalone execution
3. **Configuration-Driven**: No hardcoded business values
4. **Test Coverage**: Include standalone tests for all modules
5. **Business Context**: Apply domain knowledge and business reasoning

## Summary

The Agentic SQL Backend represents a paradigm shift from traditional BI tools to autonomous business intelligence. By combining advanced AI reasoning with sophisticated database orchestration, it delivers strategic insights that help organizations make data-driven decisions with confidence.

The 6-phase architecture ensures optimal resource utilization while maintaining the flexibility to handle everything from simple operational queries to complex strategic investigations.