# Agentic SQL Backend Architecture

> The brain and nervous system of the autonomous SQL investigation agent

## 🏗️ Backend Architecture Overview

This backend implements a Claude Code-style autonomous SQL agent that investigates data through multiple iterations until complete insights are discovered.

## 📁 Directory Structure

```python
app/
├── __init__.py
├── main.py                     # FastAPI application entry point
├── config.py                   # Configuration management
│
├── core/                       # Core autonomous agent logic
│   ├── __init__.py
│   ├── agent.py               # Single autonomous SQL agent
│   ├── investigation.py       # Investigation loop logic
│   ├── planner.py            # Task planning and strategy
│   └── memory.py             # Session and context management
│
├── mcp/                       # MCP Client Infrastructure
│   ├── __init__.py
│   ├── client_manager.py     # Manages all MCP client connections
│   ├── mariadb_client.py     # MariaDB MCP client wrapper
│   ├── postgres_client.py    # PostgreSQL MCP client wrapper
│   ├── supabase_client.py    # Supabase MCP client wrapper
│   ├── qdrant_client.py      # Qdrant vector MCP client wrapper
│   └── mcp.json             # MCP server configuration (4 databases)
│
├── database/                  # Database Layer (MCP-based)
│   ├── __init__.py
│   ├── models.py            # Data models and schemas
│   ├── memory_manager.py    # PostgreSQL memory operations via MCP
│   └── operations.py        # Safe SQL execution via MCP clients
│
├── model/                    # AI Model Integration
│   ├── __init__.py
│   ├── sonnet.py           # Claude Sonnet 4 integration
│   ├── embeddings.py       # BGE-M3 embedding service
│   └── prompts.py          # Prompt templates
│
├── rag/                     # Knowledge Base
│   ├── __init__.py
│   ├── qdrant_client.py    # Qdrant vector operations
│   ├── indexing.py         # Document indexing pipeline
│   ├── retrieval.py        # Semantic search logic
│   └── faq_matcher.py      # FAQ pattern matching
│
├── evals/                   # Evaluation System
│   ├── __init__.py
│   ├── sql_validator.py    # SQL syntax and logic validation
│   ├── result_evaluator.py # Result quality assessment
│   ├── performance.py      # Query performance analysis
│   └── metrics.py          # Success metrics tracking
│
├── guardrails/             # Safety Systems
│   ├── __init__.py
│   ├── query_guard.py      # SQL injection, dangerous operations
│   ├── rate_limiter.py     # API and query rate limiting
│   ├── permission.py       # Access control
│   └── data_privacy.py     # PII detection and masking
│
├── prompt_engineering/      # Advanced Prompting
│   ├── __init__.py
│   ├── templates.py        # Prompt templates
│   ├── chain_of_thought.py # CoT prompting for complex queries
│   ├── few_shot.py         # Dynamic example selection
│   └── adaptive.py         # Context-aware prompting
│
├── asynchronous/           # Async Patterns
│   ├── __init__.py
│   ├── task_queue.py       # Background task management
│   ├── parallel.py         # Parallel investigation execution
│   └── websocket.py        # Real-time client updates
│
├── computer_use/           # Future: Agentic UI Interaction
│   └── __init__.py         # Reserved for future features
│
├── api/                    # API Layer
│   ├── __init__.py
│   ├── routes/
│   │   ├── __init__.py
│   │   ├── agent.py        # Agent interaction endpoints
│   │   ├── investigation.py # Investigation management
│   │   └── faq.py          # FAQ management
│   ├── websocket.py        # WebSocket handlers
│   └── dependencies.py     # FastAPI dependencies
│
└── utils/                  # Utilities
    ├── __init__.py
    ├── logging.py          # Structured logging
    ├── monitoring.py       # Metrics and observability
    └── exceptions.py       # Custom exceptions
```

## 🧠 Core Components

### 1. Autonomous SQL Agent (core/agent.py)
The main brain that orchestrates investigations.

```python
class AutonomousSQLAgent:
    """Claude Code-style autonomous SQL investigation agent"""
    
    def __init__(self):
        self.planner = InvestigationPlanner()
        self.memory = MemoryManager()
        self.mcp_client = MCPClient()
        self.sonnet = SonnetClient()
    
    async def investigate(self, user_query: str, session_id: str):
        """Autonomous investigation loop"""
        # 1. Plan investigation strategy
        # 2. Execute tasks iteratively
        # 3. Adapt based on findings
        # 4. Continue until complete insights
```

### 2. MCP Dual Architecture
**External Database MCP Servers** (via MCP Python SDK):
```python
# mcp.json - External MCP servers for databases
{
  "mcpServers": {
    "postgres": {"command": "npx", "args": ["-y", "@modelcontextprotocol/server-postgres", "${POSTGRES_URL}"]},
    "mariadb": {"command": "npx", "args": ["-y", "mariadb-mcp-server"], "env": {...}},
    "supabase": {"command": "npx", "args": ["-y", "@supabase/mcp-server-supabase@latest", "--access-token", "${SUPABASE_ACCESS_TOKEN}"]},
    "qdrant": {"command": "uvx", "args": ["mcp-server-qdrant"], "env": {...}}
  }
}
```

**FastAPI as MCP Server** (via FastAPI-MCP):
```python
# app_factory.py - FastAPI exposes REST endpoints as MCP tools
from fastapi_mcp import FastAPIMCP

mcp = FastAPIMCP(app)
mcp.mount()  # Exposes /api/* endpoints as MCP tools at /mcp
```

### 3. Memory System (database/memory_manager.py)
Unified PostgreSQL-based memory management via MCP.

```python
class MemoryManager:
    """No Redis needed - PostgreSQL via MCP handles everything"""
    
    def __init__(self, postgres_client: PostgreSQLClient):
        self.postgres_client = postgres_client
    
    async def create_session(self, session_data: Dict[str, Any]):
        # Uses PostgreSQL MCP client for session storage
        await self.postgres_client.create_session(session_data)
        
    async def get_recent_context(self, session_id: str, limit: int = 10):
        # Retrieves context via MCP client
        return await self.postgres_client.get_recent_memories(session_id, limit)
```

### 4. Knowledge Base (mcp/qdrant_client.py)
Semantic search and FAQ pattern matching via MCP.

```python
class QdrantClient:
    """Multi-tier query matching strategy via MCP"""
    
    def __init__(self, session: ClientSession):
        self.session = session
    
    async def search_similar(self, query: str, limit: int = 10):
        # Uses Qdrant MCP server for semantic search
        result = await self.session.call_tool("qdrant-find", {
            "query": query, "limit": limit, "collection": "sql_knowledge"
        })
        return result.get("results", [])
```

## 🚀 Implementation Priorities

### Phase 1: Foundation (Week 1-2)

#### Week 1: Core Infrastructure
- [ ] **Day 1-2**: Project setup and configuration
  - FastAPI application structure
  - Environment configuration
  - Logging and monitoring setup
  
- [ ] **Day 3-4**: Database connections
  - MariaDB connection pool
  - PostgreSQL connection pool
  - Connection lifecycle management
  
- [ ] **Day 5**: MCP server foundation
  - Basic FastMCP server
  - Tool registration framework
  - Lifespan management

#### Week 2: Memory System
- [ ] **Day 1-2**: PostgreSQL schema design
  - Create memory tables (session_state, user_history, cache)
  - Indexes for performance
  - TTL cleanup functions
  
- [ ] **Day 3-4**: Memory manager implementation
  - CRUD operations for all memory types
  - Session context management
  - Caching layer
  
- [ ] **Day 5**: Testing and validation
  - Unit tests for memory operations
  - Performance benchmarks
  - Documentation

### Phase 2: Intelligence Layer (Week 3-4)

#### Week 3: AI Integration
- [ ] **Day 1-2**: Sonnet 4 integration
  - Anthropic API client
  - Thinking mode implementation
  - Error handling and retries
  
- [ ] **Day 3-4**: BGE-M3 embeddings
  - Local model setup
  - Embedding service
  - Batch processing optimization
  
- [ ] **Day 5**: Qdrant setup
  - Docker deployment
  - Collection creation
  - Initial indexing pipeline

#### Week 4: Knowledge Base
- [ ] **Day 1-2**: FAQ system
  - FAQ storage schema
  - Pattern matching logic
  - Confidence scoring
  
- [ ] **Day 3-4**: Semantic search
  - Vector indexing pipeline
  - Hybrid search implementation
  - Result ranking
  
- [ ] **Day 5**: Integration testing
  - End-to-end retrieval tests
  - Performance optimization
  - Cache warming strategies

### Phase 3: Autonomous Behavior (Week 5-6)

#### Week 5: Core Agent Loop
- [ ] **Day 1-2**: Investigation planner
  - Task decomposition logic
  - Strategy selection
  - Progress tracking
  
- [ ] **Day 3-4**: Autonomous execution
  - Investigation loop implementation
  - Self-correction mechanisms
  - Result synthesis
  
- [ ] **Day 5**: MCP tools
  - Database execution tools
  - Schema analysis tools
  - Safety validation tools

#### Week 6: Production Features
- [ ] **Day 1-2**: Guardrails
  - Query validation
  - Rate limiting
  - Permission system
  
- [ ] **Day 3-4**: Evaluation system
  - SQL validation
  - Result quality metrics
  - Performance tracking
  
- [ ] **Day 5**: WebSocket integration
  - Real-time progress updates
  - Streaming results
  - Error handling

### Phase 4: UI Integration (Week 7-8)

#### Week 7: API Development
- [ ] **Day 1-2**: REST endpoints
  - Agent interaction API
  - Investigation management
  - FAQ management
  
- [ ] **Day 3-4**: WebSocket handlers
  - Real-time communication
  - Progress streaming
  - Result updates
  
- [ ] **Day 5**: API documentation
  - OpenAPI specification
  - Integration examples
  - Authentication setup

#### Week 8: Testing & Optimization
- [ ] **Day 1-2**: Integration testing
  - Full system tests
  - Load testing
  - Error scenarios
  
- [ ] **Day 3-4**: Performance optimization
  - Query optimization
  - Caching strategies
  - Resource management
  
- [ ] **Day 5**: Deployment preparation
  - Docker containerization
  - Environment configurations
  - Monitoring setup

## 💡 Key Implementation Guidelines

### 1. Start Simple, Iterate Fast
- Begin with basic SQL execution
- Add autonomous features incrementally
- Test each component thoroughly

### 2. Safety First
- Implement guardrails before complex features
- Validate all SQL before execution
- Log everything for audit trails

### 3. Observable by Design
- Structured logging throughout
- Metrics for every operation
- Clear error messages

### 4. Modular Architecture
- Each component independently testable
- Clear interfaces between modules
- Dependency injection patterns

## 📊 Success Metrics

### Technical Metrics
- Query success rate > 95%
- Response time < 2s for FAQ queries
- Investigation completion rate > 80%
- Zero SQL injection vulnerabilities

### Business Metrics
- User satisfaction score > 4.5/5
- 70% reduction in time to insights
- 90% of common queries handled by FAQ
- 50% reduction in data analyst workload

## 🔧 Development Setup

### Prerequisites
```bash
# Python 3.11+
python --version

# PostgreSQL 15+
psql --version

# Docker for Qdrant
docker --version
```

### Installation
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Setup databases
python scripts/setup_databases.py
```

### Running the Backend
```bash
# Start MCP server
python -m app.mcp.server

# Start FastAPI application
uvicorn app.main:app --reload --port 8000
```

## 🧪 Testing Strategy

### Unit Tests
- Test each component in isolation
- Mock external dependencies
- Aim for 80%+ coverage

### Integration Tests
- Test component interactions
- Use test databases
- Validate full workflows

### End-to-End Tests
- Test complete investigation flows
- Include UI interaction
- Performance benchmarks

## 📈 Monitoring & Observability

### Logging
- Structured JSON logs
- Correlation IDs for request tracking
- Log levels: DEBUG, INFO, WARNING, ERROR

### Metrics
- Prometheus metrics for monitoring
- Custom dashboards in Grafana
- Alerts for critical issues

### Tracing
- OpenTelemetry for distributed tracing
- Trace autonomous investigation flows
- Performance bottleneck identification

## 🚨 Production Considerations

### Scalability
- Horizontal scaling with load balancer
- Connection pooling for databases
- Caching strategies for common queries

### Security
- API authentication (JWT tokens)
- Database connection encryption
- Input sanitization and validation

### Reliability
- Health check endpoints
- Graceful shutdown handling
- Circuit breakers for external services

---

## 🚀 Quick Start Guide

### Prerequisites

- Python 3.11+
- MariaDB/MySQL database
- PostgreSQL database  
- Anthropic API key

### Installation & Setup

1. **Environment Setup:**
   ```bash
   cd app/
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -e .
   ```

2. **Configuration:**
   ```bash
   cp .env.template .env
   # Edit .env with your database URLs and API keys
   ```

3. **Database Setup:**
   
   **MariaDB** (for company data):
   ```sql
   CREATE DATABASE company_db;
   CREATE USER 'agent'@'localhost' IDENTIFIED BY 'password';
   GRANT SELECT ON company_db.* TO 'agent'@'localhost';
   ```
   
   **PostgreSQL** (for agent memory):
   ```sql
   CREATE DATABASE agent_memory;
   CREATE USER agent WITH PASSWORD 'password';
   GRANT ALL PRIVILEGES ON DATABASE agent_memory TO agent;
   ```

4. **Run the Server:**
   ```bash
   python -m app.main
   ```
   
   Server starts on `http://localhost:8008`

## 📚 API Usage

### Core Endpoints

- **`POST /api/v1/investigations`** - Start autonomous investigation
- **`GET /api/v1/investigations/{id}`** - Get investigation status  
- **`POST /api/v1/database/execute`** - Execute SQL queries
- **`GET /api/v1/database/schema`** - Get database schema
- **`WebSocket /ws/{investigation_id}`** - Real-time progress updates

### Dual MCP Architecture

**1. FastAPI as MCP Server** (FastAPI-MCP):
- **Frontend**: Uses REST API (`/api/v1/*`)  
- **Agent**: Uses MCP tools (`/mcp`) - same endpoints, different protocol!

**2. FastAPI as MCP Client** (MCP Python SDK):
- **Database Operations**: Connects to external MCP servers
- **4 Database MCP Servers**: PostgreSQL, MariaDB, Supabase, Qdrant
- **Configuration**: `mcp.json` with environment variable substitution

### Example Usage

**Start Investigation:**
```bash
curl -X POST http://localhost:8008/api/v1/investigations \
  -H "Content-Type: application/json" \
  -d '{"query": "How many customers signed up last month?"}'
```

**Real-Time Updates:**
```javascript
const ws = new WebSocket('ws://localhost:8008/ws/investigation_id');
ws.onmessage = (event) => {
    const update = JSON.parse(event.data);
    console.log('Progress:', update);
};
```

**API Documentation:**
- Swagger UI: `http://localhost:8008/docs`
- Health Check: `http://localhost:8008/health`
- Metrics: `http://localhost:8008/metrics`

## 🤖 How Autonomous Investigation Works

### Workflow
1. **User Query** → Frontend → REST API
2. **Investigation Created** → Background task starts  
3. **Agent Uses MCP Tools** → Database operations
4. **Progress Streamed** → WebSocket → Frontend
5. **Results Delivered** → Investigation complete

### Agent Execution Steps
1. Analyze query and plan investigation
2. Discover database schema
3. Plan investigation steps  
4. Execute SQL queries autonomously
5. Synthesize insights and results

## 🔒 Security Features

- SQL injection protection
- Read-only query enforcement  
- Query timeout limits (30s default)
- Result size limits (10K rows default)
- Rate limiting (100 req/min default)
- Query validation and sanitization

## 🧪 Development Tools

**Code Quality:**
```bash
black app/
isort app/
ruff check app/
pytest
```

**Monitoring:**
- Structured JSON logging
- Prometheus metrics
- Request/response tracking
- SQL execution monitoring

## 🚧 Implementation Status

### ✅ Completed (Phase 1)
- FastAPI application with MCP integration
- Investigation management endpoints
- Database tool endpoints  
- WebSocket real-time updates
- Autonomous agent foundation
- Security guardrails
- Logging and monitoring

### ✅ MCP Integration Complete
1. **✅ Full MCP Architecture** - 4 databases via MCP clients, FastAPI as MCP server
2. **✅ External MCP Servers** - PostgreSQL, MariaDB, Supabase, Qdrant
3. **✅ FastAPI-MCP Integration** - REST endpoints exposed as MCP tools
4. **✅ MCP Client Infrastructure** - Complete client manager and database wrappers

### 🔄 Next Steps
1. **Claude Sonnet Integration** - Add intelligent planning
2. **Advanced Analytics** - Pattern discovery and statistical analysis  
3. **UI Integration** - Connect with React/Vue frontend
4. **Production Features** - Authentication, scaling, deployment

---

This backend architecture provides the foundation for a production-ready autonomous SQL investigation agent that works like Claude Code but for data analysis. The implementation is now complete and ready for development!