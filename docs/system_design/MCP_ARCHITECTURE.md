# MCP Architecture Documentation

## Table of Contents
1. [What is MCP?](#what-is-mcp)
2. [Core Concepts](#core-concepts)
3. [Our Implementation](#our-implementation)
4. [Architecture Diagrams](#architecture-diagrams)
5. [Configuration](#configuration)
6. [Code Examples](#code-examples)
7. [Deployment](#deployment)
8. [Benefits & Design Decisions](#benefits--design-decisions)

## What is MCP?

**MCP (Model Context Protocol)** is a standardized protocol that enables AI applications to securely connect to external data sources and tools. It provides a bridge between AI models and various services like databases, APIs, and file systems.

### Key Problems MCP Solves:
- **Security**: Isolates AI from direct database access
- **Standardization**: Unified protocol for different data sources
- **Modularity**: Pluggable architecture for adding new capabilities
- **Scalability**: Distributed processing across specialized services

## Core Concepts

### 1. MCP Roles

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│    HOST     │    │   CLIENT    │    │   SERVER    │
│             │    │             │    │             │
│ Coordinates │----│ Connects to │----│ Provides    │
│ AI & clients│    │ servers     │    │ tools/data  │
└─────────────┘    └─────────────┘    └─────────────┘
```

- **HOST**: Manages the overall system and AI integration
- **CLIENT**: Connects to and communicates with MCP servers
- **SERVER**: Provides specialized tools, resources, or data access

### 2. Client vs Server

| **MCP Client** | **MCP Server** |
|----------------|----------------|
| Connects TO servers | Provides tools/resources |
| Consumes services | Offers services |
| Your application | External service |
| **What we built** | What we connect to |

### 3. Protocol Features

- **Stateful Sessions**: Persistent connections with context
- **Capability Negotiation**: Dynamic feature discovery
- **Security Isolation**: Servers can't access each other
- **JSON-RPC Based**: Standard messaging protocol

## Our Implementation

### Architecture Overview

We built an **MCP Client-based business intelligence backend** that connects to multiple specialized MCP servers for different data operations.

```
Our Application (MCP HOST + CLIENTS)
├── FastAPI Frontend (HTTP Interface)
├── Business Logic Service
├── MCP Client Manager
└── 4 External MCP Servers:
    ├── MariaDB MCP Server    (Business Data)
    ├── PostgreSQL MCP Server (Memory/Cache)
    ├── Qdrant MCP Server    (Vector Search)
    └── GraphRAG MCP Server   (Knowledge Graph)
```

### Key Components

#### 1. BackendService (`server.py`)
- **Role**: MCP Host + Client coordinator
- **Purpose**: Orchestrates all MCP client connections
- **Can run**: Standalone OR embedded in FastAPI

#### 2. MCPClientManager (`fastmcp/client_manager.py`)
- **Role**: MCP Client collection
- **Purpose**: Manages connections to all 4 MCP servers
- **Features**: Health monitoring, automatic reconnection

#### 3. BusinessService (`fastmcp/service.py`)
- **Role**: Business logic layer
- **Purpose**: Provides high-level database operations
- **Uses**: MCP clients internally

#### 4. FastAPI Integration (`main.py`, `fastapi/`)
- **Role**: HTTP interface
- **Purpose**: Web API that uses the MCP backend
- **Pattern**: Embedded MCP client backend

## Architecture Diagrams

### High-Level System Architecture
```
Internet/Users
      │
      ▼
┌─────────────────────────────────────────────────────────┐
│                FastAPI App                          │
│  ┌─────────────────┐    ┌─────────────────────────┐ │
│  │ HTTP Endpoints  │    │   Service Bridge        │ │
│  │ /api/database   │----│ (Communication Layer)  │ │
│  │ /api/query      │    │                         │ │
│  └─────────────────┘    └─────────────────────────┘ │
└──────────────────────┬──────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│              Backend Service (MCP Host)             │
│  ┌─────────────────┐    ┌─────────────────────────┐ │
│  │ Business Logic  │    │   MCP Client Manager    │ │
│  │ Service         │----│                         │ │
│  │                 │    │ ┌─────┐ ┌─────┐ ┌─────┐ │ │
│  └─────────────────┘    │ │ M   │ │ P   │ │ Q   │ │ │
└─────────────────────────│ │ D   │ │ G   │ │ D   │ │─┘
                           ││ B   │ │ SQL │ │ R   │ │
                           │└─────┘ └─────┘ └─────┘ │
                           └─────────────────────────┘
                                     │ │ │
                    ┌────────────────┘ │ └────────────────┐
                    │                  │                  │
                    ▼                  ▼                  ▼
            ┌─────────────┐   ┌─────────────┐   ┌─────────────┐
            │   MariaDB   │   │ PostgreSQL  │   │   Qdrant   │
            │ MCP Server  │   │ MCP Server  │   │ MCP Server  │
            │             │   │             │   │             │
            │ (Business   │   │ (Memory/    │   │ (Vector     │
            │  Data)      │   │  Cache)     │   │  Search)    │
            └─────────────┘   └─────────────┘   └─────────────┘
```

### MCP Client Flow
```
1. Request comes to FastAPI
2. FastAPI → Service Bridge → Backend Service
3. Backend Service → Business Service
4. Business Service → MCP Client Manager
5. MCP Client Manager → Specific MCP Server
6. Response flows back through the chain
```

## Workflow Diagrams

### 1. Complete Request Flow (DOS Style)
```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│    CLIENT    │────▶│   FASTAPI    │────▶│   BACKEND    │
│   (Browser)  │     │   (main.py)  │     │ (server.py)  │
└──────────────┘     └──────────────┘     └──────────────┘
        ▲                     │                      │
        │            ┌────────▼────────┐             │
        │            │ Route Handler   │             │
        │            │ /api/query      │             │
        │            └────────┬────────┘             │
        │                     │                      │
        │            ┌────────▼────────┐             │
        │            │ Service Bridge  │             │
        │            │ (Embedded Mode) │             │
        │            └────────┬────────┘             │
        │                     │                      │
        │                     ▼               ┌──────▼──────┐
        │           ╔═══════════════════════╗ │Business     │
        │           ║    MCP PROTOCOL       ║ │Service      │
        │           ║                       ║ │             │
        │           ╚═══════════════════════╝ └─────┬─────-─┘
        │                     │                     │
        │                     │              ┌──────▼──────┐
        │                     │              │MCP Client   │
        │                     │              │Manager      │
        │                     │              └──────┬──────┘
        │                     │                     │
        │          ┌──────────▼──────────┐          │
        │          │                     │          │
        │     ┌────▼────┐ ┌────▼────┐ ┌──▼────┐ ┌───▼───┐
        │     │MariaDB  │ │PostGre  │ │Qdrant│ │GraphRAG│
        │     │MCP SVR  │ │MCP SVR  │ │MCP SVR│ │base   │
        │     └─────────┘ └─────────┘ └───────┘ └───────┘
        │           │         │         │         │
        └───────────┴─────────┴─────────┴─────────┘
                    Response flows back up
```

### 2. MCP Client Initialization Workflow
```
╔═══════════════════════════════════════════════════════════════╗
║                    SERVER STARTUP SEQUENCE                    ║
╚═══════════════════════════════════════════════════════════════╝

Step 1: Configuration Loading
┌─────────────────────────────────────────────────────────────┐
│ server.py:main()                                            │
├─────────────────────────────────────────────────────────────┤
│ 1. Load .env variables                                      │
│ 2. Initialize settings                                      │
│ 3. Setup logging                                           │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
Step 2: Backend Service Creation
┌─────────────────────────────────────────────────────────────┐
│ BackendService.__init__()                                   │
├─────────────────────────────────────────────────────────────┤
│ 1. Create empty client_manager                             │
│ 2. Create empty service                                    │
│ 3. Setup async event handlers                             │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
Step 3: MCP Client Manager Initialization
┌─────────────────────────────────────────────────────────────┐
│ MCPClientManager.initialize()                               │
├─────────────────────────────────────────────────────────────┤
│ 1. Load mcp.json configuration                             │
│ 2. Process environment variables                           │
│ 3. For each server in config:                              │
│    ├─ Create server config object                          │
│    ├─ Set environment variables                            │
│    ├─ Call stdio_client(server_config)                     │
│    ├─ Create ClientSession                                 │
│    └─ Store client & session                               │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
Step 4: Individual MCP Server Connections
╔═══════════════════════════════════════════════════════════════╗
║                    MCP SERVER STARTUP                         ║
╚═══════════════════════════════════════════════════════════════╝

┌─ MariaDB ─┐  ┌─PostgreSQL─┐  ┌─ Qdrant ─┐  ┌─GraphRAG─┐
│ npx       │  │ npx        │  │ npx      │  │ uvx      │
│ mariadb-  │  │ @model...  │  │ qdrant-   │  │ graphrag │
│ mcp-      │  │ postgres   │  │ mcp      │  │ mcp-     │
│ server    │  │            │  │          │  │ server   │
└───────────┘  └────────────┘  └──────────┘  └──────────┘
      │              │              │              │
      ▼              ▼              ▼              ▼
┌─ Process ─┐  ┌─ Process ──┐  ┌─ Process ─┐  ┌─Process─┐
│ Spawned   │  │ Spawned    │  │ Spawned   │  │Spawned  │
│ Connected │  │ Connected  │  │ Connected │  │Connected│
│ Ready     │  │ Ready      │  │ Ready     │  │Ready    │
└───────────┘  └────────────┘  └───────────┘  └─────────┘
                              │
                              ▼
Step 5: Business Service Initialization
┌─────────────────────────────────────────────────────────────┐
│ BusinessService.initialize()                                │
├─────────────────────────────────────────────────────────────┤
│ 1. Store reference to client_manager                       │
│ 2. Initialize wrapper clients                              │
│ 3. Setup business logic methods                            │
│ 4. Mark service as ready                                   │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
Step 6: Ready for Requests
╔═══════════════════════════════════════════════════════════════╗
║                  SERVER READY                                ║
║                                                               ║
║ All MCP clients connected                                     ║
║ Business service operational                                  ║
║ Ready to process requests                                     ║
╚═══════════════════════════════════════════════════════════════╝
```

### 3. Request Processing Workflow
```
╔═══════════════════════════════════════════════════════════════╗
║                   REQUEST PROCESSING FLOW                     ║
╚═══════════════════════════════════════════════════════════════╝

┌─────────────┐
│ HTTP Request│ POST /api/database/query
│ from Client │ { "sql": "SELECT * FROM users", "database": "mariadb" }
└──────┬──────┘
       │
       ▼
┌─────────────────────────────────────────────────────────────┐
│ FastAPI Route Handler                                       │ 
│ @app.post("/api/database/query")                           │
│ async def execute_query(request: QueryRequest)             │
├─────────────────────────────────────────────────────────────┤
│ 1. Validate request data                                   │
│ 2. Extract sql and database parameters                    │
│ 3. Call service bridge                                    │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│ Service Bridge (Communication Layer)                       │
│ bridge.execute_sql(sql, database)                         │
├─────────────────────────────────────────────────────────────┤
│ 1. Get embedded backend service                           │
│ 2. Forward request to business service                    │
│ 3. Handle any communication errors                        │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│ Business Service (Business Logic)                          │
│ service.execute_sql(sql, database, max_rows, timeout)     │
├─────────────────────────────────────────────────────────────┤
│ 1. Validate business rules                                │
│ 2. Apply any data transformations                         │
│ 3. Route to appropriate database client                   │
│ 4. Handle business-level errors                           │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│ MCP Client Manager                                          │
│ manager.get_client("mariadb").execute_sql(sql)            │
├─────────────────────────────────────────────────────────────┤
│ 1. Select correct MCP client (mariadb)                    │
│ 2. Check client health                                    │
│ 3. Prepare MCP protocol message                           │
│ 4. Handle connection errors                               │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
╔═══════════════════════════════════════════════════════════════╗
║                    MCP PROTOCOL LAYER                         ║
╠═══════════════════════════════════════════════════════════════╣
║ MCP Client Session                                            ║
║ session.call_tool("execute_sql", {"sql": "SELECT...", ...})   ║
║                                                               ║
║ 1. Serialize request to JSON-RPC                             ║
║ 2. Send via stdio to MCP server process                      ║
║ 3. Wait for response                                          ║
║ 4. Deserialize JSON-RPC response                             ║
╚═══════════════════════┬═══════════════════════════════════════╝
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│ External MCP Server (MariaDB)                               │
│ Running in separate process via: npx mariadb-mcp-server    │
├─────────────────────────────────────────────────────────────┤
│ 1. Receive MCP protocol message                            │
│ 2. Validate request parameters                             │
│ 3. Connect to actual MariaDB database                     │
│ 4. Execute SQL query                                       │
│ 5. Format results                                          │
│ 6. Send MCP protocol response                              │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
           ┌──────────────────────────────────────┐
           │        MariaDB Database              │
           │                                      │
           │ ┌─────────┐ ┌─────────┐ ┌─────────┐ │
           │ │ users   │ │ orders  │ │ products│ │
           │ │ table   │ │ table   │ │ table   │ │
           │ └─────────┘ └─────────┘ └─────────┘ │
           └──────────────────────────────────────┘
                          │
         ╔════════════════▼═════════════════════════════════════╗
         ║              RESPONSE FLOW                          ║
         ║         (Same path in reverse)                      ║
         ║                                                     ║
         ║ MariaDB → MCP Server → MCP Client → Business        ║
         ║ Service → Service Bridge → FastAPI → HTTP Response  ║
         ╚═════════════════════════════════════════════════════╝
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│ HTTP Response to Client                                     │
├─────────────────────────────────────────────────────────────┤
│ {                                                           │
│   "data": [{"id": 1, "name": "John"}, ...],               │
│   "columns": ["id", "name"],                               │
│   "row_count": 150,                                        │
│   "execution_time": 0.045,                                 │
│   "success": true                                           │
│ }                                                           │
└─────────────────────────────────────────────────────────────┘
```

### 4. Deployment Architecture Workflow
```
╔═══════════════════════════════════════════════════════════════╗
║                    DEPLOYMENT SCENARIOS                       ║
╚═══════════════════════════════════════════════════════════════╝

Scenario A: Standalone MCP Backend
┌─────────────────────────────────────────────────────────────┐
│ Local Development / Background Processing                   │
│                                                             │
│ Terminal: uv run python server.py                          │
├─────────────────────────────────────────────────────────────┤
│ ┌─────────────┐    ┌─────────────┐    ┌─────────────┐      │
│ │   server.py │───▶│ MCP Clients │───▶│ MCP Servers │      │
│ │             │    │             │    │             │      │
│ │ BackendSrv  │    │ MariaDB     │    │ External    │      │
│ │ Runs Forever│    │ PostgreSQL  │    │ Processes   │      │
│ │ No HTTP     │    │ Qdrant     │    │             │      │
│ └─────────────┘    │ Supabase    │    │             │      │
│                    └─────────────┘    └─────────────┘      │
└─────────────────────────────────────────────────────────────┘

Scenario B: Full Web Application (Embedded)
┌─────────────────────────────────────────────────────────────┐
│ Production Deployment                                       │
│                                                             │
│ Terminal: uv run python main.py                            │
├─────────────────────────────────────────────────────────────┤
│ ┌─────────────┐                                             │
│ │   main.py   │ ◄── HTTP Requests from Internet            │
│ │             │                                             │
│ │ ┌─────────┐ │                                             │
│ │ │FastAPI  │ │                                             │
│ │ │Routes   │ │                                             │
│ │ │/api/*   │ │                                             │
│ │ └────┬────┘ │                                             │
│ │      │      │                                             │
│ │ ┌────▼────┐ │    ┌─────────────┐    ┌─────────────┐      │
│ │ │Service  │ │───▶│ server.py   │───▶│ MCP Clients │      │
│ │ │Bridge   │ │    │             │    │             │      │
│ │ │(Embedded│ │    │ BackendSrv  │    │ MariaDB     │      │
│ │ │ Mode)   │ │    │ (Embedded)  │    │ PostgreSQL  │      │
│ │ └─────────┘ │    └─────────────┘    │ Qdrant     │      │
│ │             │                       │ Supabase    │      │
│ └─────────────┘                       └─────────────┘      │
└─────────────────────────────────────────────────────────────┘

Cloud Deployment Process:
┌─────────────────────────────────────────────────────────────┐
│ Step 1: Deploy app/ directory to cloud platform            │
├─────────────────────────────────────────────────────────────┤
│ Railway/Render/GCP detects:                                 │
│ - pyproject.toml                                            │
│ - main.py as entry point                                    │
│ - Environment variables from dashboard                      │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│ Step 2: Container/Server starts                            │
├─────────────────────────────────────────────────────────────┤
│ 1. Install dependencies (uv sync)                          │
│ 2. Set environment variables                               │
│ 3. Run: python main.py                                     │
│ 4. FastAPI starts on port 8000                            │
│ 5. FastAPI loads server.py (embedded mode)                 │
│ 6. server.py initializes MCP clients                       │
│ 7. All 4 MCP servers spawn as child processes              │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│ Step 3: Running in Production                               │
├─────────────────────────────────────────────────────────────┤
│ Cloud Container:                                            │
│ ├─ Main Process: FastAPI (main.py)                         │
│ ├─ Child Process: MariaDB MCP Server                       │
│ ├─ Child Process: PostgreSQL MCP Server                    │
│ ├─ Child Process: Qdrant MCP Server                       │
│ ├─ Child Process: Supabase MCP Server                      │
│ └─ All connected via stdio/JSON-RPC                        │
│                                                             │
│ External Access:                                            │
│ https://your-app.railway.app/api/query                     │
└─────────────────────────────────────────────────────────────┘
```



## Code Examples

### Our Implementation (MCP Client)
```python
# server.py - Our MCP CLIENT backend
class BackendService:
    def __init__(self):
        # We CREATE MCP clients
        self.client_manager = MCPClientManager()
    
    async def initialize(self):
        # We CONNECT TO external MCP servers
        await self.client_manager.initialize()
```

### Example MCP Server (For Comparison)
```python
# Example MCP SERVER (what others connect to)
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Calculator")

@mcp.tool()
def add(a: int, b: int) -> int:
    """We PROVIDE tools to clients"""
    return a + b
```

### Using Our MCP Clients
```python
# Business logic using MCP clients
async def execute_business_query(query: str):
    # Get MCP client for MariaDB
    mariadb_client = client_manager.get_client("mariadb")
    
    # Execute via MCP protocol
    result = await mariadb_client.execute_sql(query)
    
    return result
```

### Import Pattern (Handles Standalone + Module)
```python
# Fixed import pattern in our modules
try:
    # Try relative imports (when used as module)
    from .config import settings
    from .utils.logging import logger
except ImportError:
    # Fall back to absolute imports (when run standalone)
    from config import settings
    from utils.logging import logger
```

## Deployment

### Local Development
```bash
# Option 1: MCP backend only
uv run python server.py

# Option 2: Full web application  
uv run python main.py
```

### Cloud Deployment
```bash
# Deploy entire app/ directory
# Both main.py AND server.py will run together

# Railway
railway deploy

# Render  
git push origin main

# Docker
docker build -t mcp-backend .
docker run -p 8000:8000 mcp-backend
```

### What Runs Where
```
Cloud Deployment of app/:
├── main.py        → FastAPI web server (HTTP interface)
├── server.py      → MCP backend (embedded, connects to 4 MCP servers)
├── fastmcp/       → MCP client management
└── fastapi/       → HTTP route handlers
```

## Benefits & Design Decisions

### Why MCP Clients (Not Direct Database)?

| **Direct Database** | **MCP Clients** |
|---------------------|-----------------| 
| ✗ Tight coupling | ✓ Loose coupling |
| ✗ Security risks | ✓ Isolated access |
| ✗ Mixed protocols | ✓ Unified protocol |
| ✗ Hard to scale | ✓ Easy to scale |

### Security Benefits
- **Credential Isolation**: Database credentials stay in MCP servers
- **Access Control**: MCP servers handle permissions
- **Process Isolation**: Each MCP server runs separately
- **Protocol Security**: Standardized, secure communication

### Scalability Benefits
- **Independent Scaling**: Scale each MCP server independently
- **Horizontal Growth**: Add new MCP servers easily
- **Distributed Processing**: MCP servers can run anywhere
- **Load Distribution**: Spread database load across servers

### Development Benefits
- **Modularity**: Swap MCP servers without changing our code
- **Testability**: Mock MCP clients for testing
- **Reusability**: Same MCP servers used by multiple applications
- **Standardization**: Consistent interface across all databases

## Architecture Comparison

### Traditional Architecture
```
App → Database 1
    → Database 2  
    → Database 3
```
*Issues: Tight coupling, mixed protocols, security complexity*

### Our MCP Architecture
```
App → MCP Client Manager → MCP Server 1 → Database 1
                        → MCP Server 2 → Database 2
                        → MCP Server 3 → Database 3
```
*Benefits: Unified protocol, security isolation, easy scaling*

## Summary

Our implementation represents a **modern, scalable approach** to multi-database operations:

- **We are MCP Clients** that connect to specialized MCP servers
- **Single unified protocol** for all database operations
- **Security by design** with credential and process isolation
- **Scalable architecture** that can grow with business needs
- **Both standalone and web-integrated** deployment options

This architecture follows MCP best practices and provides a solid foundation for building complex business intelligence applications.