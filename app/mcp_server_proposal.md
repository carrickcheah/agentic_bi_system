# MCP Server Implementation Proposal for Agentic SQL

## Current State Analysis

The Agentic SQL system currently operates as an MCP **client-only** architecture:
- Uses MCP clients to connect to external database servers (MariaDB, PostgreSQL)
- Rich AI capabilities are internal to the application
- LLMs access functionality through FastAPI endpoints, not MCP

## Proposed MCP Server Architecture

To make the system's AI capabilities directly accessible to LLMs via MCP, we should create MCP servers for the following services:

### 1. Business Intelligence MCP Server
**Purpose**: Natural language business query processing
**Tools**:
- `analyze_business_query` - Analyze business questions and recommend approach
- `get_domain_expertise` - Get domain-specific insights
- `suggest_investigation_approach` - Recommend investigation strategy

### 2. Investigation MCP Server  
**Purpose**: Autonomous 7-step investigation framework
**Tools**:
- `create_investigation` - Start new investigation
- `execute_investigation_phase` - Run specific investigation phase
- `get_investigation_status` - Check investigation progress
- `get_investigation_results` - Retrieve investigation findings

### 3. SQL Intelligence MCP Server
**Purpose**: SQL query generation and optimization
**Tools**:
- `generate_sql_query` - Generate SQL from natural language
- `find_similar_queries` - Find similar past queries
- `optimize_sql_query` - Get optimization suggestions
- `validate_sql_query` - Check SQL validity and compliance

### 4. Insight Synthesis MCP Server
**Purpose**: Strategic insight generation
**Tools**:
- `synthesize_insights` - Generate strategic insights from data
- `create_executive_summary` - Create executive summaries
- `identify_patterns` - Find patterns in business data
- `generate_recommendations` - Create actionable recommendations

## Implementation Approach

### Phase 1: Create MCP Server Wrappers
1. Use FastMCP library to create server implementations
2. Wrap existing service modules with MCP protocol
3. Define tool schemas for each operation
4. Implement authentication and authorization

### Phase 2: Update mcp.json Configuration
Add internal MCP servers to configuration:
```json
{
  "mcpServers": {
    "mariadb": { /* existing */ },
    "postgres": { /* existing */ },
    "business-intelligence": {
      "command": "python",
      "args": ["-m", "app.mcp_servers.business_intelligence"],
      "env": {
        "ENABLE_CACHING": "true",
        "MODEL_PROVIDER": "${MODEL_PROVIDER}"
      }
    },
    "investigation": {
      "command": "python", 
      "args": ["-m", "app.mcp_servers.investigation"],
      "env": {
        "INVESTIGATION_TIMEOUT": "300",
        "ENABLE_PARALLEL_PHASES": "true"
      }
    },
    "sql-intelligence": {
      "command": "python",
      "args": ["-m", "app.mcp_servers.sql_intelligence"],
      "env": {
        "QDRANT_URL": "${QDRANT_URL}",
        "ENABLE_QUERY_LEARNING": "true"
      }
    },
    "insight-synthesis": {
      "command": "python",
      "args": ["-m", "app.mcp_servers.insight_synthesis"],
      "env": {
        "SYNTHESIS_MODEL": "${SYNTHESIS_MODEL}",
        "MAX_INSIGHTS": "10"
      }
    }
  }
}
```

### Phase 3: Enhanced FastMCP Integration
1. Update FastMCP to support bidirectional MCP communication
2. Implement MCP server discovery and registration
3. Add health monitoring for internal MCP servers
4. Create unified MCP gateway for LLM access

## Benefits

1. **Direct LLM Access**: LLMs can directly invoke business intelligence capabilities
2. **Modular Architecture**: Each service operates independently via MCP
3. **Language Agnostic**: Any LLM supporting MCP can use these services
4. **Scalability**: Services can be deployed and scaled independently
5. **Standardization**: Uses industry-standard MCP protocol

## Next Steps

1. Implement first MCP server (recommend starting with SQL Intelligence)
2. Test integration with Claude Desktop and other MCP clients
3. Gradually migrate other services to MCP servers
4. Update documentation and examples