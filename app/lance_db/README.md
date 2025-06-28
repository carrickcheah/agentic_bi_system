# LanceDB SQL Query Embeddings Module

Self-contained module for storing and searching SQL query embeddings using LanceDB vector database.

## Overview

This module provides high-performance semantic search capabilities for SQL queries in the Agentic BI system. It enables:
- **Ultra-fast cache hit detection** (5-15ms)
- **Semantic similarity matching** for natural language queries
- **Organizational learning** through pattern recognition
- **Non-blocking async operations** for real-time performance

## Architecture

LanceDB is **independent** from PostgreSQL - they serve different purposes:
- **PostgreSQL**: Stores cache, sessions, investigations (via MCP)
- **LanceDB**: Stores embeddings and performs vector search

## Implementation Guide

### Step 1: Initialize LanceDB Connection

```python
import lancedb
import pandas as pd
from datetime import datetime

# Create/connect to LanceDB database
async def init_lancedb():
    # This creates data directory within the module
    db = await lancedb.connect_async(settings.data_path)
    return db
```

### Step 2: Create SQL Query Embeddings Table

```python
async def create_sql_embeddings_table(db):
    # Define schema for SQL query embeddings
    schema = {
        "id": "string",                    # Unique query ID
        "sql_query": "string",             # Original SQL query
        "normalized_sql": "string",        # Normalized version for matching
        "vector": "vector[1024]",          # Embedding vector (size depends on model)
        "database": "string",              # Which database (mariadb/postgres)
        "query_type": "string",            # simple/analytical/investigative
        "execution_time_ms": "float",      # How long it took
        "row_count": "int32",              # Number of rows returned
        "user_id": "string",               # Who ran it
        "timestamp": "timestamp",          # When it was run
        "success": "bool",                 # Did it succeed
        "metadata": "string"               # JSON string for extra data
    }
    
    # Create table if it doesn't exist
    table_name = "sql_query_embeddings"
    existing_tables = await db.table_names()
    
    if table_name not in existing_tables:
        # Create empty dataframe with schema
        df = pd.DataFrame(columns=list(schema.keys()))
        table = await db.create_table(table_name, data=df, schema=schema)
        print(f"Created table: {table_name}")
    else:
        table = await db.open_table(table_name)
        print(f"Opened existing table: {table_name}")
    
    return table
```

### Step 3: Generate Embeddings for SQL Queries

```python
from sentence_transformers import SentenceTransformer
import hashlib
import json

class SQLEmbeddingService:
    def __init__(self):
        # BGE-M3 model as specified in your architecture
        self.model = SentenceTransformer('BAAI/bge-m3')
        self.db = None
        self.table = None
    
    async def initialize(self):
        self.db = await lancedb.connect_async("./data/agentic_bi_vectors")
        self.table = await create_sql_embeddings_table(self.db)
    
    def normalize_sql(self, sql_query: str) -> str:
        """Normalize SQL for better matching"""
        # Remove extra whitespace, lowercase, etc.
        normalized = " ".join(sql_query.split())
        normalized = normalized.lower()
        # Remove specific values but keep structure
        # Example: WHERE id = 123 � WHERE id = ?
        import re
        normalized = re.sub(r'=\s*\d+', '= ?', normalized)
        normalized = re.sub(r"=\s*'[^']*'", "= ?", normalized)
        return normalized
    
    async def store_sql_query(self, query_data: dict):
        """Store SQL query with its embedding"""
        # Generate embedding
        embedding = self.model.encode(query_data['sql_query'])
        
        # Prepare record
        record = {
            "id": hashlib.md5(query_data['sql_query'].encode()).hexdigest(),
            "sql_query": query_data['sql_query'],
            "normalized_sql": self.normalize_sql(query_data['sql_query']),
            "vector": embedding,
            "database": query_data.get('database', 'mariadb'),
            "query_type": query_data.get('query_type', 'simple'),
            "execution_time_ms": query_data.get('execution_time_ms', 0),
            "row_count": query_data.get('row_count', 0),
            "user_id": query_data.get('user_id', 'system'),
            "timestamp": datetime.now(),
            "success": query_data.get('success', True),
            "metadata": json.dumps(query_data.get('metadata', {}))
        }
        
        # Add to LanceDB
        await self.table.add([record])
        return record['id']
```

### Step 4: Search for Similar SQL Queries

```python
async def find_similar_queries(self, new_query: str, threshold: float = 0.85):
    """Find similar SQL queries for cache hits"""
    # Generate embedding for new query
    query_embedding = self.model.encode(new_query)
    
    # Search in LanceDB
    results = await (self.table
        .search(query_embedding)
        .metric("cosine")
        .limit(10)
        .to_pandas())
    
    # Filter by similarity threshold
    similar_queries = []
    for _, row in results.iterrows():
        similarity = row['_distance']  # LanceDB returns distance
        if similarity >= threshold:
            similar_queries.append({
                'sql_query': row['sql_query'],
                'similarity': similarity,
                'execution_time_ms': row['execution_time_ms'],
                'timestamp': row['timestamp']
            })
    
    return similar_queries
```

### Step 5: Integration with Your System

```python
# In your investigation engine
class InvestigationEngine:
    def __init__(self):
        self.sql_embedding_service = SQLEmbeddingService()
    
    async def process_query(self, user_query: str):
        # Check for similar queries first
        similar = await self.sql_embedding_service.find_similar_queries(
            user_query, 
            threshold=0.85
        )
        
        if similar and similar[0]['similarity'] > 0.95:
            # Cache hit! Return previous results
            return self.get_cached_results(similar[0])
        
        # Otherwise, execute new investigation
        sql_query = self.generate_sql(user_query)
        results = await self.execute_sql(sql_query)
        
        # Store for future use
        await self.sql_embedding_service.store_sql_query({
            'sql_query': sql_query,
            'query_type': self.classify_query(user_query),
            'execution_time_ms': results.execution_time,
            'row_count': len(results.data),
            'user_id': self.current_user_id,
            'success': True,
            'metadata': {
                'original_question': user_query,
                'insights': results.insights
            }
        })
        
        return results
```

## Data Flow Architecture

```
User Query � Investigation Engine
                �
        Check LanceDB for similar SQL
                �
    [Cache Hit?] � Return cached results
                �
    [Cache Miss] � Generate & Execute SQL
                �
        Store SQL + Embedding in LanceDB
                �
        Return results to user
```

## Performance Characteristics

| Operation | Performance | Notes |
|-----------|-------------|-------|
| Similarity Search | 5-15ms | With HNSW index |
| Embedding Generation | 20-50ms | BGE-M3 model |
| Store Query | 10-20ms | Async operation |
| Cache Hit Rate | 60-80% | With proper normalization |

## Installation

```bash
# Using uv (recommended)
uv add lancedb sentence-transformers

# Or using pip
pip install lancedb sentence-transformers
```

## Zeabur Deployment

### Volume Setup
1. Deploy your service to Zeabur
2. Go to service settings -> **Volumes** tab
3. Click **+ Mount Volume**
4. Configure:
   - **Mount Path**: `/data`
   - **Size**: 5GB (can resize later)
5. The module automatically detects Zeabur environment

### Environment Detection
```python
# Automatic path selection
if os.getenv("ZEABUR_ENVIRONMENT"):
    data_path = "/data/lancedb"  # Uses Zeabur volume
else:
    data_path = settings.lancedb_path  # Local development
```

## Configuration

Create `settings.env` in this directory:

```bash
# LanceDB Configuration
LANCEDB_PATH=./data/agentic_bi_vectors
EMBEDDING_MODEL=BAAI/bge-m3
SIMILARITY_THRESHOLD=0.85

# Index Configuration
ENABLE_HNSW_INDEX=true
INDEX_NPROBE=20
INDEX_REFINE_FACTOR=10

# Cache Settings
MAX_CACHE_SIZE=100000
CACHE_TTL_HOURS=24
```

## Module Structure

```
lance_db/
   config.py              # Pydantic settings configuration
   settings.env           # Environment variables
   model_logging.py       # Local logging setup
   runner.py              # Main SQLEmbeddingService
   component_1.py         # Additional components
   __init__.py           # Package exports
   test_standalone.py    # Standalone testing
   README.md             # This file
```

## Testing

Run standalone tests:

```bash
cd app/lance_db/
python test_standalone.py
```

## Best Practices

1. **Normalize SQL queries** before storing to improve cache hits
2. **Use appropriate similarity thresholds** (0.85-0.95 recommended)
3. **Index regularly** for datasets >100k queries
4. **Monitor embedding generation time** and cache if needed
5. **Clean old entries** periodically to maintain performance

## Integration Points

- **Investigation Engine**: Primary consumer of similarity search
- **Cache Manager**: Coordinates with PostgreSQL cache
- **MCP Architecture**: Independent service, no MCP server needed
- **WebSocket Updates**: Non-blocking operations maintain real-time streaming

## Future Enhancements

1. **Hybrid search** combining vector + keyword matching
2. **Query performance prediction** based on historical data
3. **Automatic index optimization** based on usage patterns
4. **Multi-model embeddings** for improved accuracy