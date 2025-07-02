# SQL Query Embeddings for LanceDB

uv run python -m lance_db.embedded_sql.ingest_sql_queries sample_queries.json

Simple system to save SQL queries in JSON format and ingest them into LanceDB for semantic similarity search.

## Quick Start

1. **Add your SQL queries** to `sql_queries.json`
2. **Run ingestion**: `python ingest_sql_queries.py sql_queries.json`
3. **Search similar queries** using the main LanceDB service

## File Structure

```
embedded_sql/
├── sql_queries.json          # Your SQL queries (add here)
├── sample_queries.json       # Example queries for testing
├── ingest_sql_queries.py     # Ingestion script
└── README.md                # This file
```

## JSON Format

### Basic Structure
```json
{
  "queries": [
    {
      "sql_query": "SELECT * FROM users WHERE age > 25",
      "database": "mariadb",
      "query_type": "simple",
      "execution_time_ms": 45.2,
      "row_count": 150,
      "user_id": "analyst",
      "success": true,
      "metadata": {
        "original_question": "Show users over 25",
        "tables_used": ["users"],
        "business_domain": "hr",
        "complexity": "simple"
      }
    }
  ]
}
```

### Required Fields
- **sql_query**: The actual SQL query string

### Optional Fields (with defaults)
- **database**: `"mariadb"` | `"postgres"` (default: `"mariadb"`)
- **query_type**: `"simple"` | `"analytical"` | `"operational"` | `"investigative"` (default: `"simple"`)
- **execution_time_ms**: Query execution time (default: `0.0`)
- **row_count**: Number of rows returned (default: `0`)
- **user_id**: User who executed the query (default: `"system"`)
- **success**: Whether query succeeded (default: `true`)
- **timestamp**: ISO timestamp (auto-generated if not provided)
- **metadata**: Additional information object (default: `{}`)

### Recommended Metadata Fields
```json
"metadata": {
  "original_question": "The natural language question",
  "tables_used": ["table1", "table2"],
  "business_domain": "sales|finance|hr|production|marketing",
  "complexity": "simple|moderate|complex",
  "notes": "Additional context"
}
```

## Usage

### 1. Add Your Queries
Edit `sql_queries.json` and add your SQL queries:

```json
{
  "queries": [
    {
      "sql_query": "SELECT customer_id, SUM(amount) FROM orders GROUP BY customer_id",
      "database": "mariadb",
      "query_type": "analytical",
      "user_id": "sales_manager",
      "metadata": {
        "original_question": "What's the total order value per customer?",
        "business_domain": "sales"
      }
    }
  ]
}
```

### 2. Run Ingestion
```bash
# From the embedded_sql directory
python ingest_sql_queries.py sql_queries.json

# With custom batch size
python ingest_sql_queries.py sql_queries.json 20

# Test with sample data
python ingest_sql_queries.py sample_queries.json
```

### 3. Check Results
The script will show:
- Progress for each query
- Success/failure count
- Processing statistics
- Any errors encountered

## Integration

Once ingested, queries are available through the main LanceDB service:

```python
from app.lance_db import SQLEmbeddingService

service = SQLEmbeddingService()
await service.initialize()

# Find similar queries
similar = await service.find_similar_queries(
    "SELECT * FROM customers WHERE status = 'active'",
    threshold=0.85
)

# Get statistics
stats = await service.get_statistics()
print(f"Total queries: {stats['total_queries']}")
```

## Query Types

### Simple
Basic CRUD operations, single table queries
```sql
SELECT * FROM users WHERE id = 123
```

### Analytical  
Aggregations, joins, business intelligence queries
```sql
SELECT region, SUM(sales) FROM orders GROUP BY region
```

### Operational
Production monitoring, status checks
```sql
SELECT COUNT(*) FROM failed_jobs WHERE created_at > NOW() - INTERVAL 1 HOUR
```

### Investigative
Complex analytical queries, troubleshooting
```sql
WITH monthly_trends AS (...) SELECT * FROM complex_analysis
```

## Business Domains

Organize queries by business area:
- **sales**: Revenue, orders, customers
- **finance**: Budgets, expenses, P&L
- **hr**: Employees, timesheets, payroll
- **production**: Manufacturing, quality, efficiency
- **marketing**: Campaigns, analytics, ROI
- **inventory**: Stock levels, procurement
- **customer_support**: Tickets, satisfaction

## Tips

1. **Normalize similar queries**: Group variations of the same query pattern
2. **Add good metadata**: Helps with future searching and organization
3. **Include execution context**: User, time, business question
4. **Test with samples**: Use `sample_queries.json` to verify ingestion works
5. **Batch process**: Large query sets can be processed in smaller batches

## Troubleshooting

### Common Issues

**Import Error**: Make sure you're running from the correct directory
```bash
cd /path/to/lance_db/embedded_sql
python ingest_sql_queries.py sql_queries.json
```

**Empty Results**: Check JSON format and required fields
```bash
# Validate your JSON
python -m json.tool sql_queries.json
```

**Service Not Initialized**: Ensure LanceDB dependencies are installed
```bash
# From the lance_db directory
python test_standalone.py
```

### Validation

The ingestion script validates:
- JSON format and structure
- Required `sql_query` field
- Data types for optional fields
- SQL query is not empty

## Examples

See `sample_queries.json` for 15 realistic business queries covering:
- Sales analytics
- HR reporting  
- Inventory management
- Production monitoring
- Quality control
- Financial analysis
- Customer management
- Marketing campaigns

These examples show best practices for:
- Query complexity levels
- Metadata usage
- Business domain organization
- Performance tracking