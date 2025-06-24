# SPS-SQL Approach: Complete Technical Guide

## Core Concept

**SPS-SQL** = **Schema-aware Pre-Synthesized SQL**

A framework that makes small language models (under 10B parameters) perform nearly as well as expensive large models on Text-to-SQL tasks.

## The Complete SPS-SQL Pipeline

### Phase 1: **One-Time Learning** (Offline)

#### Step 1: Template Extraction
```python
Input: Training SQL queries (from Spider dataset)
Process: Parse SQLs → Extract reusable patterns
Output: 728 templates with placeholders

# Example template learned:
Template = {
    structure: "SELECT [t0].[c0] FROM [t0], [t1] WHERE [t0].[c1] = [t1].[c0] AND [t1].[c1] > [value]",
    constraints: "[t0].[c1] is primary key, [t1].[c0] is foreign key",
    column_counts: [2, 2]  # table0 needs 2 columns, table1 needs 2 columns
}
```

### Phase 2: **Database-Specific Setup** (Per Database)

#### Step 2: SQL Pre-Synthesis
```python
Input: Your database schema + learned templates
Process: Apply templates to generate new SQLs
Output: ~500 SQL queries tailored to your database

# For your database:
Schema: customers(id, name, city), orders(id, customer_id, amount)
Generated: "SELECT c.name FROM customers c, orders o WHERE c.id = o.customer_id AND o.amount > 1000"
```

#### Step 3: Natural Language Generation
```python
Input: Generated SQL + database schema  
Process: LLM converts SQL to human questions
Output: Question-SQL pairs

# Example:
SQL: "SELECT c.name FROM customers c, orders o WHERE c.id = o.customer_id AND o.amount > 1000"
Question: "List customers who have orders worth more than $1000"
```

#### Step 4: Vector Embedding
```python
Input: Generated questions
Process: Embed questions into 1024-dimensional vectors
Output: Searchable vector database

# Creates searchable pool:
{
    "question": "List customers who have orders worth more than $1000",
    "sql": "SELECT c.name FROM customers c, orders o WHERE...",
    "embedding": [0.1, 0.3, -0.2, ...]
}
```

### Phase 3: **Real-Time Query Processing** (Per User Question)

#### Step 5: Similarity Search
```python
Input: User's natural language question
Process: Find most similar questions in vector database
Output: Top-k most relevant examples

User asks: "Show customers from NYC who spent over $500"
System finds: 
1. "List customers who have orders worth more than $1000" (similarity: 0.89)
2. "Show customers by city with their spending" (similarity: 0.82)
3. "Find high-value customers" (similarity: 0.76)
```

#### Step 6: Few-Shot Prompting
```python
Input: User question + similar examples + database schema
Process: Create few-shot prompt for small LLM
Output: Contextual prompt with examples

# Prompt structure:
"""
Database Schema: [your tables]
Example 1: [similar question] → [SQL]
Example 2: [similar question] → [SQL]  
Example 3: [similar question] → [SQL]
Question: [user's question]
Answer: 
"""
```

#### Step 7: SQL Generation
```python
Input: Few-shot prompt
Process: Small LLM (7B parameters) generates SQL
Output: Accurate SQL query

# Small LLM generates:
"SELECT c.name FROM customers c, orders o WHERE c.id = o.customer_id AND c.city = 'NYC' AND o.amount > 500"
```

## Key Technical Components

### 1. **Template System**
- Uses ANTLR v4 parser to convert SQL → Abstract Syntax Trees
- Extracts structural patterns with placeholders
- Captures constraints (foreign keys, data types, relationships)

### 2. **Schema-Aware Synthesis**
- Applies templates only to compatible table structures
- Respects foreign key relationships
- Generates syntactically and semantically correct SQLs

### 3. **Semantic Retrieval**
- Uses BGE embedding model for semantic similarity
- FAISS vector index for fast search
- Retrieves most relevant examples per question

### 4. **Optimized Prompting**
- Tests different prompt formats (Code Representation vs OpenAI style)
- Optimizes few-shot count (5-20 examples)
- Balances context length vs. accuracy

## The Innovation Stack

```
🔄 Real-time: User Question → Similarity Search → Few-shot Prompting → SQL Generation
    ↑
📊 Database-specific: Schema → Template Application → SQL Synthesis → NL Generation → Vector Embedding
    ↑  
🧠 One-time Learning: Training Data → Template Extraction
```

## Detailed Example Walkthrough

### Your Database Schema
```sql
-- Your database tables
Table: customers (id, name, email, city, registration_date)
Table: orders (id, customer_id, total_amount, order_date, status)
Table: products (id, name, price, category, stock_quantity)
Table: order_items (order_id, product_id, quantity, unit_price)
```

### Template Extraction (Done Once)
```sql
-- Template 1:
SELECT [table0].[column0] FROM [table0], [table1] 
WHERE [table0].[id_column] = [table1].[foreign_key] 
AND [table1].[column1] > [value]

-- Template 2:
SELECT COUNT(*) FROM [table0] 
WHERE [table0].[date_column] > '[date_value]'

-- Template 3:
SELECT [table0].[column0], SUM([table1].[column1]) 
FROM [table0], [table1] 
WHERE [table0].[id] = [table1].[foreign_key] 
GROUP BY [table0].[column0]
```

### Auto-Generated Examples
```sql
-- From Template 1:
SELECT c.name FROM customers c, orders o 
WHERE c.id = o.customer_id AND o.total_amount > 500
-- Question: "List customers who have orders worth more than $500"

-- From Template 2:
SELECT COUNT(*) FROM orders 
WHERE order_date > '2024-01-01'
-- Question: "How many orders were placed after January 1st, 2024?"

-- From Template 3:
SELECT c.city, SUM(o.total_amount) 
FROM customers c, orders o 
WHERE c.id = o.customer_id 
GROUP BY c.city
-- Question: "What is the total order value for each city?"
```

### Real-Time Processing
```python
# User asks:
"Show me customers from New York who spent over $1000"

# System finds similar examples:
1. "List customers who have orders worth more than $500" (similarity: 0.89)
2. "Show customers by city with their total spending" (similarity: 0.76)  
3. "Find high-value customers in each region" (similarity: 0.71)

# Creates prompt with examples and generates:
SELECT c.name FROM customers c, orders o 
WHERE c.id = o.customer_id 
AND c.city = 'New York' 
AND o.total_amount > 1000
```

## Performance Results

### Benchmark Performance
- **Qwen 2.5 Coder (7B)**: 81.7% dev set, 82.1% test set
- **Llama 3.1 (8B)**: 80.5% dev set, 78.7% test set  
- **GLM-4 (9B)**: 78.1% dev set, 79.0% test set

### Efficiency Metrics
- **Speed**: 1.5-2 seconds per query on single RTX 4090 GPU
- **Hardware**: Runs on consumer-grade hardware
- **Cost**: Significantly cheaper than large model solutions

### Comparison with Other Methods
| Method | Model | Dev Accuracy | Test Accuracy | Speed (s/query) |
|--------|-------|--------------|---------------|-----------------|
| SPS-SQL | Llama 3.1 | 80.5% | 78.7% | 1.5 |
| SPS-SQL | GLM-4 | 78.1% | 79.0% | 2.0 |
| DAIL-SQL | Llama 3.1 | 76.6% | 73.5% | 3.1 |
| C3 | Llama 3.1 | 73.8% | 74.1% | 78.7 |
| MAC-SQL | Llama 3.1 | 65.8% | 46.1% | 10.8 |

## Why SPS-SQL Works

### Traditional Approach Problems:
- ❌ Small LLMs lack sufficient SQL knowledge
- ❌ Generic examples don't match user's database
- ❌ No relevant context for complex queries
- ❌ Expensive multi-turn reasoning required

### SPS-SQL Solutions:
- ✅ Provides database-specific examples
- ✅ Uses semantic similarity for relevant context
- ✅ Gives small LLMs exactly what they need to succeed
- ✅ Single-turn efficiency (no multi-step reasoning)
- ✅ Schema-aware generation ensures valid SQL

## Key Innovations

### 1. **Democratizing Text-to-SQL**
- Makes high-quality Text-to-SQL accessible on consumer hardware
- Reduces dependency on expensive large models
- Enables deployment in resource-constrained environments

### 2. **Schema-Only Synthesis**
- Generates training examples using only database structure
- No need for existing domain-specific SQL training data
- Works with any new database immediately

### 3. **Intelligent Template Learning**
- Automatically extracts 728 complex patterns from training data
- Goes beyond simple fill-in-the-blank templates
- Captures semantic relationships and constraints

### 4. **Dynamic Example Retrieval**
- Creates personalized examples for each database
- Uses semantic similarity to find most relevant examples
- Provides perfect context for each user question

## Implementation Requirements

### Hardware Requirements
- Single GPU (RTX 4090 or similar)
- 24GB VRAM recommended
- Standard CPU and RAM

### Software Dependencies
- ANTLR v4 for SQL parsing
- BGE embedding model (bge-large-en-1.5)
- FAISS for vector indexing
- Small LLM (Qwen 2.5 Coder, Llama 3.1, GLM-4)

### Setup Process
1. Extract templates from training data (one-time)
2. For each target database:
   - Apply templates to schema
   - Generate SQL-question pairs
   - Create vector embeddings
   - Build searchable index
3. Deploy real-time query processing

## Conclusion

**SPS-SQL represents a paradigm shift in Text-to-SQL generation, making enterprise-grade capabilities accessible through intelligent pre-synthesis and retrieval, rather than relying on expensive large-scale models.**

The approach essentially creates a "personalized SQL tutor" for each database that can instantly provide relevant examples for any question, enabling small models to achieve performance comparable to much larger, more expensive alternatives.

## References

Yan, L., Wan, Q., Liu, C., Duan, S., Han, P., & Xu, Y. (2025). SPS-SQL: Enhancing Text-to-SQL generation on small-scale LLMs with pre-synthesized queries. *Pattern Recognition Letters*, *196*, 45–51. https://doi.org/10.1016/j.patrec.2025.04.016