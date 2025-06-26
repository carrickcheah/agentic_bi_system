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




        ===============================================================================

# Summary

## Core Innovation

### Definition
- **SPS-SQL**: Schema-aware Pre-Synthesized SQL framework
- **Purpose**: Enable small language models (<10B parameters) to perform Text-to-SQL at enterprise-grade accuracy
- **Key Insight**: Pre-synthesize database-specific examples rather than relying on generic training data

### Paradigm Shift
- **Traditional Approach**: Large expensive models with generic SQL knowledge
- **SPS-SQL Approach**: Small efficient models with personalized, database-specific examples
- **Result**: 82.1% accuracy with 7B parameter models vs. expensive large model dependencies

## Three-Phase Architecture

### Phase 1: One-Time Learning (Offline)
#### Template Extraction Process
- **Input**: Training SQL queries from Spider dataset
- **Process**: ANTLR v4 parser converts SQL → Abstract Syntax Trees → Extract reusable patterns
- **Output**: 728 templates with placeholders and constraints
- **Template Structure**: SQL patterns + foreign key relationships + column count requirements

#### Example Template Format
```python
Template = {
    structure: "SELECT [t0].[c0] FROM [t0], [t1] WHERE [t0].[c1] = [t1].[c0] AND [t1].[c1] > [value]",
    constraints: "[t0].[c1] is primary key, [t1].[c0] is foreign key",
    column_counts: [2, 2]  # table requirements
}
```

### Phase 2: Database-Specific Setup (Per Database)
#### SQL Pre-Synthesis
- **Input**: Target database schema + learned templates
- **Process**: Apply templates only to compatible table structures
- **Output**: ~500 tailored SQL queries per database
- **Schema Awareness**: Respects foreign keys, data types, relationships

#### Natural Language Generation
- **Input**: Generated SQL + database schema context
- **Process**: LLM converts synthetic SQL to natural language questions
- **Output**: Question-SQL pairs specific to database domain
- **Quality**: Semantically coherent questions matching SQL complexity

#### Vector Embedding Creation
- **Model**: BGE embedding model (bge-large-en-1.5)
- **Dimensions**: 1024-dimensional vectors
- **Index**: FAISS vector database for fast similarity search
- **Storage**: Question-SQL pairs with searchable embeddings

### Phase 3: Real-Time Query Processing (Per User Question)
#### Similarity Search
- **Input**: User's natural language question
- **Process**: Semantic similarity matching against pre-synthesized examples
- **Output**: Top-k most relevant question-SQL pairs
- **Speed**: Sub-second retrieval from vector database

#### Few-Shot Prompting
- **Context Assembly**: User question + similar examples + database schema
- **Prompt Structure**: Schema definition + 5-20 relevant examples + user question
- **Optimization**: Balances context length vs. accuracy
- **Format Testing**: Code representation vs. OpenAI-style prompting

#### SQL Generation
- **Model**: Small LLM (7B-9B parameters)
- **Input**: Contextual few-shot prompt
- **Output**: Syntactically and semantically correct SQL
- **Speed**: 1.5-2 seconds per query on single GPU

## Technical Components Deep Dive

### Template Learning System
- **Parser**: ANTLR v4 for robust SQL parsing
- **Pattern Extraction**: Identifies structural SQL patterns across training data
- **Constraint Capture**: Records table relationships, data type requirements
- **Placeholder System**: Generic tokens for tables, columns, values
- **Coverage**: 728 distinct templates covering complex SQL operations

### Schema-Aware Synthesis Engine
- **Compatibility Checking**: Templates applied only to matching table structures
- **Relationship Preservation**: Maintains foreign key constraints in generated SQLs
- **Type Safety**: Ensures data type compatibility in WHERE clauses and JOINs
- **Semantic Validity**: Generated SQLs are both syntactically correct and logically meaningful

### Semantic Retrieval System
- **Embedding Model**: BGE-large-en-1.5 for high-quality semantic representations
- **Vector Store**: FAISS for efficient similarity search
- **Similarity Metrics**: Cosine similarity for question matching
- **Retrieval Strategy**: Top-k examples based on semantic closeness

### Optimized Prompting Framework
- **Context Management**: Balances example quantity with model context limits
- **Example Selection**: Most relevant examples based on similarity scores
- **Schema Integration**: Database structure included in every prompt
- **Format Optimization**: Tested prompt structures for maximum accuracy

## Performance Benchmarks

### Accuracy Results (Spider Dataset)
- **Qwen 2.5 Coder (7B)**: 81.7% dev set, 82.1% test set
- **Llama 3.1 (8B)**: 80.5% dev set, 78.7% test set
- **GLM-4 (9B)**: 78.1% dev set, 79.0% test set

### Speed Performance
- **Query Processing**: 1.5-2 seconds per query
- **Hardware**: Single RTX 4090 GPU
- **Latency**: Sub-second similarity search + 1-2s SQL generation
- **Throughput**: Suitable for real-time business applications

### Comparative Analysis
| Method | Model | Dev Accuracy | Test Accuracy | Speed (s/query) | Hardware Cost |
|--------|-------|--------------|---------------|-----------------|---------------|
| **SPS-SQL** | **Llama 3.1** | **80.5%** | **78.7%** | **1.5** | **Low** |
| **SPS-SQL** | **GLM-4** | **78.1%** | **79.0%** | **2.0** | **Low** |
| DAIL-SQL | Llama 3.1 | 76.6% | 73.5% | 3.1 | Medium |
| C3 | Llama 3.1 | 73.8% | 74.1% | 78.7 | High |
| MAC-SQL | Llama 3.1 | 65.8% | 46.1% | 10.8 | High |

### Cost Efficiency
- **Hardware Requirements**: Consumer-grade GPU vs. enterprise-scale infrastructure
- **Model Size**: 7-9B parameters vs. 70B+ parameter models
- **Operational Cost**: Significantly lower than large model solutions
- **Deployment**: Single-machine deployment possible

## Implementation Architecture

### Hardware Requirements
- **GPU**: RTX 4090 or equivalent (24GB VRAM recommended)
- **CPU**: Standard multi-core processor
- **Memory**: 32GB+ RAM for optimal performance
- **Storage**: SSD for vector database and model storage

### Software Stack
```python
Dependencies:
- ANTLR v4: SQL parsing and AST generation
- BGE Model: Semantic embedding generation
- FAISS: Vector similarity search
- PyTorch: Model inference
- Transformers: LLM integration
```

### Database Setup Process
1. **One-time Learning**: Extract templates from training data
2. **Schema Analysis**: Parse target database structure
3. **SQL Synthesis**: Apply templates to generate ~500 queries
4. **NL Generation**: Convert SQLs to natural language questions
5. **Vector Indexing**: Create searchable embedding database
6. **Deployment**: Real-time query processing setup

### Deployment Patterns
- **Single Database**: Dedicated setup per database schema
- **Multi-tenant**: Shared template system with database-specific examples
- **Cloud Deployment**: Container-based deployment with GPU acceleration
- **Edge Deployment**: Local deployment for data privacy requirements

## Key Technical Innovations

### Database Personalization
- **Schema-Only Approach**: Works with any database using only structural information
- **No Training Data Required**: Generates examples without existing domain SQL
- **Immediate Deployment**: New databases supported without additional training
- **Custom Examples**: Every database gets tailored question-SQL pairs

### Template-Driven Synthesis
- **Pattern Learning**: Automatically extracts complex SQL patterns
- **Constraint Awareness**: Captures semantic relationships between tables
- **Structural Validity**: Ensures generated SQLs follow database constraints
- **Coverage Breadth**: 728 templates covering diverse SQL operations

### Semantic Context Retrieval
- **Dynamic Examples**: Each query gets most relevant context
- **Similarity-Based**: Semantic matching rather than keyword matching
- **Quality Context**: High-quality examples improve small model performance
- **Personalized Learning**: Each database becomes a specialized tutor

### Small Model Optimization
- **Targeted Context**: Provides exactly what small models need to succeed
- **Single-Turn Efficiency**: No multi-step reasoning required
- **Memory Efficiency**: Optimized for consumer hardware constraints
- **Fast Inference**: Real-time performance with small model benefits

## Practical Example Walkthrough

### Sample Database Schema
```sql
Tables:
- customers (id, name, email, city, registration_date)
- orders (id, customer_id, total_amount, order_date, status)
- products (id, name, price, category, stock_quantity)
- order_items (order_id, product_id, quantity, unit_price)
```

### Template Application Examples
```sql
-- Template 1 Applied:
SELECT c.name FROM customers c, orders o 
WHERE c.id = o.customer_id AND o.total_amount > 500
-- Generated Question: "List customers who have orders worth more than $500"

-- Template 2 Applied:
SELECT COUNT(*) FROM orders WHERE order_date > '2024-01-01'
-- Generated Question: "How many orders were placed after January 1st, 2024?"

-- Template 3 Applied:
SELECT c.city, SUM(o.total_amount) FROM customers c, orders o 
WHERE c.id = o.customer_id GROUP BY c.city
-- Generated Question: "What is the total order value for each city?"
```

### Real-Time Processing Flow
```python
# User Query: "Show me customers from New York who spent over $1000"

# Step 1: Vector similarity search finds:
Similar_Examples = [
    ("List customers who have orders worth more than $500", similarity: 0.89),
    ("Show customers by city with their total spending", similarity: 0.76),
    ("Find high-value customers in each region", similarity: 0.71)
]

# Step 2: Construct few-shot prompt with examples + schema

# Step 3: Small LLM generates:
Generated_SQL = """
SELECT c.name FROM customers c, orders o 
WHERE c.id = o.customer_id 
AND c.city = 'New York' 
AND o.total_amount > 1000
"""
```

## Advantages Over Traditional Approaches

### Efficiency Benefits
- **Hardware Accessibility**: Runs on consumer GPUs vs. enterprise infrastructure
- **Speed Optimization**: 1.5-2s queries vs. 10+ second traditional methods
- **Cost Reduction**: 90%+ cost savings compared to large model solutions
- **Energy Efficiency**: Lower power consumption for equivalent performance

### Accuracy Improvements
- **Personalized Context**: Database-specific examples improve relevance
- **Semantic Matching**: Better example retrieval than keyword-based methods
- **Schema Awareness**: Generated SQL respects actual database constraints
- **Quality Control**: Pre-synthesized examples ensure high-quality context

### Deployment Advantages
- **Immediate Setup**: Works with any database schema without training data
- **Privacy Preservation**: Can run entirely on-premises
- **Scalability**: Easy replication across multiple databases
- **Maintenance**: Automatic updates when schema changes

### Business Impact
- **Democratization**: Makes enterprise Text-to-SQL accessible to smaller organizations
- **Resource Optimization**: Reduces infrastructure requirements significantly
- **Performance Reliability**: Consistent sub-2-second response times
- **Cost Predictability**: Fixed hardware costs vs. per-query API charges

## Research Contributions

### Novel Methodologies
- **Pre-Synthesis Strategy**: First framework to pre-generate database-specific examples
- **Template Learning**: Automated extraction of SQL patterns from training data
- **Semantic Retrieval**: Integration of vector similarity with SQL generation
- **Small Model Optimization**: Specialized approach for resource-constrained models

### Empirical Validation
- **Benchmark Results**: State-of-the-art performance on Spider dataset
- **Ablation Studies**: Demonstrates effectiveness of each component
- **Cross-Model Testing**: Validates approach across different small LLMs
- **Speed Analysis**: Comprehensive performance characterization

### Industry Applications
- **Enterprise Deployment**: Production-ready architecture for business use
- **Cost Analysis**: Detailed comparison with existing solutions
- **Hardware Optimization**: Specific recommendations for deployment infrastructure
- **Scalability Testing**: Multi-database deployment patterns

## Future Directions

### Technical Enhancements
- **Multi-Modal Integration**: Extend to chart and visualization understanding
- **Real-Time Learning**: Continuous improvement from user feedback
- **Cross-Database**: Templates that work across different database systems
- **Advanced Constraints**: Support for complex business rules and validations

### Performance Optimizations
- **Model Compression**: Further reduce model size while maintaining accuracy
- **Hardware Acceleration**: Optimize for specific GPU architectures
- **Batch Processing**: Support for multiple simultaneous queries
- **Cache Optimization**: Intelligent caching of frequent query patterns

### Business Applications
- **BI Tool Integration**: Direct integration with Tableau, Power BI, etc.
- **Natural Language Analytics**: Expand beyond SQL to full analytical workflows
- **Domain Specialization**: Industry-specific template libraries
- **Multi-Language Support**: Support for non-English natural language queries

## Implementation Best Practices

### Setup Recommendations
- **Hardware Sizing**: Match GPU memory to expected concurrent users
- **Database Preparation**: Ensure clean schema documentation for better synthesis
- **Template Customization**: Adapt templates for domain-specific patterns
- **Performance Tuning**: Optimize vector search parameters for specific use cases

### Quality Assurance
- **Example Validation**: Review generated question-SQL pairs for accuracy
- **Schema Testing**: Verify template applications produce valid SQLs
- **Performance Monitoring**: Track accuracy and speed metrics in production
- **User Feedback**: Implement feedback loops for continuous improvement

### Security Considerations
- **Data Privacy**: Ensure sensitive data doesn't leak through examples
- **Access Control**: Implement proper database permissions
- **Audit Logging**: Track all generated queries for compliance
- **Model Security**: Protect against prompt injection attacks

## Conclusion

**SPS-SQL represents a fundamental breakthrough in making enterprise-grade Text-to-SQL capabilities accessible through intelligent pre-synthesis and retrieval, achieving 82.1% accuracy with 7B parameter models while running on consumer hardware.**

### Key Impact
- **Democratizes Access**: High-quality Text-to-SQL no longer requires expensive infrastructure
- **Proves Efficiency**: Small models with smart context outperform large models with generic knowledge
- **Enables Innovation**: Opens Text-to-SQL to organizations previously limited by cost or infrastructure
- **Sets New Standard**: Establishes template-driven synthesis as viable alternative to scale-based approaches

### Strategic Implications
- **Cost Revolution**: 90%+ reduction in operational costs for Text-to-SQL systems
- **Deployment Freedom**: On-premises deployment preserves data privacy
- **Performance Reliability**: Consistent sub-2-second response times enable real-time applications
- **Technology Access**: Consumer-grade hardware brings enterprise capabilities to broader market

The SPS-SQL approach fundamentally challenges the assumption that better AI performance requires larger models, instead demonstrating that intelligent context provision can achieve superior results with dramatically lower resource requirements.












