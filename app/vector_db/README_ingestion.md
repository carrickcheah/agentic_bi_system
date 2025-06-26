# Pattern Ingestion Guide

Production-grade ingestion of business intelligence patterns into Qdrant vector database.

## Overview

The `ingest_all_patterns.py` script provides enterprise-level pattern ingestion with:
- **Deduplication**: Skips existing patterns automatically
- **Batch Processing**: 10x faster embedding generation
- **Error Recovery**: Checkpoint system for failed ingestions
- **Progress Tracking**: Real-time progress bars and metrics
- **Flexible CLI**: Multiple ingestion modes

## Quick Start

### 1. Prerequisites

Ensure you have the required dependencies:
```bash
cd /Users/carrickcheah/Project/agentic_sql/app
uv add qdrant-client sentence-transformers tqdm
```

### 2. Environment Setup

Ensure your `.env` file contains:
```bash
QDRANT_URL=your_qdrant_cloud_url
QDRANT_API_KEY=your_api_key
```

### 3. Basic Usage

```bash
# Ingest all patterns (default)
python vector_db/ingest_all_patterns.py

# Ingest specific domain only
python vector_db/ingest_all_patterns.py --domain sales_revenue

# Dry run (see what would be ingested)
python vector_db/ingest_all_patterns.py --dry-run

# Force re-ingestion (skip deduplication)
python vector_db/ingest_all_patterns.py --force

# Verify after ingestion
python vector_db/ingest_all_patterns.py --verify
```

### 4. Adding New Patterns

#### For New Patterns in Existing Domains

When you add new patterns to an existing domain file (e.g., add patterns to `sales_revenue.json`):

```bash
# Test what would be ingested
python vector_db/ingest_all_patterns.py --domain sales_revenue --dry-run

# Ingest only the new patterns (automatic deduplication)
python vector_db/ingest_all_patterns.py --domain sales_revenue
```

The script automatically detects which patterns are new and only ingests those.

#### For Completely New Domains

When you create a new pattern file (e.g., `customer_service.json`):

```bash
# Ingest the new domain
python vector_db/ingest_all_patterns.py --domain customer_service --verify
```

#### Example: Adding One New Pattern

1. **Add your pattern** to the appropriate JSON file:
   ```json
   // In sales_revenue.json
   {
     "information": "Your new sales pattern...",
     "metadata": {
       "pattern": "analysis_workflow",
       "business_domain": "sales",
       "complexity": "simple"
     }
   }
   ```

2. **Test the ingestion**:
   ```bash
   python vector_db/ingest_all_patterns.py --domain sales_revenue --dry-run
   ```
   Output will show: `Ready to ingest 1 new patterns`

3. **Ingest the new pattern**:
   ```bash
   python vector_db/ingest_all_patterns.py --domain sales_revenue
   ```

#### Expected Output for Single Pattern:
```
📄 Loaded 15 patterns from 1 domains
🔍 Loading existing pattern hashes...
📝 Found 284 existing patterns
📋 Ready to ingest 1 new patterns

🧮 Generating embeddings for 1 patterns...
💾 Uploading 1 points to Qdrant...
✅ Successfully uploaded 1 patterns

Ingestion Stats:
  New patterns: 1
  Skipped duplicates: 14
  Successfully uploaded: 1
```

## Command-Line Options

| Option | Description | Example |
|--------|-------------|---------|
| `--all` | Ingest all domains (default) | `--all` |
| `--domain` | Ingest specific domain | `--domain hr_workforce` |
| `--verify` | Run verification queries after ingestion | `--verify` |
| `--dry-run` | Show what would be ingested without uploading | `--dry-run` |
| `--force` | Skip deduplication check | `--force` |
| `--batch-size` | Embedding batch size (default: 32) | `--batch-size 64` |

## Available Domains

The script automatically discovers patterns from these domains:

- `asset_equipment` - Asset and equipment management
- `cost_management` - Cost analysis and budgeting  
- `customer_demand` - Customer behavior and demand
- `finance_budgeting` - Financial planning and budgets
- `hr_workforce` - Human resources and workforce
- `marketing_campaigns` - Marketing and campaigns
- `operations_efficiency` - Operational efficiency
- `planning_scheduling` - Planning and scheduling
- `product_management` - Product lifecycle management
- `production_operations` - Production and manufacturing
- `quality_management` - Quality assurance and control
- `safety_compliance` - Safety and regulatory compliance
- `sales_revenue` - Sales performance and revenue
- `supply_chain_inventory` - Supply chain and inventory

## Performance Features

### Batch Processing
- Processes embeddings in batches of 32 (configurable)
- ~10x faster than sequential processing
- Optimal memory usage

### Deduplication
- Content-based hashing prevents duplicates
- Safe to run multiple times
- Preserves existing data

### Error Recovery
- Automatic checkpoint system
- Resume from last successful upload
- Exponential backoff on failures

## Output Examples

### Normal Ingestion
```
🎯 Production Pattern Ingestion
==================================================
🔧 Initializing production pattern ingester...
📡 Loading embedding model: sentence-transformers/all-MiniLM-L6-v2...
📊 Collection 'valiant_vector' exists with 150 points
🔍 Loading existing pattern hashes...
📝 Found 150 existing patterns
📄 Loaded 284 patterns from 14 domains

📋 Ready to ingest 134 new patterns

🧮 Generating embeddings for 134 patterns...
Embedding batches: 100%|████████| 5/5 [00:12<00:00,  2.34s/it]

💾 Uploading 134 points to Qdrant...
Upload chunks: 100%|████████| 2/2 [00:03<00:00,  1.67s/it]
✅ Successfully uploaded 134 patterns

📊 Ingestion Summary
==================================================
Collection: valiant_vector
Total points: 284
Vector size: 384

Ingestion Stats:
  New patterns: 134
  Skipped duplicates: 150
  Successfully uploaded: 134
  Upload failures: 0

⏱️  Total time: 18.42 seconds
🎉 Pattern ingestion completed successfully!
```

### Dry Run
```
🔍 DRY RUN - Not uploading to Qdrant

1. sales_revenue_001 (sales_revenue)
   Info: Monthly revenue performance vs target analysis...
   Complexity: simple

2. hr_workforce_003 (hr_workforce)
   Info: Employee performance review and development planning...
   Complexity: moderate

... and 132 more patterns
```

## Verification

The script includes built-in verification with sample queries:
- "employee performance management"
- "sales revenue analysis" 
- "supply chain optimization"
- "customer satisfaction trends"

Verification confirms patterns are searchable and returns relevant results.

## Troubleshooting

### Common Issues

1. **Missing Dependencies**
   ```bash
   ❌ Missing required package: sentence_transformers
   ```
   **Solution**: Run `uv add sentence-transformers`

2. **Environment Variables**
   ```bash
   ❌ Missing required environment variables: ['QDRANT_URL']
   ```
   **Solution**: Check your `.env` file

3. **Collection Not Found**
   ```bash
   🔨 Creating collection 'valiant_vector'...
   ```
   **Normal**: Script creates collection automatically

4. **Upload Failures**
   ```bash
   ⚠️  Retry 1/3 after error: Connection timeout
   ```
   **Normal**: Script retries with exponential backoff

### Debug Mode

For detailed debugging, check the checkpoint file:
```bash
cat .ingestion_checkpoint.json
```

## Testing

Test the ingestion system:
```bash
python testing/scripts/test_pattern_ingestion.py
```

## Architecture Notes

### ID Strategy
- Numeric IDs: `domain_index * 1000 + pattern_index`
- String IDs: `{domain}_{index:03d}`
- Supports 1000 patterns per domain

### Metadata Structure
Each pattern includes:
- Pattern ID and domain
- Business metadata (complexity, timeframe, roles)
- Content hash for deduplication
- Ingestion timestamp
- Source file reference

### Security
- No secrets in code
- Environment variable configuration
- Credential isolation in settings