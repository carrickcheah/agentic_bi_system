#!/usr/bin/env python3
"""
SQL Query Ingestion Script for LanceDB

Reads SQL queries from JSON file and ingests them into LanceDB using the SQLEmbeddingService.
This script can handle large batches of queries efficiently.
"""

import asyncio
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any

from ..runner import SQLEmbeddingService
from ..lance_logging import logger


class SQLQueryIngester:
    """Handles batch ingestion of SQL queries from JSON files into LanceDB."""
    
    def __init__(self):
        self.service = None
        self.stats = {
            "total_queries": 0,
            "successful_ingestions": 0,
            "failed_ingestions": 0,
            "start_time": None,
            "end_time": None,
            "errors": []
        }
    
    async def initialize(self):
        """Initialize the SQL embedding service."""
        print("Initializing SQL Embedding Service...")
        self.service = SQLEmbeddingService()
        await self.service.initialize()
        
        # Verify service health
        health = await self.service.health_check()
        if not all(health.values()):
            print(f"Warning: Some components are not healthy: {health}")
        else:
            print("Service initialized successfully!")
    
    async def load_queries_from_json(self, json_file_path: Path) -> List[Dict[str, Any]]:
        """Load SQL queries from JSON file."""
        if not json_file_path.exists():
            raise FileNotFoundError(f"JSON file not found: {json_file_path}")
        
        print(f"Loading queries from: {json_file_path}")
        
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Handle different JSON formats
        if isinstance(data, list):
            queries = data
        elif isinstance(data, dict) and 'queries' in data:
            queries = data['queries']
        elif isinstance(data, dict) and 'query_content' in data:
            # This is an enhanced template format (like template_moq.json)
            queries = [self._convert_enhanced_template_to_query(data)]
            print(f"Detected enhanced template format")
        else:
            raise ValueError("JSON file must contain an array of queries, an object with 'queries' key, or an enhanced template")
        
        print(f"Loaded {len(queries)} queries from JSON file")
        return queries
    
    def _convert_enhanced_template_to_query(self, template: Dict[str, Any]) -> Dict[str, Any]:
        """Convert enhanced template format to simple query format."""
        query = {
            "sql_query": template["query_content"]["sql_query"],
            "database": template.get("technical_metadata", {}).get("database", "mariadb"),
            "query_type": template["query_content"].get("query_type", "analytical"),
            "user_id": template.get("user_context", {}).get("user_id", "system"),
            "metadata": {
                "readable_description": template["query_content"].get("readable_description", ""),
                "business_question": template["query_content"].get("business_question", ""),
                "query_intent": template["query_content"].get("query_intent", ""),
                "business_domain": template.get("semantic_context", {}).get("business_domain", ""),
                "tags": template.get("tags", []),
                "kpi_category": template.get("business_intelligence", {}).get("kpi_category", ""),
                "entities": template.get("semantic_context", {}).get("entities", []),
                "keywords": template.get("semantic_context", {}).get("keywords", [])
            }
        }
        
        # Add any custom fields (like moq_specific_metadata)
        for key in template:
            if key not in ["query_content", "semantic_context", "technical_metadata", 
                          "user_context", "investigation_context", "execution_results",
                          "learning_metadata", "business_intelligence", "collaboration",
                          "version_control", "caching", "monitoring", "security",
                          "automation", "embeddings", "tags", "_id"]:
                query["metadata"][key] = template[key]
        
        return query
    
    def validate_query(self, query: Dict[str, Any], index: int) -> Dict[str, Any]:
        """Validate and normalize a single query object."""
        # Required fields
        if 'sql_query' not in query:
            raise ValueError(f"Query {index}: Missing required field 'sql_query'")
        
        if not query['sql_query'].strip():
            raise ValueError(f"Query {index}: 'sql_query' cannot be empty")
        
        # Set defaults for optional fields
        normalized_query = {
            "sql_query": query['sql_query'].strip(),
            "database": query.get('database', 'mariadb'),
            "query_type": query.get('query_type', 'simple'),
            "execution_time_ms": float(query.get('execution_time_ms', 0.0)),
            "row_count": int(query.get('row_count', 0)),
            "user_id": query.get('user_id', 'system'),
            "success": bool(query.get('success', True)),
            "metadata": query.get('metadata', {})
        }
        
        # Add timestamp if not provided
        if 'timestamp' in query:
            # If timestamp is provided, keep it (assume it's a valid ISO string)
            normalized_query['timestamp'] = query['timestamp']
        
        return normalized_query
    
    async def ingest_queries(self, queries: List[Dict[str, Any]], batch_size: int = 10):
        """Ingest queries in batches."""
        self.stats["total_queries"] = len(queries)
        self.stats["start_time"] = datetime.now()
        
        print(f"\nStarting ingestion of {len(queries)} queries (batch size: {batch_size})")
        print("-" * 60)
        
        for i in range(0, len(queries), batch_size):
            batch = queries[i:i + batch_size]
            batch_start = i + 1
            batch_end = min(i + batch_size, len(queries))
            
            print(f"Processing batch {batch_start}-{batch_end}...")
            
            for j, query in enumerate(batch):
                query_index = i + j + 1
                try:
                    # Validate and normalize query
                    normalized_query = self.validate_query(query, query_index)
                    
                    # Store in LanceDB
                    query_id = await self.service.store_sql_query(normalized_query)
                    
                    self.stats["successful_ingestions"] += 1
                    print(f"  ✓ Query {query_index}: {query_id[:8]}... | {normalized_query['sql_query'][:60]}...")
                    
                except Exception as e:
                    self.stats["failed_ingestions"] += 1
                    error_msg = f"Query {query_index}: {str(e)}"
                    self.stats["errors"].append(error_msg)
                    print(f"  ✗ {error_msg}")
            
            # Small delay between batches to avoid overwhelming the system
            if i + batch_size < len(queries):
                await asyncio.sleep(0.1)
        
        self.stats["end_time"] = datetime.now()
    
    def print_summary(self):
        """Print ingestion summary statistics."""
        duration = (self.stats["end_time"] - self.stats["start_time"]).total_seconds()
        success_rate = (self.stats["successful_ingestions"] / self.stats["total_queries"]) * 100
        
        print("\n" + "=" * 60)
        print("INGESTION SUMMARY")
        print("=" * 60)
        print(f"Total Queries:        {self.stats['total_queries']}")
        print(f"Successful:           {self.stats['successful_ingestions']}")
        print(f"Failed:               {self.stats['failed_ingestions']}")
        print(f"Success Rate:         {success_rate:.1f}%")
        print(f"Processing Time:      {duration:.2f} seconds")
        print(f"Queries per Second:   {self.stats['total_queries'] / duration:.2f}")
        
        if self.stats["errors"]:
            print(f"\nErrors ({len(self.stats['errors'])}):")
            for error in self.stats["errors"][:5]:  # Show first 5 errors
                print(f"  - {error}")
            if len(self.stats["errors"]) > 5:
                print(f"  ... and {len(self.stats['errors']) - 5} more")
    
    async def cleanup(self):
        """Clean up resources."""
        if self.service:
            await self.service.cleanup()


async def main():
    """Main ingestion workflow."""
    print("SQL Query Ingestion Tool")
    print("=" * 40)
    
    # Parse command line arguments
    if len(sys.argv) < 2:
        print("Usage: python ingest_sql_queries.py <json_file_path> [batch_size]")
        print("\nExample:")
        print("  python ingest_sql_queries.py sql_queries.json")
        print("  python ingest_sql_queries.py sample_queries.json 20")
        sys.exit(1)
    
    json_file_path = Path(sys.argv[1])
    batch_size = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    
    # Resolve path relative to script location if not absolute
    if not json_file_path.is_absolute():
        json_file_path = Path(__file__).parent / json_file_path
    
    ingester = SQLQueryIngester()
    
    try:
        # Initialize service
        await ingester.initialize()
        
        # Load queries from JSON
        queries = await ingester.load_queries_from_json(json_file_path)
        
        if not queries:
            print("No queries found in JSON file.")
            return
        
        # Confirm before proceeding
        print(f"\nReady to ingest {len(queries)} queries into LanceDB.")
        response = input("Continue? (y/N): ").strip().lower()
        
        if response != 'y':
            print("Ingestion cancelled.")
            return
        
        # Run ingestion
        await ingester.ingest_queries(queries, batch_size)
        
        # Print summary
        ingester.print_summary()
        
        # Show database statistics
        if ingester.stats["successful_ingestions"] > 0:
            print("\nDatabase Statistics:")
            db_stats = await ingester.service.get_statistics()
            print(f"  Total queries in DB:  {db_stats['total_queries']}")
            print(f"  Databases:            {list(db_stats['databases'].keys())}")
            print(f"  Query types:          {list(db_stats['query_types'].keys())}")
    
    except KeyboardInterrupt:
        print("\nIngestion interrupted by user.")
    except Exception as e:
        print(f"\nError during ingestion: {e}")
        import traceback
        traceback.print_exc()
    finally:
        await ingester.cleanup()


if __name__ == "__main__":
    asyncio.run(main())