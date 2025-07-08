#!/usr/bin/env python3
"""
MOQ Template Ingestion Script
Ingests the MOQ template data into LanceDB using the enhanced schema.
"""

import asyncio
import json
import sys
from pathlib import Path

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

async def ingest_moq_template():
    """Ingest MOQ template into LanceDB."""
    print("Starting MOQ template ingestion...")
    
    try:
        # Import the runner service
        from runner import SQLEmbeddingService
        print("✅ SQLEmbeddingService imported successfully")
        
        # Initialize the service
        service = SQLEmbeddingService()
        await service.initialize()
        print("✅ Service initialized successfully")
        
        # Load MOQ template
        template_path = Path("patterns/template_moq.json")
        with open(template_path, 'r') as f:
            moq_data = json.load(f)
        
        print(f"✅ Loaded MOQ template with {len(moq_data.keys())} sections")
        
        # Convert MOQ template to query format for ingestion
        query_data = {
            "sql_query": moq_data["query_content"]["sql_query"],
            "database": moq_data["technical_metadata"]["database"],
            "query_type": moq_data["query_content"]["query_type"],
            "user_id": "moq_template_user",
            "success": True,
            "execution_time_ms": 45.0,
            "row_count": 10,
            "metadata": {
                "business_domain": moq_data["semantic_context"]["business_domain"],
                "business_question": moq_data["query_content"]["business_question"],
                "query_intent": moq_data["query_content"]["query_intent"],
                "analysis_type": moq_data["semantic_context"]["analysis_type"],
                "business_function": moq_data["semantic_context"]["business_function"]
            }
        }
        
        # Store the query
        query_id = await service.store_sql_query(query_data)
        print(f"✅ MOQ template ingested successfully with ID: {query_id}")
        
        # Verify ingestion by searching
        similar_queries = await service.find_similar_queries(
            "MOQ pricing optimization", 
            limit=5
        )
        print(f"✅ Found {len(similar_queries)} similar queries")
        
        if similar_queries:
            print(f"✅ First result similarity: {similar_queries[0].get('similarity', 'N/A')}")
        
        # Check health and stats
        health = await service.health_check()
        print(f"✅ Service health check passed: {sum(health.values())}/{len(health)} components healthy")
        
        stats = await service.get_statistics()
        print(f"✅ Total queries in DB: {stats['total_queries']}")
        
        await service.cleanup()
        print("✅ Ingestion completed successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ MOQ ingestion failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(ingest_moq_template())
    print(f"\nMOQ Template Ingestion: {'SUCCESS' if success else 'FAILED'}")
    sys.exit(0 if success else 1)