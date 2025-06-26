#!/usr/bin/env python3
"""
Example: Search Business Patterns in Qdrant

This script demonstrates how to search for business intelligence patterns
after they've been loaded into Qdrant vector database.
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.fastmcp.client_manager import MCPClientManager
from app.utils.logging import logger


async def search_patterns(query: str, limit: int = 5):
    """Search for patterns matching the query."""
    try:
        logger.info(f"🔍 Searching patterns for: '{query}'")
        
        # Initialize MCP client manager
        client_manager = MCPClientManager()
        await client_manager.initialize()
        
        # Get Qdrant client
        qdrant_client = client_manager.get_client("qdrant")
        if not qdrant_client:
            logger.error("❌ Qdrant client not available")
            return []
        
        # Search for patterns
        result = await qdrant_client.call_tool("qdrant-find", {
            "query": query,
            "collection_name": "valiant_vector",
            "limit": limit
        })
        
        if result.get("success", False):
            results = result.get("results", [])
            logger.info(f"✅ Found {len(results)} matching patterns:")
            
            for i, pattern in enumerate(results, 1):
                score = pattern.get("score", 0)
                metadata = pattern.get("metadata", {})
                info = metadata.get("information", "No information")
                domain = metadata.get("business_domain", "unknown")
                pattern_type = metadata.get("pattern", "unknown")
                
                print(f"\n{i}. Score: {score:.3f}")
                print(f"   Domain: {domain}")
                print(f"   Pattern: {pattern_type}")
                print(f"   Info: {info[:100]}...")
                
            return results
        else:
            logger.error(f"❌ Search failed: {result}")
            return []
            
    except Exception as e:
        logger.error(f"❌ Search error: {e}")
        return []
    
    finally:
        if 'client_manager' in locals():
            await client_manager.close()


async def main():
    """Run example searches."""
    example_queries = [
        "budget variance analysis",
        "customer satisfaction measurement",
        "equipment maintenance optimization",
        "sales revenue forecasting",
        "inventory management",
        "cost reduction strategies",
        "employee performance tracking",
        "quality control processes"
    ]
    
    print("🔍 Business Intelligence Pattern Search Examples")
    print("=" * 50)
    
    for query in example_queries:
        print(f"\n🔍 Query: '{query}'")
        print("-" * 40)
        
        results = await search_patterns(query, limit=3)
        
        if not results:
            print("   No matching patterns found")
        
        await asyncio.sleep(0.5)  # Small delay between searches


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Search business patterns in Qdrant")
    parser.add_argument("query", nargs="?", help="Search query")
    parser.add_argument("--limit", "-l", type=int, default=5, help="Max results")
    parser.add_argument("--examples", action="store_true", help="Run example searches")
    
    args = parser.parse_args()
    
    if args.examples or not args.query:
        asyncio.run(main())
    else:
        asyncio.run(search_patterns(args.query, args.limit)) 