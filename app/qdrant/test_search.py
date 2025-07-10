#!/usr/bin/env python3
"""Test Qdrant search functionality and diagnose indexing issues."""

import asyncio
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from qdrant.runner import get_qdrant_service
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from qdrant.config import settings


async def diagnose_qdrant():
    """Comprehensive Qdrant diagnostics."""
    print("Qdrant Diagnostics")
    print("="*60)
    
    service = await get_qdrant_service()
    
    try:
        # 1. Collection info
        info = await service.client.get_collection(settings.collection_name)
        print(f"\n1. Collection: {settings.collection_name}")
        print(f"   - Points stored: {info.points_count}")
        print(f"   - Vectors indexed: {info.indexed_vectors_count}")
        print(f"   - Status: {info.status}")
        
        # 2. Get some points to verify data
        print("\n2. Sample data (first 3 points):")
        points = await service.client.scroll(
            collection_name=settings.collection_name,
            limit=3,
            with_payload=True,
            with_vectors=False
        )
        
        for i, point in enumerate(points[0], 1):
            print(f"\n   Point {i}:")
            print(f"   - ID: {point.id}")
            print(f"   - Question: {point.payload.get('business_question', 'N/A')[:60]}...")
            print(f"   - SQL: {point.payload.get('sql_query', 'N/A')[:60]}...")
        
        # 3. Test different search methods
        print("\n3. Testing search methods:")
        
        test_queries = [
            "What were yesterday's sales?",
            "sales revenue",
            "customer",
            "show"
        ]
        
        for query in test_queries:
            print(f"\n   Query: '{query}'")
            
            # Standard search
            results = await service.search_similar_queries(
                query=query,
                limit=3,
                threshold=0.1  # Very low threshold to catch any matches
            )
            
            print(f"   - Results: {len(results)}")
            if results:
                best = results[0]
                print(f"   - Best match: score={best['score']:.3f}, question='{best['business_question'][:40]}...'")
            
        # 4. Try direct client search (bypass service layer)
        print("\n4. Direct client search test:")
        try:
            from model import create_embedding_model
            embedding_model = create_embedding_model()
            test_embedding = await embedding_model.embed_text_async("sales revenue yesterday")
            
            direct_results = await service.client.search(
                collection_name=settings.collection_name,
                query_vector=test_embedding,
                limit=5,
                score_threshold=0.0  # No threshold
            )
            
            print(f"   - Direct search returned: {len(direct_results)} results")
            if direct_results:
                for r in direct_results[:2]:
                    print(f"   - Score: {r.score:.3f}, ID: {r.id}")
                    
        except Exception as e:
            print(f"   - Direct search failed: {e}")
        
        # 5. Collection optimizer status
        print("\n5. Checking optimizer configuration:")
        try:
            # Get collection config through API
            collection_info = await service.client.get_collection(settings.collection_name)
            if hasattr(collection_info, 'config') and hasattr(collection_info.config, 'optimizer_config'):
                optimizer = collection_info.config.optimizer_config
                print(f"   - Indexing threshold: {getattr(optimizer, 'indexing_threshold', 'unknown')}")
            else:
                print("   - Optimizer config not accessible")
        except Exception as e:
            print(f"   - Could not get optimizer config: {e}")
            
    finally:
        await service.close()


async def quick_search_test():
    """Quick search test."""
    service = await get_qdrant_service()
    
    query = input("Enter search query: ")
    
    print(f"\nSearching for: '{query}'")
    print("-"*40)
    
    results = await service.search_similar_queries(
        query=query,
        limit=5,
        threshold=0.1
    )
    
    if results:
        print(f"\nFound {len(results)} matches:")
        for i, result in enumerate(results, 1):
            print(f"\n{i}. Score: {result['score']:.3f}")
            print(f"   Question: {result['business_question']}")
            print(f"   SQL: {result['sql_query'][:100]}...")
    else:
        print("\nNo matches found!")
        
    await service.close()


async def main():
    """Main menu."""
    print("\nQdrant Search Testing")
    print("="*50)
    print("1. Run full diagnostics")
    print("2. Quick search test")
    print("3. Exit")
    
    choice = input("\nSelect option (1-3): ").strip()
    
    if choice == "1":
        await diagnose_qdrant()
    elif choice == "2":
        await quick_search_test()
    else:
        print("Goodbye!")


if __name__ == "__main__":
    asyncio.run(main())