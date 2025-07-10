#!/usr/bin/env python3
"""
Recreate Qdrant collection with immediate indexing enabled.
This script will delete and recreate the collection with proper indexing settings.
"""

import asyncio
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Distance, VectorParams, OptimizersConfigDiff
from qdrant.config import settings
from qdrant.runner import get_qdrant_service


async def recreate_collection_with_indexing():
    """Delete and recreate collection with immediate indexing."""
    print(f"Recreating collection '{settings.collection_name}' with immediate indexing...")
    
    # Create client
    client = AsyncQdrantClient(
        url=settings.qdrant_url,
        api_key=settings.api_key,
        timeout=settings.timeout_seconds
    )
    
    try:
        # Delete existing collection
        try:
            await client.delete_collection(settings.collection_name)
            print(f"✓ Deleted existing collection '{settings.collection_name}'")
        except Exception as e:
            print(f"  Collection doesn't exist or couldn't delete: {e}")
        
        # Create new collection with immediate indexing
        await client.create_collection(
            collection_name=settings.collection_name,
            vectors_config=VectorParams(
                size=settings.embedding_dim,
                distance=Distance.COSINE
            ),
            optimizers_config=OptimizersConfigDiff(
                indexing_threshold=0  # Index immediately
            )
        )
        
        print(f"✓ Created collection '{settings.collection_name}' with immediate indexing")
        
        # Verify collection settings
        info = await client.get_collection(settings.collection_name)
        print(f"\nCollection info:")
        print(f"  - Points count: {info.points_count}")
        print(f"  - Indexed vectors: {info.indexed_vectors_count}")
        print(f"  - Config: {info.config}")
        
        # Re-ingest data
        print("\n" + "="*60)
        print("Now re-ingest your data:")
        print("cd /Users/carrickcheah/Project/agentic_sql/app/qdrant")
        print("python runner.py")
        print("="*60)
        
    finally:
        await client.close()


async def check_indexing_status():
    """Check current indexing status of the collection."""
    print(f"Checking indexing status for '{settings.collection_name}'...")
    
    service = await get_qdrant_service()
    
    try:
        info = await service.client.get_collection(settings.collection_name)
        print(f"\nCollection '{settings.collection_name}':")
        print(f"  - Points count: {info.points_count}")
        print(f"  - Indexed vectors count: {info.indexed_vectors_count}")
        print(f"  - Status: {info.status}")
        
        if info.config and hasattr(info.config, 'optimizer_config'):
            print(f"  - Optimizer config: {info.config.optimizer_config}")
        
        # Try a search to verify
        print("\nTesting search functionality...")
        results = await service.search_similar_queries(
            "What were yesterday's sales?",
            limit=5,
            threshold=0.5
        )
        print(f"  - Search returned {len(results)} results")
        
        if results:
            print("\n✓ Indexing is working! Found matches:")
            for i, result in enumerate(results[:3], 1):
                print(f"  {i}. Score: {result['score']:.3f} - {result['business_question'][:50]}...")
        else:
            print("\n✗ No results found - indexing may still be incomplete")
            
    finally:
        await service.close()


async def force_index_optimization():
    """Force Qdrant to optimize and build indexes."""
    print(f"Forcing index optimization for '{settings.collection_name}'...")
    
    client = AsyncQdrantClient(
        url=settings.qdrant_url,
        api_key=settings.api_key,
        timeout=settings.timeout_seconds
    )
    
    try:
        # Update collection to force optimization
        await client.update_collection(
            collection_name=settings.collection_name,
            optimizers_config=OptimizersConfigDiff(
                indexing_threshold=0
            )
        )
        
        print("✓ Triggered index optimization")
        
        # Wait a bit for indexing
        print("Waiting 5 seconds for indexing to complete...")
        await asyncio.sleep(5)
        
        # Check status
        await check_indexing_status()
        
    finally:
        await client.close()


async def main():
    """Main menu for Qdrant indexing operations."""
    print("Qdrant Indexing Management")
    print("="*50)
    print("1. Check current indexing status")
    print("2. Force index optimization")
    print("3. Recreate collection with immediate indexing")
    print("4. Exit")
    
    choice = input("\nSelect option (1-4): ").strip()
    
    if choice == "1":
        await check_indexing_status()
    elif choice == "2":
        await force_index_optimization()
    elif choice == "3":
        confirm = input("\n⚠️  This will DELETE all data! Continue? (yes/no): ")
        if confirm.lower() == "yes":
            await recreate_collection_with_indexing()
        else:
            print("Cancelled.")
    elif choice == "4":
        print("Goodbye!")
    else:
        print("Invalid choice.")


if __name__ == "__main__":
    asyncio.run(main())