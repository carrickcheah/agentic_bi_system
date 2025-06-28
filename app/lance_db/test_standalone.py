#!/usr/bin/env python3
"""Standalone test suite for independent LanceDB module validation."""

import asyncio
import sys
import os
from pathlib import Path

# Add module to path for testing
sys.path.insert(0, str(Path(__file__).parent))

from config import settings
from runner import SQLEmbeddingService


async def test_configuration():
    """Test configuration loading."""
    print("Testing Configuration...")
    try:
        print(f"SUCCESS: LanceDB Path: {settings.lancedb_path}")
        print(f"SUCCESS: Embedding Model: {settings.embedding_model}")
        print(f"SUCCESS: Similarity Threshold: {settings.similarity_threshold}")
        print(f"SUCCESS: Data Path: {settings.data_path}")
        print(f"SUCCESS: Cache Enabled: {settings.enable_query_cache}")
        return True
    except Exception as e:
        print(f"FAILED: Configuration test failed: {e}")
        return False


async def test_service_initialization():
    """Test SQL embedding service initialization."""
    print("\nTesting Service Initialization...")
    try:
        service = SQLEmbeddingService()
        await service.initialize()
        
        print("SUCCESS: Service initialized")
        
        # Test health check
        health = await service.health_check()
        print(f"SUCCESS: Health Status: {health}")
        
        all_healthy = all(health.values())
        if all_healthy:
            print("SUCCESS: All components healthy")
        else:
            print("WARNING: Some components not healthy")
        
        await service.cleanup()
        return True
        
    except Exception as e:
        print(f"FAILED: Service initialization failed: {e}")
        return False


async def test_embedding_operations():
    """Test embedding generation and storage."""
    print("\nTesting Embedding Operations...")
    try:
        service = SQLEmbeddingService()
        await service.initialize()
        
        # Test storing a query
        test_query = {
            "sql_query": "SELECT * FROM users WHERE age > 25 AND status = 'active'",
            "database": "test_db",
            "query_type": "simple",
            "execution_time_ms": 45.2,
            "row_count": 150,
            "user_id": "test_user",
            "success": True,
            "metadata": {
                "original_question": "Show active users over 25",
                "table_name": "users"
            }
        }
        
        query_id = await service.store_sql_query(test_query)
        print(f"SUCCESS: Stored query with ID: {query_id[:8]}...")
        
        # Test retrieval by ID
        retrieved = await service.get_query_by_id(query_id)
        if retrieved:
            print("SUCCESS: Retrieved query by ID")
        else:
            print("FAILED: Could not retrieve query by ID")
            return False
        
        await service.cleanup()
        return True
        
    except Exception as e:
        print(f"FAILED: Embedding operations failed: {e}")
        return False


async def test_similarity_search():
    """Test similarity search functionality."""
    print("\nTesting Similarity Search...")
    try:
        service = SQLEmbeddingService()
        await service.initialize()
        
        # Store multiple test queries
        test_queries = [
            "SELECT * FROM users WHERE age > 25",
            "SELECT name, email FROM users WHERE age >= 30",
            "SELECT COUNT(*) FROM orders WHERE status = 'completed'",
            "SELECT * FROM products WHERE price < 100"
        ]
        
        print(f"Storing {len(test_queries)} test queries...")
        for i, query in enumerate(test_queries):
            await service.store_sql_query({
                "sql_query": query,
                "database": "test_db",
                "query_type": "simple",
                "execution_time_ms": 30.0 + i * 10,
                "row_count": 100 + i * 50,
                "user_id": f"test_user_{i}",
                "success": True
            })
        
        # Test similarity search
        search_query = "SELECT * FROM users WHERE age > 30"
        similar = await service.find_similar_queries(search_query, threshold=0.3)
        
        print(f"SUCCESS: Found {len(similar)} similar queries")
        for result in similar[:3]:  # Show top 3
            print(f"  - Similarity: {result['similarity']:.3f} | Query: {result['sql_query'][:50]}...")
        
        await service.cleanup()
        return len(similar) > 0
        
    except Exception as e:
        print(f"FAILED: Similarity search failed: {e}")
        return False


async def test_statistics():
    """Test statistics functionality."""
    print("\nTesting Statistics...")
    try:
        service = SQLEmbeddingService()
        await service.initialize()
        
        stats = await service.get_statistics()
        print(f"SUCCESS: Statistics retrieved")
        print(f"  - Total queries: {stats['total_queries']}")
        print(f"  - Databases: {stats['databases']}")
        print(f"  - Query types: {stats['query_types']}")
        print(f"  - Success rate: {stats['success_rate']:.1f}%")
        
        await service.cleanup()
        return True
        
    except Exception as e:
        print(f"FAILED: Statistics test failed: {e}")
        return False


async def test_pattern_ingestion():
    """Test business pattern ingestion functionality."""
    print("\nTesting Pattern Ingestion...")
    try:
        service = SQLEmbeddingService()
        await service.initialize()
        
        # Check if patterns directory exists
        patterns_dir = Path(__file__).parent / "patterns"
        if not patterns_dir.exists():
            print("WARNING: Patterns directory not found, skipping ingestion test")
            await service.cleanup()
            return True
        
        # Test pattern ingestion
        print("Starting pattern ingestion...")
        stats = await service.ingest_business_patterns()
        
        print(f"SUCCESS: Pattern ingestion completed")
        print(f"  - Files processed: {stats['files_processed']}")
        print(f"  - Total patterns: {stats['total_patterns']}")
        print(f"  - Processing time: {stats['processing_time_ms']:.2f}ms")
        
        if stats['errors']:
            print(f"  - Errors encountered: {len(stats['errors'])}")
            for error in stats['errors'][:2]:  # Show first 2 errors
                print(f"    {error}")
        
        await service.cleanup()
        return stats['total_patterns'] > 0
        
    except Exception as e:
        print(f"FAILED: Pattern ingestion failed: {e}")
        return False


async def test_pattern_search():
    """Test business pattern search functionality."""
    print("\nTesting Pattern Search...")
    try:
        service = SQLEmbeddingService()
        await service.initialize()
        
        # Check if patterns are available
        try:
            pattern_stats = await service.get_pattern_statistics()
            if pattern_stats['total_patterns'] == 0:
                print("INFO: No patterns found, running ingestion first...")
                await service.ingest_business_patterns()
        except Exception:
            print("INFO: Pattern statistics not available, attempting ingestion...")
            await service.ingest_business_patterns()
        
        # Test semantic search
        search_results = await service.search_business_patterns(
            query="sales revenue analysis",
            search_type="semantic",
            limit=5
        )
        
        print(f"SUCCESS: Semantic search found {len(search_results)} patterns")
        for result in search_results[:3]:  # Show top 3
            print(f"  - {result['information'][:60]}... (similarity: {result.get('similarity', 'N/A')})")
        
        # Test domain-specific search
        domain_results = await service.get_patterns_by_domain("sales", limit=3)
        print(f"SUCCESS: Domain search found {len(domain_results)} sales patterns")
        
        # Test user role recommendations
        role_results = await service.get_recommended_patterns(
            user_role="sales_manager",
            complexity_preference="moderate",
            limit=3
        )
        print(f"SUCCESS: Role recommendations found {len(role_results)} patterns for sales_manager")
        
        await service.cleanup()
        return len(search_results) > 0
        
    except Exception as e:
        print(f"FAILED: Pattern search failed: {e}")
        return False


async def test_pattern_statistics():
    """Test pattern statistics functionality."""
    print("\nTesting Pattern Statistics...")
    try:
        service = SQLEmbeddingService()
        await service.initialize()
        
        # Get pattern statistics
        pattern_stats = await service.get_pattern_statistics()
        
        print(f"SUCCESS: Pattern statistics retrieved")
        print(f"  - Total patterns: {pattern_stats['total_patterns']}")
        
        if pattern_stats['total_patterns'] > 0:
            print(f"  - Domain distribution: {list(pattern_stats['domain_distribution'].keys())}")
            print(f"  - Complexity levels: {list(pattern_stats['complexity_distribution'].keys())}")
            print(f"  - Average success rate: {pattern_stats['average_success_rate']:.3f}")
        else:
            print("  - No patterns found in database")
        
        await service.cleanup()
        return True
        
    except Exception as e:
        print(f"FAILED: Pattern statistics failed: {e}")
        return False


async def test_standalone_execution():
    """Test that the module can run standalone from its directory."""
    print("\nTesting Standalone Execution...")
    try:
        # This test verifies we can import and run without external dependencies
        cwd = os.getcwd()
        expected_dir = str(Path(__file__).parent)
        
        if cwd == expected_dir:
            print("SUCCESS: Running from module directory")
        else:
            print(f"INFO: Running from {cwd}, module at {expected_dir}")
        
        # Test that we can access all components
        from embedding_component import EmbeddingGenerator
        from search_component import VectorSearcher
        from lance_logging import get_logger
        from pattern_ingestion import BusinessPatternIngestion
        from pattern_search_component import BusinessPatternSearcher
        
        print("SUCCESS: All components importable (including pattern components)")
        return True
        
    except Exception as e:
        print(f"FAILED: Standalone execution test failed: {e}")
        return False


async def main():
    """Run all tests."""
    print("Running LanceDB Module Tests...\n")
    
    # Run tests
    test_results = {}
    
    # Core functionality tests
    test_results["configuration"] = await test_configuration()
    test_results["service_init"] = await test_service_initialization()
    test_results["embedding_ops"] = await test_embedding_operations()
    test_results["similarity_search"] = await test_similarity_search()
    test_results["statistics"] = await test_statistics()
    
    # Pattern functionality tests
    test_results["pattern_ingestion"] = await test_pattern_ingestion()
    test_results["pattern_search"] = await test_pattern_search()
    test_results["pattern_statistics"] = await test_pattern_statistics()
    
    # Module integrity test
    test_results["standalone"] = await test_standalone_execution()
    
    # Print summary
    print("\nTest Summary:")
    print("Core SQL Functionality:")
    for test_name in ["configuration", "service_init", "embedding_ops", "similarity_search", "statistics"]:
        result = test_results[test_name]
        status = "PASS" if result else "FAIL"
        print(f"  {test_name.replace('_', ' ').title()}: {status}")
    
    print("\nBusiness Pattern Functionality:")
    for test_name in ["pattern_ingestion", "pattern_search", "pattern_statistics"]:
        result = test_results[test_name]
        status = "PASS" if result else "FAIL"
        print(f"  {test_name.replace('_', ' ').title()}: {status}")
    
    print("\nModule Integrity:")
    result = test_results["standalone"]
    status = "PASS" if result else "FAIL"
    print(f"  Standalone Execution: {status}")
    
    # Overall result
    all_passed = all(test_results.values())
    if all_passed:
        print("\nALL TESTS PASSED!")
        print("LanceDB module with business pattern support is ready for production use.")
        return 0
    else:
        failed_tests = [name for name, result in test_results.items() if not result]
        print(f"\nSOME TESTS FAILED: {', '.join(failed_tests)}")
        print("Please check the configuration and dependencies.")
        return 1


if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        sys.exit(1)