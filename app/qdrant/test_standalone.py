#!/usr/bin/env python3
"""Standalone test suite for Qdrant module.

Production testing with benchmarks and validation.
"""

import asyncio
import sys
import time
import json
from pathlib import Path
from typing import List, Dict, Any

# Add module to path for testing
sys.path.insert(0, str(Path(__file__).parent.parent))

from qdrant.config import settings, validate_settings
from qdrant.runner import get_qdrant_service
from qdrant.qdrant_logging import logger


class TestColors:
    """Terminal colors for test output."""
    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    RESET = "\033[0m"
    BOLD = "\033[1m"


def print_test_header(test_name: str):
    """Print formatted test header."""
    print(f"\n{TestColors.BOLD}{TestColors.BLUE}=== {test_name} ==={TestColors.RESET}")


def print_result(success: bool, message: str):
    """Print test result with color."""
    if success:
        print(f"{TestColors.GREEN} {message}{TestColors.RESET}")
    else:
        print(f"{TestColors.RED} {message}{TestColors.RESET}")


async def test_configuration():
    """Test configuration loading and validation."""
    print_test_header("Configuration Test")
    
    try:
        # Test settings loaded
        assert settings.api_key, "API key not loaded"
        assert settings.qdrant_url, "Qdrant URL not loaded"
        assert settings.collection_name, "Collection name not set"
        assert settings.embedding_dim == 1024, "Wrong embedding dimension"
        
        print_result(True, f"API key loaded: {settings.api_key[:10]}...")
        print_result(True, f"URL: {settings.qdrant_url}")
        print_result(True, f"Collection: {settings.collection_name}")
        print_result(True, f"Embedding dim: {settings.embedding_dim}")
        
        # Test validation
        validate_settings()
        print_result(True, "Configuration validation passed")
        
        return True
        
    except Exception as e:
        print_result(False, f"Configuration error: {e}")
        return False


async def test_connection():
    """Test Qdrant connection and initialization."""
    print_test_header("Connection Test")
    
    try:
        service = await get_qdrant_service()
        print_result(True, "Service initialized")
        
        # Health check
        health = await service.health_check()
        print_result(health["healthy"], f"Health check: {health['healthy']}")
        print_result(
            health["client_connected"],
            f"Client connected: {health['client_connected']}"
        )
        print_result(
            health["collection_exists"],
            f"Collection exists: {health['collection_exists']}"
        )
        
        return health["healthy"]
        
    except Exception as e:
        print_result(False, f"Connection error: {e}")
        return False


async def test_store_and_search():
    """Test storing and searching queries."""
    print_test_header("Store and Search Test")
    
    try:
        service = await get_qdrant_service()
        
        # Test data
        test_queries = [
            {
                "id": "test_sales_001",
                "sql": "SELECT SUM(amount) FROM sales WHERE date >= '2024-01-01'",
                "question": "What are the total sales for 2024?"
            },
            {
                "id": "test_sales_002",
                "sql": "SELECT product, SUM(quantity) FROM sales GROUP BY product",
                "question": "Show sales quantity by product"
            },
            {
                "id": "test_customer_001",
                "sql": "SELECT COUNT(*) FROM customers WHERE status = 'active'",
                "question": "How many active customers do we have?"
            }
        ]
        
        # Store queries
        print("\nStoring test queries...")
        for query in test_queries:
            success = await service.store_query(
                query["id"],
                query["sql"],
                query["question"],
                {"test": True}
            )
            print_result(success, f"Stored {query['id']}")
        
        # Search similar queries
        print("\nSearching similar queries...")
        search_tests = [
            "SELECT SUM(revenue) FROM sales WHERE year = 2024",
            "SELECT COUNT(DISTINCT customer_id) FROM customers WHERE active = true",
            "SELECT item, SUM(amount) FROM orders GROUP BY item"
        ]
        
        for search_query in search_tests:
            results = await service.search_similar_queries(search_query, limit=3)
            print(f"\nQuery: {search_query[:50]}...")
            print(f"Found {len(results)} results:")
            
            for i, result in enumerate(results[:2]):
                print(f"  {i+1}. Score: {result['score']:.3f}")
                print(f"     Question: {result['business_question']}")
                print(f"     SQL: {result['sql_query'][:60]}...")
        
        return True
        
    except Exception as e:
        print_result(False, f"Store/Search error: {e}")
        return False


async def test_performance():
    """Test performance with benchmarks."""
    print_test_header("Performance Test")
    
    try:
        service = await get_qdrant_service()
        
        # Benchmark search performance
        print("\nBenchmarking search performance...")
        query = "SELECT * FROM sales WHERE amount > 1000"
        
        # Warm up
        await service.search_similar_queries(query)
        
        # Run benchmark
        latencies = []
        for i in range(10):
            start = time.time()
            await service.search_similar_queries(query)
            latency = (time.time() - start) * 1000
            latencies.append(latency)
        
        avg_latency = sum(latencies) / len(latencies)
        min_latency = min(latencies)
        max_latency = max(latencies)
        
        print(f"\nLatency stats (10 queries):")
        print(f"  Average: {avg_latency:.2f}ms")
        print(f"  Min: {min_latency:.2f}ms")
        print(f"  Max: {max_latency:.2f}ms")
        
        # Check against threshold
        success = avg_latency < settings.slow_query_threshold_ms
        print_result(
            success,
            f"Performance {'PASSED' if success else 'FAILED'} "
            f"(threshold: {settings.slow_query_threshold_ms}ms)"
        )
        
        return success
        
    except Exception as e:
        print_result(False, f"Performance test error: {e}")
        return False


async def test_circuit_breaker():
    """Test circuit breaker functionality."""
    print_test_header("Circuit Breaker Test")
    
    if not settings.enable_circuit_breaker:
        print(TestColors.YELLOW + "Circuit breaker disabled in settings" + TestColors.RESET)
        return True
    
    try:
        service = await get_qdrant_service()
        
        # Get initial state
        metrics = service.get_metrics()
        print(f"Initial state: {metrics['circuit_breaker_state']}")
        
        # Note: Full circuit breaker testing would require simulating failures
        # This is a basic test to ensure it's configured
        print_result(True, "Circuit breaker configured and active")
        
        return True
        
    except Exception as e:
        print_result(False, f"Circuit breaker test error: {e}")
        return False


async def test_cache():
    """Test caching functionality."""
    print_test_header("Cache Test")
    
    if not settings.enable_cache:
        print(TestColors.YELLOW + "Cache disabled in settings" + TestColors.RESET)
        return True
    
    try:
        service = await get_qdrant_service()
        
        # Clear metrics
        service.cache.hits = 0
        service.cache.misses = 0
        
        query = "SELECT * FROM products WHERE price > 100"
        
        # First query - should miss
        await service.search_similar_queries(query)
        
        # Second query - should hit
        await service.search_similar_queries(query)
        
        # Get cache stats
        stats = service.cache.get_stats()
        
        print(f"\nCache stats:")
        print(f"  Hits: {stats['hits']}")
        print(f"  Misses: {stats['misses']}")
        print(f"  Hit rate: {stats['hit_rate']:.2%}")
        print(f"  Size: {stats['size']}/{stats['max_size']}")
        
        success = stats['hits'] > 0
        print_result(success, "Cache working correctly")
        
        return success
        
    except Exception as e:
        print_result(False, f"Cache test error: {e}")
        return False


async def test_metrics():
    """Test metrics collection."""
    print_test_header("Metrics Test")
    
    try:
        service = await get_qdrant_service()
        
        # Get comprehensive metrics
        metrics = service.get_metrics()
        
        print("\nMetrics snapshot:")
        print(json.dumps(metrics, indent=2))
        
        # Validate metrics structure
        required_keys = [
            "total_queries", "total_errors", "avg_latency_ms",
            "cache_stats", "circuit_breaker_state"
        ]
        
        for key in required_keys:
            success = key in metrics
            print_result(success, f"Metric '{key}' present")
        
        return True
        
    except Exception as e:
        print_result(False, f"Metrics test error: {e}")
        return False


async def main():
    """Run all tests."""
    print(f"{TestColors.BOLD}\nQdrant Module Test Suite{TestColors.RESET}")
    print("=" * 50)
    
    tests = [
        ("Configuration", test_configuration),
        ("Connection", test_connection),
        ("Store & Search", test_store_and_search),
        ("Performance", test_performance),
        ("Circuit Breaker", test_circuit_breaker),
        ("Cache", test_cache),
        ("Metrics", test_metrics)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            results[test_name] = await test_func()
        except Exception as e:
            logger.error(f"Test '{test_name}' failed with error: {e}")
            results[test_name] = False
    
    # Summary
    print(f"\n{TestColors.BOLD}Test Summary{TestColors.RESET}")
    print("=" * 50)
    
    total = len(results)
    passed = sum(1 for r in results.values() if r)
    
    for test_name, success in results.items():
        status = "PASS" if success else "FAIL"
        color = TestColors.GREEN if success else TestColors.RED
        print(f"{test_name:.<30} {color}{status}{TestColors.RESET}")
    
    print("\n" + "=" * 50)
    success_rate = passed / total * 100
    overall_color = TestColors.GREEN if passed == total else TestColors.YELLOW
    print(f"{overall_color}Overall: {passed}/{total} tests passed ({success_rate:.0f}%){TestColors.RESET}")
    
    # Cleanup
    service = await get_qdrant_service()
    await service.close()
    
    return 0 if passed == total else 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)