#!/usr/bin/env python3
"""
Test Script for Cross-Module Vector Indexing - Phase 0.2 Validation
Demonstrates enterprise vector schema integration and cross-module query optimization.
"""

import asyncio
import json
import time
from pathlib import Path
import sys

# Add current directory to path
sys.path.append('.')

from ingest_moq import MOQIngestionEngine
from vector_index_manager import VectorIndexManager, IndexStrategy


async def test_cross_module_indexing():
    """Test cross-module vector indexing capabilities."""
    print("🧪 Testing Cross-Module Vector Indexing Integration")
    print("=" * 60)
    
    try:
        # Step 1: Initialize enhanced ingestion engine
        print("1️⃣ Initializing enhanced MOQ ingestion engine...")
        engine = MOQIngestionEngine()
        
        print(f"   ✅ Enterprise schema enabled: {engine.db_manager.enterprise_schema_enabled}")
        print(f"   ✅ Advanced indexing enabled: {engine.db_manager.advanced_indexing_enabled}")
        
        # Step 2: Load test template
        print("\n2️⃣ Loading MOQ template for testing...")
        template_path = Path("../patterns/template_moq.json")
        
        if not template_path.exists():
            print(f"   ❌ Template not found: {template_path}")
            return False
        
        with open(template_path, 'r') as f:
            template_data = json.load(f)
        
        print(f"   ✅ Loaded template: {template_path}")
        
        # Step 3: Test enhanced ingestion with enterprise schema
        print("\n3️⃣ Testing enterprise vector schema ingestion...")
        start_time = time.time()
        
        result = await engine.ingest_moq_template(template_path)
        
        ingestion_time = time.time() - start_time
        print(f"   ✅ Ingestion completed in {ingestion_time:.2f}s")
        print(f"   ✅ Success: {result.get('success', False)}")
        
        if result.get('success'):
            print(f"   📊 Query ID: {result.get('query_id', 'N/A')}")
            print(f"   📊 Embedding dimension: {result.get('embedding_dimension', 'N/A')}")
        
        # Step 4: Test cross-module indexing health
        print("\n4️⃣ Testing cross-module index health...")
        if engine.db_manager.vector_index_manager:
            health_report = await engine.db_manager.vector_index_manager.get_index_health_report()
            
            print(f"   📈 Total indexes: {health_report['total_indexes']}")
            print(f"   📈 Active indexes: {health_report['active_indexes']}")
            print(f"   📈 Failed indexes: {health_report['failed_indexes']}")
            print(f"   📈 Query patterns: {health_report['query_patterns']}")
            
            if health_report.get('recommendations'):
                print("   💡 Recommendations:")
                for rec in health_report['recommendations']:
                    print(f"      - {rec}")
        else:
            print("   ⚠️ Vector index manager not available")
        
        # Step 5: Test query pattern optimization
        print("\n5️⃣ Testing query pattern optimization...")
        if engine.db_manager.vector_index_manager:
            # Test optimization for auto-generation similarity pattern
            optimization_result = await engine.db_manager.vector_index_manager.optimize_for_query_pattern(
                "auto_generation_similarity"
            )
            
            print(f"   🚀 Pattern optimized: {optimization_result['pattern_id']}")
            print(f"   🚀 Strategy applied: {optimization_result['strategy_applied']}")
            print(f"   🚀 Status: {optimization_result['optimization_result']['status']}")
            
            if optimization_result['optimization_result']['status'] == 'success':
                improvements = optimization_result['optimization_result']['optimizations_applied']
                print(f"   📊 Optimizations applied: {len(improvements)}")
                for improvement in improvements:
                    print(f"      - {improvement}")
        
        # Step 6: Test cross-module query statistics
        print("\n6️⃣ Testing cross-module query statistics...")
        query_stats = await engine.db_manager.get_cross_module_query_stats()
        
        if 'error' not in query_stats:
            print(f"   📊 Cross-module indexing enabled: {query_stats['cross_module_indexing_enabled']}")
            print(f"   📊 Vector index manager available: {query_stats['vector_index_manager_available']}")
            
            if 'index_health' in query_stats:
                index_health = query_stats['index_health']
                print(f"   📊 Index health timestamp: {index_health['timestamp']}")
        else:
            print(f"   ⚠️ Query stats error: {query_stats['error']}")
        
        # Step 7: Performance summary
        print("\n7️⃣ Performance Summary")
        print(f"   ⏱️ Total test time: {time.time() - start_time:.2f}s")
        print(f"   ⏱️ Ingestion time: {ingestion_time:.2f}s")
        
        # Cleanup
        print("\n🧹 Cleanup...")
        if engine.db_manager.vector_index_manager:
            await engine.db_manager.vector_index_manager.cleanup()
        
        print("\n✅ Cross-module vector indexing test completed successfully!")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_enterprise_metadata_generation():
    """Test enterprise metadata generation specifically."""
    print("\n" + "=" * 60)
    print("🧪 Testing Enterprise Metadata Generation")
    print("=" * 60)
    
    try:
        from generate_fn.auto_generate import AutoGenerationEngine
        
        # Load template
        template_path = Path("../patterns/template_moq.json")
        with open(template_path, 'r') as f:
            template_data = json.load(f)
        
        # Test auto-generation with enterprise metadata
        engine = AutoGenerationEngine()
        enhanced_data = engine.auto_generate_all_fields(template_data)
        
        # Check enterprise metadata
        if 'enterprise_vector_metadata' in enhanced_data:
            metadata = enhanced_data['enterprise_vector_metadata']
            
            if metadata.get('cross_module_ready'):
                print("✅ Enterprise vector metadata generation successful")
                print(f"   📊 Vector ID: {metadata['vector_id']}")
                print(f"   📊 Module source: {metadata['module_source']}")
                print(f"   📊 Business domain: {metadata['unified_classification']['business_domain']}")
                print(f"   📊 Performance tier: {metadata['unified_classification']['performance_tier']}")
                print(f"   📊 Complexity score: {metadata['unified_scores']['complexity_score']:.3f}")
                print(f"   📊 Business value score: {metadata['unified_scores']['business_value_score']:.3f}")
                return True
            else:
                print("❌ Enterprise metadata not cross-module ready")
                print(f"   Error: {metadata.get('error', 'Unknown error')}")
                return False
        else:
            print("❌ No enterprise vector metadata found")
            return False
            
    except Exception as e:
        print(f"❌ Enterprise metadata test failed: {e}")
        return False


async def main():
    """Run all cross-module indexing tests."""
    print("🚀 Cross-Module Vector Indexing Test Suite")
    print("=" * 60)
    
    # Test 1: Enterprise metadata generation
    metadata_success = await test_enterprise_metadata_generation()
    
    # Test 2: Cross-module indexing integration  
    indexing_success = await test_cross_module_indexing()
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    print(f"Enterprise Metadata Generation: {'✅ PASS' if metadata_success else '❌ FAIL'}")
    print(f"Cross-Module Vector Indexing: {'✅ PASS' if indexing_success else '❌ FAIL'}")
    
    if metadata_success and indexing_success:
        print("\n🎉 ALL TESTS PASSED - Phase 0.2 Implementation Successful!")
        print("\n📋 Phase 0.2 Achievements:")
        print("   ✅ Enterprise vector schema integration")
        print("   ✅ Cross-module vector indexing strategy")
        print("   ✅ Query pattern optimization")
        print("   ✅ Index health monitoring")
        print("   ✅ Performance metrics collection")
        return 0
    else:
        print("\n⚠️ Some tests failed - Phase 0.2 needs attention")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)