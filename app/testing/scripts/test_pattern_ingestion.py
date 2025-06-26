"""
Test Script for Pattern Ingestion

Simple test to verify the pattern ingestion system works correctly.
"""

import asyncio
import sys
from pathlib import Path

# Add parent directories to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from vector_db.ingest_all_patterns import ProductionPatternIngester


async def test_pattern_ingestion():
    """Test pattern ingestion functionality."""
    print("🧪 Testing Pattern Ingestion System")
    print("=" * 50)
    
    try:
        # Initialize ingester
        ingester = ProductionPatternIngester()
        
        # Test environment validation
        print("1. Testing environment validation...")
        ingester._validate_environment()
        print("   ✅ Environment validation passed")
        
        # Test pattern loading
        print("\n2. Testing pattern loading...")
        patterns_by_domain = ingester.load_patterns_from_directory()
        
        total_patterns = sum(len(patterns) for patterns in patterns_by_domain.values())
        print(f"   ✅ Loaded {total_patterns} patterns from {len(patterns_by_domain)} domains")
        
        # Test deduplication preparation
        print("\n3. Testing pattern preparation...")
        # Don't load existing hashes for test
        patterns_to_ingest = ingester.prepare_patterns_for_ingestion(patterns_by_domain)
        print(f"   ✅ Prepared {len(patterns_to_ingest)} patterns for ingestion")
        
        # Test embedding text generation
        print("\n4. Testing embedding text generation...")
        if patterns_to_ingest:
            sample_pattern = patterns_to_ingest[0][3]  # Get first pattern data
            embedding_text = ingester.create_embedding_text(sample_pattern)
            print(f"   ✅ Generated embedding text: {embedding_text[:100]}...")
        
        # Test content hash generation
        print("\n5. Testing content hash generation...")
        if patterns_to_ingest:
            sample_pattern = patterns_to_ingest[0][3]
            content_hash = ingester.generate_content_hash(sample_pattern)
            print(f"   ✅ Generated content hash: {content_hash[:16]}...")
        
        print(f"\n🎉 All tests passed! Ready to ingest {len(patterns_to_ingest)} patterns")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False


if __name__ == "__main__":
    success = asyncio.run(test_pattern_ingestion())
    exit(0 if success else 1)