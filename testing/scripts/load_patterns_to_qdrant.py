#!/usr/bin/env python3
"""
Load Business Intelligence Patterns into Qdrant Vector Database

This script loads all business patterns from JSON files into Qdrant
for semantic search and pattern matching in business intelligence investigations.
"""

import asyncio
import sys
from pathlib import Path
import os

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Now import our modules
from app.fastmcp.client_manager import MCPClientManager
from app.intelligence.pattern_loader import PatternLoader
from app.utils.logging import logger


async def load_patterns_to_qdrant():
    """Load all business patterns into Qdrant vector database."""
    try:
        logger.info("🚀 Starting pattern loading to Qdrant...")
        
        # Verify environment variables
        required_env = ["QDRANT_URL", "QDRANT_API_KEY"]
        missing_env = [var for var in required_env if not os.getenv(var)]
        if missing_env:
            logger.error(f"❌ Missing environment variables: {missing_env}")
            return False
        
        # Initialize MCP client manager
        logger.info("🔧 Initializing MCP client manager...")
        client_manager = MCPClientManager()
        await client_manager.initialize()
        
        # Verify Qdrant client is available
        qdrant_client = client_manager.get_client("qdrant")
        if not qdrant_client:
            logger.error("❌ Qdrant client not available")
            return False
        
        logger.info("✅ Qdrant client connected")
        
        # Create pattern loader with patterns directory
        patterns_dir = PROJECT_ROOT / "app" / "data" / "patterns"
        loader = PatternLoader(client_manager, patterns_dir)
        
        # Get list of pattern files to show what we're loading
        pattern_files = list(patterns_dir.glob("*.json"))
        logger.info(f"📁 Found {len(pattern_files)} pattern files:")
        for file in pattern_files:
            logger.info(f"  • {file.name}")
        
        # Load all patterns
        logger.info("📊 Loading patterns into Qdrant...")
        success = await loader.load_all_patterns()
        
        if success:
            logger.info("✅ All patterns loaded successfully!")
            
            # Verify the loading with test searches
            logger.info("🔍 Verifying pattern loading...")
            verification = await loader.verify_patterns()
            
            if verification["success"]:
                logger.info("✅ Pattern verification successful!")
                logger.info("🔍 Test search results:")
                for query, result in verification["test_queries"].items():
                    if result["success"]:
                        logger.info(f"  • '{query}': {result['results_count']} results found")
                    else:
                        logger.warning(f"  • '{query}': {result['error']}")
                
                return True
            else:
                logger.warning("⚠️ Pattern verification failed")
                return False
        else:
            logger.error("❌ Pattern loading failed")
            return False
            
    except Exception as e:
        logger.error(f"❌ Pattern loading error: {e}")
        return False
    
    finally:
        # Cleanup
        if 'client_manager' in locals():
            try:
                await client_manager.close()
                logger.info("🔧 MCP client manager closed")
            except Exception as e:
                logger.warning(f"Warning during cleanup: {e}")


async def check_qdrant_status():
    """Check Qdrant connection and collection status."""
    try:
        logger.info("🔍 Checking Qdrant status...")
        
        client_manager = MCPClientManager()
        await client_manager.initialize()
        
        qdrant_client = client_manager.get_client("qdrant")
        if not qdrant_client:
            logger.error("❌ Qdrant client not available")
            return
        
        # Test basic connection
        try:
            result = await qdrant_client.call_tool("qdrant-find", {
                "query": "test",
                "collection_name": "valiant_vector",
                "limit": 1
            })
            logger.info(f"✅ Qdrant connection OK: {result}")
        except Exception as e:
            logger.warning(f"⚠️ Qdrant test query failed: {e}")
        
        await client_manager.close()
        
    except Exception as e:
        logger.error(f"❌ Qdrant status check failed: {e}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Load business patterns into Qdrant")
    parser.add_argument("--check", action="store_true", help="Check Qdrant status only")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")
    
    args = parser.parse_args()
    
    if args.verbose:
        import logging
        logging.getLogger().setLevel(logging.DEBUG)
    
    if args.check:
        asyncio.run(check_qdrant_status())
    else:
        success = asyncio.run(load_patterns_to_qdrant())
        sys.exit(0 if success else 1) 