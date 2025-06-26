"""
Simple test script to verify pattern loading functionality.

This script tests the basic pattern loading process without requiring
the full test infrastructure.
"""

import asyncio
import sys
from pathlib import Path

# Add the app directory to Python path for imports
app_dir = Path(__file__).parent.parent.parent / "app"
sys.path.insert(0, str(app_dir))

from utils.logging import setup_logging, logger
from fastmcp.client_manager import MCPClientManager
from intelligence.pattern_loader import PatternLoader


async def test_pattern_loading():
    """Simple test of pattern loading functionality."""
    
    logger.info("🧪 Starting simple pattern loading test...")
    
    try:
        # Initialize MCP client manager
        logger.info("Initializing MCP client manager...")
        client_manager = MCPClientManager()
        await client_manager.initialize()
        
        # Check if Qdrant client is available
        qdrant_client = client_manager.get_client("qdrant")
        if not qdrant_client:
            logger.warning("⚠️  Qdrant client not available - this test requires Qdrant MCP server")
            return False
        
        logger.info("✅ Qdrant client available")
        
        # Initialize pattern loader
        logger.info("Initializing pattern loader...")
        loader = PatternLoader(client_manager)
        
        # Load patterns
        logger.info("Loading patterns into Qdrant...")
        success = await loader.load_all_patterns()
        
        if success:
            logger.info("✅ Pattern loading successful!")
            
            # Verify patterns
            logger.info("Verifying patterns...")
            verification = await loader.verify_patterns()
            
            if verification["success"]:
                logger.info("✅ Pattern verification successful!")
                logger.info(f"📊 Test queries results: {verification['test_queries']}")
                return True
            else:
                logger.error(f"❌ Pattern verification failed: {verification}")
                return False
        else:
            logger.error("❌ Pattern loading failed")
            return False
            
    except Exception as e:
        logger.error(f"❌ Test error: {e}")
        return False
    
    finally:
        # Cleanup
        try:
            await client_manager.close()
        except:
            pass


async def main():
    """Main entry point."""
    setup_logging()
    
    logger.info("🚀 Simple Pattern Loader Test")
    logger.info("=" * 50)
    
    success = await test_pattern_loading()
    
    if success:
        logger.info("🎉 Pattern loading test PASSED!")
        return 0
    else:
        logger.error("💥 Pattern loading test FAILED!")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)