"""
Pattern Data Loader for Qdrant Vector Database

Loads business intelligence investigation patterns from JSON files into Qdrant
for semantic search and pattern matching capabilities.
"""

import json
import asyncio
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

try:
    # Try relative imports first (when used as module)
    from ..utils.logging import logger
    from ..fastmcp.client_manager import MCPClientManager
except ImportError:
    # Fall back to absolute imports (when run directly)
    import sys
    sys.path.append(str(Path(__file__).parent.parent))
    from utils.logging import logger
    from fastmcp.client_manager import MCPClientManager


@dataclass
class PatternData:
    """Represents a business intelligence investigation pattern."""
    information: str
    metadata: Dict[str, Any]
    source_file: str
    pattern_id: str


class PatternLoader:
    """
    Loads business intelligence patterns into Qdrant vector database.
    
    This service reads pattern JSON files and stores them in Qdrant with
    semantic embeddings for intelligent pattern matching and recommendations.
    """
    
    def __init__(self, client_manager: MCPClientManager, patterns_dir: Optional[Path] = None):
        self.client_manager = client_manager
        self.patterns_dir = patterns_dir or Path(__file__).parent.parent / "data" / "patterns"
        self.collection_name = "valiant_vector"  # From mcp.json configuration
        
    async def load_all_patterns(self) -> bool:
        """
        Load all pattern files into Qdrant vector database.
        
        Returns:
            bool: True if all patterns loaded successfully, False otherwise
        """
        try:
            logger.info("Starting pattern loading process")
            
            # Get Qdrant client
            qdrant_client = self.client_manager.get_client("qdrant")
            if not qdrant_client:
                logger.error("Qdrant client not available")
                return False
            
            # Load all pattern files
            pattern_files = list(self.patterns_dir.glob("*.json"))
            if not pattern_files:
                logger.warning(f"No pattern files found in {self.patterns_dir}")
                return False
            
            logger.info(f"Found {len(pattern_files)} pattern files to load")
            
            total_patterns = 0
            successful_loads = 0
            
            for pattern_file in pattern_files:
                try:
                    patterns = await self._load_pattern_file(pattern_file)
                    total_patterns += len(patterns)
                    
                    for pattern in patterns:
                        if await self._store_pattern(qdrant_client, pattern):
                            successful_loads += 1
                        else:
                            logger.warning(f"Failed to store pattern: {pattern.pattern_id}")
                
                except Exception as e:
                    logger.error(f"Error processing file {pattern_file}: {e}")
                    continue
            
            logger.info(f"Pattern loading completed: {successful_loads}/{total_patterns} patterns loaded successfully")
            return successful_loads == total_patterns
            
        except Exception as e:
            logger.error(f"Pattern loading failed: {e}")
            return False
    
    async def _load_pattern_file(self, file_path: Path) -> List[PatternData]:
        """Load patterns from a single JSON file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            patterns = []
            for i, item in enumerate(data):
                pattern_id = f"{file_path.stem}_{i:03d}"
                pattern = PatternData(
                    information=item["information"],
                    metadata=item["metadata"],
                    source_file=file_path.name,
                    pattern_id=pattern_id
                )
                patterns.append(pattern)
            
            logger.info(f"Loaded {len(patterns)} patterns from {file_path.name}")
            return patterns
            
        except Exception as e:
            logger.error(f"Error loading pattern file {file_path}: {e}")
            return []
    
    async def _store_pattern(self, qdrant_client, pattern: PatternData) -> bool:
        """Store a single pattern in Qdrant."""
        try:
            # Prepare metadata for Qdrant storage
            enhanced_metadata = {
                **pattern.metadata,
                "source_file": pattern.source_file,
                "pattern_id": pattern.pattern_id,
                "information": pattern.information  # Include in metadata for filtering
            }
            
            # Call qdrant-store tool via MCP
            result = await qdrant_client.call_tool(
                "qdrant-store",
                {
                    "information": pattern.information,
                    "metadata": enhanced_metadata,
                    "collection_name": self.collection_name
                }
            )
            
            if result.get("success", False):
                logger.debug(f"Successfully stored pattern: {pattern.pattern_id}")
                return True
            else:
                logger.warning(f"Failed to store pattern {pattern.pattern_id}: {result}")
                return False
                
        except Exception as e:
            logger.error(f"Error storing pattern {pattern.pattern_id}: {e}")
            return False
    
    async def clear_patterns(self) -> bool:
        """
        Clear all patterns from the collection.
        
        Note: This is a destructive operation for development/testing.
        """
        try:
            logger.warning("Clearing all patterns from Qdrant collection")
            
            qdrant_client = self.client_manager.get_client("qdrant")
            if not qdrant_client:
                logger.error("Qdrant client not available")
                return False
            
            # Note: The MCP server may not have a direct clear tool
            # This is mainly for documentation of the intended functionality
            logger.warning("Clear operation not implemented - manual collection management required")
            return False
            
        except Exception as e:
            logger.error(f"Error clearing patterns: {e}")
            return False
    
    async def verify_patterns(self) -> Dict[str, Any]:
        """
        Verify that patterns are properly loaded and searchable.
        
        Returns:
            Dict with verification results and statistics
        """
        try:
            logger.info("Verifying pattern loading")
            
            qdrant_client = self.client_manager.get_client("qdrant")
            if not qdrant_client:
                return {"success": False, "error": "Qdrant client not available"}
            
            # Test search with a sample query
            test_queries = [
                "revenue analysis",
                "equipment maintenance",
                "customer satisfaction",
                "cost optimization"
            ]
            
            verification_results = {
                "success": True,
                "test_queries": {},
                "total_searches": len(test_queries)
            }
            
            for query in test_queries:
                try:
                    result = await qdrant_client.call_tool(
                        "qdrant-find",
                        {
                            "query": query,
                            "collection_name": self.collection_name,
                            "limit": 5
                        }
                    )
                    
                    if result.get("success", False):
                        verification_results["test_queries"][query] = {
                            "success": True,
                            "results_count": len(result.get("results", []))
                        }
                    else:
                        verification_results["test_queries"][query] = {
                            "success": False,
                            "error": result.get("error", "Unknown error")
                        }
                        verification_results["success"] = False
                
                except Exception as e:
                    verification_results["test_queries"][query] = {
                        "success": False,
                        "error": str(e)
                    }
                    verification_results["success"] = False
            
            logger.info(f"Pattern verification completed: {verification_results['success']}")
            return verification_results
            
        except Exception as e:
            logger.error(f"Pattern verification failed: {e}")
            return {"success": False, "error": str(e)}


async def main():
    """Main entry point for standalone pattern loading."""
    try:
        logger.info("Starting standalone pattern loader")
        
        # Initialize MCP client manager
        client_manager = MCPClientManager()
        await client_manager.initialize()
        
        # Create pattern loader
        loader = PatternLoader(client_manager)
        
        # Load all patterns
        logger.info("Loading all patterns into Qdrant...")
        success = await loader.load_all_patterns()
        
        if success:
            logger.info("✅ All patterns loaded successfully!")
            
            # Verify the loading
            verification = await loader.verify_patterns()
            if verification["success"]:
                logger.info("✅ Pattern verification successful!")
                for query, result in verification["test_queries"].items():
                    if result["success"]:
                        logger.info(f"  • '{query}': {result['results_count']} results found")
                    else:
                        logger.warning(f"  • '{query}': {result['error']}")
            else:
                logger.warning("⚠️ Pattern verification failed")
        else:
            logger.error("❌ Pattern loading failed")
            
    except Exception as e:
        logger.error(f"Main execution failed: {e}")
        raise


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())