#!/usr/bin/env python3
"""Batch ingestion script for Qdrant patterns.

Run this script to ingest all JSON files from the configured patterns directory.
"""

import asyncio
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from qdrant.runner import get_qdrant_service
from qdrant.config import settings
from qdrant.qdrant_logging import logger


async def main():
    """Perform batch ingestion from configured directory."""
    service = await get_qdrant_service()
    
    print(f"\nStarting batch ingestion from: {settings.file_path}")
    print("=" * 60)
    
    try:
        stats = await service.ingest_from_directory()
        
        print("\nIngestion Results:")
        print(f"  Total files: {stats['total_files']}")
        print(f"  Total entries: {stats['total_entries']}")
        print(f"  Successful: {stats['success']}")
        print(f"  Failed: {stats['failed']}")
        if stats['total_entries'] > 0:
            print(f"  Success rate: {stats['success']/stats['total_entries']*100:.1f}%")
        print(f"\nFiles processed: {', '.join(stats['files_processed'])}")
        
        return 0 if stats['failed'] == 0 else 1
        
    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        return 1
    finally:
        await service.close()


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)