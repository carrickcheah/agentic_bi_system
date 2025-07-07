#!/usr/bin/env python3
"""
Standalone Business Pattern Ingestion Script.
Processes all business intelligence patterns from JSON files into LanceDB.
"""

import asyncio
import sys
import time
import argparse
from pathlib import Path
from typing import Dict, Any

# Add module to path for standalone execution
sys.path.insert(0, str(Path(__file__).parent))

try:
    from .pattern_ingestion import BusinessPatternIngestion
    from .lance_logging import get_logger
    from ..config import settings
except ImportError:
    from pattern_ingestion import BusinessPatternIngestion
    from lance_logging import get_logger
    from config import settings

logger = get_logger("pattern_ingestion_script")


async def run_ingestion(force_reingest: bool = False, verbose: bool = False) -> Dict[str, Any]:
    """
    Run the complete pattern ingestion process.
    
    Args:
        force_reingest: Whether to clear existing patterns and re-ingest
        verbose: Enable detailed logging output
    
    Returns:
        Dictionary with ingestion results and statistics
    """
    start_time = time.time()
    
    try:
        logger.info("Initializing Business Pattern Ingestion System...")
        
        # Initialize ingestion system
        ingestion = BusinessPatternIngestion()
        await ingestion.initialize()
        
        # Check patterns directory
        patterns_dir = Path(__file__).parent / "patterns"
        if not patterns_dir.exists():
            raise RuntimeError(f"Patterns directory not found: {patterns_dir}")
        
        pattern_files = list(patterns_dir.glob("*.json"))
        if not pattern_files:
            raise RuntimeError(f"No pattern JSON files found in {patterns_dir}")
        
        logger.info(f"Found {len(pattern_files)} pattern files to process")
        
        # Check existing patterns
        if not force_reingest:
            try:
                existing_stats = await ingestion.get_ingestion_statistics()
                if existing_stats['total_patterns'] > 0:
                    logger.info(f"Found {existing_stats['total_patterns']} existing patterns")
                    
                    response = input("Existing patterns found. Continue with ingestion? (y/N): ")
                    if response.lower() not in ['y', 'yes']:
                        print("Ingestion cancelled by user")
                        await ingestion.cleanup()
                        return {"status": "cancelled", "reason": "user_choice"}
            except Exception:
                logger.info("No existing patterns found or statistics unavailable")
        
        # Run ingestion
        logger.info("Starting pattern ingestion process...")
        ingestion_stats = await ingestion.ingest_all_patterns()
        
        # Get detailed statistics
        detailed_stats = await ingestion.get_ingestion_statistics()
        
        # Calculate total time
        total_time_ms = (time.time() - start_time) * 1000
        
        # Compile final results
        results = {
            "status": "success",
            "ingestion_stats": ingestion_stats,
            "detailed_stats": detailed_stats,
            "total_time_ms": total_time_ms,
            "patterns_found": len(pattern_files)
        }
        
        logger.info("Pattern ingestion completed successfully")
        await ingestion.cleanup()
        
        return results
        
    except Exception as e:
        logger.error(f"Pattern ingestion failed: {e}")
        return {
            "status": "error",
            "error": str(e),
            "total_time_ms": (time.time() - start_time) * 1000
        }


def print_results(results: Dict[str, Any], verbose: bool = False):
    """Print ingestion results in a formatted way."""
    
    print("\n" + "="*60)
    print("BUSINESS PATTERN INGESTION RESULTS")
    print("="*60)
    
    if results["status"] == "success":
        ingestion_stats = results["ingestion_stats"]
        detailed_stats = results["detailed_stats"]
        
        print(f"\nSUCCESS: Ingestion completed successfully!")
        print(f"Total time: {results['total_time_ms']:.2f}ms")
        print(f"Pattern files found: {results['patterns_found']}")
        
        print(f"\nIngestion Summary:")
        print(f"  Files processed: {ingestion_stats['files_processed']}")
        print(f"  Patterns ingested: {ingestion_stats['total_patterns']}")
        print(f"  Processing time: {ingestion_stats['processing_time_ms']:.2f}ms")
        
        if ingestion_stats['errors']:
            print(f"  Errors encountered: {len(ingestion_stats['errors'])}")
            if verbose:
                for i, error in enumerate(ingestion_stats['errors'], 1):
                    print(f"    {i}. {error}")
        
        print(f"\nPattern Distribution by File:")
        for domain, count in ingestion_stats['patterns_by_domain'].items():
            print(f"  {domain}: {count} patterns")
        
        print(f"\nDatabase Statistics:")
        print(f"  Total patterns in DB: {detailed_stats['total_patterns']}")
        print(f"  Average success rate: {detailed_stats['average_success_rate']:.3f}")
        
        if verbose:
            print(f"\nDomain Categories:")
            for domain, count in detailed_stats['domain_distribution'].items():
                print(f"  {domain}: {count}")
            
            print(f"\nComplexity Distribution:")
            for complexity, count in detailed_stats['complexity_distribution'].items():
                print(f"  {complexity}: {count}")
    
    elif results["status"] == "cancelled":
        print(f"\nCANCELLED: {results['reason']}")
    
    else:
        print(f"\nERROR: Ingestion failed!")
        print(f"Error: {results['error']}")
        print(f"Time before failure: {results['total_time_ms']:.2f}ms")
    
    print("="*60)


async def health_check():
    """Perform a health check of the ingestion system."""
    print("\nPerforming ingestion system health check...")
    
    try:
        # Test configuration
        print(f"Configuration check:")
        print(f"  Data path: {settings.data_path}")
        print(f"  Embedding model: {settings.embedding_model}")
        
        # Test patterns directory
        patterns_dir = Path(__file__).parent / "patterns"
        pattern_files = list(patterns_dir.glob("*.json")) if patterns_dir.exists() else []
        print(f"  Patterns directory: {patterns_dir} ({'EXISTS' if patterns_dir.exists() else 'NOT FOUND'})")
        print(f"  Pattern files found: {len(pattern_files)}")
        
        if pattern_files:
            print(f"  Sample files: {[f.name for f in pattern_files[:3]]}")
        
        # Test ingestion system initialization
        ingestion = BusinessPatternIngestion()
        await ingestion.initialize()
        
        # Get current statistics
        stats = await ingestion.get_ingestion_statistics()
        print(f"  Current patterns in DB: {stats['total_patterns']}")
        
        await ingestion.cleanup()
        
        print("SUCCESS: Health check passed")
        return True
        
    except Exception as e:
        print(f"FAILED: Health check failed - {e}")
        return False


def main():
    """Main entry point with command line argument parsing."""
    parser = argparse.ArgumentParser(
        description="Ingest business intelligence patterns into LanceDB",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python ingest_patterns.py                    # Normal ingestion
  python ingest_patterns.py --verbose          # Verbose output
  python ingest_patterns.py --force            # Force re-ingestion
  python ingest_patterns.py --health-check     # Run health check only
        """
    )
    
    parser.add_argument(
        "--force", 
        action="store_true",
        help="Force re-ingestion even if patterns already exist"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true", 
        help="Enable verbose output with detailed statistics"
    )
    
    parser.add_argument(
        "--health-check",
        action="store_true",
        help="Run health check only, don't ingest patterns"
    )
    
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress all output except errors"
    )
    
    args = parser.parse_args()
    
    if not args.quiet:
        print("Business Pattern Ingestion Script")
        print("=================================")
    
    try:
        if args.health_check:
            success = asyncio.run(health_check())
            sys.exit(0 if success else 1)
        else:
            results = asyncio.run(run_ingestion(
                force_reingest=args.force,
                verbose=args.verbose
            ))
            
            if not args.quiet:
                print_results(results, verbose=args.verbose)
            
            # Exit with appropriate code
            if results["status"] == "success":
                sys.exit(0)
            elif results["status"] == "cancelled":
                sys.exit(2)
            else:
                sys.exit(1)
                
    except KeyboardInterrupt:
        if not args.quiet:
            print("\nIngestion interrupted by user")
        sys.exit(130)
    except Exception as e:
        if not args.quiet:
            print(f"\nUnexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()