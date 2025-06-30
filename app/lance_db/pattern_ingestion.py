"""
Business Pattern Ingestion System for LanceDB.
Processes business intelligence patterns into searchable vector embeddings.
"""

import json
import hashlib
import time
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional

import pandas as pd
import lancedb

try:
    from .config import settings
    from .lance_logging import get_logger, log_operation, log_error, log_performance
    from .embedding_component import EmbeddingGenerator
except ImportError:
    # For standalone execution
    from config import settings
    from lance_logging import get_logger, log_operation, log_error, log_performance
    from embedding_component import EmbeddingGenerator


class BusinessPatternIngestion:
    """
    Production-grade business pattern ingestion system.
    Processes business intelligence patterns into LanceDB for semantic search.
    """
    
    def __init__(self):
        self.db = None
        self.patterns_table = None
        self.embedding_generator = None
        self._initialized = False
        
        # Pattern source directory
        self.patterns_dir = Path(__file__).parent / "patterns"
        
        # Domain mapping for normalization
        self.domain_mapping = {
            "sales_revenue": "sales",
            "customer_demand": "customer", 
            "production_operations": "production",
            "supply_chain_inventory": "supply_chain",
            "quality_management": "quality",
            "hr_workforce": "human_resources",
            "finance_budgeting": "finance",
            "marketing_campaigns": "marketing",
            "asset_equipment": "assets",
            "cost_management": "cost",
            "product_management": "product",
            "safety_compliance": "safety",
            "planning_scheduling": "planning",
            "operations_efficiency": "operations"
        }
        
        self.logger = get_logger("pattern_ingestion")
    
    async def initialize(self):
        """Initialize LanceDB connection and embedding generator."""
        if self._initialized:
            self.logger.warning("Pattern ingestion already initialized")
            return
        
        try:
            start_time = time.time()
            
            # Initialize LanceDB connection
            log_operation("Connecting to LanceDB for pattern ingestion", {"path": settings.data_path})
            self.db = await lancedb.connect_async(settings.data_path)
            
            # Initialize embedding generator
            self.embedding_generator = EmbeddingGenerator()
            await self.embedding_generator.initialize()
            
            # Create or open business patterns table
            await self._init_patterns_table()
            
            self._initialized = True
            
            duration_ms = (time.time() - start_time) * 1000
            log_performance("Pattern ingestion initialization", duration_ms)
            
        except Exception as e:
            log_error("Pattern ingestion initialization", e)
            raise RuntimeError(f"Failed to initialize pattern ingestion: {e}")
    
    async def _init_patterns_table(self):
        """Initialize the business patterns table with comprehensive schema."""
        table_name = "business_patterns"
        
        try:
            existing_tables = await self.db.table_names()
            
            if table_name not in existing_tables:
                log_operation("Creating business patterns table", {"name": table_name})
                
                # Create table with sample data to establish proper schema
                # Generate a dummy embedding vector to establish vector field types
                dummy_embedding = self.embedding_generator.generate_embedding("sample text for schema")
                
                sample_data = [{
                    "id": "schema_sample_id",
                    "information": "Sample business pattern information",
                    "information_vector": dummy_embedding,
                    "pattern_workflow": "sample_workflow → process → result",
                    "pattern_vector": dummy_embedding,
                    "user_roles": '["sample_role"]',
                    "business_domain": "sample_domain",
                    "timeframe": "sample_timeframe",
                    "complexity": "moderate",
                    "success_rate": 0.75,
                    "confidence_indicators": '["sample", "indicator"]',
                    "expected_deliverables": '["sample_deliverable"]',
                    "data_source": "schema_initialization",
                    "sample_size": 0,
                    "domain_category": "sample",
                    "created_at": datetime.now(),
                    "source_file": "schema_init.json"
                }]
                
                df = pd.DataFrame(sample_data)
                self.patterns_table = await self.db.create_table(table_name, data=df)
                
                # Remove the sample data immediately after table creation
                await self.patterns_table.delete("id = 'schema_sample_id'")
                self.logger.info(f"Created business patterns table: {table_name}")
            else:
                self.patterns_table = await self.db.open_table(table_name)
                self.logger.info(f"Opened existing patterns table: {table_name}")
                
        except Exception as e:
            log_error("Business patterns table initialization", e)
            raise
    
    async def ingest_all_patterns(self) -> Dict[str, Any]:
        """
        Ingest all business pattern files into LanceDB.
        
        Returns:
            Dictionary with ingestion statistics
        """
        if not self._initialized:
            raise RuntimeError("Pattern ingestion not initialized. Call initialize() first.")
        
        try:
            start_time = time.time()
            
            # Find all pattern JSON files
            pattern_files = list(self.patterns_dir.glob("*.json"))
            
            if not pattern_files:
                raise RuntimeError(f"No pattern files found in {self.patterns_dir}")
            
            log_operation("Starting pattern ingestion", {
                "files_found": len(pattern_files),
                "source_directory": str(self.patterns_dir)
            })
            
            total_patterns = 0
            ingestion_stats = {
                "files_processed": 0,
                "total_patterns": 0,
                "patterns_by_domain": {},
                "processing_time_ms": 0,
                "errors": []
            }
            
            # Process each pattern file
            for pattern_file in pattern_files:
                try:
                    domain = pattern_file.stem
                    patterns_ingested = await self._ingest_pattern_file(pattern_file, domain)
                    
                    total_patterns += patterns_ingested
                    ingestion_stats["files_processed"] += 1
                    ingestion_stats["patterns_by_domain"][domain] = patterns_ingested
                    
                    self.logger.info(f"Ingested {patterns_ingested} patterns from {pattern_file.name}")
                    
                except Exception as e:
                    error_msg = f"Failed to process {pattern_file.name}: {str(e)}"
                    ingestion_stats["errors"].append(error_msg)
                    log_error(f"Pattern file processing: {pattern_file.name}", e)
            
            ingestion_stats["total_patterns"] = total_patterns
            ingestion_stats["processing_time_ms"] = (time.time() - start_time) * 1000
            
            log_performance("Complete pattern ingestion", ingestion_stats["processing_time_ms"])
            self.logger.info(f"Pattern ingestion completed: {total_patterns} patterns from {len(pattern_files)} files")
            
            return ingestion_stats
            
        except Exception as e:
            log_error("Pattern ingestion", e)
            raise
    
    async def _ingest_pattern_file(self, file_path: Path, domain: str) -> int:
        """
        Ingest patterns from a single JSON file.
        
        Args:
            file_path: Path to the JSON pattern file
            domain: Domain category for the patterns
        
        Returns:
            Number of patterns successfully ingested
        """
        try:
            # Load JSON patterns
            with open(file_path, 'r', encoding='utf-8') as f:
                patterns = json.load(f)
            
            if not isinstance(patterns, list):
                raise ValueError(f"Expected list of patterns, got {type(patterns)}")
            
            # Process each pattern
            processed_patterns = []
            for i, pattern in enumerate(patterns):
                try:
                    processed_pattern = await self._process_single_pattern(
                        pattern, file_path.name, domain, i
                    )
                    processed_patterns.append(processed_pattern)
                    
                except Exception as e:
                    self.logger.warning(f"Failed to process pattern {i} in {file_path.name}: {e}")
                    continue
            
            # Insert patterns with upsert logic (simpler approach for reliability)
            if processed_patterns:
                # Store pattern IDs for this batch
                pattern_ids_in_batch = [p["id"] for p in processed_patterns]
                
                # Delete existing patterns with these IDs (if any exist)
                try:
                    for pattern_id in pattern_ids_in_batch:
                        await self.patterns_table.delete(f"id = '{pattern_id}'")
                except Exception as e:
                    # Ignore errors - patterns might not exist yet
                    pass
                
                # Insert all patterns fresh
                await self.patterns_table.add(processed_patterns)
                
                log_operation(f"Upserted patterns", {
                    "file": file_path.name,
                    "count": len(processed_patterns),
                    "operation": "delete_then_insert"
                })
            
            return len(processed_patterns)
            
        except Exception as e:
            log_error(f"Pattern file ingestion: {file_path}", e)
            return 0
    
    async def _process_single_pattern(
        self, 
        pattern_data: Dict[str, Any], 
        source_file: str, 
        domain: str,
        index: int
    ) -> Dict[str, Any]:
        """
        Process and enhance a single business pattern.
        
        Args:
            pattern_data: Raw pattern data from JSON
            source_file: Source file name
            domain: Business domain
            index: Pattern index in file
        
        Returns:
            Processed pattern ready for LanceDB insertion
        """
        # Validate pattern structure
        if "information" not in pattern_data:
            raise ValueError("Pattern missing 'information' field")
        
        if "metadata" not in pattern_data:
            raise ValueError("Pattern missing 'metadata' field")
        
        metadata = pattern_data["metadata"]
        
        # Generate dual embeddings
        info_embedding = self.embedding_generator.generate_embedding(
            pattern_data["information"]
        )
        
        workflow_embedding = self.embedding_generator.generate_embedding(
            metadata.get("pattern", "")
        )
        
        # Generate unique pattern ID
        pattern_id = hashlib.md5(
            f"{source_file}_{index}_{pattern_data['information'][:50]}".encode()
        ).hexdigest()
        
        # Process and clean metadata
        processed_pattern = {
            "id": pattern_id,
            "information": pattern_data["information"],
            "information_vector": info_embedding,
            "pattern_workflow": metadata.get("pattern", ""),
            "pattern_vector": workflow_embedding,
            "user_roles": json.dumps(metadata.get("user_roles", [])),
            "business_domain": metadata.get("business_domain", domain),
            "timeframe": metadata.get("timeframe", "unknown"),
            "complexity": metadata.get("complexity", "unknown"),
            "success_rate": float(metadata.get("success_rate", 0.0)),
            "confidence_indicators": json.dumps(metadata.get("confidence_indicators", [])),
            "expected_deliverables": json.dumps(metadata.get("expected_deliverables", [])),
            "data_source": metadata.get("data_source", "unknown"),
            "sample_size": int(metadata.get("sample_size", 0)),
            "domain_category": self._normalize_domain(domain),
            "created_at": datetime.now(),
            "source_file": source_file
        }
        
        return processed_pattern
    
    def _normalize_domain(self, domain: str) -> str:
        """Normalize domain names for consistent categorization."""
        return self.domain_mapping.get(domain, domain)
    
    async def get_ingestion_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics about ingested patterns."""
        if not self._initialized:
            raise RuntimeError("Pattern ingestion not initialized")
        
        try:
            df = await self.patterns_table.to_pandas()
            
            if len(df) == 0:
                return {
                    "total_patterns": 0,
                    "message": "No patterns ingested yet"
                }
            
            stats = {
                "total_patterns": len(df),
                "domain_distribution": df["domain_category"].value_counts().to_dict(),
                "complexity_distribution": df["complexity"].value_counts().to_dict(),
                "timeframe_distribution": df["timeframe"].value_counts().to_dict(),
                "business_domain_distribution": df["business_domain"].value_counts().to_dict(),
                "average_success_rate": float(df["success_rate"].mean()),
                "success_rate_by_complexity": df.groupby("complexity")["success_rate"].mean().to_dict(),
                "patterns_by_source_file": df["source_file"].value_counts().to_dict(),
                "latest_ingestion": df["created_at"].max().isoformat() if "created_at" in df.columns else "unknown"
            }
            
            return stats
            
        except Exception as e:
            log_error("Get ingestion statistics", e)
            raise
    
    async def cleanup(self):
        """Clean up resources."""
        try:
            if self.embedding_generator:
                await self.embedding_generator.cleanup()
            
            self._initialized = False
            self.logger.info("Pattern ingestion system cleaned up")
            
        except Exception as e:
            log_error("Pattern ingestion cleanup", e)


# Standalone execution for testing
if __name__ == "__main__":
    import asyncio
    
    async def main():
        """Test pattern ingestion system."""
        print("Initializing Business Pattern Ingestion System...")
        
        ingestion = BusinessPatternIngestion()
        await ingestion.initialize()
        
        print("Starting pattern ingestion...")
        stats = await ingestion.ingest_all_patterns()
        
        print(f"\nIngestion Results:")
        print(f"Files Processed: {stats['files_processed']}")
        print(f"Total Patterns: {stats['total_patterns']}")
        print(f"Processing Time: {stats['processing_time_ms']:.2f}ms")
        
        if stats['errors']:
            print(f"\nErrors: {len(stats['errors'])}")
            for error in stats['errors'][:3]:  # Show first 3 errors
                print(f"  - {error}")
        
        print(f"\nPatterns by Domain:")
        for domain, count in stats['patterns_by_domain'].items():
            print(f"  {domain}: {count} patterns")
        
        # Get detailed statistics
        detailed_stats = await ingestion.get_ingestion_statistics()
        print(f"\nDetailed Statistics:")
        print(f"Domain Categories: {list(detailed_stats['domain_distribution'].keys())}")
        print(f"Complexity Levels: {list(detailed_stats['complexity_distribution'].keys())}")
        print(f"Average Success Rate: {detailed_stats['average_success_rate']:.3f}")
        
        await ingestion.cleanup()
        print("\nPattern ingestion completed successfully!")
    
    asyncio.run(main())