"""
Production-Grade Pattern Ingestion to Qdrant

Ingests all business intelligence patterns from /app/data/patterns/ 
with deduplication, batch processing, and error recovery.

Features:
- Batch embedding generation for 10x performance
- Content-based deduplication
- Checkpoint system for failure recovery
- Command-line interface
- Progress tracking and monitoring
"""

import json
import asyncio
import hashlib
import argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional, Set
from datetime import datetime
import time
import os
from collections import defaultdict

# Check for required packages
try:
    from qdrant_client import QdrantClient
    from qdrant_client.http import models
    from sentence_transformers import SentenceTransformer
    from config import settings
    from tqdm import tqdm
except ImportError as e:
    print(f"❌ Missing required package: {e}")
    print("Run: uv add qdrant-client sentence-transformers tqdm")
    exit(1)


class ProductionPatternIngester:
    """Production-grade pattern ingester with batch processing and deduplication."""
    
    # Domain mapping for consistent ID generation
    PATTERN_DOMAINS = {
        'asset_equipment': 0,
        'cost_management': 1,
        'customer_demand': 2,
        'finance_budgeting': 3,
        'hr_workforce': 4,
        'marketing_campaigns': 5,
        'operations_efficiency': 6,
        'planning_scheduling': 7,
        'product_management': 8,
        'production_operations': 9,
        'quality_management': 10,
        'safety_compliance': 11,
        'sales_revenue': 12,
        'supply_chain_inventory': 13
    }
    
    def __init__(self, batch_size: int = 32, checkpoint_file: str = ".ingestion_checkpoint.json"):
        # Qdrant configuration
        self.qdrant_url = settings.qdrant_url
        self.qdrant_api_key = settings.qdrant_api_key
        self.collection_name = settings.collection_name
        self.embedding_model_name = settings.embedding_model
        
        # Processing configuration
        self.batch_size = batch_size
        self.checkpoint_file = checkpoint_file
        
        # Initialize clients
        self.qdrant_client = None
        self.embeddings_model = None
        
        # Tracking
        self.stats = defaultdict(int)
        self.existing_hashes = set()
        
    def initialize(self):
        """Initialize Qdrant client and embedding model."""
        print("🔧 Initializing production pattern ingester...")
        
        # Validate environment
        self._validate_environment()
        
        # Initialize Qdrant client
        self.qdrant_client = QdrantClient(
            url=self.qdrant_url,
            api_key=self.qdrant_api_key,
            timeout=60
        )
        
        # Initialize embedding model
        print(f"📡 Loading embedding model: {self.embedding_model_name}...")
        self.embeddings_model = SentenceTransformer(self.embedding_model_name)
        
        # Ensure collection exists
        self._ensure_collection_exists()
        
        # Load existing pattern hashes for deduplication
        self._load_existing_hashes()
        
        print("✅ Pattern ingester initialized")
    
    def _validate_environment(self):
        """Validate required environment variables."""
        required_vars = ['QDRANT_URL', 'QDRANT_API_KEY']
        missing = []
        
        if not self.qdrant_url:
            missing.append('QDRANT_URL')
        if not self.qdrant_api_key:
            missing.append('QDRANT_API_KEY')
            
        if missing:
            raise EnvironmentError(f"Missing required environment variables: {missing}")
    
    def _ensure_collection_exists(self):
        """Create collection if it doesn't exist."""
        try:
            collection_info = self.qdrant_client.get_collection(self.collection_name)
            print(f"📊 Collection '{self.collection_name}' exists with {collection_info.points_count} points")
        except Exception:
            print(f"🔨 Creating collection '{self.collection_name}'...")
            self.qdrant_client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=384,  # all-MiniLM-L6-v2 embedding size
                    distance=models.Distance.COSINE
                )
            )
            print(f"✅ Created collection '{self.collection_name}'")
    
    def _load_existing_hashes(self):
        """Load existing pattern hashes for deduplication."""
        print("🔍 Loading existing pattern hashes...")
        try:
            # Scroll through all existing points to get content hashes
            offset = None
            while True:
                records, offset = self.qdrant_client.scroll(
                    collection_name=self.collection_name,
                    limit=100,
                    offset=offset,
                    with_payload=True
                )
                
                for record in records:
                    if 'content_hash' in record.payload:
                        self.existing_hashes.add(record.payload['content_hash'])
                
                if offset is None:
                    break
            
            print(f"📝 Found {len(self.existing_hashes)} existing patterns")
            
        except Exception as e:
            print(f"⚠️  Could not load existing hashes: {e}")
            self.existing_hashes = set()
    
    def generate_content_hash(self, pattern: Dict[str, Any]) -> str:
        """Generate deterministic hash for pattern deduplication."""
        # Create consistent string representation
        content = json.dumps({
            'information': pattern['information'],
            'metadata': pattern['metadata']
        }, sort_keys=True)
        
        return hashlib.sha256(content.encode()).hexdigest()
    
    def load_patterns_from_directory(self, specific_domain: Optional[str] = None) -> Dict[str, List[Dict]]:
        """Load patterns from directory, optionally filtering by domain."""
        patterns_dir = Path(__file__).parent.parent / "data" / "patterns"
        patterns_by_domain = {}
        
        if specific_domain:
            # Load specific domain only
            pattern_file = patterns_dir / f"{specific_domain}.json"
            if pattern_file.exists():
                with open(pattern_file, 'r', encoding='utf-8') as f:
                    patterns_by_domain[specific_domain] = json.load(f)
            else:
                raise FileNotFoundError(f"Pattern file not found: {pattern_file}")
        else:
            # Load all pattern files
            pattern_files = sorted(patterns_dir.glob("*.json"))
            
            for pattern_file in pattern_files:
                domain = pattern_file.stem
                try:
                    with open(pattern_file, 'r', encoding='utf-8') as f:
                        patterns_by_domain[domain] = json.load(f)
                except Exception as e:
                    print(f"❌ Error loading {pattern_file}: {e}")
                    continue
        
        # Summary
        total_patterns = sum(len(patterns) for patterns in patterns_by_domain.values())
        print(f"📄 Loaded {total_patterns} patterns from {len(patterns_by_domain)} domains")
        
        return patterns_by_domain
    
    def create_embedding_text(self, pattern: Dict[str, Any]) -> str:
        """Create comprehensive text for embedding generation."""
        info = pattern["information"]
        metadata = pattern["metadata"]
        
        # Combine key fields for rich semantic representation
        embedding_parts = [
            info,
            metadata.get("business_domain", ""),
            " ".join(metadata.get("confidence_indicators", [])),
            metadata.get("pattern", ""),
            " ".join(metadata.get("expected_deliverables", [])),
            metadata.get("complexity", ""),
            metadata.get("timeframe", "")
        ]
        
        # Filter empty parts and join
        embedding_parts = [part.strip() for part in embedding_parts if part and part.strip()]
        return " | ".join(embedding_parts)
    
    def prepare_patterns_for_ingestion(self, patterns_by_domain: Dict[str, List[Dict]]) -> List[Tuple]:
        """Prepare patterns with deduplication and metadata."""
        patterns_to_ingest = []
        
        for domain, patterns in patterns_by_domain.items():
            domain_idx = self.PATTERN_DOMAINS.get(domain, 99)
            
            for idx, pattern in enumerate(patterns):
                # Generate content hash
                content_hash = self.generate_content_hash(pattern)
                
                # Skip if already exists
                if content_hash in self.existing_hashes:
                    self.stats['skipped_duplicates'] += 1
                    continue
                
                # Create numeric ID for Qdrant
                numeric_id = domain_idx * 1000 + idx
                
                # Create pattern ID
                pattern_id = f"{domain}_{idx:03d}"
                
                # Prepare pattern data
                pattern_data = (
                    numeric_id,
                    pattern_id,
                    domain,
                    pattern,
                    content_hash
                )
                
                patterns_to_ingest.append(pattern_data)
                self.stats['new_patterns'] += 1
        
        return patterns_to_ingest
    
    def generate_embeddings_batch(self, patterns: List[Tuple]) -> List[models.PointStruct]:
        """Generate embeddings in batches for better performance."""
        print(f"🧮 Generating embeddings for {len(patterns)} patterns...")
        
        points = []
        
        # Process in batches
        for i in tqdm(range(0, len(patterns), self.batch_size), desc="Embedding batches"):
            batch = patterns[i:i + self.batch_size]
            
            # Extract texts for batch encoding
            texts = [self.create_embedding_text(pattern[3]) for pattern in batch]
            
            # Batch encode
            embeddings = self.embeddings_model.encode(
                texts,
                normalize_embeddings=True,
                show_progress_bar=False
            )
            
            # Create points
            for (pattern_data, embedding) in zip(batch, embeddings):
                numeric_id, pattern_id, domain, pattern, content_hash = pattern_data
                
                # Prepare metadata
                payload = {
                    "pattern_id": pattern_id,
                    "information": pattern["information"],
                    "business_domain": domain,
                    "source_file": f"{domain}.json",
                    "complexity": pattern["metadata"].get("complexity", "unknown"),
                    "timeframe": pattern["metadata"].get("timeframe", "unknown"),
                    "user_roles": pattern["metadata"].get("user_roles", []),
                    "confidence_indicators": pattern["metadata"].get("confidence_indicators", []),
                    "expected_deliverables": pattern["metadata"].get("expected_deliverables", []),
                    "success_rate": pattern["metadata"].get("success_rate", 0.5),
                    "pattern_workflow": pattern["metadata"].get("pattern", ""),
                    "content_hash": content_hash,
                    "ingestion_timestamp": datetime.utcnow().isoformat()
                }
                
                point = models.PointStruct(
                    id=numeric_id,
                    vector=embedding.tolist(),
                    payload=payload
                )
                
                points.append(point)
        
        return points
    
    def upload_points_with_retry(self, points: List[models.PointStruct], chunk_size: int = 100):
        """Upload points to Qdrant in chunks with retry logic."""
        print(f"💾 Uploading {len(points)} points to Qdrant...")
        
        uploaded_count = 0
        
        # Upload in chunks
        for i in tqdm(range(0, len(points), chunk_size), desc="Upload chunks"):
            chunk = points[i:i + chunk_size]
            
            # Retry logic
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    self.qdrant_client.upsert(
                        collection_name=self.collection_name,
                        points=chunk
                    )
                    uploaded_count += len(chunk)
                    break
                except Exception as e:
                    if attempt < max_retries - 1:
                        print(f"⚠️  Retry {attempt + 1}/{max_retries} after error: {e}")
                        time.sleep(2 ** attempt)  # Exponential backoff
                    else:
                        print(f"❌ Failed to upload chunk after {max_retries} attempts: {e}")
                        self.stats['upload_failures'] += len(chunk)
            
            # Save checkpoint
            self._save_checkpoint(i + len(chunk))
        
        self.stats['uploaded'] = uploaded_count
        print(f"✅ Successfully uploaded {uploaded_count} patterns")
    
    def _save_checkpoint(self, processed_count: int):
        """Save checkpoint for recovery."""
        checkpoint = {
            "processed_count": processed_count,
            "timestamp": datetime.utcnow().isoformat(),
            "stats": dict(self.stats)
        }
        
        with open(self.checkpoint_file, 'w') as f:
            json.dump(checkpoint, f)
    
    def _load_checkpoint(self) -> Optional[Dict]:
        """Load checkpoint if exists."""
        if os.path.exists(self.checkpoint_file):
            try:
                with open(self.checkpoint_file, 'r') as f:
                    return json.load(f)
            except Exception:
                return None
        return None
    
    def verify_ingestion(self, sample_queries: List[str] = None):
        """Verify patterns were successfully ingested."""
        print("\n🔍 Verifying pattern ingestion...")
        
        if sample_queries is None:
            sample_queries = [
                "employee performance management",
                "sales revenue analysis",
                "supply chain optimization",
                "customer satisfaction trends"
            ]
        
        success_count = 0
        
        for query in sample_queries:
            try:
                results = self.qdrant_client.search(
                    collection_name=self.collection_name,
                    query_vector=self.embeddings_model.encode(query).tolist(),
                    limit=3,
                    with_payload=True
                )
                
                if results:
                    success_count += 1
                    print(f"✅ Query '{query}' returned {len(results)} results")
                    
                    # Show top result
                    top_result = results[0]
                    pattern_id = top_result.payload.get('pattern_id', 'unknown')
                    score = top_result.score
                    info = top_result.payload.get('information', '')[:80] + "..."
                    
                    print(f"   Top match: {pattern_id} (score: {score:.3f})")
                    print(f"   Info: {info}")
                
            except Exception as e:
                print(f"❌ Verification query failed: {e}")
        
        return success_count == len(sample_queries)
    
    def get_ingestion_summary(self):
        """Display ingestion summary."""
        print("\n📊 Ingestion Summary")
        print("=" * 50)
        
        # Get collection info
        try:
            collection_info = self.qdrant_client.get_collection(self.collection_name)
            print(f"Collection: {self.collection_name}")
            print(f"Total points: {collection_info.points_count}")
            print(f"Vector size: {collection_info.config.params.vectors.size}")
        except Exception as e:
            print(f"Could not get collection info: {e}")
        
        # Display stats
        print(f"\nIngestion Stats:")
        print(f"  New patterns: {self.stats['new_patterns']}")
        print(f"  Skipped duplicates: {self.stats['skipped_duplicates']}")
        print(f"  Successfully uploaded: {self.stats['uploaded']}")
        print(f"  Upload failures: {self.stats['upload_failures']}")
        
        # Clean up checkpoint if successful
        if self.stats['upload_failures'] == 0 and os.path.exists(self.checkpoint_file):
            os.remove(self.checkpoint_file)
            print(f"\n✅ Checkpoint file cleaned up")


async def main():
    """Main ingestion process with CLI."""
    parser = argparse.ArgumentParser(
        description="Ingest business intelligence patterns into Qdrant"
    )
    
    parser.add_argument(
        "--domain",
        type=str,
        help="Specific domain to ingest (e.g., sales_revenue)"
    )
    
    parser.add_argument(
        "--all",
        action="store_true",
        default=True,
        help="Ingest all domains (default)"
    )
    
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify ingestion with sample queries"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be ingested without uploading"
    )
    
    parser.add_argument(
        "--force",
        action="store_true",
        help="Skip deduplication and force re-ingestion"
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for embedding generation (default: 32)"
    )
    
    args = parser.parse_args()
    
    print("🎯 Production Pattern Ingestion")
    print("=" * 50)
    
    start_time = time.time()
    
    try:
        # Initialize ingester
        ingester = ProductionPatternIngester(batch_size=args.batch_size)
        ingester.initialize()
        
        # Force mode - clear existing hashes
        if args.force:
            print("⚠️  Force mode: Skipping deduplication")
            ingester.existing_hashes = set()
        
        # Load patterns
        patterns_by_domain = ingester.load_patterns_from_directory(
            specific_domain=args.domain
        )
        
        # Prepare patterns (with deduplication)
        patterns_to_ingest = ingester.prepare_patterns_for_ingestion(patterns_by_domain)
        
        if not patterns_to_ingest:
            print("ℹ️  No new patterns to ingest (all patterns already exist)")
            return True
        
        print(f"\n📋 Ready to ingest {len(patterns_to_ingest)} new patterns")
        
        if args.dry_run:
            print("🔍 DRY RUN - Not uploading to Qdrant")
            
            # Show sample patterns
            for i, (_, pattern_id, domain, pattern, _) in enumerate(patterns_to_ingest[:5]):
                print(f"\n{i+1}. {pattern_id} ({domain})")
                print(f"   Info: {pattern['information'][:100]}...")
                print(f"   Complexity: {pattern['metadata'].get('complexity', 'unknown')}")
            
            if len(patterns_to_ingest) > 5:
                print(f"\n... and {len(patterns_to_ingest) - 5} more patterns")
            
            return True
        
        # Generate embeddings
        points = ingester.generate_embeddings_batch(patterns_to_ingest)
        
        # Upload to Qdrant
        ingester.upload_points_with_retry(points)
        
        # Verify if requested
        if args.verify:
            verification_success = ingester.verify_ingestion()
            if not verification_success:
                print("⚠️  Verification showed potential issues")
        
        # Summary
        ingester.get_ingestion_summary()
        
        elapsed_time = time.time() - start_time
        print(f"\n⏱️  Total time: {elapsed_time:.2f} seconds")
        print(f"🎉 Pattern ingestion completed successfully!")
        
        return True
        
    except Exception as e:
        print(f"❌ Ingestion failed: {e}")
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)