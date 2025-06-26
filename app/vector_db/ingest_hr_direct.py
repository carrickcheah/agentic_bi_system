"""
Direct HR Workforce Pattern Ingestion to Qdrant

Bypasses MCP and connects directly to Qdrant cloud instance.
Ingests hr_workforce.json patterns with embeddings.
"""

import json
import asyncio
from pathlib import Path
from typing import List, Dict, Any

# Check for required packages
try:
    from qdrant_client import QdrantClient
    from qdrant_client.http import models
    from sentence_transformers import SentenceTransformer
    from config import settings
except ImportError as e:
    print(f"❌ Missing required package: {e}")
    print("Run: uv add qdrant-client sentence-transformers")
    exit(1)


class DirectQdrantIngester:
    """Direct Qdrant ingestion without MCP layer."""
    
    def __init__(self):
        # Qdrant configuration from Pydantic settings
        self.qdrant_url = settings.qdrant_url
        self.qdrant_api_key = settings.qdrant_api_key
        self.collection_name = settings.collection_name
        self.embedding_model_name = settings.embedding_model
        
        # Initialize clients
        self.qdrant_client = None
        self.embeddings_model = None
        
    def initialize(self):
        """Initialize Qdrant client and embedding model."""
        print("🔧 Initializing direct Qdrant connection...")
        
        # Initialize Qdrant client
        self.qdrant_client = QdrantClient(
            url=self.qdrant_url,
            api_key=self.qdrant_api_key,
            timeout=60
        )
        
        # Initialize embedding model
        print(f"📡 Loading sentence transformer model: {self.embedding_model_name}...")
        self.embeddings_model = SentenceTransformer(self.embedding_model_name)
        
        # Create collection if it doesn't exist
        self._ensure_collection_exists()
        
        print("✅ Direct Qdrant connection initialized")
    
    def _ensure_collection_exists(self):
        """Create collection if it doesn't exist."""
        try:
            # Check if collection exists
            collection_info = self.qdrant_client.get_collection(self.collection_name)
            print(f"📊 Collection '{self.collection_name}' already exists")
        except Exception:
            # Collection doesn't exist, create it
            print(f"🔨 Creating collection '{self.collection_name}'...")
            self.qdrant_client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=384,  # all-MiniLM-L6-v2 embedding size
                    distance=models.Distance.COSINE
                )
            )
            print(f"✅ Created collection '{self.collection_name}'")
        
    def load_hr_patterns(self) -> List[Dict[str, Any]]:
        """Load HR workforce patterns from JSON file."""
        patterns_file = Path(__file__).parent.parent / "data" / "patterns" / "hr_workforce.json"
        
        if not patterns_file.exists():
            raise FileNotFoundError(f"HR patterns file not found: {patterns_file}")
        
        with open(patterns_file, 'r', encoding='utf-8') as f:
            patterns = json.load(f)
        
        print(f"📄 Loaded {len(patterns)} HR workforce patterns")
        return patterns
    
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
        embedding_parts = [part.strip() for part in embedding_parts if part.strip()]
        return " | ".join(embedding_parts)
    
    def ingest_patterns(self, patterns: List[Dict[str, Any]]) -> int:
        """Ingest patterns directly into Qdrant."""
        print(f"🚀 Starting ingestion of {len(patterns)} HR patterns...")
        
        points = []
        successfully_processed = 0
        
        for i, pattern in enumerate(patterns):
            try:
                # Create unique ID
                point_id = f"hr_workforce_{i:03d}"
                
                # Generate embedding text
                embedding_text = self.create_embedding_text(pattern)
                
                # Generate embedding vector
                embedding_vector = self.embeddings_model.encode(embedding_text).tolist()
                
                # Prepare metadata
                payload = {
                    "pattern_id": point_id,
                    "information": pattern["information"],
                    "business_domain": "hr_workforce",
                    "source_file": "hr_workforce.json",
                    "complexity": pattern["metadata"].get("complexity", "unknown"),
                    "timeframe": pattern["metadata"].get("timeframe", "unknown"),
                    "user_roles": pattern["metadata"].get("user_roles", []),
                    "confidence_indicators": pattern["metadata"].get("confidence_indicators", []),
                    "expected_deliverables": pattern["metadata"].get("expected_deliverables", []),
                    "success_rate": pattern["metadata"].get("success_rate", 0.5),
                    "pattern_workflow": pattern["metadata"].get("pattern", "")
                }
                
                # Create point
                point = models.PointStruct(
                    id=i,  # Use numeric ID
                    vector=embedding_vector,
                    payload=payload
                )
                
                points.append(point)
                successfully_processed += 1
                
                if (i + 1) % 5 == 0:
                    print(f"  📝 Processed {i + 1}/{len(patterns)} patterns")
                
            except Exception as e:
                print(f"❌ Error processing pattern {i}: {e}")
                continue
        
        # Upsert all points to Qdrant
        if points:
            print(f"💾 Uploading {len(points)} points to Qdrant...")
            
            try:
                self.qdrant_client.upsert(
                    collection_name=self.collection_name,
                    points=points
                )
                print(f"✅ Successfully uploaded {len(points)} HR patterns to Qdrant!")
                
            except Exception as e:
                print(f"❌ Failed to upload to Qdrant: {e}")
                return 0
        
        return successfully_processed
    
    def verify_ingestion(self) -> bool:
        """Verify that patterns were successfully ingested."""
        print("🔍 Verifying pattern ingestion...")
        
        try:
            # Test search for HR-related content  
            search_result = self.qdrant_client.search(
                collection_name=self.collection_name,
                query_vector=self.embeddings_model.encode("employee performance management").tolist(),
                limit=3,
                with_payload=True
            )
            
            if search_result:
                print(f"✅ Search verification successful: {len(search_result)} results found")
                
                # Show sample results
                for i, result in enumerate(search_result[:2]):
                    payload = result.payload
                    score = result.score
                    pattern_id = payload.get("pattern_id", "unknown")
                    info = payload.get("information", "")[:80] + "..."
                    
                    print(f"  {i+1}. {pattern_id} (score: {score:.3f})")
                    print(f"     {info}")
                
                return True
            else:
                print("❌ No search results found - ingestion may have failed")
                return False
                
        except Exception as e:
            print(f"❌ Verification failed: {e}")
            return False
    
    def get_collection_info(self):
        """Get information about the collection."""
        try:
            collection_info = self.qdrant_client.get_collection(self.collection_name)
            print(f"📊 Collection '{self.collection_name}' info:")
            print(f"  Points count: {collection_info.points_count}")
            print(f"  Vector size: {collection_info.config.params.vectors.size}")
            print(f"  Distance metric: {collection_info.config.params.vectors.distance}")
            
        except Exception as e:
            print(f"⚠️  Could not get collection info: {e}")


async def main():
    """Main ingestion process."""
    print("🎯 HR Workforce Pattern Ingestion - Direct Qdrant")
    print("=" * 50)
    
    try:
        # Initialize ingester
        ingester = DirectQdrantIngester()
        ingester.initialize()
        
        # Load patterns
        patterns = ingester.load_hr_patterns()
        
        # Ingest patterns
        ingested_count = ingester.ingest_patterns(patterns)
        
        if ingested_count > 0:
            # Verify ingestion
            verification_success = ingester.verify_ingestion()
            
            # Show collection info
            ingester.get_collection_info()
            
            if verification_success:
                print(f"🎉 SUCCESS: {ingested_count} HR patterns ingested and verified!")
                return True
            else:
                print(f"⚠️  WARNING: {ingested_count} patterns ingested but verification failed")
                return False
        else:
            print("❌ FAILED: No patterns were successfully ingested")
            return False
            
    except Exception as e:
        print(f"❌ Ingestion failed: {e}")
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)