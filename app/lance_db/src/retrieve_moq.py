#!/usr/bin/env python3
"""
MOQ Data Retrieval System
Simple script to retrieve and search the ingested MOQ template data.
"""

import asyncio
import json
import sys
from pathlib import Path

# Add parent directory to path for config access
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import lancedb
    import pandas as pd
    from sentence_transformers import SentenceTransformer
except ImportError as e:
    print(f"ERROR: Missing dependency: {e}")
    sys.exit(1)

try:
    from config import settings
except ImportError:
    print("ERROR: Cannot import config")
    sys.exit(1)


class MOQRetriever:
    """Simple retrieval system for MOQ template data."""
    
    def __init__(self):
        self.db = None
        self.table = None
        self.model = None
    
    async def initialize(self):
        """Initialize connections."""
        print("🔧 Initializing retriever...")
        
        # Connect to LanceDB
        self.db = await lancedb.connect_async(settings.data_path)
        self.table = await self.db.open_table("moq_sql_embeddings")
        
        # Load embedding model for similarity search
        self.model = SentenceTransformer(settings.embedding_model)
        
        print("✅ Retriever initialized")
    
    async def get_all_records(self):
        """Get all records in the table."""
        print("📋 Retrieving all records...")
        
        df = await self.table.to_pandas()
        
        print(f"📊 Found {len(df)} records:")
        for i, row in df.iterrows():
            print(f"  {i+1}. ID: {row['id'][:12]}... | Domain: {row['business_domain']} | Type: {row['query_type']}")
        
        return df
    
    async def get_moq_record(self):
        """Get the specific MOQ template record."""
        print("🎯 Retrieving MOQ template record...")
        
        # Search for MOQ-related records
        df = await self.table.to_pandas()
        moq_records = df[df['business_domain'] == 'manufacturing_sales']
        
        if len(moq_records) > 0:
            moq_record = moq_records.iloc[0]
            
            print("📋 MOQ Template Details:")
            print(f"  🆔 ID: {moq_record['id']}")
            print(f"  🏢 Business Domain: {moq_record['business_domain']}")
            print(f"  🏷️ Query Type: {moq_record['query_type']}")
            print(f"  ❓ Business Question: {moq_record['business_question']}")
            print(f"  🎯 Query Intent: {moq_record['query_intent']}")
            print(f"  👤 User Role: {moq_record['user_role']}")
            print(f"  🗄️ Database: {moq_record['database']}")
            print(f"  ⏰ Created: {moq_record['created_at']}")
            
            # Show SQL query (truncated)
            sql_query = moq_record['sql_query']
            print(f"  📝 SQL Query: {sql_query[:200]}...")
            
            # Parse and show technical metadata
            tech_metadata = json.loads(moq_record['technical_metadata_json'])
            print(f"  🔧 Tables Used: {tech_metadata.get('tables_used', [])}")
            print(f"  🔗 Join Count: {tech_metadata.get('join_count', 0)}")
            print(f"  🪟 Has Window Functions: {tech_metadata.get('has_window_functions', False)}")
            
            return moq_record
        else:
            print("❌ No MOQ records found")
            return None
    
    async def similarity_search(self, query_text: str, limit: int = 3):
        """Perform similarity search for MOQ-related queries."""
        print(f"🔍 Searching for: '{query_text}'")
        
        # Generate embedding for search query
        query_embedding = self.model.encode(query_text, convert_to_numpy=True, normalize_embeddings=True)
        
        # Perform similarity search
        search_query = self.table.search(query_embedding)
        results_df = await search_query.limit(limit).to_pandas()
        
        print(f"📊 Found {len(results_df)} similar queries:")
        
        for i, row in results_df.iterrows():
            similarity = 1 - row.get('_distance', 0)
            print(f"\n  {i+1}. Similarity: {similarity:.4f}")
            print(f"     ID: {row['id'][:12]}...")
            print(f"     Domain: {row['business_domain']}")
            print(f"     Question: {row['business_question'][:100]}...")
            print(f"     Intent: {row['query_intent']}")
        
        return results_df
    
    async def get_custom_metadata(self):
        """Show custom metadata (like MOQ-specific fields)."""
        print("🔍 Retrieving custom metadata...")
        
        df = await self.table.to_pandas()
        
        for i, row in df.iterrows():
            custom_fields = json.loads(row.get('custom_fields_json', '{}'))
            
            if custom_fields:
                print(f"\n📋 Record {i+1} Custom Fields:")
                for key, value in custom_fields.items():
                    if isinstance(value, dict):
                        print(f"  {key}: {json.dumps(value, indent=2)[:200]}...")
                    else:
                        print(f"  {key}: {value}")


async def main():
    """Main retrieval interface."""
    print("🚀 MOQ Data Retrieval System")
    print("=" * 40)
    
    retriever = MOQRetriever()
    await retriever.initialize()
    
    while True:
        print("\n📋 Available options:")
        print("1. Show all records")
        print("2. Show MOQ template details")
        print("3. Similarity search")
        print("4. Show custom metadata")
        print("5. Exit")
        
        try:
            choice = input("\nEnter choice (1-5): ").strip()
            
            if choice == "1":
                await retriever.get_all_records()
            
            elif choice == "2":
                await retriever.get_moq_record()
            
            elif choice == "3":
                query = input("Enter search query: ").strip()
                if query:
                    await retriever.similarity_search(query)
            
            elif choice == "4":
                await retriever.get_custom_metadata()
            
            elif choice == "5":
                print("👋 Goodbye!")
                break
            
            else:
                print("❌ Invalid choice. Please enter 1-5.")
                
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")


# Quick retrieval function for non-interactive use
async def quick_retrieve():
    """Quick retrieval for testing."""
    retriever = MOQRetriever()
    await retriever.initialize()
    
    print("\n" + "=" * 40)
    print("📊 QUICK DATA OVERVIEW")
    print("=" * 40)
    
    # Show all records
    df = await retriever.get_all_records()
    
    print("\n" + "=" * 40)
    print("🎯 MOQ TEMPLATE DETAILS")
    print("=" * 40)
    
    # Show MOQ details
    moq_record = await retriever.get_moq_record()
    
    print("\n" + "=" * 40)
    print("🔍 SIMILARITY SEARCH TEST")
    print("=" * 40)
    
    # Test similarity search
    await retriever.similarity_search("MOQ pricing optimization", limit=2)


if __name__ == "__main__":
    # Run quick retrieval by default
    asyncio.run(quick_retrieve())