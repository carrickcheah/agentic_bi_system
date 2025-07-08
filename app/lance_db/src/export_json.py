#!/usr/bin/env python3
"""
Export MOQ Template Data to JSON
Exports the ingested MOQ template data back to JSON format.
"""

import asyncio
import json
import sys
from pathlib import Path
from datetime import datetime

# Add parent directory to path for config access
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import lancedb
    import pandas as pd
    import numpy as np
except ImportError as e:
    print(f"ERROR: Missing dependency: {e}")
    sys.exit(1)

try:
    from config import settings
except ImportError:
    print("ERROR: Cannot import config")
    sys.exit(1)


async def export_moq_to_json():
    """Export MOQ template data to JSON format."""
    print("📤 Exporting MOQ Template Data to JSON")
    print("=" * 50)
    
    # Connect to LanceDB
    db = await lancedb.connect_async(settings.data_path)
    table = await db.open_table("moq_sql_embeddings")
    
    # Get all data
    df = await table.to_pandas()
    print(f"📊 Found {len(df)} records in table")
    
    # Find MOQ records
    moq_records = df[df['business_domain'] == 'manufacturing_sales']
    
    if len(moq_records) == 0:
        print("❌ No MOQ records found")
        return
    
    # Get the latest MOQ record (most recently ingested)
    moq = moq_records.iloc[-1]  # Get the last record (most recent)
    print(f"🎯 Exporting latest MOQ record: {moq['id'][:12]}...")
    
    # Reconstruct the original MOQ template structure
    moq_json = {
        "_id": moq['id'],
        
        # Parse JSON fields back to objects
        "query_content": json.loads(moq['query_content_json']),
        "semantic_context": json.loads(moq['semantic_context_json']),
        "technical_metadata": json.loads(moq['technical_metadata_json']),
        "user_context": json.loads(moq['user_context_json']),
        "investigation_context": json.loads(moq['investigation_context_json']),
        "execution_results": json.loads(moq['execution_results_json']),
        "learning_metadata": json.loads(moq['learning_metadata_json']),
        "business_intelligence": json.loads(moq['business_intelligence_json']),
        "collaboration": json.loads(moq['collaboration_json']),
        "version_control": json.loads(moq['version_control_json']),
        "caching": json.loads(moq['caching_json']),
        "monitoring": json.loads(moq['monitoring_json']),
        "security": json.loads(moq['security_json']),
        "automation": json.loads(moq['automation_json']),
        "embeddings": json.loads(moq['embeddings_json']),
        "tags": json.loads(moq['tags_json']),
        
        # Add custom fields
        **json.loads(moq['custom_fields_json'])
    }
    
    # Add ingestion metadata
    moq_json["_ingestion_metadata"] = {
        "ingested_at": moq['created_at'].isoformat() if pd.notna(moq['created_at']) else None,
        "ingestion_system": "single_file_moq_ingester",
        "vector_dimension": len(moq['vector']) if moq['vector'] is not None else None,
        "lancedb_record_id": moq['id'],
        "business_domain": moq['business_domain'],
        "query_type": moq['query_type']
    }
    
    # Convert numpy arrays to lists for JSON serialization
    def convert_numpy(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(item) for item in obj]
        else:
            return obj
    
    moq_json = convert_numpy(moq_json)
    
    # Save to file
    output_file = Path(__file__).parent.parent / "exported_moq_template.json"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(moq_json, f, indent=2, ensure_ascii=False)
    
    print(f"✅ MOQ template exported to: {output_file}")
    print(f"📏 File size: {output_file.stat().st_size:,} bytes")
    
    # Also print to console
    print("\n" + "=" * 50)
    print("📋 JSON EXPORT (formatted)")
    print("=" * 50)
    
    print(json.dumps(moq_json, indent=2, ensure_ascii=False))
    
    print("\n" + "=" * 50)
    print("📊 EXPORT SUMMARY")
    print("=" * 50)
    print(f"Record ID: {moq_json['_id']}")
    print(f"Business Domain: {moq_json['semantic_context']['business_domain']}")
    print(f"Query Type: {moq_json['query_content']['query_type']}")
    print(f"SQL Query Length: {len(moq_json['query_content']['sql_query'])} characters")
    print(f"Vector Dimension: {moq_json['_ingestion_metadata']['vector_dimension']}")
    print(f"Custom Fields: {len(moq_json.get('moq_specific_metadata', {}))}")
    print(f"Total JSON Size: {len(json.dumps(moq_json)):,} characters")


async def export_all_records():
    """Export all records to JSON array."""
    print("📤 Exporting All Records to JSON Array")
    print("=" * 50)
    
    # Connect to LanceDB
    db = await lancedb.connect_async(settings.data_path)
    table = await db.open_table("moq_sql_embeddings")
    
    # Get all data
    df = await table.to_pandas()
    print(f"📊 Found {len(df)} records in table")
    
    all_records = []
    
    for idx, row in df.iterrows():
        record = {
            "_id": row['id'],
            "_lancedb_metadata": {
                "created_at": row['created_at'].isoformat() if pd.notna(row['created_at']) else None,
                "business_domain": row['business_domain'],
                "query_type": row['query_type'],
                "vector_dimension": len(row['vector']) if row['vector'] is not None else None
            },
            "query_content": json.loads(row['query_content_json']),
            "semantic_context": json.loads(row['semantic_context_json']),
            "technical_metadata": json.loads(row['technical_metadata_json']),
            "user_context": json.loads(row['user_context_json']),
            "investigation_context": json.loads(row['investigation_context_json']),
            "execution_results": json.loads(row['execution_results_json']),
            "custom_fields": json.loads(row['custom_fields_json'])
        }
        
        all_records.append(record)
    
    # Save to file
    output_file = Path(__file__).parent.parent / "all_records_export.json"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_records, f, indent=2, ensure_ascii=False)
    
    print(f"✅ All records exported to: {output_file}")
    print(f"📏 File size: {output_file.stat().st_size:,} bytes")
    print(f"📊 Records exported: {len(all_records)}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Export MOQ template data to JSON")
    parser.add_argument("--all", action="store_true", help="Export all records")
    parser.add_argument("--moq-only", action="store_true", help="Export MOQ template only (default)")
    
    args = parser.parse_args()
    
    if args.all:
        asyncio.run(export_all_records())
    else:
        asyncio.run(export_moq_to_json())