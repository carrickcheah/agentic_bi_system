#!/usr/bin/env python3
"""
Simple MOQ Data Retrieval
Quick script to check and retrieve MOQ template data.
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
except ImportError as e:
    print(f"ERROR: Missing dependency: {e}")
    sys.exit(1)

try:
    from config import settings
except ImportError:
    print("ERROR: Cannot import config")
    sys.exit(1)


async def show_moq_data():
    """Show MOQ template data in detail."""
    print("🔍 MOQ Template Data Retrieval")
    print("=" * 50)
    
    # Connect to LanceDB
    db = await lancedb.connect_async(settings.data_path)
    table = await db.open_table("moq_sql_embeddings")
    
    # Get all data
    df = await table.to_pandas()
    print(f"📊 Total records in table: {len(df)}")
    
    # Find MOQ record
    moq_records = df[df['business_domain'] == 'manufacturing_sales']
    
    if len(moq_records) == 0:
        print("❌ No MOQ records found")
        return
    
    # Get the first MOQ record
    moq = moq_records.iloc[0]
    
    print("\n" + "=" * 50)
    print("🎯 MOQ TEMPLATE DETAILS")
    print("=" * 50)
    
    print(f"🆔 Record ID: {moq['id']}")
    print(f"🏢 Business Domain: {moq['business_domain']}")
    print(f"🏷️ Query Type: {moq['query_type']}")
    print(f"❓ Business Question: {moq['business_question']}")
    print(f"🎯 Intent: {moq['query_intent']}")
    print(f"👤 User Role: {moq['user_role']}")
    print(f"🗄️ Database: {moq['database']}")
    print(f"⏰ Created: {moq['created_at']}")
    print(f"📏 Vector Dimension: {len(moq['vector'])}")
    
    print("\n" + "=" * 50)
    print("📝 SQL QUERY")
    print("=" * 50)
    print(moq['sql_query'])
    
    print("\n" + "=" * 50)
    print("🔧 TECHNICAL METADATA")
    print("=" * 50)
    
    tech_metadata = json.loads(moq['technical_metadata_json'])
    print(f"Tables Used: {tech_metadata.get('tables_used', [])}")
    print(f"Join Count: {tech_metadata.get('join_count', 0)}")
    print(f"Has Subqueries: {tech_metadata.get('has_subqueries', False)}")
    print(f"Has Window Functions: {tech_metadata.get('has_window_functions', False)}")
    print(f"Aggregation Functions: {tech_metadata.get('aggregation_functions', [])}")
    print(f"Query Pattern: {tech_metadata.get('query_pattern', 'N/A')}")
    
    print("\n" + "=" * 50)
    print("🏢 BUSINESS CONTEXT")
    print("=" * 50)
    
    semantic_context = json.loads(moq['semantic_context_json'])
    print(f"Business Function: {semantic_context.get('business_function', 'N/A')}")
    print(f"Analysis Type: {semantic_context.get('analysis_type', 'N/A')}")
    print(f"Time Dimension: {semantic_context.get('time_dimension', 'N/A')}")
    print(f"Metrics: {semantic_context.get('metrics', [])}")
    print(f"Business Concepts: {semantic_context.get('business_concepts', [])}")
    
    print("\n" + "=" * 50)
    print("👤 USER CONTEXT")
    print("=" * 50)
    
    user_context = json.loads(moq['user_context_json'])
    print(f"User ID: {user_context.get('user_id', 'N/A')}")
    print(f"Department: {user_context.get('user_department', 'N/A')}")
    print(f"Skill Level: {user_context.get('user_skill_level', 'N/A')}")
    print(f"Organization: {user_context.get('organization_id', 'N/A')}")
    print(f"Industry: {user_context.get('industry', 'N/A')}")
    
    print("\n" + "=" * 50)
    print("🔍 INVESTIGATION CONTEXT")
    print("=" * 50)
    
    investigation_context = json.loads(moq['investigation_context_json'])
    print(f"Investigation Phase: {investigation_context.get('investigation_phase', 'N/A')}")
    print(f"Hypothesis: {investigation_context.get('hypothesis', 'N/A')}")
    print(f"Confidence Level: {investigation_context.get('confidence_level', 'N/A')}")
    print(f"Business Impact: {investigation_context.get('business_impact', 'N/A')}")
    print(f"Urgency: {investigation_context.get('urgency', 'N/A')}")
    
    related_questions = investigation_context.get('related_questions', [])
    if related_questions:
        print(f"Related Questions:")
        for i, question in enumerate(related_questions, 1):
            print(f"  {i}. {question}")
    
    print("\n" + "=" * 50)
    print("🔧 CUSTOM FIELDS")
    print("=" * 50)
    
    custom_fields = json.loads(moq['custom_fields_json'])
    if custom_fields:
        for key, value in custom_fields.items():
            print(f"{key}:")
            if isinstance(value, dict):
                for subkey, subvalue in value.items():
                    print(f"  {subkey}: {subvalue}")
            else:
                print(f"  {value}")
    else:
        print("No custom fields found")
    
    print("\n✅ MOQ template data successfully retrieved and displayed!")


if __name__ == "__main__":
    asyncio.run(show_moq_data())