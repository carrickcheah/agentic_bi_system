"""
Interactive chat interface for Agentic BI.
"""

import asyncio
import uuid


async def simple_chat():
    """Simple chat interface using the 5-phase flow."""
    # Import the clean workflow
    from core import AgenticWorkflow
    from main import initialize_async_services
    
    print("=" * 60)
    print("Agentic BI Chat - Clean Workflow")
    print("=" * 60)
    print("\n💡 Type 'exit' to quit, 'help' for examples\n")
    
    # CRITICAL: Initialize async services first (especially Qdrant!)
    await initialize_async_services()
    
    session_id = str(uuid.uuid4())
    
    # Initialize workflow
    workflow = AgenticWorkflow()
    await workflow.initialize()
    
    while True:
        try:
            # Get user input
            query = input("\n🤖 You: ").strip()
            
            if not query:
                continue
                
            if query.lower() in ['exit', 'quit', 'bye']:
                print("\n👋 Goodbye!")
                break
                
            if query.lower() == 'help':
                print("\n📚 Example questions:")
                print("  • What were yesterday's sales?")
                print("  • Why did customer satisfaction drop?")
                print("  • Show me top products by revenue")
                continue
            
            # Process query using AgenticBiFlow
            print("\n" + "-" * 40)
            
            async for update in workflow.process_query(query):
                # Handle different update types
                if update.get("type") == "non_business_response":
                    print(f"\n💬 {update.get('response')}")
                    break
                    
                elif update.get("type") == "cached_response":
                    print(f"\n⚡ Cache Hit! (saved ~15s)")
                    insights = update.get("insights", {})
                    print(f"\n📋 {insights.get('executive_summary', 'Cached result')}")
                    break
                    
                elif update.get("type") == "qdrant_fast_response":
                    print(f"\n🎯 Qdrant Exact Match! (confidence: {update.get('confidence', 0):.2f})")
                    print(f"   Original: {update.get('original_question', 'N/A')}")
                    print(f"   SQL: {update.get('cached_sql', 'N/A')[:100]}...")
                    print(f"\n📋 {update.get('result')}")
                    break
                    
                elif "phase" in update:
                    phase = update.get("phase")
                    status = update.get("status", "")
                    name = update.get("name", "")
                    
                    if phase == "pre-check" and status == "rejected":
                        # Already handled by non_business_response
                        continue
                    elif phase == "qdrant_match":
                        status = update.get("status", "")
                        if status == "exact_match":
                            print(f"\n🎯 Qdrant Match Found! Confidence: {update.get('confidence', 0):.2f}")
                        elif status == "low_confidence":
                            print(f"   ✓ Qdrant best match: {update.get('best_score', 0):.2f} (below threshold)")
                    elif phase == "routing":
                        route = update.get("route", "")
                        complexity = update.get("complexity", 0)
                        print(f"\n🔀 Routing: {route} (complexity: {complexity:.2f})")
                    elif status == "starting":
                        print(f"\n🔄 Phase {phase}: {name}...")
                    elif status == "completed":
                        if phase == 2:  # Intelligence planning
                            data = update.get("data", {})
                            print(f"   ✓ Complexity: {data.get('complexity', 0):.2f}")
                            print(f"   ✓ Method: {data.get('methodology', 'unknown')}")
                
                elif update.get("type") == "fast_response":
                    print(f"\n🚀 Fast Response (Simple Query Path)")
                    insights = update.get("insights", {})
                    print(f"\n📋 {insights.get('executive_summary', 'Query completed')}")
                    if insights.get('data'):
                        print(f"   Results: {len(insights.get('data', []))} rows")
                    
                elif update.get("type") == "investigation_completed":
                    insights = update.get("insights", {})
                    print(f"\n✅ Investigation Complete!")
                    print(f"\n📋 Summary: {insights.get('executive_summary', 'No summary')}")
                    
                    # Show insights
                    strategic_insights = insights.get("strategic_insights", [])
                    if strategic_insights:
                        print(f"\n💡 Insights ({len(strategic_insights)}):")
                        for i, insight in enumerate(strategic_insights[:3], 1):
                            print(f"   {i}. {insight.get('title', 'Untitled')}")
                    
                    # Show recommendations
                    recommendations = insights.get("recommendations", [])
                    if recommendations:
                        print(f"\n🎯 Recommendations ({len(recommendations)}):")
                        for i, rec in enumerate(recommendations[:3], 1):
                            print(f"   {i}. {rec.get('title', 'Untitled')} [{rec.get('priority', 'Medium')}]")
            
        except KeyboardInterrupt:
            print("\n\n⚠️  Interrupted. Type 'exit' to quit.")
            continue
        except Exception as e:
            print(f"\n❌ Error: {e}")
            continue