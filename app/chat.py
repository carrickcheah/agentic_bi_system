#!/usr/bin/env python3
"""
Interactive chat interface for Agentic BI using AgenticBiFlow.
Provides a simple command-line interface for business intelligence queries.
"""

import asyncio
import sys
from datetime import datetime
import uuid
from typing import Optional

# Import the high-level interface and question checker from main
from main import AgenticBiFlow, question_checker


async def run_chat():
    """Run the interactive chat interface."""
    print("=" * 60)
    print("Welcome to Agentic BI - Business Intelligence Assistant")
    print("=" * 60)
    print("\nInitializing services...")
    
    # Initialize the flow
    flow = AgenticBiFlow()
    await flow.initialize()
    
    print("✅ All services initialized successfully!")
    print("\n💡 Tips:")
    print("  - Ask business questions like 'What were last month's sales?'")
    print("  - Type 'help' for examples")
    print("  - Type 'exit' to quit")
    print("  - Type 'stats' to see cache statistics")
    print("\n" + "=" * 60 + "\n")
    
    # Session tracking
    session_id = str(uuid.uuid4())
    query_count = 0
    
    while True:
        try:
            # Get user input
            user_query = input("\n🤖 You: ").strip()
            
            if not user_query:
                continue
            
            # Handle special commands
            if user_query.lower() in ['exit', 'quit', 'bye', 'q']:
                print("\n👋 Goodbye! Thank you for using Agentic BI.")
                break
            
            elif user_query.lower() == 'help':
                print("\n📚 Example Questions:")
                print("  • What were yesterday's sales?")
                print("  • Show me production efficiency by line")
                print("  • Why did customer satisfaction drop last quarter?")
                print("  • Compare revenue across product categories")
                print("  • What's the trend in order volume this month?")
                continue
            
            elif user_query.lower() == 'stats':
                stats = await flow.get_cache_stats()
                print(f"\n📊 Cache Statistics:")
                print(f"  • Total requests: {stats.get('total_requests', 0)}")
                print(f"  • Cache hits: {stats.get('anthropic_hits', 0) + stats.get('postgresql_hits', 0)}")
                print(f"  • Cache misses: {stats.get('cache_misses', 0)}")
                if stats.get('total_requests', 0) > 0:
                    hit_rate = ((stats.get('anthropic_hits', 0) + stats.get('postgresql_hits', 0)) / 
                               stats.get('total_requests', 1)) * 100
                    print(f"  • Hit rate: {hit_rate:.1f}%")
                continue
            
            # Check if it's a valid business question first
            is_valid, rejection_message = await question_checker.is_database_question(user_query)
            
            if not is_valid:
                # Not a business question - show appropriate response
                print(f"\n💬 Assistant: {rejection_message}")
                continue
            
            # Process business query
            query_count += 1
            correlation_id = f"{session_id}-{query_count}"
            start_time = datetime.now()
            
            print("\n🔍 Processing your business query...")
            
            # Track if we got a cached response
            used_cache = False
            investigation_phases = []
            
            # Use streaming to show progress
            async for update in flow.investigate_query(
                question=user_query,
                user_context={
                    "correlation_id": correlation_id,
                    "session_id": session_id,
                    "query_number": query_count
                },
                organization_context={
                    "company": "demo_corp",
                    "industry": "manufacturing"
                },
                stream_progress=True
            ):
                # Handle different update types
                if update.get("type") == "cached_response":
                    used_cache = True
                    print("⚡ Found cached response!")
                
                elif update.get("type") == "non_business_response":
                    # Handle greetings, help, etc.
                    print(f"\n💬 Assistant: {update.get('response', 'How can I help you?')}")
                    break
                
                elif update.get("type") == "progress_update":
                    # Show investigation progress
                    phase = update.get("current_phase", "")
                    progress = update.get("progress_percentage", 0)
                    message = update.get("message", "")
                    
                    if phase not in investigation_phases:
                        investigation_phases.append(phase)
                        print(f"\n📍 Phase {update.get('phase_number', 0)}/5: {phase}")
                    
                    print(f"   [{progress:3.0f}%] {message}")
                
                elif update.get("type") == "investigation_completed":
                    # Show final results
                    elapsed = (datetime.now() - start_time).total_seconds()
                    insights = update.get("insights", {})
                    
                    print(f"\n✅ Investigation completed in {elapsed:.2f}s")
                    
                    # Executive summary
                    print(f"\n📋 Executive Summary:")
                    print(f"{insights.get('executive_summary', 'No summary available')}")
                    
                    # Strategic insights
                    strategic_insights = insights.get("strategic_insights", [])
                    if strategic_insights:
                        print(f"\n💡 Strategic Insights ({len(strategic_insights)}):")
                        for i, insight in enumerate(strategic_insights[:3], 1):
                            print(f"  {i}. {insight.get('title', 'Untitled')}")
                            print(f"     {insight.get('description', '')[:100]}...")
                    
                    # Recommendations
                    recommendations = insights.get("recommendations", [])
                    if recommendations:
                        print(f"\n🎯 Key Recommendations ({len(recommendations)}):")
                        for i, rec in enumerate(recommendations[:3], 1):
                            print(f"  {i}. {rec.get('title', 'Untitled')}")
                            print(f"     Priority: {rec.get('priority', 'Medium')}")
                    
                    # Performance note
                    if used_cache:
                        print(f"\n⚡ Response served from cache (saved ~{elapsed-0.1:.1f}s)")
                
        except KeyboardInterrupt:
            print("\n\n⚠️  Interrupted. Type 'exit' to quit or continue with a new query.")
            continue
        
        except Exception as e:
            print(f"\n❌ Error: {str(e)}")
            print("Please try again with a different query.")
            continue


async def main():
    """Main entry point."""
    print("Starting Agentic BI Chat Interface...\n")
    
    try:
        await run_chat()
    except KeyboardInterrupt:
        print("\n\nShutting down...")
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    # Run the chat interface
    asyncio.run(main())