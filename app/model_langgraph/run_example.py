#!/usr/bin/env python3
"""
Simple example to run the LangGraph model.
Shows how to use the graph directly and through the compatibility layer.
"""

import asyncio
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from model_langgraph.runner import ModelManager
from model_langgraph.graph import create_model_graph, visualize_graph
from model_langgraph.state import ModelState


async def example_1_basic_usage():
    """Example 1: Basic usage through ModelManager (compatibility layer)."""
    print("\n" + "="*60)
    print("Example 1: Basic Usage with ModelManager")
    print("="*60)
    
    # Initialize the model manager
    manager = ModelManager()
    
    # Generate a simple response
    prompt = "What is the capital of Japan? Answer in one sentence."
    print(f"\nPrompt: {prompt}")
    print("Generating response...")
    
    try:
        response = await manager.generate_response(
            prompt=prompt,
            max_tokens=100,
            temperature=0
        )
        print(f"\nResponse: {response}")
        print(f"Current model used: {manager.get_current_model()}")
    except Exception as e:
        print(f"Error: {e}")


async def example_2_streaming():
    """Example 2: Streaming response."""
    print("\n" + "="*60)
    print("Example 2: Streaming Response")
    print("="*60)
    
    manager = ModelManager()
    
    prompt = "List 3 benefits of using LangGraph for AI orchestration."
    print(f"\nPrompt: {prompt}")
    print("Streaming response: ", end="", flush=True)
    
    try:
        async for chunk in manager.generate_response_stream(
            prompt=prompt,
            max_tokens=200,
            temperature=0.7
        ):
            print(chunk, end="", flush=True)
        print("\n")
    except Exception as e:
        print(f"\nError: {e}")


async def example_3_direct_graph_usage():
    """Example 3: Direct graph usage for advanced control."""
    print("\n" + "="*60)
    print("Example 3: Direct Graph Usage (Advanced)")
    print("="*60)
    
    # Create the graph
    graph = create_model_graph()
    
    # Create initial state with custom settings
    initial_state: ModelState = {
        "prompt": "Explain what LangGraph is in 2 sentences.",
        "max_tokens": 100,
        "temperature": 0.5,
        "use_system_prompt": False,
        "correlation_id": "example-123",
        "is_streaming": False,
        "request_priority": "high",
        "business_domain": "technology",
        # Initialize required fields
        "system_prompt": None,
        "schema_info": None,
        "available_models": {},
        "current_model": None,
        "attempted_models": [],
        "model_health_scores": {},
        "response": None,
        "response_chunks": [],
        "error": None,
        "error_count": 0,
        "last_error_timestamp": None,
        "retry_count": 0,
        "max_retries": 3,
        "start_time": None,
        "end_time": None,
        "model_latencies": {},
        "total_tokens_used": 0,
        "cache_key": None,
        "cache_hit": False,
        "anthropic_cache_enabled": False,
        "should_retry": False,
        "should_fallback": False,
        "next_model": None,
        "final_status": None,
        "user_role": None
    }
    
    print("\nRunning graph with custom state...")
    print(f"Priority: {initial_state['request_priority']}")
    print(f"Domain: {initial_state['business_domain']}")
    
    try:
        # Run the graph
        final_state = await graph.ainvoke(initial_state)
        
        # Extract results
        if final_state.get("response"):
            print(f"\nResponse: {final_state['response']}")
            print(f"\nMetrics:")
            print(f"  - Model used: {final_state.get('current_model')}")
            print(f"  - Latency: {final_state.get('model_latencies', {}).get(final_state.get('current_model'), 'N/A')}s")
            print(f"  - Total duration: {(final_state.get('end_time') - final_state.get('start_time')).total_seconds():.2f}s")
            print(f"  - Attempted models: {final_state.get('attempted_models')}")
        else:
            print(f"Error: {final_state.get('error')}")
    except Exception as e:
        print(f"Error: {e}")


async def example_4_graph_visualization():
    """Example 4: Visualize the graph structure."""
    print("\n" + "="*60)
    print("Example 4: Graph Visualization")
    print("="*60)
    
    try:
        # Try to create a visual representation
        print("\nAttempting to generate graph visualization...")
        result = visualize_graph("langgraph_model_flow.png")
        
        if result:
            print("✅ Graph visualization saved to: langgraph_model_flow.png")
            print("   You can open this file to see the flow diagram")
        else:
            print("⚠️  Could not generate visualization (requires graphviz)")
            
        # Show graph structure in text
        print("\nGraph Structure:")
        print("  1. Initialize → Set defaults and correlation ID")
        print("  2. Validate Input → Check parameters")
        print("  3. Check Health → Verify model availability")
        print("  4. Select Model → Choose best available model")
        print("  5. Generate Response → Call model API")
        print("  6. Handle Error → Retry or fallback if needed")
        print("  7. Collect Metrics → Track performance")
        
    except Exception as e:
        print(f"Error: {e}")


async def main():
    """Run all examples."""
    print("\n🚀 LangGraph Model Examples")
    print("=" * 60)
    print("This demonstrates the LangGraph-based model orchestration")
    print("=" * 60)
    
    # Check configuration
    from model_langgraph.config import settings
    print("\n📋 Configuration Status:")
    print(f"  Anthropic: {'✅' if settings.anthropic_api_key else '❌'}")
    print(f"  DeepSeek: {'✅' if settings.deepseek_api_key else '❌'}")
    print(f"  OpenAI: {'✅' if settings.openai_api_key else '❌'}")
    
    if not any([settings.anthropic_api_key, settings.deepseek_api_key, settings.openai_api_key]):
        print("\n❌ No models configured! Please set API keys in settings.env")
        return
    
    # Run examples
    await example_1_basic_usage()
    await example_2_streaming()
    await example_3_direct_graph_usage()
    await example_4_graph_visualization()
    
    print("\n✅ All examples completed!")
    print("\nKey Benefits of LangGraph Architecture:")
    print("  • Visual flow representation")
    print("  • State-based execution tracking")
    print("  • Easy testing of individual nodes")
    print("  • Flexible routing and error handling")
    print("  • Built-in performance metrics")


if __name__ == "__main__":
    asyncio.run(main())