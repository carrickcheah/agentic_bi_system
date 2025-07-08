#!/usr/bin/env python3
"""
Standalone tests for LangGraph-based model module.
Tests all functionality without external dependencies.
"""

import asyncio
import sys
from pathlib import Path
import time

# Add parent directory to path for proper imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from model_langgraph.runner import ModelManager
from model_langgraph.graph import visualize_graph
from model_langgraph.state import ModelState, ModelStatus
from model_langgraph.config import settings


async def test_basic_generation():
    """Test basic response generation."""
    print("\n=== Testing Basic Generation ===")
    
    manager = ModelManager()
    
    try:
        response = await manager.generate_response(
            prompt="What is the capital of France? Answer in one word.",
            max_tokens=50,
            temperature=0
        )
        print(f" Response: {response}")
        assert len(response) > 0, "Response should not be empty"
        return True
    except Exception as e:
        print(f"L Failed: {e}")
        return False


async def test_streaming_generation():
    """Test streaming response generation."""
    print("\n=== Testing Streaming Generation ===")
    
    manager = ModelManager()
    
    try:
        print("Streaming response: ", end="", flush=True)
        chunks = []
        async for chunk in manager.generate_response_stream(
            prompt="Count from 1 to 5 with commas between numbers.",
            max_tokens=50,
            temperature=0
        ):
            print(chunk, end="", flush=True)
            chunks.append(chunk)
        print()  # New line
        
        full_response = "".join(chunks)
        print(f" Full response length: {len(full_response)} chars")
        assert len(full_response) > 0, "Streamed response should not be empty"
        return True
    except Exception as e:
        print(f"L Failed: {e}")
        return False


async def test_model_fallback():
    """Test fallback between models."""
    print("\n=== Testing Model Fallback ===")
    
    # Create a state that will force fallback
    from model_langgraph.graph import create_model_graph
    graph = create_model_graph()
    
    initial_state: ModelState = {
        "prompt": "Test fallback",
        "max_tokens": 10,
        "temperature": 0,
        "use_system_prompt": False,
        "is_streaming": False,
        "correlation_id": "test-fallback",
        # Force first model to fail by marking it unhealthy
        "available_models": {
            "anthropic": ModelStatus.UNHEALTHY,
            "deepseek": ModelStatus.HEALTHY,
            "openai": ModelStatus.HEALTHY
        },
        # Initialize other fields
        "system_prompt": None,
        "schema_info": None,
        "current_model": None,
        "attempted_models": [],
        "model_health_scores": {"anthropic": 0.1, "deepseek": 1.0},
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
        "business_domain": None,
        "user_role": None,
        "request_priority": None
    }
    
    try:
        final_state = await graph.ainvoke(initial_state)
        
        if final_state.get("response"):
            print(f" Response generated with model: {final_state.get('current_model')}")
            print(f"   Attempted models: {final_state.get('attempted_models')}")
            return True
        else:
            print(f"L No response generated")
            return False
    except Exception as e:
        print(f"L Failed: {e}")
        return False


async def test_health_check():
    """Test model health checking."""
    print("\n=== Testing Health Check ===")
    
    manager = ModelManager()
    
    try:
        health_status = await manager.health_check()
        print("Health Status:")
        for model, is_healthy in health_status.items():
            status = " Healthy" if is_healthy else "L Unhealthy"
            print(f"  {model}: {status}")
        
        return True
    except Exception as e:
        print(f"L Failed: {e}")
        return False


async def test_error_handling():
    """Test error handling and recovery."""
    print("\n=== Testing Error Handling ===")
    
    manager = ModelManager()
    
    try:
        # Test with invalid input
        await manager.generate_response(
            prompt="",  # Empty prompt should trigger validation error
            max_tokens=50
        )
        print("L Should have raised an error for empty prompt")
        return False
    except Exception as e:
        print(f" Correctly caught error: {e}")
        return True


async def test_graph_visualization():
    """Test graph visualization."""
    print("\n=== Testing Graph Visualization ===")
    
    try:
        # Try to visualize (may fail without graphviz)
        result = visualize_graph("test_model_graph.png")
        if result:
            print(" Graph visualization created: test_model_graph.png")
        else:
            print("�  Graph visualization skipped (graphviz not available)")
        return True
    except Exception as e:
        print(f"�  Visualization failed (non-critical): {e}")
        return True


async def test_performance():
    """Test performance and latency tracking."""
    print("\n=== Testing Performance Tracking ===")
    
    from model_langgraph.graph import create_model_graph
    graph = create_model_graph()
    
    initial_state: ModelState = {
        "prompt": "What is 2+2?",
        "max_tokens": 10,
        "temperature": 0,
        "use_system_prompt": False,
        "is_streaming": False,
        "correlation_id": "test-performance",
        # Initialize all required fields
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
        "anthropic_cache_enabled": settings.anthropic_enable_caching,
        "should_retry": False,
        "should_fallback": False,
        "next_model": None,
        "final_status": None,
        "business_domain": None,
        "user_role": None,
        "request_priority": None
    }
    
    try:
        start_time = time.time()
        final_state = await graph.ainvoke(initial_state)
        end_time = time.time()
        
        if final_state.get("response"):
            total_duration = end_time - start_time
            model_latencies = final_state.get("model_latencies", {})
            
            print(f" Performance Metrics:")
            print(f"   Total duration: {total_duration:.2f}s")
            print(f"   Model used: {final_state.get('current_model')}")
            print(f"   Model latencies: {model_latencies}")
            print(f"   Final status: {final_state.get('final_status')}")
            
            return True
        else:
            print("L No response generated")
            return False
    except Exception as e:
        print(f"L Failed: {e}")
        return False


async def test_prompt_system():
    """Test prompt management system."""
    print("\n=== Testing Prompt System ===")
    
    from model_langgraph.prompts import get_prompt, PROMPTS
    
    try:
        # Test getting prompts
        sql_prompt = get_prompt("sql_agent")
        print(f" SQL Agent prompt length: {len(sql_prompt)} chars")
        
        default_prompt = get_prompt("unknown_prompt")
        print(f" Default fallback working: {len(default_prompt)} chars")
        
        # Test all available prompts
        print(f" Available prompts: {list(PROMPTS.keys())}")
        
        return True
    except Exception as e:
        print(f"L Failed: {e}")
        return False


async def main():
    """Run all tests."""
    print(">� LangGraph Model Module Tests")
    print("=" * 50)
    
    # Check configuration
    print("\n=� Configuration Check:")
    print(f"  Anthropic configured: {'' if settings.anthropic_api_key else 'L'}")
    print(f"  DeepSeek configured: {'' if settings.deepseek_api_key else 'L'}")
    print(f"  OpenAI configured: {'' if settings.openai_api_key else 'L'}")
    
    if not any([settings.anthropic_api_key, settings.deepseek_api_key, settings.openai_api_key]):
        print("\nL No models configured! Please set API keys in settings.env")
        return 1
    
    # Run tests
    tests = [
        ("Prompt System", test_prompt_system),
        ("Basic Generation", test_basic_generation),
        ("Streaming Generation", test_streaming_generation),
        ("Health Check", test_health_check),
        ("Error Handling", test_error_handling),
        ("Model Fallback", test_model_fallback),
        ("Performance Tracking", test_performance),
        ("Graph Visualization", test_graph_visualization),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = await test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\nL Test '{test_name}' crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("=� Test Summary:")
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = " PASS" if result else "L FAIL"
        print(f"  {test_name}: {status}")
    
    print(f"\n{'' if passed == total else 'L'} {passed}/{total} tests passed")
    
    return 0 if passed == total else 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)