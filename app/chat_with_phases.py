#!/usr/bin/env python3
"""
Interactive Chat with 5-Phase Orchestrator
Combines the model module with the 5-phase business analyst for comprehensive analysis.
"""
import asyncio
import sys
from pathlib import Path
import logging
import os

# Add app directory to path
sys.path.insert(0, str(Path(__file__).parent))

# Configure logging
logging.basicConfig(
    level=logging.WARNING,
    format='%(message)s'
)

async def chat_with_model():
    """Simple chat using just the model module with streaming support."""
    from model.runner import ModelManager
    from model.config import get_prompt
    from datetime import datetime
    
    class StreamingModelManager(ModelManager):
        """Extended ModelManager with streaming support for all models."""
        
        async def generate_response_stream(self, prompt, max_tokens=2048, temperature=0.7, use_system_prompt=True, schema_info=None):
            """Generate streaming response from any model."""
            model_name, model = self.current_model
            
            try:
                if model_name == "anthropic":
                    # Anthropic streaming with extended thinking support
                    messages = [{"role": "user", "content": prompt}]
                    system_prompt = get_prompt("sql_agent") if use_system_prompt else None
                    
                    api_params = {
                        "model": model.model,
                        "max_tokens": max_tokens,
                        "temperature": temperature,
                        "messages": messages
                    }
                    
                    if system_prompt:
                        api_params["system"] = [{"type": "text", "text": system_prompt}]
                    
                    # Add extended thinking if enabled
                    supports_thinking = any(ver in model.model for ver in ["claude-opus-4", "claude-sonnet-4", "claude-sonnet-3.7"])
                    if model.enable_thinking and supports_thinking:
                        api_params["thinking"] = {
                            "type": "enabled",
                            "budget_tokens": model.thinking_budget
                        }
                        api_params["temperature"] = 1.0  # Required for extended thinking
                        print("\n🧠 Extended thinking enabled...\n", flush=True)
                    
                    # Proper Anthropic async streaming implementation
                    async with model.client.messages.stream(**api_params) as stream:
                        async for text in stream.text_stream:
                            yield text
                            
                elif model_name in ["deepseek", "openai"]:
                    # DeepSeek and OpenAI streaming
                    messages = []
                    if use_system_prompt:
                        system_prompt = get_prompt("sql_agent")
                        messages.append({"role": "system", "content": system_prompt})
                    messages.append({"role": "user", "content": prompt})
                    
                    stream = await model.client.chat.completions.create(
                        model=model.model,
                        messages=messages,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        stream=True
                    )
                    
                    async for chunk in stream:
                        if chunk.choices and chunk.choices[0].delta.content:
                            yield chunk.choices[0].delta.content
                            
                else:
                    # Fallback to non-streaming
                    response = await self.generate_response(prompt, max_tokens, temperature, use_system_prompt, schema_info)
                    yield response
                    
            except Exception as e:
                print(f"\n⚠️ {model_name} failed: {str(e)[:200]}...")
                # Try all remaining models in order (avoid infinite recursion)
                original_model = model_name
                for fallback_name, fallback_model in self.models:
                    if fallback_name != original_model:
                        try:
                            print(f"\n🔄 Falling back to {fallback_name}...")
                            self.current_model = (fallback_name, fallback_model)
                            
                            # Use direct model call instead of recursive stream call
                            if fallback_name == "anthropic":
                                # Skip anthropic if it already failed
                                continue
                            elif fallback_name in ["deepseek", "openai"]:
                                # Try OpenAI-compatible models directly
                                messages = []
                                if use_system_prompt:
                                    system_prompt = get_prompt("sql_agent")
                                    messages.append({"role": "system", "content": system_prompt})
                                messages.append({"role": "user", "content": prompt})
                                
                                stream = await fallback_model.client.chat.completions.create(
                                    model=fallback_model.model,
                                    messages=messages,
                                    max_tokens=max_tokens,
                                    temperature=temperature,
                                    stream=True
                                )
                                
                                async for chunk in stream:
                                    if chunk.choices and chunk.choices[0].delta.content:
                                        yield chunk.choices[0].delta.content
                                return
                                
                        except Exception as e2:
                            print(f"\n⚠️ {fallback_name} also failed: {str(e2)[:100]}...")
                            continue
                            
                # If we get here, all models failed
                yield f"\n\nError: All models failed. Original error: {e}"
    
    print("Initializing Model-based Assistant...")
    model_manager = StreamingModelManager()
    print(f"✅ Using {model_manager.get_current_model()} model")
    print(f"📋 Available models: {', '.join(model_manager.get_available_models())}\n")
    
    print("Hi! I am your Business Intelligence Assistant (Direct Mode).")
    print("✨ Responses will stream in real-time")
    print("(Type 'exit' to quit)\n")
    
    while True:
        try:
            user_query = input("User: ").strip()
            if not user_query:
                continue
                
            if user_query.lower() in ['exit', 'quit', 'bye']:
                print("\nGoodbye!")
                break
            
            initial_model = model_manager.get_current_model()
            print(f"\n🤔 Processing with {initial_model}...")
            start_time = datetime.now()
            
            prompt = f"""You are a business intelligence analyst. Answer this question concisely: "{user_query}"
Provide key insights and actionable recommendations where relevant."""
            
            print("\nAssistant: ", end="", flush=True)
            
            char_count = 0
            async for chunk in model_manager.generate_response_stream(
                prompt=prompt,
                max_tokens=16000,  # Must be > thinking_budget (10000)
                temperature=0.7,
                use_system_prompt=True
            ):
                print(chunk, end="", flush=True)
                char_count += len(chunk)
            
            total_time = (datetime.now() - start_time).total_seconds()
            final_model = model_manager.get_current_model()
            
            if final_model != initial_model:
                print(f"\n\n📊 {char_count} chars in {total_time:.1f}s | {final_model} model (⚠️ fell back from {initial_model})")
            else:
                print(f"\n\n📊 {char_count} chars in {total_time:.1f}s | {final_model} model")
            print()
            
        except KeyboardInterrupt:
            print("\n\nExiting...")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}\n")

async def chat_with_phases():
    """Chat using the full 5-phase orchestrator."""
    # Set required environment variables if not present
    if not os.getenv("ANTHROPIC_API_KEY"):
        # Try to load from model settings
        model_settings_path = Path(__file__).parent / "model" / "settings.env"
        if model_settings_path.exists():
            print("Loading API keys from model settings...")
            with open(model_settings_path) as f:
                for line in f:
                    if line.strip() and not line.startswith("#"):
                        key, value = line.strip().split("=", 1)
                        os.environ[key] = value
    
    try:
        from core.business_analyst import AutonomousBusinessAnalyst
        
        print("Initializing 5-Phase Business Intelligence System...")
        analyst = AutonomousBusinessAnalyst()
        await analyst.initialize()
        
        print("\nHi! I am your Business Intelligence Assistant (5-Phase Analysis).")
        print("I will analyze your questions through:")
        print("  1. Query Understanding")
        print("  2. Strategic Planning")
        print("  3. Service Orchestration")
        print("  4. Investigation Execution")
        print("  5. Insight Synthesis")
        print("\n(Type 'exit' to quit)\n")
        
        while True:
            try:
                user_query = input("User: ").strip()
                if not user_query:
                    continue
                    
                if user_query.lower() in ['exit', 'quit', 'bye']:
                    print("\nGoodbye!")
                    break
                
                print("\nAnalyzing through 5 phases...")
                
                result = await analyst.conduct_investigation(
                    business_question=user_query,
                    user_context={},
                    stream_progress=False
                )
                
                print("\nAssistant:")
                
                if "executive_summary" in result:
                    print(f"\n{result['executive_summary']}")
                
                if "key_findings" in result and result["key_findings"]:
                    print("\nKey Findings:")
                    for i, finding in enumerate(result["key_findings"], 1):
                        print(f"  {i}. {finding}")
                
                if "strategic_recommendations" in result and result["strategic_recommendations"]:
                    print("\nRecommendations:")
                    for i, rec in enumerate(result["strategic_recommendations"], 1):
                        if isinstance(rec, dict):
                            print(f"  {i}. {rec.get('recommendation', rec)}")
                        else:
                            print(f"  {i}. {rec}")
                
                print()
                
            except KeyboardInterrupt:
                print("\n\nExiting...")
                break
            except Exception as e:
                print(f"\nError in 5-phase analysis: {e}")
                print("Falling back to direct model mode...\n")
                await chat_with_model()
                break
                
        await analyst.cleanup()
        
    except ImportError as e:
        print(f"Cannot load 5-phase orchestrator: {e}")
        print("Falling back to direct model mode...\n")
        await chat_with_model()
    except Exception as e:
        print(f"Error initializing 5-phase system: {e}")
        print("Falling back to direct model mode...\n")
        await chat_with_model()

async def main():
    """Main entry point with mode selection."""
    print("Business Intelligence Chat Interface")
    print("=" * 50)
    print("\nSelect mode:")
    print("1. Direct Model (Simple Q&A)")
    print("2. 5-Phase Analysis (Comprehensive investigation)")
    print("3. Exit")
    
    choice = input("\nEnter choice (1-3): ").strip()
    
    if choice == "1":
        await chat_with_model()
    elif choice == "2":
        await chat_with_phases()
    elif choice == "3":
        print("Goodbye!")
    else:
        print("Invalid choice. Starting direct mode...")
        await chat_with_model()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nChat terminated.")
    except Exception as e:
        print(f"Fatal error: {e}")
        sys.exit(1)