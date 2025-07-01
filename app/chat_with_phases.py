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
    """Simple chat using just the model module."""
    from model.runner import ModelManager
    
    print("Initializing Model-based Assistant...")
    model_manager = ModelManager()
    print(f"Using {model_manager.get_current_model()} model\n")
    
    print("Hi! I am your Business Intelligence Assistant (Direct Mode).")
    print("(Type 'exit' to quit)\n")
    
    while True:
        try:
            user_query = input("User: ").strip()
            if not user_query:
                continue
                
            if user_query.lower() in ['exit', 'quit', 'bye']:
                print("\nGoodbye!")
                break
            
            print("\nAssistant: ", end="", flush=True)
            
            prompt = f"""You are a business intelligence analyst. Answer this question concisely: "{user_query}"
Provide key insights and actionable recommendations where relevant."""
            
            response = await model_manager.generate_response(
                prompt=prompt,
                max_tokens=1024,
                temperature=0.7,
                use_system_prompt=True
            )
            
            print(response)
            print()
            
        except KeyboardInterrupt:
            print("\n\nExiting...")
            break
        except Exception as e:
            print(f"\nError: {e}\n")

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