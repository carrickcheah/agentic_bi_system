#!/usr/bin/env python3
"""
Simple Interactive Chat Interface using Model Module
Direct Q&A with the model manager for business intelligence questions.
"""
import asyncio
import sys
from pathlib import Path
import logging

# Add app directory to path
sys.path.insert(0, str(Path(__file__).parent))

from model.runner import ModelManager

# Configure logging
logging.basicConfig(
    level=logging.WARNING,
    format='%(message)s'
)

async def main():
    """Run the simple chat interface."""
    print("Initializing Business Intelligence Assistant...")
    
    # Initialize the model manager
    try:
        model_manager = ModelManager()
        print(f"\nUsing {model_manager.get_current_model()} model")
        print(f"Available models: {', '.join(model_manager.get_available_models())}")
    except Exception as e:
        print(f"Error initializing models: {e}")
        print("Please check your API keys in app/model/settings.env")
        return
    
    print("\nHi! I am your Business Intelligence Assistant. How can I help you today?")
    print("(Type 'exit' or 'quit' to end the conversation)\n")
    
    while True:
        try:
            # Get user input
            user_query = input("User: ").strip()
            if not user_query:
                continue
                
            # Check for exit commands
            if user_query.lower() in ['exit', 'quit', 'bye']:
                print("\nGoodbye! Thank you for using Business Intelligence Assistant.")
                break
            
            # Generate response
            print("\nAssistant: ", end="", flush=True)
            
            try:
                # Create a business-focused prompt
                prompt = f"""You are a business intelligence analyst. The user has asked: "{user_query}"

Please provide a concise, actionable response that includes:
1. A brief answer to their question
2. Key insights or findings (if applicable)
3. Strategic recommendations (if applicable)

Keep the response focused and practical."""

                # Generate response using the model manager
                response = await model_manager.generate_response(
                    prompt=prompt,
                    max_tokens=1024,
                    temperature=0.7,
                    use_system_prompt=True
                )
                
                print(response)
                print()  # Empty line for readability
                
            except Exception as e:
                print(f"\nError generating response: {e}")
                print("Please try again.\n")
                
        except KeyboardInterrupt:
            print("\n\nExiting chat...")
            break
        except Exception as e:
            logging.error(f"Error in chat loop: {str(e)}")
            print(f"\nError: {str(e)}\n")

if __name__ == "__main__":
    # Run the chat interface
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nChat terminated.")
    except Exception as e:
        print(f"Fatal error: {e}")
        sys.exit(1)