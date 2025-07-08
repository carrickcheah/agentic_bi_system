#!/usr/bin/env python3
"""
Enhanced Interactive Chat with Proper Error Handling
Production-ready chat interface with comprehensive error handling and monitoring.
"""
import asyncio
import sys
from pathlib import Path
import logging
import os
from datetime import datetime
from typing import Optional, AsyncGenerator

# Add app directory to path
sys.path.insert(0, str(Path(__file__).parent))

# Import our new error handling system
from core.error_handling import (
    error_boundary, with_error_handling, validate_input,
    ValidationError, ExternalServiceError, ResourceExhaustedError,
    ErrorCategory, ErrorSeverity, error_tracker
)

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s - %(correlation_id)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/chat.log')
    ]
)

logger = logging.getLogger(__name__)

class EnhancedStreamingModelManager:
    """Production-ready model manager with proper error handling."""
    
    def __init__(self):
        self.correlation_id = None
        self.model_manager = None
        self._initialize_sync()
    
    def _initialize_sync(self):
        """Initialize model manager with error handling"""
        try:
            from model.runner import ModelManager
            self.model_manager = ModelManager()
        except Exception as e:
            logger.critical(
                "Failed to initialize ModelManager",
                extra={"error": str(e), "correlation_id": "init"}
            )
            raise ExternalServiceError(
                "Model system initialization failed",
                component="chat",
                operation="initialize",
                recoverable=False
            )
    
    def validate_input_parameters(self, prompt: str, max_tokens: int, temperature: float) -> dict:
        """Validate all input parameters"""
        def validator(data):
            prompt, max_tokens, temperature = data
            
            # Validate prompt
            if not prompt or not isinstance(prompt, str):
                raise ValueError("Prompt must be a non-empty string")
            if len(prompt.strip()) == 0:
                raise ValueError("Prompt cannot be empty")
            if len(prompt) > 50000:  # Reasonable limit
                raise ValueError("Prompt too long (max 50,000 chars)")
            
            # Validate max_tokens
            if not isinstance(max_tokens, int) or max_tokens <= 0:
                raise ValueError("max_tokens must be a positive integer")
            if max_tokens > 16000:
                raise ValueError("max_tokens exceeds maximum limit (16,000)")
            
            # Validate temperature
            if not isinstance(temperature, (int, float)):
                raise ValueError("temperature must be a number")
            if not 0.0 <= temperature <= 2.0:
                raise ValueError("temperature must be between 0.0 and 2.0")
            
            return {
                "prompt": prompt.strip(),
                "max_tokens": min(max_tokens, 16000),
                "temperature": max(0.0, min(temperature, 2.0))
            }
        
        return validate_input(
            (prompt, max_tokens, temperature),
            validator,
            operation="validate_chat_input",
            component="chat",
            correlation_id=self.correlation_id
        )
    
    @with_error_handling(
        operation="generate_streaming_response",
        component="chat",
        timeout_seconds=120.0  # 2 minute timeout
    )
    async def generate_response_stream(
        self,
        prompt: str,
        max_tokens: int = 2048,
        temperature: float = 0.7,
        use_system_prompt: bool = True,
        correlation_id: Optional[str] = None
    ) -> AsyncGenerator[str, None]:
        """Generate streaming response with comprehensive error handling."""
        
        self.correlation_id = correlation_id
        
        # Validate inputs
        validated_params = self.validate_input_parameters(prompt, max_tokens, temperature)
        
        logger.info(
            "Starting streaming response generation",
            extra={
                "correlation_id": correlation_id,
                "prompt_length": len(validated_params["prompt"]),
                "max_tokens": validated_params["max_tokens"],
                "temperature": validated_params["temperature"]
            }
        )
        
        model_name, model = self.model_manager.current_model
        fallback_attempted = False
        
        try:
            async for chunk in self._stream_from_model(
                model_name, model, validated_params, use_system_prompt
            ):
                yield chunk
                
        except ExternalServiceError as e:
            if not fallback_attempted:
                logger.warning(
                    f"Primary model {model_name} failed, attempting fallback",
                    extra={"correlation_id": correlation_id, "error": str(e)}
                )
                fallback_attempted = True
                
                # Try fallback models
                async for chunk in self._attempt_fallback(validated_params, use_system_prompt, model_name):
                    yield chunk
            else:
                # All models failed
                error_msg = f"All models failed. Last error: {str(e)}"
                logger.error(error_msg, extra={"correlation_id": correlation_id})
                yield f"\n❌ Error: {error_msg}\n"
    
    async def _stream_from_model(
        self, 
        model_name: str, 
        model, 
        params: dict, 
        use_system_prompt: bool
    ) -> AsyncGenerator[str, None]:
        """Stream from specific model with error handling."""
        
        try:
            if model_name == "anthropic":
                async for chunk in self._stream_anthropic(model, params, use_system_prompt):
                    yield chunk
                    
            elif model_name in ["deepseek", "openai"]:
                async for chunk in self._stream_openai_compatible(model_name, model, params, use_system_prompt):
                    yield chunk
                    
            else:
                # Fallback to non-streaming
                response = await self.model_manager.generate_response(
                    params["prompt"],
                    params["max_tokens"],
                    params["temperature"],
                    use_system_prompt
                )
                yield response
                
        except Exception as e:
            # Convert any error to our structured error type
            if "401" in str(e) or "authentication" in str(e).lower():
                raise ExternalServiceError(
                    f"{model_name} authentication failed",
                    service=model_name,
                    correlation_id=self.correlation_id,
                    component="chat",
                    operation="stream_response"
                )
            elif "429" in str(e) or "rate limit" in str(e).lower():
                raise ExternalServiceError(
                    f"{model_name} rate limited",
                    service=model_name,
                    correlation_id=self.correlation_id,
                    component="chat",
                    operation="stream_response",
                    retry_after=60
                )
            elif "timeout" in str(e).lower():
                raise ExternalServiceError(
                    f"{model_name} request timeout",
                    service=model_name,
                    correlation_id=self.correlation_id,
                    component="chat",
                    operation="stream_response",
                    retry_after=30
                )
            else:
                raise ExternalServiceError(
                    f"{model_name} service error: {str(e)}",
                    service=model_name,
                    correlation_id=self.correlation_id,
                    component="chat",
                    operation="stream_response"
                )
    
    async def _stream_anthropic(self, model, params: dict, use_system_prompt: bool):
        """Stream from Anthropic with proper error handling."""
        from model.config import get_prompt
        
        messages = [{"role": "user", "content": params["prompt"]}]
        api_params = {
            "model": model.model,
            "max_tokens": params["max_tokens"],
            "temperature": params["temperature"],
            "messages": messages
        }
        
        if use_system_prompt:
            system_prompt = get_prompt("sql_agent")
            api_params["system"] = [{"type": "text", "text": system_prompt}]
        
        # Add extended thinking if supported
        supports_thinking = any(ver in model.model for ver in ["claude-opus-4", "claude-sonnet-4", "claude-sonnet-3.7"])
        if model.enable_thinking and supports_thinking:
            api_params["thinking"] = {
                "type": "enabled",
                "budget_tokens": model.thinking_budget
            }
            api_params["temperature"] = 1.0
        
        async with model.client.messages.stream(**api_params) as stream:
            async for text in stream.text_stream:
                yield text
    
    async def _stream_openai_compatible(self, model_name: str, model, params: dict, use_system_prompt: bool):
        """Stream from OpenAI-compatible models."""
        from model.config import get_prompt
        
        messages = []
        if use_system_prompt:
            system_prompt = get_prompt("sql_agent")
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": params["prompt"]})
        
        stream = await model.client.chat.completions.create(
            model=model.model,
            messages=messages,
            max_tokens=params["max_tokens"],
            temperature=params["temperature"],
            stream=True
        )
        
        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
    
    async def _attempt_fallback(self, params: dict, use_system_prompt: bool, failed_model: str):
        """Attempt fallback to other models."""
        for fallback_name, fallback_model in self.model_manager.models:
            if fallback_name != failed_model:
                try:
                    logger.info(
                        f"Attempting fallback to {fallback_name}",
                        extra={"correlation_id": self.correlation_id}
                    )
                    
                    self.model_manager.current_model = (fallback_name, fallback_model)
                    
                    async for chunk in self._stream_from_model(
                        fallback_name, fallback_model, params, use_system_prompt
                    ):
                        yield chunk
                    return  # Success, exit fallback loop
                    
                except Exception as e:
                    logger.warning(
                        f"Fallback to {fallback_name} also failed: {str(e)}",
                        extra={"correlation_id": self.correlation_id}
                    )
                    continue
        
        # All models failed
        raise ExternalServiceError(
            "All models failed after fallback attempts",
            correlation_id=self.correlation_id,
            component="chat",
            operation="fallback_attempt",
            recoverable=False
        )

@with_error_handling(
    operation="chat_with_model",
    component="chat",
    timeout_seconds=300.0  # 5 minute timeout for full chat session
)
async def chat_with_model(correlation_id: Optional[str] = None):
    """Enhanced chat with proper error handling and monitoring."""
    
    logger.info("Starting chat session", extra={"correlation_id": correlation_id})
    
    try:
        model_manager = EnhancedStreamingModelManager()
        print("Initializing Enhanced Business Intelligence Assistant...")
        print(f"✅ Using {model_manager.model_manager.get_current_model()} model")
        print(f"📋 Available models: {', '.join(model_manager.model_manager.get_available_models())}")
        print("✨ Enhanced error handling and monitoring enabled")
        print("(Type 'exit' to quit)\n")
        
        session_stats = {
            "queries_processed": 0,
            "errors_encountered": 0,
            "total_chars_generated": 0
        }
        
        while True:
            try:
                user_query = input("User: ").strip()
                if not user_query:
                    continue
                    
                if user_query.lower() in ['exit', 'quit', 'bye']:
                    break
                
                # Generate new correlation ID for each query
                query_correlation_id = correlation_id + f"_q{session_stats['queries_processed']}"
                
                async with error_boundary(
                    operation="process_user_query",
                    component="chat",
                    correlation_id=query_correlation_id,
                    timeout_seconds=60.0
                ) as ctx:
                    
                    initial_model = model_manager.model_manager.get_current_model()
                    start_time = datetime.now()
                    
                    # Enhanced prompt with business focus
                    enhanced_prompt = f"""You are a business intelligence analyst. Answer this question concisely: "{user_query}"
Provide key insights and actionable recommendations where relevant."""
                    
                    print(f"\n🤔 Processing with {initial_model}...")
                    print("Assistant: ", end="", flush=True)
                    
                    char_count = 0
                    async for chunk in model_manager.generate_response_stream(
                        prompt=enhanced_prompt,
                        max_tokens=16000,
                        temperature=0.7,
                        use_system_prompt=True,
                        correlation_id=query_correlation_id
                    ):
                        print(chunk, end="", flush=True)
                        char_count += len(chunk)
                    
                    total_time = (datetime.now() - start_time).total_seconds()
                    final_model = model_manager.model_manager.get_current_model()
                    
                    # Update statistics
                    session_stats["queries_processed"] += 1
                    session_stats["total_chars_generated"] += char_count
                    
                    # Display metrics
                    if final_model != initial_model:
                        print(f"\n\n📊 {char_count} chars in {total_time:.1f}s | {final_model} (⚠️ fell back from {initial_model})")
                    else:
                        print(f"\n\n📊 {char_count} chars in {total_time:.1f}s | {final_model}")
                    
                    logger.info(
                        "Query processed successfully",
                        extra={
                            "correlation_id": query_correlation_id,
                            "response_chars": char_count,
                            "processing_time_seconds": total_time,
                            "model_used": final_model
                        }
                    )
                    print()
            
            except KeyboardInterrupt:
                print("\n\nChat interrupted by user")
                break
                
            except Exception as e:
                session_stats["errors_encountered"] += 1
                logger.error(
                    "Error processing query",
                    extra={
                        "correlation_id": correlation_id,
                        "error": str(e),
                        "error_type": type(e).__name__
                    }
                )
                print(f"\n❌ Error: {str(e)}\n")
    
    except Exception as e:
        logger.critical(
            "Critical error in chat session",
            extra={"correlation_id": correlation_id, "error": str(e)}
        )
        print(f"\n💥 Critical error: {str(e)}")
        raise
    
    finally:
        # Log session statistics
        logger.info(
            "Chat session ended",
            extra={
                "correlation_id": correlation_id,
                "session_stats": session_stats,
                "error_stats": error_tracker.get_error_stats()
            }
        )
        
        print(f"\n📈 Session Stats:")
        print(f"   Queries processed: {session_stats['queries_processed']}")
        print(f"   Errors encountered: {session_stats['errors_encountered']}")
        print(f"   Total chars generated: {session_stats['total_chars_generated']}")
        print("\nGoodbye!")

async def main():
    """Main entry point with comprehensive error handling."""
    import uuid
    correlation_id = str(uuid.uuid4())
    
    try:
        await chat_with_model(correlation_id=correlation_id)
    except KeyboardInterrupt:
        print("\nChat terminated by user.")
    except Exception as e:
        logger.critical(
            "Fatal error in chat application",
            extra={"correlation_id": correlation_id, "error": str(e)}
        )
        print(f"Fatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    # Ensure logs directory exists
    os.makedirs('logs', exist_ok=True)
    
    try:
        asyncio.run(main())
    except Exception as e:
        print(f"Failed to start chat: {e}")
        sys.exit(1)