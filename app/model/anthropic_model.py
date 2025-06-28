"""
Anthropic Claude Model Integration

Provides async Claude API access using Pydantic settings.
"""

from typing import List, Dict, Any, Optional, Union
from anthropic import AsyncAnthropic

from .config import settings, get_prompt
from .model_logging import logger


class AnthropicModel:
    """
    Claude API client with async support.
    
    Uses Pydantic settings for configuration - much better than manual env loading!
    """
    
    def __init__(self):
        self.client = AsyncAnthropic(
            api_key=settings.anthropic_api_key  # From pydantic settings!
        )
        self.model = settings.anthropic_model
        self.enable_caching = settings.anthropic_enable_caching
        self.cache_system_prompt = settings.cache_system_prompt
        self.cache_schema_info = settings.cache_schema_info
        logger.info(f"Anthropic model initialized: {self.model}, caching: {self.enable_caching}")
    
    def _build_cached_messages(
        self, 
        prompt: str, 
        schema_info: Optional[Dict] = None,
        use_system_prompt: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Build messages with cache_control for prompt caching.
        
        Args:
            prompt: User prompt
            schema_info: Database schema information to cache
            use_system_prompt: Whether to include and cache system prompt
            
        Returns:
            Messages list with cache_control directives
        """
        if not self.enable_caching:
            # Return simple message structure if caching disabled
            return [{"role": "user", "content": prompt}]
        
        content_parts = []
        
        # Add cached system prompt if enabled
        if use_system_prompt and self.cache_system_prompt:
            sql_agent_prompt = get_prompt("sql_agent")
            content_parts.append({
                "type": "text",
                "text": sql_agent_prompt,
                "cache_control": {"type": "ephemeral"}
            })
        
        # Add cached schema info if provided
        if schema_info and self.cache_schema_info:
            schema_text = f"Database Schema:\n{schema_info}"
            content_parts.append({
                "type": "text", 
                "text": schema_text,
                "cache_control": {"type": "ephemeral"}
            })
        
        # Add user prompt (not cached)
        content_parts.append({
            "type": "text",
            "text": prompt
        })
        
        return [{"role": "user", "content": content_parts}]
    
    async def generate_response(
        self, 
        prompt: str, 
        max_tokens: int = 2048,
        temperature: float = 0.7,
        use_system_prompt: bool = True,
        schema_info: Optional[Dict] = None
    ) -> str:
        """
        Generate a response from Claude with prompt caching support.
        
        Args:
            prompt: The user prompt
            max_tokens: Maximum tokens in response
            temperature: Response creativity (0-1)
            use_system_prompt: Whether to include SQL agent system prompt
            schema_info: Database schema info to cache (optional)
            
        Returns:
            Claude's response text
        """
        try:
            # Build messages with caching if enabled
            if self.enable_caching and (use_system_prompt or schema_info):
                messages = self._build_cached_messages(prompt, schema_info, use_system_prompt)
                system_prompt = None  # System prompt is now in message content
                logger.debug("Using prompt caching for request")
            else:
                # Fallback to original behavior
                messages = [{"role": "user", "content": prompt}]
                system_prompt = get_prompt("sql_agent") if use_system_prompt else None
                logger.debug("Using standard request (no caching)")
            
            # Handle system prompt format for new Anthropic API
            api_params = {
                "model": self.model,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "messages": messages
            }
            
            # Add system prompt if provided (new API expects list format)
            if system_prompt:
                if isinstance(system_prompt, str):
                    api_params["system"] = [{"type": "text", "text": system_prompt}]
                else:
                    api_params["system"] = system_prompt
            
            response = await self.client.messages.create(**api_params)
            
            return response.content[0].text
            
        except Exception as e:
            logger.error(f"Anthropic API error: {e}")
            raise
    
    async def analyze_sql_query(self, query: str, schema_info: Dict) -> Dict[str, Any]:
        """
        Analyze a SQL query for investigation planning with cached schema.
        
        Args:
            query: User's SQL query or question
            schema_info: Database schema information (will be cached)
            
        Returns:
            Analysis with suggestions, intent, etc.
        """
        prompt = f"""
        Analyze this SQL query/question for autonomous investigation:
        
        Query: {query}
        
        Provide analysis in JSON format with:
        - intent: What the user wants to find
        - entities: Which tables/columns are relevant  
        - approach: How to investigate this
        - sql_suggestions: Suggested SQL queries
        - confidence: How confident you are (0-1)
        """
        
        try:
            # Pass schema_info for caching
            response = await self.generate_response(
                prompt, 
                max_tokens=1500,
                schema_info=schema_info
            )
            # In a real implementation, you'd parse this as JSON
            return {
                "raw_response": response,
                "analysis_type": "sql_query_analysis"
            }
            
        except Exception as e:
            logger.error(f"SQL analysis error: {e}")
            return {
                "error": str(e),
                "analysis_type": "sql_query_analysis"
            }
    
    async def synthesize_investigation_results(
        self, 
        original_query: str,
        findings: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Synthesize investigation findings into final insights.
        
        Args:
            original_query: Original user question
            findings: List of investigation step results
            
        Returns:
            Final synthesized insights
        """
        prompt = f"""
        Synthesize these investigation findings into clear insights:
        
        Original question: {original_query}
        
        Investigation findings: {findings}
        
        Provide synthesis in JSON format with:
        - key_insights: Main discoveries
        - answer: Direct answer to original question
        - confidence: How confident in the answer (0-1)
        - recommendations: What user should do next
        - data_quality_notes: Any data quality concerns
        """
        
        try:
            response = await self.generate_response(prompt, max_tokens=2000)
            return {
                "raw_response": response,
                "synthesis_type": "investigation_results"
            }
            
        except Exception as e:
            logger.error(f"Synthesis error: {e}")
            return {
                "error": str(e),
                "synthesis_type": "investigation_results"
            }
    
    async def health_check(self) -> bool:
        """
        Check if Anthropic API is accessible.
        
        Returns:
            True if API is working, False otherwise
        """
        try:
            # Use simple request without caching for health check
            health_prompt = get_prompt("health_check")
            response = await self.generate_response(
                health_prompt,
                max_tokens=10,
                use_system_prompt=False  # Disable system prompt for simple health check
            )
            return "OK" in response or "ok" in response.lower()
            
        except Exception as e:
            logger.error(f"Anthropic health check failed: {e}")
            return False


if __name__ == "__main__":
    async def main():
        model = AnthropicModel()
        print(await model.health_check())

    import asyncio
    asyncio.run(main())
